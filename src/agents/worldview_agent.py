"""Worldview Agent - 独立世界观工作流。

符合需求：Input -> Initial Expansion -> World Rule Review -> Worldview Consistency Review -> Human -> Commit；
人工不同意或任一审查失败进入 Modify Content，再回到 World Rule Review。
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.review_nodes.world_review import make_world_review_node, make_world_review_route
from src.agents.review_nodes.worldview_review import make_worldview_review_node, make_worldview_review_route
from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, get_unified_context, parse_json_safely
from src.common.revision_mode_prompt import build_revision_mode_instruction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


AGENT_NAME = "worldview_agent"
INITIAL_EXPANSION_AGENT_NAME = "worldview_agent_initial_expansion"
MODIFY_CONTENT_AGENT_NAME = "worldview_agent_modify_content"
HUMAN_FEEDBACK_AGENT_NAME = "worldview_agent_human_feedback_modify_content"
ENTITY_TYPE = "worldview"
PRIMARY_FIELD = "summary"
MAX_AUTO_REVIEW_ITERATIONS = 3
WORKFLOW_DESCRIPTION = "世界观 Agent 流程：接收设定输入 -> 初始扩充设定内容 -> 世界规则审查 -> 既有世界观一致性审查 -> 人工确认 -> 批准后写入该世界唯一 worldview 库下的 lore(type=worldview) 设定条目；人工不同意或任一审查失败进入修改内容节点，再从世界规则审查重新开始。"
WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收世界观输入",
        "function": "接收设定条目 payload 与父级 world_id",
        "description": "记录世界观名称、摘要、world_id、worldview_id 或 target_id，以及用户消息，确保本次任务只处理该世界下的 Canon 设定。",
    },
    "initial_expansion": {
        "step_index": 2,
        "step_title": "步骤 2：初始扩充",
        "function": "调用 worldview_agent 专属 LLM 整理世界观输入",
        "description": "对碎片化设定进行初步补全并直接生成可审查的世界观 payload，明确条目名称、分类、核心规则、父级 world_id、需要检索的 Canon 关键词和不可改写的用户原意；不得写库，不得跳过 LLM，不得使用通用 Prompt。",
    },
    "world_rule_review": {
        "step_index": 3,
        "step_title": "步骤 3：世界规则审查",
        "function": "检查是否违反世界禁止规则与基本设定",
        "description": "基于 worlds 中的 forbidden_rules 与 basic_settings 审查当前世界观内容是否违反世界根禁令、基础时代、力量体系、地理边界、资源机制或组织结构；失败时写入 world_rule_review_feedback 并进入修改内容节点。",
    },
    "worldview_consistency_review": {
        "step_index": 4,
        "step_title": "步骤 4：既有世界观一致性审查",
        "function": "检查是否违反同一世界下已有世界观设定",
        "description": "基于同一 world_id 下已经入库的世界观设定审查 Canon 冲突、逻辑漏洞和前后矛盾；失败时写入 worldview_consistency_feedback 并进入修改内容节点。",
    },
    "human": {
        "step_index": 5,
        "step_title": "步骤 5：人工确认",
        "function": "等待用户批准、重写或中止",
        "description": "两个审查节点都通过后进入人工节点。用户可批准写库，也可选择修改模式并提交反馈进入修改内容节点；修改后必须重新通过两个审查节点。",
    },
    "modify_content": {
        "step_index": 6,
        "step_title": "步骤 6：修改内容",
        "function": "根据审查意见或人工反馈调用 worldview_agent 专属 LLM 修改世界观内容",
        "description": "仅在世界规则审查失败、既有世界观一致性审查失败或人工不同意时执行。根据审查反馈或用户反馈修正世界观 payload，保留未要求修改的内容和业务 ID；不得写库，不得使用通用 Prompt。",
    },
    "commit": {
        "step_index": 7,
        "step_title": "步骤 7：写库固化",
        "function": "执行 worldview 设定条目写入",
        "description": "人工批准后写入或更新该世界唯一 worldview 库下的 MongoDB lore(type=worldview) 设定条目，并保留 world_id、worldview_id 和真实写库结果。",
    },
}
NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入必须包含 world_id，并可包含 worldview_id 或 target_id；本节点确认任务只处理该世界下的世界观设定。",
        "output_annotation": "输出 accepted=true，并把 payload 固定为后续初始扩充与审查的基准。",
        "next_step_annotation": "下一步进入初始扩充节点，先整理设定意图、分类和 Canon 检索关键词。",
    },
    "initial_expansion": {
        "input_annotation": "输入是用户消息、世界观 payload、人工反馈和修改模式；必须保留 world_id 和用户原始设定。",
        "output_annotation": "输出可审查的世界观 payload、expanded_input、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步进入世界规则审查，先检查是否违反世界禁止规则与基本设定。",
    },
    "world_rule_review": {
        "input_annotation": "输入是当前世界观 payload、world_id 和 worlds 中的 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 world_rule_review_feedback。",
        "next_step_annotation": "通过则进入既有世界观一致性审查；失败且未超出自动迭代上限则进入修改内容节点。",
    },
    "worldview_consistency_review": {
        "input_annotation": "输入是通过世界规则审查后的世界观 payload 和同一世界下已有世界观上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 worldview_consistency_feedback。",
        "next_step_annotation": "通过则进入人工确认；失败且未超出自动迭代上限则进入修改内容节点。",
    },
    "human": {
        "input_annotation": "输入是两次审查通过后的世界观内容、审查意见、用户决策和修改模式。",
        "output_annotation": "输出记录 approve、request_changes 或 reject，以及用户反馈内容。",
        "next_step_annotation": "批准则写库；要求修改则进入修改内容节点并重新审查；中止则结束。",
    },
    "modify_content": {
        "input_annotation": "输入是当前世界观 payload、world_rule_review_feedback、worldview_consistency_feedback、人工反馈和修改模式。",
        "output_annotation": "输出修改后的世界观 payload、llm_call、raw_response、parsed_response 和 change_summary。",
        "next_step_annotation": "下一步回到世界规则审查，必须连续通过两个审查节点后才能进入人工确认。",
    },
    "commit": {
        "input_annotation": "输入是人工批准后的最终 worldview 设定 payload，必须挂到该世界唯一 worldview 库下。",
        "output_annotation": "输出是真实 MongoDB lore(type=worldview) 写入结果，包含设定条目 id、world_id 与 worldview_id。",
        "next_step_annotation": "写库完成后工作流结束。",
    },
}


class WorldviewAgentState(TypedDict, total=False):
    action: str
    message: str
    payload: Dict[str, Any]
    pending_payload: Dict[str, Any]
    feedback: str
    review_feedback: str
    revision_mode: str
    decision: str
    manual_edit: bool
    expanded_input: Dict[str, Any]
    initial_expansion: Dict[str, Any]
    modification: Dict[str, Any]
    review_passed: bool
    review_errors: List[str]
    world_rule_review_passed: bool
    world_rule_review_errors: List[str]
    world_rule_review_feedback: str
    worldview_consistency_passed: bool
    worldview_consistency_errors: List[str]
    worldview_consistency_feedback: str
    nodes: List[Dict[str, Any]]
    conversation: List[Dict[str, Any]]
    iterations: int
    status: str
    current_node: str
    commit_result: Dict[str, Any]
    committed: bool


def _extract_llm_content(response: Any) -> str:
    """提取 LLM 返回正文，兼容字符串、Message 和分段 content。"""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "")


def _llm_metadata(raw_content: str, prompt: str, llm_agent_name: str) -> Dict[str, Any]:
    """生成本次 worldview_agent LLM 调用的中文可审计元数据。"""
    config = get_config()
    provider = str(config.get("LLM_PROVIDER", "ollama")).lower()
    agent_config = (config.get("AGENT_MODELS") or {}).get(llm_agent_name) or {}
    if not agent_config:
        agent_config = (config.get("AGENT_MODELS") or {}).get(AGENT_NAME) or {}
    model_name = agent_config.get("model") if isinstance(agent_config, dict) else agent_config
    provider_config = (config.get("LLM_MODELS") or {}).get(provider) or {}
    if isinstance(provider_config, dict) and not model_name:
        model_name = provider_config.get("default")
    return {"llm_invoked": True, "llm_agent_name": llm_agent_name, "provider": provider, "model": model_name or config.get("DEFAULT_MODEL"), "json_mode": True, "raw_response_chars": len(raw_content), "prompt": prompt, "prompt_chars": len(prompt)}


def _invoke_llm(prompt: str, *, llm_agent_name: str) -> tuple[str, Dict[str, Any]]:
    """真实调用 worldview_agent 对应 LLM；空响应直接报错，禁止伪成功。"""
    llm = get_llm(json_mode=True, agent_name=llm_agent_name)
    config: Dict[str, Any] = {}
    callback = get_langfuse_callback()
    if callback:
        config["callbacks"] = [callback]
    response = llm.invoke(prompt, config=config if config else None)
    raw_content = _extract_llm_content(response)
    if not raw_content.strip():
        raise ValueError(f"{llm_agent_name} returned empty LLM response")
    return raw_content, _llm_metadata(raw_content, prompt, llm_agent_name)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    """构造带中文步骤说明、节点注解、输入输出说明的工作流节点。"""
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    """构造 worldview_agent 初始扩充 Prompt，使用结构化模板明确世界观任务。"""
    world_id = str(payload.get("world_id", "") or "")
    world_rules = _load_world_forbidden_rules(world_id)
    return f"""【角色设定】
你是一名世界观编辑和设定守门人。你的唯一职责是整理和扩充世界观条目，确保所有内容严格遵循父级世界的禁止规则。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查世界观草稿与世界禁止规则是否冲突，确认 name、summary、world_id 和业务 ID 是否完整、自洽。
2. 扩展（Expand）：在不偏离用户原始设定的前提下，扩充 summary，补全分类、核心规则、佳能关键词和冲突风险。

【输入信息】
【所属世界 world_id】{world_id}
【业务动作】{action}
【用户消息】{message}
【世界观草稿】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}
【世界禁止规则】
{json.dumps(world_rules, ensure_ascii=False, indent=2)}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留用户原始设定、父级 world_id、worldview_id、target_id。
3. 必须扩充世界观 `summary`，并补全分类、核心规则、佳能关键词和冲突风险。
4. 所有扩展都不得违反世界禁止规则。

只返回合法 JSON：
{{
  "metadata": {{"agent": "worldview_agent", "node": "initial_expansion", "entity_type": "worldview", "action": "{action}"}},
  "payload": {{
    "world_id": "{world_id}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "世界观名称",
    "summary": "构造世界观设定边界"
  }},
  "expanded_input": {{
    "world_id": "{world_id}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "{payload.get("name", "")}",
    "summary_seed": "{payload.get("summary", "")}",
    "category": "从用户输入提炼的设置分类",
    "canon_keywords": ["需要搜索的正典关键词"],
    "must_keep": ["不可改写的用户原意"],
    "review_focus": "审查节点需要重点检查的佳能约束"
  }},
  "expansion_notes": "自然资源新增整理了哪些设定约束"
}}
"""


def _load_world_forbidden_rules(world_id: str) -> List[str]:
    """读取父级世界的禁止规则；缺失时回退到世界观修改节点默认硬约束。"""
    default_rules = [
        "不得出现魔法",
        "不得出现神",
        "不得超越科学，出现法则，创造物质，创建世界等",
    ]
    if not world_id:
        return default_rules
    world = get_mongodb_db()["worlds"].find_one({"world_id": world_id}) or {}
    forbidden_rules = world.get("forbidden_rules")
    if isinstance(forbidden_rules, list) and forbidden_rules:
        return [str(rule) for rule in forbidden_rules if str(rule).strip()]
    return default_rules


def _ensure_worldview_library(db: Any, world_id: str) -> Dict[str, Any]:
    """确保父级世界存在唯一 worldview 库；缺失时按世界自动补建。"""
    worldview = db["worldviews"].find_one({"world_id": world_id})
    if worldview:
        return worldview
    world = db["worlds"].find_one({"world_id": world_id})
    if not world:
        raise ValueError(f"Parent world not found for worldview setting: {world_id}")
    worldview = {
        "worldview_id": f"wv_{world_id}",
        "world_id": world_id,
        "name": f"{world.get('name', world_id)} 世界观设定集",
        "summary": world.get("summary", ""),
        "forbidden_rules": [],
        "basic_settings": {},
        "auto_created": True,
        "created_at": _now(),
        "updated_at": _now(),
    }
    db["worldviews"].insert_one(worldview)
    return worldview


def _normalize_hierarchy_path(raw_path: Any) -> List[str]:
    """统一世界观层级路径写法，始终使用 `A > B > C` 语义。"""
    if raw_path in (None, ""):
        return []
    return [part.strip() for part in str(raw_path).replace("/", ">").split(">") if part.strip()]


def _resolve_worldview_entry_path(
    payload: Dict[str, Any],
    *,
    entry_name: str,
    fallback_path: Any = "",
    existing_name: str = "",
) -> str:
    """根据 parent_path/path/category 解析世界观设定条目的最终层级路径。"""
    explicit_path = _normalize_hierarchy_path(payload.get("path"))
    explicit_parent = _normalize_hierarchy_path(payload.get("parent_path"))
    category_parts = _normalize_hierarchy_path(payload.get("category"))
    fallback_parts = _normalize_hierarchy_path(fallback_path)

    if explicit_parent:
        full_parts = explicit_parent + [entry_name]
    elif explicit_path:
        if explicit_path[-1] == entry_name:
            full_parts = explicit_path
        elif existing_name and explicit_path[-1] == existing_name:
            full_parts = explicit_path[:-1] + [entry_name]
        else:
            full_parts = explicit_path + [entry_name]
    elif category_parts:
        full_parts = category_parts if category_parts[-1] == entry_name else category_parts + [entry_name]
    elif fallback_parts:
        if fallback_parts[-1] == entry_name:
            full_parts = fallback_parts
        elif existing_name and fallback_parts[-1] == existing_name:
            full_parts = fallback_parts[:-1] + [entry_name]
        else:
            full_parts = fallback_parts + [entry_name]
    else:
        full_parts = [entry_name]

    return " > ".join(full_parts)


def _preserve_routing_fields(source_payload: Dict[str, Any], target_payload: Dict[str, Any]) -> None:
    """保留层级挂载必需字段，禁止 LLM 草案把世界观目录上下文改丢。"""
    for key in ("world_id", "worldview_id", "target_id", "parent_path", "path"):
        if source_payload.get(key):
            target_payload[key] = source_payload[key]


def generate_initial_expansion(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "") -> Dict[str, Any]:
    """调用 LLM 生成世界观初始扩充结果，确保第二节点真实使用 worldview_agent LLM。"""
    prompt = build_initial_expansion_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=INITIAL_EXPANSION_AGENT_NAME)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} initial expansion returned non-object JSON: {raw_content[:500]}")
    expanded_input = parsed.get("expanded_input") or {}
    initial_payload = parsed.get("payload")
    if not isinstance(initial_payload, dict):
        initial_payload = parsed.get("有效载荷")
    if not isinstance(initial_payload, dict):
        initial_payload = expanded_input
    if not isinstance(initial_payload, dict):
        raise ValueError(f"{AGENT_NAME} initial expansion missing payload object: {raw_content[:500]}")
    if not isinstance(expanded_input, dict):
        expanded_input = {}
    _preserve_routing_fields(payload, initial_payload)
    _preserve_routing_fields(payload, expanded_input)
    if payload.get("name") and revision_mode != "full_rewrite":
        initial_payload["name"] = payload["name"]
    return {"payload": initial_payload, "expanded_input": expanded_input, "llm_invoked": True, "agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    """构造 worldview_agent 修改内容 Prompt，使用结构化模板明确局部修正任务。"""
    rag_context = get_unified_context(
        f"{message}\n{payload.get('name', '')}\n{payload.get('summary', '')}",
        worldview_id=str(payload.get("worldview_id") or payload.get("target_id") or "default_wv"),
    )
    world_id = str(payload.get("world_id", "") or "")
    world_rules = _load_world_forbidden_rules(world_id)
    review_feedback = feedback or expansion_error or "无额外修改意见。"
    retry_clause = f"\n【附加审查失败原因】\n{expansion_error}\n" if expansion_error else ""
    mode_instruction = build_revision_mode_instruction(
        revision_mode,
        entity_label="世界观条目",
        primary_field_label="payload.summary",
        content_rewrite_scope="允许围绕世界观摘要 summary 做大段重写；如 name 未被明确要求修改，默认保留原值。",
        full_rewrite_scope="允许整体重写 name、summary 和 expanded_input 中的分类与佳能关键词等世界观业务内容。",
    )
    return f"""【角色设定】
你是一名世界观编辑和设定守门人。你的唯一职责是修正世界观条目，确保所有内容严格遵循父级世界的禁止规则与已有设定。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查当前世界观与世界禁止规则、审查意见、人工反馈之间的冲突点。
2. 修正（Modify）：根据【修改意见】和下面的【修改模式说明】修正世界观条目。
3. 扩展（Expand）：仅在修正完成后，补充必要细节、分类、核心规则和佳能关键词，但不得偏离反馈要求。

【输入信息】
【所属世界 world_id】{world_id}
【业务动作】{action}
【修改模式】{revision_mode or "partial_rewrite"}
【用户消息】{message}
【世界观草稿】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}
【修改意见】
{review_feedback}
【修改模式说明】
{mode_instruction}
【世界禁止规则】
{json.dumps(world_rules, ensure_ascii=False, indent=2)}{retry_clause}
【参考上下文】
{rag_context}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留 `world_id`、`worldview_id`、`target_id`；`name` 仅允许在 `full_rewrite` 或用户明确点名时修改。
3. `summary` 必须消除超自然、超科学、违反世界禁止规则的内容。
4. 补充细节时不得引入新的设定漂移或逻辑冲突。

最终输出必须严格遵守以下结构：
{{
  "metadata": {{
    "agent": "worldview_agent",
    "node": "modify_content",
    "entity_type": "worldview",
    "action": "{action}"
  }},
  "payload": {{
    "world_id": "{world_id}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "世界观名称",
    "summary": "修改后的世界观摘要"
  }},
  "expanded_input": {{
    "world_id": "{world_id}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "世界观名称",
    "summary_seed": "修改后的摘要种子",
    "category": "提炼的分类",
    "canon_keywords": ["搜索关键词"],
    "must_keep": ["不可改写的用户原意/硬性规则"],
    "review_focus": "审查节点需要重点检查的佳能约束"
  }},
  "expansion_notes": "根据操作流程执行后的总结和补充说明。"
}}
"""


def generate_content_modification(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "", expansion_error: str = "", llm_agent_name: Optional[str] = None) -> Dict[str, Any]:
    """调用 LLM 根据审查意见或人工反馈修改世界观内容。"""
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback, expansion_error=expansion_error)
    agent_name = llm_agent_name or MODIFY_CONTENT_AGENT_NAME
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=agent_name)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} modification returned non-object JSON: {raw_content[:500]}")
    modified_payload = parsed.get("payload")
    if not isinstance(modified_payload, dict):
        modified_payload = parsed.get("有效载荷")
    if not isinstance(modified_payload, dict):
        raise ValueError(f"{AGENT_NAME} modification missing payload object: {raw_content[:500]}")
    expanded_input = parsed.get("expanded_input")
    if not isinstance(expanded_input, dict):
        expanded_input = {}
    expansion_notes = str(parsed.get("expansion_notes") or "")
    _preserve_routing_fields(payload, modified_payload)
    _preserve_routing_fields(payload, expanded_input)
    if payload.get("name") and revision_mode != "full_rewrite":
        modified_payload["name"] = payload["name"]
    return {
        "payload": modified_payload,
        "expanded_input": expanded_input,
        "llm_invoked": True,
        "agent_name": agent_name,
        "llm_agent_name": agent_name,
        "llm_call": llm_call,
        "raw_response": raw_content,
        "parsed_response": parsed,
        "expansion_notes": expansion_notes,
        "modification_notes": parsed.get("modification_notes", "") or expansion_notes,
        "change_summary": parsed.get("change_summary", "") or expansion_notes,
    }


def input_node(state: WorldviewAgentState) -> WorldviewAgentState:
    """输入节点：记录世界观 payload、父级 world_id 和用户消息。"""
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: WorldviewAgentState) -> WorldviewAgentState:
    """初始扩充节点：调用 worldview_agent LLM 扩充世界观内容并提交世界规则审查。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=state.get("feedback", ""))
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    return {"initial_expansion": expansion, "expanded_input": expansion["expanded_input"], "pending_payload": expansion["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_rule_review", "status": "reviewing_world_rules"}


def modify_content_node(state: WorldviewAgentState) -> WorldviewAgentState:
    """修改内容节点：按审查意见或人工反馈调用 worldview_agent LLM 修改世界观内容。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    manual_edit = bool(state.get("manual_edit"))
    user_feedback = str(state.get("feedback") or "")
    review_feedback = str(
        state.get("world_rule_review_feedback")
        or state.get("worldview_consistency_feedback")
        or state.get("review_feedback")
        or ""
    )
    feedback = user_feedback if manual_edit and user_feedback else review_feedback or user_feedback
    llm_agent_name = HUMAN_FEEDBACK_AGENT_NAME if manual_edit else MODIFY_CONTENT_AGENT_NAME
    modification = generate_content_modification(
        state.get("action", "create"),
        payload,
        state.get("message", ""),
        revision_mode=state.get("revision_mode"),
        feedback=feedback,
        expansion_error=feedback,
        llm_agent_name=llm_agent_name,
    )
    iteration = int(state.get("iterations") or 0) + 1
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("modify_content", "completed", {"payload": payload, "feedback": feedback, "revision_mode": state.get("revision_mode"), "manual_edit": manual_edit}, {**modification, "iteration": iteration}))
    return {"modification": modification, "pending_payload": modification["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_rule_review", "status": "reviewing_world_rules"}


world_rule_review_node = make_world_review_node(
    node_id="world_rule_review",
    entity_type="worldview_world_rules",
    reviewer="worldview_world_rules_review_agent",
    passed_key="world_rule_review_passed",
    errors_key="world_rule_review_errors",
    feedback_key="world_rule_review_feedback",
    next_node="worldview_consistency_review",
    next_status="reviewing_worldview_consistency",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_world_rule_review = make_world_review_route(
    passed_key="world_rule_review_passed",
    next_node="worldview_consistency_review",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)
worldview_consistency_review_node = make_worldview_review_node(
    node_id="worldview_consistency_review",
    entity_type="worldview_consistency",
    reviewer="worldview_consistency_review_agent",
    passed_key="worldview_consistency_passed",
    errors_key="worldview_consistency_errors",
    feedback_key="worldview_consistency_feedback",
    next_node="human",
    next_status="waiting_human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_worldview_consistency_review = make_worldview_review_route(
    passed_key="worldview_consistency_passed",
    next_node="human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)


def human_node(state: WorldviewAgentState) -> WorldviewAgentState:
    """人工节点：等待批准、重写或中止，并记录反馈与修改模式。"""
    decision = state.get("decision")
    feedback = state.get("feedback", "")
    revision_mode = state.get("revision_mode") or "partial_rewrite"
    if not decision:
        user_input = interrupt({
            "agent": AGENT_NAME,
            "status": "waiting_human",
            "payload": state.get("pending_payload"),
            "review_errors": state.get("review_errors", []),
            "world_rule_review_errors": state.get("world_rule_review_errors", []),
            "worldview_consistency_errors": state.get("worldview_consistency_errors", []),
            "actions": ["approve", "request_changes", "reject"],
            "revision_modes": ["partial_rewrite", "content_rewrite", "full_rewrite"],
        })
        if isinstance(user_input, dict):
            decision = user_input.get("decision")
            feedback = user_input.get("feedback", "")
            revision_mode = user_input.get("revision_mode") or revision_mode
        else:
            decision = "approve" if str(user_input).lower() in {"approve", "批准", "ok", "yes"} else "request_changes"
            feedback = str(user_input)
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("human", "completed", {"decision": decision, "feedback": feedback, "revision_mode": revision_mode}, {"received": True}))
    return {"decision": decision, "feedback": feedback, "revision_mode": revision_mode, "nodes": nodes}


def route_after_human(state: WorldviewAgentState) -> str:
    """人工节点路由：批准进入写库，要求修改进入修改内容节点，中止结束。"""
    if state.get("decision") == "approve":
        return "commit"
    if state.get("decision") == "reject":
        return "end"
    return "modify_content"


def commit_node(state: WorldviewAgentState) -> WorldviewAgentState:
    """写库节点：人工批准后真实创建或更新该世界唯一 worldview 库下的 lore 设定条目。"""
    db = get_mongodb_db()
    action = state.get("action", "create")
    payload = dict(state.get("pending_payload") or {})
    if action == "create":
        world_id = str(payload["world_id"])
        worldview_library = _ensure_worldview_library(db, world_id)
        entry_id = str(payload.get("id") or payload.get("entry_id") or f"wv_entry_{uuid.uuid4().hex[:8]}")
        if db["lore"].find_one({"id": entry_id}):
            raise ValueError(f"Worldview setting already exists: {entry_id}")
        entry_name = str(payload["name"]).strip()
        hierarchy_path = _resolve_worldview_entry_path(payload, entry_name=entry_name)
        doc = {
            "id": entry_id,
            "type": "worldview",
            "world_id": world_id,
            "worldview_id": worldview_library["worldview_id"],
            "name": entry_name,
            "content": payload.get("summary", ""),
            "category": hierarchy_path,
            "path": hierarchy_path,
            "created_at": _now(),
            "updated_at": _now(),
        }
        db["lore"].insert_one(doc)
        result = doc
    elif action == "update":
        target_id = payload["target_id"]
        worldview_entry = db["lore"].find_one({"id": target_id, "type": "worldview"})
        if worldview_entry:
            next_name = str(payload.get("name") or worldview_entry.get("name") or "").strip()
            update = {
                **({"name": next_name} if "name" in payload else {}),
                **({"content": payload["summary"]} if "summary" in payload else {}),
                "updated_at": _now(),
            }
            if any(key in payload for key in ("name", "category", "path", "parent_path")):
                hierarchy_path = _resolve_worldview_entry_path(
                    payload,
                    entry_name=next_name or str(worldview_entry.get("name") or ""),
                    fallback_path=worldview_entry.get("path") or worldview_entry.get("category") or "",
                    existing_name=str(worldview_entry.get("name") or ""),
                )
                update["category"] = hierarchy_path
                update["path"] = hierarchy_path
            db["lore"].update_one({"id": target_id, "type": "worldview"}, {"$set": update})
            result = {
                "id": target_id,
                "type": "worldview",
                "world_id": worldview_entry["world_id"],
                "worldview_id": worldview_entry["worldview_id"],
                **update,
            }
        else:
            library = db["worldviews"].find_one({"worldview_id": target_id})
            if not library:
                raise ValueError(f"Worldview setting not found: {target_id}")
            update = {k: payload[k] for k in ("name", "summary") if k in payload}
            if "summary" in update:
                update["summary"] = update.pop("summary")
            update["updated_at"] = _now()
            db["worldviews"].update_one({"worldview_id": target_id}, {"$set": update})
            result = {"worldview_id": target_id, **update}
    else:
        raise ValueError(f"{AGENT_NAME} does not handle delete operations")
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": result}))
    return {"commit_result": result, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}


workflow = StateGraph(WorldviewAgentState)
workflow.add_node("input", input_node)
workflow.add_node("initial_expansion", initial_expansion_node)
workflow.add_node("world_rule_review", world_rule_review_node)
workflow.add_node("worldview_consistency_review", worldview_consistency_review_node)
workflow.add_node("human", human_node)
workflow.add_node("modify_content", modify_content_node)
workflow.add_node("commit", commit_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "initial_expansion")
workflow.add_edge("initial_expansion", "world_rule_review")
workflow.add_conditional_edges("world_rule_review", route_after_world_rule_review, {"worldview_consistency_review": "worldview_consistency_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("worldview_consistency_review", route_after_worldview_consistency_review, {"modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("human", route_after_human, {"modify_content": "modify_content", "commit": "commit", "end": END})
workflow.add_edge("modify_content", "world_rule_review")
workflow.add_edge("commit", END)

app = workflow.compile(checkpointer=MemorySaver())
