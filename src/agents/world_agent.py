"""World Agent - 独立世界工作流。

符合 docs/product_world_hierarchy_requirements.md：
Input -> Initial Expansion -> Human -> Commit；人工不同意则进入 Modify Content -> Human。世界不包含 Review 节点。
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, get_unified_context, parse_json_safely
from src.common.revision_mode_prompt import build_revision_mode_instruction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


AGENT_NAME = "world_agent"
INITIAL_EXPANSION_AGENT_NAME = "world_agent_initial_expansion"
MODIFY_CONTENT_AGENT_NAME = "world_agent_modify_content"
ENTITY_TYPE = "world"
PRIMARY_FIELD = "summary"
WORKFLOW_DESCRIPTION = "世界 Agent 流程：接收世界输入 -> 初始扩充世界内容 -> 等待人工确认 -> 批准后写入 worlds；人工不同意则进入修改内容节点，再回到人工确认。世界模块不包含审查节点，但必须维护世界禁止规则与基本设定。"
WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收世界输入",
        "function": "接收用户消息、业务动作和世界 payload",
        "description": "记录本次创建或修改世界的名称、摘要、世界禁止规则、世界基本设定、目标 ID、人工意图和 URL/表单来源，作为 world_agent 后续初始扩充和内容修改的唯一输入基准。",
    },
    "initial_expansion": {
        "step_index": 2,
        "step_title": "步骤 2：初始扩充",
        "function": "调用 world_agent 专属 LLM 整理世界输入",
        "description": "对用户输入进行初步理解、补全和结构化，直接生成可供人工确认的世界名称、摘要、禁止规则与基本设定；不得写库，不得跳过 LLM，不得使用通用 Prompt。",
    },
    "human": {
        "step_index": 3,
        "step_title": "步骤 3：人工确认",
        "function": "等待用户同意或不同意",
        "description": "初始扩充结果在此暂停，不写库。用户同意则进入写库；用户不同意则携带反馈进入修改内容节点。",
    },
    "modify_content": {
        "step_index": 4,
        "step_title": "步骤 4：修改内容",
        "function": "根据人工反馈调用 world_agent 专属 LLM 修改世界内容",
        "description": "仅在人工不同意时执行。根据用户反馈和修改模式更新世界名称、摘要、禁止规则或基本设定，保留未要求修改的内容和业务 ID；不得写库，不得使用通用 Prompt。",
    },
    "commit": {
        "step_index": 5,
        "step_title": "步骤 5：写库固化",
        "function": "执行 worlds 集合写入",
        "description": "只有人工批准后才会写入 MongoDB worlds 集合，并记录真实 world_id、世界禁止规则、基本设定、更新字段和写库结果。",
    },
}
NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入应包含 action、message、payload；payload 只允许承载世界名称、摘要、forbidden_rules、basic_settings、world_id 或 target_id。",
        "output_annotation": "输出 accepted=true，表示输入已被记录为本轮 world_agent 的工作流基准。",
        "next_step_annotation": "下一步进入初始扩充节点，先整理用户原始意图和缺失字段。",
    },
    "initial_expansion": {
        "input_annotation": "输入是用户消息、世界 payload、人工反馈和修改模式；只能处理世界根实体。",
        "output_annotation": "输出可人工确认的世界 payload、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步进入人工确认；世界模块不经过预览生成节点或审查节点。",
    },
    "human": {
        "input_annotation": "输入是当前世界内容、用户同意/不同意决策、反馈文本和修改模式。",
        "output_annotation": "输出记录用户是否 approve、request_changes 或 stop，以及反馈内容。",
        "next_step_annotation": "同意则进入写库固化；不同意则进入修改内容节点；中止则结束。",
    },
    "modify_content": {
        "input_annotation": "输入是当前世界 payload、用户不同意原因、反馈文本和修改模式。",
        "output_annotation": "输出修改后的世界 payload、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步回到人工确认，等待用户再次同意或继续提出修改。",
    },
    "commit": {
        "input_annotation": "输入是人工批准后的最终 world payload。",
        "output_annotation": "输出是真实 MongoDB worlds 写入结果，包括 world_id、name、summary、forbidden_rules、basic_settings。",
        "next_step_annotation": "写库完成后工作流结束。",
    },
}


class WorldAgentState(TypedDict, total=False):
    action: str
    message: str
    payload: Dict[str, Any]
    pending_payload: Dict[str, Any]
    feedback: str
    revision_mode: str
    decision: str
    manual_edit: bool
    expanded_input: Dict[str, Any]
    initial_expansion: Dict[str, Any]
    modification: Dict[str, Any]
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
    """生成本次 world_agent LLM 调用的中文可审计元数据。"""
    config = get_config()
    provider = str(config.get("LLM_PROVIDER", "ollama")).lower()
    agent_config = (config.get("AGENT_MODELS") or {}).get(llm_agent_name) or {}
    if not agent_config:
        agent_config = (config.get("AGENT_MODELS") or {}).get(AGENT_NAME) or {}
    model_name = agent_config.get("model") if isinstance(agent_config, dict) else agent_config
    provider_config = (config.get("LLM_MODELS") or {}).get(provider) or {}
    if isinstance(provider_config, dict) and not model_name:
        model_name = provider_config.get("default")
    return {
        "llm_invoked": True,
        "llm_agent_name": llm_agent_name,
        "provider": provider,
        "model": model_name or config.get("DEFAULT_MODEL"),
        "json_mode": True,
        "raw_response_chars": len(raw_content),
        "prompt": prompt,
        "prompt_chars": len(prompt),
    }


def _invoke_llm(prompt: str, *, llm_agent_name: str) -> tuple[str, Dict[str, Any]]:
    """真实调用 world_agent 对应 LLM；空响应直接报错，禁止伪成功。"""
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
    return {
        "node_id": node_id,
        "label": step["step_title"],
        **step,
        "node_annotation": f"{step['step_title']}：{step['description']}",
        **annotations,
        "status": status,
        "input": node_input,
        "output": output,
    }


def build_initial_expansion_prompt(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str],
    feedback: str,
) -> str:
    """构造 world_agent 初始扩充 Prompt，使用结构化模板明确世界根任务。"""
    world_id = payload.get("world_id") or payload.get("target_id") or ""
    world_id_hint = world_id or "写库时自动生成"
    return f"""【角色设定】
你是一名世界根设定编辑。你的唯一职责是整理和扩充“世界（World）”根实体，不得越权生成世界观、小说、大纲或章节内容。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查原始世界草稿是否缺少关键根设定，确认名称、摘要方向、业务 ID 与已有禁止规则/基本设定输入是否自洽。
2. 扩展（Expand）：在不改变用户原始创意方向的前提下，扩充世界摘要，并补全 forbidden_rules 与 basic_settings，明确后续所有世界观与小说都必须遵守的根约束。

【输入信息】
【业务动作】{action}
【世界 ID】{world_id_hint}
【用户消息】{message}
【人工反馈】{feedback}
【修改模式】{revision_mode or "initial_expansion"}
【世界草稿】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留用户原始世界创意、名称、摘要方向和业务 ID。
3. 必须生成 `forbidden_rules` 与 `basic_settings`，供后续世界观与小说审查强制校验。
4. `summary` 必须聚焦世界根设定，避免漂移到世界观条目、小说项目、大纲或章节正文。

只返回合法 JSON：
{{
  "metadata": {{"agent": "world_agent", "node": "initial_expansion", "entity_type": "world", "action": "{action}"}},
  "payload": {{
    "world_id": "{world_id}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "世界名称",
    "summary": "50-200 字扩充后的世界根设定摘要",
    "forbidden_rules": ["世界内绝对禁止出现或绕开的规则"],
    "basic_settings": {{
      "era": "时代与文明阶段",
      "power_system": "力量体系或科技/魔法边界",
      "geography": "核心地理边界",
      "organizations": "主要组织结构",
      "resource_rules": "关键资源与限制",
      "baseline_constraints": ["后续世界观与小说必须遵守的基础约束"]
    }}
  }},
  "expanded_input": {{
    "world_id": "{world_id}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "{payload.get("name", "")}",
    "summary_seed": "{payload.get("summary", "")}",
    "forbidden_rules_seed": {json.dumps(payload.get("forbidden_rules", []), ensure_ascii=False)},
    "basic_settings_seed": {json.dumps(payload.get("basic_settings", {}), ensure_ascii=False)},
    "tone": "从用户输入中提炼的创作基调",
    "must_keep": ["必须保留的用户原始意图"],
    "missing_fields": ["初始扩充需要补全的字段"],
    "confirmation_focus": "人工确认时需要重点检查的世界内容"
  }},
  "expansion_notes": "初始扩充节点整理和扩充了哪些世界内容"
}}
"""


def generate_initial_expansion(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str] = None,
    feedback: str = "",
) -> Dict[str, Any]:
    """调用 LLM 生成世界初始扩充结果，确保第二节点真实使用 world_agent LLM。"""
    prompt = build_initial_expansion_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=INITIAL_EXPANSION_AGENT_NAME)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} initial expansion returned non-object JSON: {raw_content[:500]}")
    expanded_input = parsed.get("expanded_input") or {}
    initial_payload = parsed.get("payload") or expanded_input
    if not isinstance(initial_payload, dict):
        raise ValueError(f"{AGENT_NAME} initial expansion missing payload object: {raw_content[:500]}")
    if not isinstance(expanded_input, dict):
        expanded_input = {}
    if payload.get("name") and revision_mode != "full_rewrite":
        initial_payload["name"] = payload["name"]
    for field in ("forbidden_rules", "basic_settings"):
        if field in payload and field not in initial_payload:
            initial_payload[field] = payload[field]
    return {
        "payload": initial_payload,
        "expanded_input": expanded_input,
        "llm_invoked": True,
        "agent_name": INITIAL_EXPANSION_AGENT_NAME,
        "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME,
        "llm_call": llm_call,
        "raw_response": raw_content,
        "parsed_response": parsed,
        "expansion_notes": parsed.get("expansion_notes", ""),
    }


def build_modification_prompt(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str],
    feedback: str,
) -> str:
    """构造 world_agent 修改内容 Prompt，使用结构化模板明确局部修正任务。"""
    world_id = payload.get("world_id") or payload.get("target_id") or "world_default"
    rag_context = get_unified_context(
        f"{message}\n{payload.get('name', '')}\n{payload.get('summary', '')}",
        worldview_id=str(world_id),
    )
    mode_instruction = build_revision_mode_instruction(
        revision_mode,
        entity_label="世界根实体",
        primary_field_label="payload.summary",
        content_rewrite_scope="允许围绕世界摘要 summary 做较大幅重写，但 forbidden_rules 与 basic_settings 只可做保持一致性所必需的最小联动。",
        full_rewrite_scope="允许整体重写 name、summary、forbidden_rules 和 basic_settings 等世界级业务内容。",
    )
    return f"""【角色设定】
你是一名世界根设定编辑。你的唯一职责是修正“世界（World）”根实体，不得越权生成世界观、小说、大纲或章节内容。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查当前世界内容与人工反馈是否存在明确冲突，确认哪些字段必须调整、哪些字段必须保留。
2. 修正（Modify）：根据【人工不同意原因/修改意见】和下面的【修改模式说明】修正世界内容。
3. 扩展（Expand）：仅在完成修正后，补充必要细节，让世界根摘要、forbidden_rules 与 basic_settings 更完整，但不得偏离反馈要求。

【输入信息】
【业务动作】{action}
【世界 ID】{world_id}
【用户消息】{message}
【人工不同意原因/修改意见】{feedback}
【修改模式】{revision_mode or "partial_rewrite"}
【当前世界 payload】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}
【检索上下文】
{rag_context}
【修改模式说明】
{mode_instruction}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. `partial_rewrite` / `content_rewrite` 必须保留未点名字段；`full_rewrite` 也必须保留 world_id、target_id 等业务 ID。
3. `summary` 只描述世界根设定，不生成世界观、小说、大纲或章节。
4. `forbidden_rules` 与 `basic_settings` 是后续审查依据，必须保留或按反馈精确修改。

只返回合法 JSON：
{{
  "metadata": {{"agent": "world_agent", "node": "modify_content", "entity_type": "world", "action": "{action}"}},
  "payload": {{
    "world_id": "{world_id}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "世界名称",
    "summary": "扩充后的世界根设定摘要",
    "forbidden_rules": ["修改后的世界禁止规则"],
    "basic_settings": {{
      "era": "修改后的时代与文明阶段",
      "power_system": "修改后的力量体系或科技/魔法边界",
      "geography": "修改后的核心地理边界",
      "organizations": "修改后的主要组织结构",
      "resource_rules": "修改后的关键资源与限制",
      "baseline_constraints": ["修改后的基础约束"]
    }}
  }},
  "modification_notes": "world_agent 本轮根据人工反馈修改了哪些内容",
  "change_summary": "相对上一版世界 payload 的变化摘要"
}}
"""


def generate_content_modification(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str] = None,
    feedback: str = "",
    llm_agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 LLM 根据人工反馈修改世界内容，并返回可再次确认的 payload。"""
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    agent_name = llm_agent_name or MODIFY_CONTENT_AGENT_NAME
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=agent_name)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} modification returned non-object JSON: {raw_content[:500]}")
    modified_payload = parsed.get("payload")
    if not isinstance(modified_payload, dict):
        raise ValueError(f"{AGENT_NAME} modification missing payload object: {raw_content[:500]}")
    if payload.get("name") and revision_mode != "full_rewrite":
        modified_payload["name"] = payload["name"]
    for field in ("forbidden_rules", "basic_settings"):
        if field in payload and field not in modified_payload:
            modified_payload[field] = payload[field]
    return {
        "payload": modified_payload,
        "llm_invoked": True,
        "agent_name": agent_name,
        "llm_agent_name": agent_name,
        "llm_call": llm_call,
        "raw_response": raw_content,
        "parsed_response": parsed,
        "modification_notes": parsed.get("modification_notes", ""),
        "change_summary": parsed.get("change_summary", ""),
    }


def input_node(state: WorldAgentState) -> WorldAgentState:
    """输入节点：记录用户消息和世界 payload，建立本轮工作流基准。"""
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: WorldAgentState) -> WorldAgentState:
    """初始扩充节点：调用 world_agent LLM 扩充世界内容并提交人工确认。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(
        state.get("action", "create"),
        payload,
        state.get("message", ""),
        revision_mode=state.get("revision_mode"),
        feedback=state.get("feedback", ""),
    )
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "agent", "message": "world_agent 初始扩充已完成，等待人工确认。", "payload": expansion["payload"]})
    return {
        "initial_expansion": expansion,
        "expanded_input": expansion["expanded_input"],
        "pending_payload": expansion["payload"],
        "nodes": nodes,
        "conversation": conversation,
        "iterations": iteration,
        "current_node": "human",
        "status": "waiting_human",
    }


def modify_content_node(state: WorldAgentState) -> WorldAgentState:
    """修改内容节点：人工不同意后调用 world_agent LLM 修改世界内容。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    modification = generate_content_modification(
        state.get("action", "create"),
        payload,
        state.get("message", ""),
        revision_mode=state.get("revision_mode"),
        feedback=state.get("feedback", ""),
    )
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("modify_content", "completed", {"payload": payload, "feedback": state.get("feedback", ""), "revision_mode": state.get("revision_mode")}, {**modification, "iteration": iteration}))
    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "agent", "message": "world_agent 已根据反馈修改内容，等待人工再次确认。", "payload": modification["payload"]})
    return {
        "modification": modification,
        "pending_payload": modification["payload"],
        "nodes": nodes,
        "conversation": conversation,
        "iterations": iteration,
        "current_node": "human",
        "status": "waiting_human",
    }


def human_node(state: WorldAgentState) -> WorldAgentState:
    """人工节点：等待同意、不同意或中止，并记录用户反馈和修改模式。"""
    decision = state.get("decision")
    feedback = state.get("feedback", "")
    revision_mode = state.get("revision_mode") or "partial_rewrite"
    if not decision:
        user_input = interrupt({
            "agent": AGENT_NAME,
            "status": "waiting_human",
            "payload": state.get("pending_payload"),
            "actions": ["approve", "request_changes", "stop"],
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


def route_after_human(state: WorldAgentState) -> str:
    """人工节点路由：同意进入写库，不同意进入修改内容节点，中止结束。"""
    decision = state.get("decision")
    if decision == "approve":
        return "commit"
    if decision == "stop":
        return "end"
    return "modify_content"


def commit_node(state: WorldAgentState) -> WorldAgentState:
    """写库节点：人工批准后真实创建或更新 MongoDB worlds 集合，并自动维护唯一 worldview 库。"""
    db = get_mongodb_db()
    action = state.get("action", "create")
    payload = dict(state.get("pending_payload") or {})
    if action == "create":
        original_payload = dict(state.get("payload") or {})
        explicit_world_id = original_payload.get("world_id") or original_payload.get("target_id")
        world_id = explicit_world_id or f"world_{uuid.uuid4().hex[:8]}"
        if db["worlds"].find_one({"world_id": world_id}):
            raise ValueError(f"World already exists: {world_id}")
        doc = {
            "world_id": world_id,
            "name": payload["name"],
            "summary": payload.get("summary", ""),
            "forbidden_rules": payload.get("forbidden_rules", []),
            "basic_settings": payload.get("basic_settings", {}),
        }
        db["worlds"].insert_one(doc)
        worldview_doc = {
            "worldview_id": f"wv_{world_id}",
            "world_id": world_id,
            "name": f"{payload['name']} 世界观设定集",
            "summary": payload.get("summary", ""),
            "forbidden_rules": [],
            "basic_settings": {},
            "auto_created": True,
            "created_at": _now(),
            "updated_at": _now(),
        }
        db["worldviews"].insert_one(worldview_doc)
        result = {**doc, "worldview_id": worldview_doc["worldview_id"]}
    elif action == "update":
        target_id = payload["target_id"]
        update = {k: payload[k] for k in ("name", "summary", "forbidden_rules", "basic_settings") if k in payload}
        db["worlds"].update_one({"world_id": target_id}, {"$set": update})
        result = {"world_id": target_id, **update}
    else:
        raise ValueError(f"{AGENT_NAME} does not handle delete operations")
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": result}))
    return {"commit_result": result, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}


workflow = StateGraph(WorldAgentState)
workflow.add_node("input", input_node)
workflow.add_node("initial_expansion", initial_expansion_node)
workflow.add_node("human", human_node)
workflow.add_node("modify_content", modify_content_node)
workflow.add_node("commit", commit_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "initial_expansion")
workflow.add_edge("initial_expansion", "human")
workflow.add_conditional_edges("human", route_after_human, {"modify_content": "modify_content", "commit": "commit", "end": END})
workflow.add_edge("modify_content", "human")
workflow.add_edge("commit", END)

app = workflow.compile(checkpointer=MemorySaver())
