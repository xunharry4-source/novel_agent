"""Novel Agent - 独立小说项目工作流。

流程：Input -> Initial Expansion -> Review -> Human -> Commit；人工不同意或审查失败进入 Modify Content -> Review。
"""

import json
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.review_nodes.novel_review import make_novel_review_node, make_novel_review_route
from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, get_unified_context, parse_json_safely


AGENT_NAME = "novel_agent"
ENTITY_TYPE = "novel"
PRIMARY_FIELD = "summary"
MAX_AUTO_REVIEW_ITERATIONS = 3
WORKFLOW_DESCRIPTION = "小说 Agent 流程：接收小说项目输入 -> 初始扩充故事项目内容与小说级规则 -> 世界规则与背景契合审查 -> 人工确认 -> 批准后写入 novels；人工不同意或审查失败进入修改内容节点，再重新审查。"
WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收小说输入",
        "function": "接收小说 payload 与父级 world_id",
        "description": "记录小说名称、摘要、小说禁止规则、小说基本设定、world_id、可选 worldview_id、novel_id 或 target_id，明确小说只能归属单一世界。",
    },
    "initial_expansion": {
        "step_index": 2,
        "step_title": "步骤 2：初始扩充",
        "function": "调用 novel_agent 专属 LLM 整理小说输入",
        "description": "对标题、简介、父级世界和可选世界观进行初步整合，直接生成可审查的小说项目 payload，明确故事类型、核心卖点、主角方向、父级约束、小说禁止规则、小说基本设定和不可偏离的用户指令；不得写库，不得跳过 LLM，不得使用通用 Prompt。",
    },
    "review": {
        "step_index": 3,
        "step_title": "步骤 3：世界规则与背景契合审查",
        "function": "检查是否违反世界禁止规则、基本设定或世界观约束",
        "description": "审查故事背景、主角动机和核心主线是否违反父级世界 forbidden_rules、basic_settings、已选世界观设定或小说级规则，发现反吃设定、绕开禁令或方向偏离时写入 review_feedback 并自动进入修改内容节点。",
    },
    "human": {
        "step_index": 4,
        "step_title": "步骤 4：人工确认",
        "function": "等待用户批准、重写或中止",
        "description": "审查通过后等待用户批准写库；用户不同意则通过 partial_rewrite、content_rewrite、full_rewrite 提交反馈进入修改内容节点，修改后必须再次审查。",
    },
    "modify_content": {
        "step_index": 5,
        "step_title": "步骤 5：修改内容",
        "function": "根据审查意见或人工反馈调用 novel_agent 专属 LLM 修改小说内容",
        "description": "仅在审查失败或人工不同意时执行。根据 review_feedback 或用户反馈修正小说项目 payload、小说禁止规则或小说基本设定，保留未要求修改的内容和业务 ID；不得写库，不得使用通用 Prompt。",
    },
    "commit": {
        "step_index": 6,
        "step_title": "步骤 6：写库固化",
        "function": "执行 novels 集合写入",
        "description": "人工批准后创建或更新 MongoDB novels 集合，保留 world_id、novel_id、小说禁止规则、小说基本设定和真实写库结果。",
    },
}
NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入必须包含 world_id，并可包含 novel_id、worldview_id、target_id、forbidden_rules 或 basic_settings；小说只能归属一个世界。",
        "output_annotation": "输出 accepted=true，并把小说项目 payload 固定为本次工作流基准。",
        "next_step_annotation": "下一步进入初始扩充节点，先整理故事类型、卖点和父级约束。",
    },
    "initial_expansion": {
        "input_annotation": "输入是用户消息、小说 payload、小说禁止规则、小说基本设定、人工反馈和修改模式；必须保留 world_id、可选 worldview_id 和用户原始指令。",
        "output_annotation": "输出可审查的小说 payload、小说禁止规则、小说基本设定、expanded_input、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步进入世界规则与背景契合审查，检查是否违反世界禁止规则、基本设定或世界观约束。",
    },
    "review": {
        "input_annotation": "输入是当前小说项目 payload、父级世界 forbidden_rules/basic_settings、小说级 forbidden_rules/basic_settings 及可选世界观约束。",
        "output_annotation": "输出包含 passed、errors 和 reviewer，用于判断是否违反世界禁止规则、小说禁止规则、基本设定、背景约束或存在反吃设定。",
        "next_step_annotation": "通过则进入人工确认；失败且未超出上限则进入修改内容节点。",
    },
    "human": {
        "input_annotation": "输入是审查后的小说项目内容、审查意见、用户反馈和修改模式。",
        "output_annotation": "输出记录用户决策、反馈文本和本轮是否需要重新生成。",
        "next_step_annotation": "批准则写库；要求修改则进入修改内容节点并再次审查；中止则结束。",
    },
    "modify_content": {
        "input_annotation": "输入是当前小说 payload、小说禁止规则、小说基本设定、review_feedback、人工反馈和修改模式。",
        "output_annotation": "输出修改后的小说 payload、llm_call、raw_response、parsed_response 和 change_summary。",
        "next_step_annotation": "下一步回到背景契合审查，只有审查通过后才能进入人工确认。",
    },
    "commit": {
        "input_annotation": "输入是人工批准后的最终 novel payload。",
        "output_annotation": "输出是真实 MongoDB novels 写入结果，包含 world_id、novel_id、forbidden_rules 与 basic_settings。",
        "next_step_annotation": "写库完成后工作流结束。",
    },
}


class NovelAgentState(TypedDict, total=False):
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


def _llm_metadata(raw_content: str, prompt: str) -> Dict[str, Any]:
    """生成本次 novel_agent LLM 调用的中文可审计元数据。"""
    config = get_config()
    provider = str(config.get("LLM_PROVIDER", "ollama")).lower()
    agent_config = (config.get("AGENT_MODELS") or {}).get(AGENT_NAME) or {}
    model_name = agent_config.get("model") if isinstance(agent_config, dict) else agent_config
    provider_config = (config.get("LLM_MODELS") or {}).get(provider) or {}
    if isinstance(provider_config, dict) and not model_name:
        model_name = provider_config.get("default")
    return {"llm_invoked": True, "llm_agent_name": AGENT_NAME, "provider": provider, "model": model_name or config.get("DEFAULT_MODEL"), "json_mode": True, "raw_response_chars": len(raw_content), "prompt": prompt, "prompt_chars": len(prompt)}


def _invoke_llm(prompt: str) -> tuple[str, Dict[str, Any]]:
    """真实调用 novel_agent 对应 LLM；空响应直接报错，禁止伪成功。"""
    llm = get_llm(json_mode=True, agent_name=AGENT_NAME)
    config: Dict[str, Any] = {}
    callback = get_langfuse_callback()
    if callback:
        config["callbacks"] = [callback]
    response = llm.invoke(prompt, config=config if config else None)
    raw_content = _extract_llm_content(response)
    if not raw_content.strip():
        raise ValueError(f"{AGENT_NAME} returned empty LLM response")
    return raw_content, _llm_metadata(raw_content, prompt)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    """构造带中文步骤说明、节点注解、输入输出说明的工作流节点。"""
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    """构造 novel_agent 初始扩充 Prompt，使用结构化模板明确小说项目任务。"""
    return f"""【角色设定】
你是一名小说项目编辑。你的唯一职责是整理和扩充小说项目本身，不得越权生成世界观条目、大纲或章节正文。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查小说草稿与父级 world/worldview 约束是否冲突，确认标题、简介、摘要方向和业务 ID 是否完整、自洽。
2. 扩展（Expand）：在不偏离用户原始创意的前提下，补全故事类型、主角方向、核心卖点、小说级 forbidden_rules 与 basic_settings，明确后续大纲和章节必须遵守的约束。

【输入信息】
【业务动作】{action}
【所属世界 world_id】{payload.get("world_id", "")}
【小说 ID】{payload.get("novel_id", "") or payload.get("target_id", "")}
【可选约束 worldview_id】{payload.get("worldview_id", "")}
【用户消息】{message}
【人工反馈】{feedback}
【修改模式】{revision_mode or "initial_expansion"}
【小说草稿】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留小说标题、介绍、简介、world_id、worldview_id、novel_id、target_id。
3. 必须生成小说级 `forbidden_rules` 与 `basic_settings`，供大纲与章节审查强制校验。
4. 小说级规则不得违反父级世界禁止规则与基本设定。

只返回合法 JSON：
{{
  "metadata": {{"agent": "novel_agent", "node": "initial_expansion", "entity_type": "novel", "action": "{action}"}},
  "payload": {{
    "world_id": "{payload.get("world_id", "")}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "novel_id": "{payload.get("novel_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "小说项目名称",
    "introduction": "小说项目介绍",
    "summary": "小说项目构思摘要",
    "forbidden_rules": ["本小说内绝对禁止出现或绕开的叙事/设定规则"],
    "basic_settings": {{
      "genre": "小说类型与叙事范式",
      "protagonist_baseline": "主角基础设定与不可破坏的人物底线",
      "main_conflict": "主线冲突与边界",
      "tone": "叙事基调",
      "timeline": "故事时间线与阶段边界",
      "relationship_rules": "人物关系与阵营规则",
      "plot_constraints": ["后续大纲与章节必须遵守的小说级约束"]
    }}
  }},
  "expanded_input": {{
    "world_id": "{payload.get("world_id", "")}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "novel_id": "{payload.get("novel_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "{payload.get("name", "")}",
    "introduction_seed": "{payload.get("introduction", "")}",
    "summary_seed": "{payload.get("summary", "")}",
    "forbidden_rules_seed": {json.dumps(payload.get("forbidden_rules", []), ensure_ascii=False)},
    "basic_settings_seed": {json.dumps(payload.get("basic_settings", {}), ensure_ascii=False)},
    "genre": "从用户输入提炼的故事类型",
    "selling_points": ["核心卖点"],
    "protagonist_direction": "主角方向",
    "parent_constraints": ["父级世界或世界观约束"],
    "must_keep": ["不可偏离的用户指令"]
  }},
  "expansion_notes": "初始扩充节点整理了哪些小说项目约束"
}}
"""


def generate_initial_expansion(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "") -> Dict[str, Any]:
    """调用 LLM 生成小说初始扩充结果，确保第二节点真实使用 novel_agent LLM。"""
    prompt = build_initial_expansion_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    raw_content, llm_call = _invoke_llm(prompt)
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
    return {"payload": initial_payload, "expanded_input": expanded_input, "llm_invoked": True, "agent_name": AGENT_NAME, "llm_agent_name": AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    """构造 novel_agent 修改内容 Prompt，使用结构化模板明确局部修正任务。"""
    rag_context = get_unified_context(
        f"{message}\n{payload.get('name', '')}\n{payload.get('summary', '')}",
        worldview_id=str(payload.get("worldview_id") or "default_wv"),
    )
    retry_clause = f"\n【审查失败原因】{expansion_error}\n必须根据失败原因重新修正小说项目内容。" if expansion_error else ""
    return f"""【角色设定】
你是一名小说项目编辑。你的唯一职责是修正小说项目本身，不得越权生成世界观条目、大纲或章节正文。

【操作流程 (Mandatory Workflow)】
1. 审查（Review）：检查当前小说项目与审查意见、人工反馈、父级 world/worldview 约束之间的冲突点。
2. 修正（Modify）：根据【人工反馈或审查意见】和【修改模式】对小说项目做局部、精准修改，不得超范围改写。
3. 扩展（Expand）：仅在修正完成后，补充必要细节，让故事主旨、主角方向、核心冲突和小说级规则更完整，但不得偏离反馈要求。 

【输入信息】
【业务动作】{action}
【所属世界 world_id】{payload.get("world_id", "")}
【小说 ID】{payload.get("novel_id", "") or payload.get("target_id", "")}
【可选约束 worldview_id】{payload.get("worldview_id", "")}
【用户消息】{message}
【人工反馈或审查意见】{feedback}
【修改模式】{revision_mode or "partial_rewrite"}
【当前 payload】
{json.dumps(payload or {}, ensure_ascii=False, indent=2)}
【RAG 上下文】
{rag_context}{retry_clause}

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 只修正审查失败原因或人工反馈要求修改的内容，必须保留 world_id、novel_id、target_id，且不得破坏未点名内容。
3. `forbidden_rules` 与 `basic_settings` 是后续大纲、章节审查依据，必须保留或按反馈精确修改。
4. 小说级规则不得违反父级世界禁止规则与基本设定。

只返回合法 JSON：
{{
  "metadata": {{"agent": "novel_agent", "node": "modify_content", "entity_type": "novel", "action": "{action}"}},
  "payload": {{
    "world_id": "{payload.get("world_id", "")}",
    "worldview_id": "{payload.get("worldview_id", "")}",
    "novel_id": "{payload.get("novel_id", "")}",
    "target_id": "{payload.get("target_id", "")}",
    "name": "小说项目名称",
    "introduction": "修改后的小说项目介绍",
    "summary": "小说项目构思摘要",
    "forbidden_rules": ["修改后的小说禁止规则"],
    "basic_settings": {{
      "genre": "修改后的小说类型与叙事范式",
      "protagonist_baseline": "修改后的主角基础设定与人物底线",
      "main_conflict": "修改后的主线冲突与边界",
      "tone": "修改后的叙事基调",
      "timeline": "修改后的故事时间线与阶段边界",
      "relationship_rules": "修改后的人物关系与阵营规则",
      "plot_constraints": ["修改后的小说级约束"]
    }}
  }},
  "modification_notes": "novel_agent 本轮修正的故事项目维度",
  "change_summary": "相对输入 payload 的变化摘要"
}}
"""


def generate_content_modification(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "", expansion_error: str = "") -> Dict[str, Any]:
    """调用 LLM 根据审查意见或人工反馈修改小说项目内容。"""
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback, expansion_error=expansion_error)
    raw_content, llm_call = _invoke_llm(prompt)
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
    return {"payload": modified_payload, "llm_invoked": True, "agent_name": AGENT_NAME, "llm_agent_name": AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "modification_notes": parsed.get("modification_notes", ""), "change_summary": parsed.get("change_summary", "")}


def input_node(state: NovelAgentState) -> NovelAgentState:
    """输入节点：记录小说 payload、父级 world_id、可选 worldview_id 和用户消息。"""
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: NovelAgentState) -> NovelAgentState:
    """初始扩充节点：调用 novel_agent LLM 扩充小说项目内容并提交审查。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=state.get("feedback", ""))
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    return {"initial_expansion": expansion, "expanded_input": expansion["expanded_input"], "pending_payload": expansion["payload"], "nodes": nodes, "iterations": iteration, "current_node": "review", "status": "reviewing"}


def modify_content_node(state: NovelAgentState) -> NovelAgentState:
    """修改内容节点：按审查意见或人工反馈调用 novel_agent LLM 修改小说内容。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    feedback = state.get("review_feedback") or state.get("feedback", "")
    modification = generate_content_modification(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=feedback, expansion_error=state.get("review_feedback", ""))
    iteration = int(state.get("iterations") or 0) + 1
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("modify_content", "completed", {"payload": payload, "feedback": feedback, "revision_mode": state.get("revision_mode")}, {**modification, "iteration": iteration}))
    return {"modification": modification, "pending_payload": modification["payload"], "nodes": nodes, "iterations": iteration, "current_node": "review", "status": "reviewing"}


review_node = make_novel_review_node(
    node_id="review",
    entity_type="novel_world_rules",
    reviewer="novel_world_rules_review_agent",
    passed_key="review_passed",
    errors_key="review_errors",
    feedback_key="review_feedback",
    next_node="human",
    next_status="waiting_human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_review = make_novel_review_route(
    passed_key="review_passed",
    next_node="human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)


def human_node(state: NovelAgentState) -> NovelAgentState:
    """人工节点：等待批准、重写或中止，并记录反馈与修改模式。"""
    decision = state.get("decision")
    feedback = state.get("feedback", "")
    revision_mode = state.get("revision_mode") or "partial_rewrite"
    if not decision:
        user_input = interrupt({"agent": AGENT_NAME, "status": "waiting_human", "payload": state.get("pending_payload"), "review_errors": state.get("review_errors", []), "actions": ["approve", "request_changes", "reject"], "revision_modes": ["partial_rewrite", "content_rewrite", "full_rewrite"]})
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


def route_after_human(state: NovelAgentState) -> str:
    """人工节点路由：批准进入写库，要求修改进入修改内容节点，中止结束。"""
    if state.get("decision") == "approve":
        return "commit"
    if state.get("decision") == "reject":
        return "end"
    return "modify_content"


def commit_node(state: NovelAgentState) -> NovelAgentState:
    """写库节点：人工批准后真实创建或更新 MongoDB novels 集合。"""
    db = get_mongodb_db()
    action = state.get("action", "create")
    payload = dict(state.get("pending_payload") or {})
    if action == "create":
        novel_id = payload.get("novel_id") or f"novel_{uuid.uuid4().hex[:8]}"
        doc = {
            "novel_id": novel_id,
            "world_id": payload["world_id"],
            "name": payload["name"],
            "introduction": payload.get("introduction", ""),
            "summary": payload.get("summary", ""),
            "forbidden_rules": payload.get("forbidden_rules", []),
            "basic_settings": payload.get("basic_settings", {}),
        }
        db["novels"].insert_one(doc)
        result = doc
    elif action == "update":
        target_id = payload["target_id"]
        update = {k: payload[k] for k in ("name", "introduction", "summary", "worldview_id", "forbidden_rules", "basic_settings") if k in payload}
        db["novels"].update_one({"novel_id": target_id}, {"$set": update})
        result = {"novel_id": target_id, **update}
    else:
        raise ValueError(f"{AGENT_NAME} does not handle delete operations")
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": result}))
    return {"commit_result": result, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}


workflow = StateGraph(NovelAgentState)
workflow.add_node("input", input_node)
workflow.add_node("initial_expansion", initial_expansion_node)
workflow.add_node("review", review_node)
workflow.add_node("human", human_node)
workflow.add_node("modify_content", modify_content_node)
workflow.add_node("commit", commit_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "initial_expansion")
workflow.add_edge("initial_expansion", "review")
workflow.add_conditional_edges("review", route_after_review, {"modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("human", route_after_human, {"modify_content": "modify_content", "commit": "commit", "end": END})
workflow.add_edge("modify_content", "review")
workflow.add_edge("commit", END)

app = workflow.compile(checkpointer=MemorySaver())
