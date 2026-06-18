"""Chapter outline summary create agent."""

import json
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.types import interrupt

from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, parse_json_safely


AGENT_NAME = "chapter_outline_summary_create_agent"
INITIAL_EXPANSION_AGENT_NAME = "chapter_outline_summary_create_llm"
MODIFY_CONTENT_AGENT_NAME = "chapter_outline_summary_create_modify_llm"
ENTITY_TYPE = "chapter_outline_summary_create"
PRIMARY_FIELD = "content"
SUMMARY_SCOPE = "chapter_outline"
SUMMARY_ACTION = "create"
WORKFLOW_DESCRIPTION = "新增章节大纲总结工作流：输入章节大纲全文 -> 生成下游摘要 -> 检查是否过度简化 -> 人工确认 -> 写入 downstream_summaries。"
WORKFLOW_STEPS = {
    "input": {"step_index": 1, "step_title": "步骤 1：接收章节大纲输入", "function": "记录原始章节大纲和父级上下文", "description": "保留 outline_id、novel_id、world_id、worldview_id、chapter_id、name 和原始 content。"},
    "initial_expansion": {"step_index": 2, "step_title": "步骤 2：调用 LLM 简化章节大纲", "function": "为新增章节大纲生成下游摘要", "description": "调用章节大纲总结专属 LLM，把章节大纲压缩成下游更容易消费的摘要。"},
    "review": {"step_index": 3, "step_title": "步骤 3：检查摘要是否错误简化", "function": "检查是否丢失章节任务、关键冲突和阶段结果", "description": "如果摘要太短、没完成简化或者丢失核心任务链，就不允许直接入库。"},
    "human": {"step_index": 4, "step_title": "步骤 4：人工确认", "function": "等待用户批准或要求重做", "description": "用户批准则写库；否则进入修改节点。"},
    "modify_content": {"step_index": 5, "step_title": "步骤 5：按意见重做章节大纲摘要", "function": "根据人工意见返工 downstream_summary", "description": "只允许调整 downstream_summary，不得改动原始 content。"},
    "commit": {"step_index": 6, "step_title": "步骤 6：写入总结库", "function": "写入 downstream_summaries", "description": "把章节大纲原文快照和最终摘要写入 downstream_summaries。"},
}
NODE_ANNOTATIONS = {
    "input": {"input_annotation": "输入应包含 outline_id 和原始 content。", "output_annotation": "输出 accepted=true。", "next_step_annotation": "下一步进入章节大纲摘要 LLM 节点。"},
    "initial_expansion": {"input_annotation": "输入是完整章节大纲正文。", "output_annotation": "输出 downstream_summary。", "next_step_annotation": "下一步进入摘要质量检查节点。"},
    "review": {"input_annotation": "输入是原始章节大纲与候选摘要。", "output_annotation": "输出 passed、errors、review_feedback。", "next_step_annotation": "通过则进入人工确认；失败则进入修改节点。"},
    "human": {"input_annotation": "输入是候选摘要和检查结果。", "output_annotation": "输出 decision 和 feedback。", "next_step_annotation": "批准则写库；要求修改则进入修改节点。"},
    "modify_content": {"input_annotation": "输入是原始章节大纲和人工修改意见。", "output_annotation": "输出返工后的 downstream_summary。", "next_step_annotation": "下一步回到检查节点。"},
    "commit": {"input_annotation": "输入是人工批准后的最终章节大纲摘要。", "output_annotation": "输出 downstream_summaries 写库结果。", "next_step_annotation": "写库完成后结束。"},
}


class ChapterOutlineSummaryCreateState(TypedDict, total=False):
    action: str
    message: str
    payload: Dict[str, Any]
    pending_payload: Dict[str, Any]
    feedback: str
    review_feedback: str
    revision_mode: str
    decision: str
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
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "")


def _llm_metadata(raw_content: str, prompt: str, llm_agent_name: str) -> Dict[str, Any]:
    config = get_config()
    provider = str(config.get("LLM_PROVIDER", "ollama")).lower()
    agent_config = (config.get("AGENT_MODELS") or {}).get(llm_agent_name) or {}
    model_name = agent_config.get("model") if isinstance(agent_config, dict) else agent_config
    provider_config = (config.get("LLM_MODELS") or {}).get(provider) or {}
    if isinstance(provider_config, dict) and not model_name:
        model_name = provider_config.get("default")
    return {"llm_invoked": True, "llm_agent_name": llm_agent_name, "provider": provider, "model": model_name or config.get("DEFAULT_MODEL"), "json_mode": True, "raw_response_chars": len(raw_content), "prompt": prompt, "prompt_chars": len(prompt)}


def _invoke_llm(prompt: str, *, llm_agent_name: str) -> tuple[str, Dict[str, Any]]:
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


def _normalize_text(text: Any) -> str:
    return "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())


def _chapter_outline_markers(text: str) -> List[str]:
    markers: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("第", "开篇", "主要人物", "主要矛盾", "主要事件", "转折", "高潮", "结局")):
            markers.append(stripped)
    return markers


def _check_summary_quality(source_text: Any, candidate_text: Any) -> List[str]:
    source = _normalize_text(source_text)
    candidate = _normalize_text(candidate_text)
    errors: List[str] = []
    if not candidate:
        return ["章节大纲下游摘要为空。"]
    source_len = len(source)
    candidate_len = len(candidate)
    if source_len >= 400 and candidate_len < max(110, int(source_len * 0.14)):
        errors.append(f"摘要过短：原文 {source_len} 字，摘要只有 {candidate_len} 字。")
    if source_len >= 200 and candidate_len > int(source_len * 0.97):
        errors.append("摘要没有真正压缩。")
    source_markers = _chapter_outline_markers(source)
    if len(source_markers) >= 3:
        kept = [marker for marker in source_markers if marker in candidate]
        if len(kept) < max(2, len(source_markers) // 2):
            errors.append("摘要丢失了过多章节大纲结构。")
    return errors


def _build_task_context(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    return json.dumps({"action": action, "message": message, "revision_mode": revision_mode or SUMMARY_ACTION, "feedback": feedback, "payload": payload or {}}, ensure_ascii=False, indent=2)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    task_context = _build_task_context(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    return f"""【角色设定】
你是一名章节大纲下游摘要编辑。

你的任务是：
把章节大纲压缩成下游更容易消费的稳定摘要，
但不能删掉本章任务链。

【任务上下文】
{task_context}

【摘要要求】
1. 保留本章任务、主要冲突、关键事件、阶段结果和后续钩子。
2. 允许压缩展开过程，不允许删掉任务链。
3. 禁止改写成正式章节正文。
4. 禁止修改 payload.content 原文。

【长度要求】
1. 结果必须明显短于原文。
2. 不能短到只剩空泛概述。

【输出要求】
1. 只返回合法 JSON。
2. payload.content 保留输入原文。
3. payload.downstream_summary 输出章节大纲下游摘要。
4. 必须保留 outline_id、novel_id、world_id、worldview_id、chapter_id、target_id、name。

只返回合法 JSON：
{{
  "metadata": {{"agent": "{AGENT_NAME}", "node": "initial_expansion", "entity_type": "{ENTITY_TYPE}", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入 outline_id]",
    "novel_id": "[保留输入 novel_id]",
    "world_id": "[保留输入 world_id]",
    "worldview_id": "[保留输入 worldview_id]",
    "chapter_id": "[保留输入 chapter_id]",
    "target_id": "[保留输入 target_id]",
    "name": "[保留输入 name]",
    "content": "[保留输入原始章节大纲全文]",
    "downstream_summary": "[章节大纲下游摘要]"
  }},
  "expanded_input": {{
    "content_seed": "[原始章节大纲全文]",
    "summary_goal": "[这次摘要重点保留的章节任务]",
    "must_keep": ["[必须保留的信息]"]
  }},
  "expansion_notes": "[本次摘要保留了哪些章节任务链]"
}}
"""


def generate_initial_expansion(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "") -> Dict[str, Any]:
    prompt = build_initial_expansion_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=INITIAL_EXPANSION_AGENT_NAME)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} initial summary returned non-object JSON: {raw_content[:500]}")
    summarized_payload = parsed.get("payload") or {}
    if not isinstance(summarized_payload, dict):
        raise ValueError(f"{AGENT_NAME} initial summary missing payload object: {raw_content[:500]}")
    summarized_payload["content"] = payload.get("content", "") or payload.get("summary", "")
    if payload.get("name"):
        summarized_payload["name"] = payload["name"]
    return {"payload": summarized_payload, "expanded_input": parsed.get("expanded_input") or {}, "llm_invoked": True, "agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    task_context = _build_task_context(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    return f"""【角色设定】
你是一名章节大纲摘要返工编辑。

你的任务是：
保留原始章节大纲全文，
根据人工意见重做 downstream_summary。

【任务上下文】
{task_context}

【已有问题】
{expansion_error or feedback}

【返工要求】
1. 只修改 downstream_summary。
2. 不得改动 payload.content 原文。
3. 必须修复丢任务链、丢主要冲突、摘要过短等问题。

【输出要求】
1. 只返回合法 JSON。
2. payload.content 保留原文。
3. payload.downstream_summary 输出返工后的章节大纲摘要。
4. 保留所有业务 ID 和 name。

只返回合法 JSON：
{{
  "metadata": {{"agent": "{AGENT_NAME}", "node": "modify_content", "entity_type": "{ENTITY_TYPE}", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入 outline_id]",
    "novel_id": "[保留输入 novel_id]",
    "world_id": "[保留输入 world_id]",
    "worldview_id": "[保留输入 worldview_id]",
    "chapter_id": "[保留输入 chapter_id]",
    "target_id": "[保留输入 target_id]",
    "name": "[保留输入 name]",
    "content": "[保留输入原始章节大纲全文]",
    "downstream_summary": "[返工后的章节大纲摘要]"
  }},
  "modification_notes": "[本次返工修正了什么]",
  "change_summary": "[相对上一版摘要的变化]"
}}
"""


def generate_content_modification(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "", expansion_error: str = "") -> Dict[str, Any]:
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback, expansion_error=expansion_error)
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=MODIFY_CONTENT_AGENT_NAME)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} modification returned non-object JSON: {raw_content[:500]}")
    summarized_payload = parsed.get("payload") or {}
    if not isinstance(summarized_payload, dict):
        raise ValueError(f"{AGENT_NAME} modification missing payload object: {raw_content[:500]}")
    summarized_payload["content"] = payload.get("content", "") or payload.get("summary", "")
    if payload.get("name"):
        summarized_payload["name"] = payload["name"]
    return {"payload": summarized_payload, "llm_invoked": True, "agent_name": MODIFY_CONTENT_AGENT_NAME, "llm_agent_name": MODIFY_CONTENT_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "modification_notes": parsed.get("modification_notes", ""), "change_summary": parsed.get("change_summary", "")}


def input_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=state.get("feedback", ""))
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    return {"pending_payload": expansion["payload"], "nodes": nodes, "iterations": iteration, "current_node": "review", "status": "reviewing"}


def review_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    payload = dict(state.get("pending_payload") or {})
    errors = _check_summary_quality(payload.get("content", ""), payload.get("downstream_summary", ""))
    passed = not errors
    feedback = "；".join(errors)
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("review", "completed", {"payload": payload}, {"passed": passed, "errors": errors, "reviewer": AGENT_NAME, "llm_invoked": False}))
    return {"review_passed": passed, "review_errors": errors, "review_feedback": feedback, "nodes": nodes, "current_node": "human" if passed else "modify_content", "status": "waiting_human" if passed else "review_failed"}


def modify_content_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    feedback = state.get("review_feedback") or state.get("feedback", "")
    modification = generate_content_modification(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=feedback, expansion_error=feedback)
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("modify_content", "completed", {"payload": payload, "feedback": feedback, "revision_mode": state.get("revision_mode")}, {**modification, "iteration": iteration}))
    return {"pending_payload": modification["payload"], "nodes": nodes, "iterations": iteration, "current_node": "review", "status": "reviewing"}


def human_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    decision = state.get("decision")
    feedback = state.get("feedback", "")
    revision_mode = state.get("revision_mode") or "summary_rewrite"
    if not decision:
        user_input = interrupt({"agent": AGENT_NAME, "status": "waiting_human", "payload": state.get("pending_payload"), "review_errors": state.get("review_errors", []), "actions": ["approve", "request_changes", "reject"], "revision_modes": ["summary_rewrite"]})
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


def route_after_human(state: ChapterOutlineSummaryCreateState) -> str:
    if state.get("decision") == "approve":
        return "commit"
    if state.get("decision") == "reject":
        return "end"
    return "modify_content"


def commit_node(state: ChapterOutlineSummaryCreateState) -> ChapterOutlineSummaryCreateState:
    db = get_mongodb_db()
    payload = dict(state.get("pending_payload") or {})
    now = payload.get("updated_at") or str(uuid.uuid4())
    summary_key = f"{AGENT_NAME}:{payload.get('target_id') or payload.get('chapter_id') or payload.get('outline_id') or payload.get('name')}"
    summary_id = f"dws_{uuid.uuid5(uuid.NAMESPACE_URL, summary_key).hex[:16]}"
    doc = {
        "summary_id": summary_id,
        "summary_key": summary_key,
        "agent_type": AGENT_NAME,
        "summary_scope": SUMMARY_SCOPE,
        "summary_action": SUMMARY_ACTION,
        "world_id": payload.get("world_id"),
        "worldview_id": payload.get("worldview_id"),
        "novel_id": payload.get("novel_id"),
        "outline_id": payload.get("outline_id"),
        "chapter_id": payload.get("chapter_id"),
        "target_id": payload.get("target_id"),
        "name": payload.get("name"),
        "source_field": PRIMARY_FIELD,
        "source_text": payload.get("content", "") or payload.get("summary", ""),
        "downstream_summary": payload.get("downstream_summary", ""),
        "updated_at": now,
    }
    db["downstream_summaries"].update_one({"summary_key": summary_key}, {"$set": doc, "$setOnInsert": {"created_at": now}}, upsert=True)
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": doc}))
    return {"commit_result": doc, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}
