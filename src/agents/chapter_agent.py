"""Chapter Agent - 独立章节正文工作流。

流程：Input -> Initial Expansion -> World Review -> Worldview Review -> Novel Review -> Outline Review -> Chapter Review -> Human -> Commit；
人工不同意或任一审查失败进入 Modify Content，再回到 World Review。
"""

import json
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.review_nodes.chapter_review import build_chapter_review_nodes
from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, get_unified_context, parse_json_safely


AGENT_NAME = "chapter_agent"
INITIAL_EXPANSION_AGENT_NAME = "chapter_agent_initial_expansion"
MODIFY_CONTENT_AGENT_NAME = "chapter_agent_modify_content"
ENTITY_TYPE = "chapter"
PRIMARY_FIELD = "content"
MAX_AUTO_REVIEW_ITERATIONS = 3
WORKFLOW_DESCRIPTION = "章节 Agent 流程：接收章节输入 -> 初始扩充正文内容 -> 世界审查 -> 世界观审查 -> 小说审查 -> 大纲审查 -> 章节审查 -> 人工确认 -> 批准后写入 prose；人工不同意或任一审查失败进入修改内容节点，再从世界审查重新开始。"
WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收章节输入",
        "function": "接收章节 payload、父级大纲和上下文",
        "description": "记录章节标题、正文要求、outline_id、novel_id、worldview_id、world_id、chapter_id 或 target_id，确保章节继承完整父级关系。",
    },
    "initial_expansion": {
        "step_index": 2,
        "step_title": "步骤 2：初始扩充",
        "function": "调用 chapter_agent 专属 LLM 整理章节输入",
        "description": "对大纲节点、前文上下文、目标片段和重写范围进行初步整理，直接生成可审查的章节 payload，明确场景目标、人物状态、叙事视角、上下文承接、父级 outline_id/novel_id/worldview_id/world_id 和不得违反的设定约束；不得写库，不得跳过 LLM，不得使用通用 Prompt。",
    },
    "world_review": {
        "step_index": 3,
        "step_title": "步骤 3：世界审查",
        "function": "检查章节是否违反世界禁止规则与基本设定",
        "description": "基于所属世界的 forbidden_rules 与 basic_settings 审查章节正文是否违反世界根禁令、时代边界、力量体系、地理边界、组织结构或资源机制；失败时写入 world_review_feedback 并进入修改内容节点。",
    },
    "worldview_review": {
        "step_index": 4,
        "step_title": "步骤 4：世界观审查",
        "function": "检查章节是否违反关联世界观设定",
        "description": "基于 worldview_id 与同一 world_id 下已有世界观 Canon 审查章节正文是否出现设定冲突、规则冲突、历史地理矛盾或 Lore 使用错误；失败时写入 worldview_review_feedback 并进入修改内容节点。",
    },
    "novel_review": {
        "step_index": 5,
        "step_title": "步骤 5：小说审查",
        "function": "检查章节是否违反小说禁止规则、基本设定和主线约束",
        "description": "基于 novel_id 对应小说的 forbidden_rules、basic_settings、主角底线、主线冲突、叙事基调、时间线和人物关系规则审查章节是否偏离小说设计；失败时写入 novel_review_feedback 并进入修改内容节点。",
    },
    "outline_review": {
        "step_index": 6,
        "step_title": "步骤 6：大纲审查",
        "function": "检查章节是否严格执行父级大纲任务",
        "description": "基于 outline_id 对应大纲的标题、摘要、结构目标和剧情任务审查章节是否偏离大纲节点、删改大纲安排或提前/延后关键事件；失败时写入 outline_review_feedback 并进入修改内容节点。",
    },
    "chapter_review": {
        "step_index": 7,
        "step_title": "步骤 7：章节审查",
        "function": "检查当前章节与之前章节内容是否一致",
        "description": "基于同一 outline_id/novel_id/world_id 下已入库的前置章节，审查当前章节在剧情承接、时间线、人物状态、地点变化、资源装备、伏笔和叙事视角上是否连续一致；失败时写入 chapter_review_feedback 并进入修改内容节点。",
    },
    "human": {
        "step_index": 8,
        "step_title": "步骤 8：人工确认",
        "function": "等待用户批准、局部重写或中止",
        "description": "世界审查、世界观审查、小说审查、大纲审查、章节审查均通过后等待用户批准正文入库；用户不同意则标记段落并选择 partial_rewrite、content_rewrite 或 full_rewrite 进入修改内容节点，修改后必须重新通过五个审查节点。",
    },
    "modify_content": {
        "step_index": 9,
        "step_title": "步骤 9：修改内容",
        "function": "根据审查意见或人工反馈调用 chapter_agent 专属 LLM 修改章节正文",
        "description": "仅在世界审查失败、世界观审查失败、小说审查失败、大纲审查失败、章节审查失败或人工不同意时执行。根据审查反馈或用户反馈修正章节 payload，保留未要求修改的内容和业务 ID；不得写库，不得使用通用 Prompt。",
    },
    "commit": {
        "step_index": 10,
        "step_title": "步骤 10：写库固化",
        "function": "执行 prose 集合写入",
        "description": "人工批准后写入 MongoDB prose 集合，继承 outline_id、novel_id、worldview_id、world_id，并保存最终正文。",
    },
}
NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入必须包含 outline_id，并应继承 novel_id、worldview_id、world_id、chapter_id 或 target_id。",
        "output_annotation": "输出 accepted=true，并锁定章节正文任务和完整父级关系。",
        "next_step_annotation": "下一步进入初始扩充节点，先整理场景目标、人物状态、上下文承接和重写范围。",
    },
    "initial_expansion": {
        "input_annotation": "输入是用户消息、章节 payload、人工反馈和修改模式；必须保留 outline_id、novel_id、worldview_id、world_id 和目标片段。",
        "output_annotation": "输出可审查的章节 payload、expanded_input、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步进入世界审查，先检查是否违反世界禁止规则与基本设定。",
    },
    "world_review": {
        "input_annotation": "输入是当前章节 payload、world_id 和世界 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 world_review_feedback。",
        "next_step_annotation": "通过则进入世界观审查；失败且未超出上限则进入修改内容节点。",
    },
    "worldview_review": {
        "input_annotation": "输入是通过世界审查后的章节 payload、worldview_id 和已有世界观 Canon 上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 worldview_review_feedback。",
        "next_step_annotation": "通过则进入小说审查；失败且未超出上限则进入修改内容节点。",
    },
    "novel_review": {
        "input_annotation": "输入是通过世界观审查后的章节 payload、novel_id 和小说 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 novel_review_feedback。",
        "next_step_annotation": "通过则进入大纲审查；失败且未超出上限则进入修改内容节点。",
    },
    "outline_review": {
        "input_annotation": "输入是通过小说审查后的章节 payload、outline_id 和父级大纲任务约束。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 outline_review_feedback。",
        "next_step_annotation": "通过则进入章节审查；失败且未超出上限则进入修改内容节点。",
    },
    "chapter_review": {
        "input_annotation": "输入是通过大纲审查后的章节 payload、前置章节内容和同一作品上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 chapter_review_feedback。",
        "next_step_annotation": "通过则进入人工确认；失败且未超出上限则进入修改内容节点。",
    },
    "human": {
        "input_annotation": "输入是五个审查节点通过后的章节正文、审查意见、用户标记段落、反馈和修改模式。",
        "output_annotation": "输出记录用户决策；局部重写时必须保留未点名正文和父级约束。",
        "next_step_annotation": "批准则写库；要求修改则进入修改内容节点并再次审查；中止则结束。",
    },
    "modify_content": {
        "input_annotation": "输入是当前章节 payload、world_review_feedback、worldview_review_feedback、novel_review_feedback、outline_review_feedback、chapter_review_feedback、人工反馈、修改模式和目标片段。",
        "output_annotation": "输出修改后的章节 payload、llm_call、raw_response、parsed_response 和 change_summary。",
        "next_step_annotation": "下一步回到世界审查，必须连续通过世界、世界观、小说、大纲、章节五个审查节点后才能进入人工确认。",
    },
    "commit": {
        "input_annotation": "输入是人工批准后的最终 chapter payload。",
        "output_annotation": "输出是真实 MongoDB prose 写入结果，包含章节 ID、outline_id、novel_id、worldview_id、world_id 和 content。",
        "next_step_annotation": "写库完成后工作流结束。",
    },
}


class ChapterAgentState(TypedDict, total=False):
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
    world_review_passed: bool
    world_review_errors: List[str]
    world_review_feedback: str
    worldview_review_passed: bool
    worldview_review_errors: List[str]
    worldview_review_feedback: str
    novel_review_passed: bool
    novel_review_errors: List[str]
    novel_review_feedback: str
    outline_review_passed: bool
    outline_review_errors: List[str]
    outline_review_feedback: str
    chapter_review_passed: bool
    chapter_review_errors: List[str]
    chapter_review_feedback: str
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
    """生成本次 chapter_agent LLM 调用的中文可审计元数据。"""
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
    """真实调用 chapter_agent 对应 LLM；空响应直接报错，禁止伪成功。"""
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


def _chapter_seed(payload: Dict[str, Any]) -> str:
    return str(payload.get("content", "") or payload.get("summary", "") or "")


def _normalize_chapter_text(text: Any) -> str:
    return "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())


def _assert_chapter_not_simplified(source_text: Any, candidate_text: Any, *, stage: str) -> None:
    source = _normalize_chapter_text(source_text)
    candidate = _normalize_chapter_text(candidate_text)
    if not source or not candidate:
        return
    source_len = len(source)
    candidate_len = len(candidate)
    if source_len >= 1200 and candidate_len < int(source_len * 0.85):
        raise ValueError(
            f"{AGENT_NAME} {stage} compressed long chapter too aggressively: "
            f"input_chars={source_len}, output_chars={candidate_len}"
        )


def _build_chapter_task_context(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str],
    feedback: str,
    expansion_error: str = "",
) -> str:
    task_context = {
        "action": action,
        "message": message,
        "revision_mode": revision_mode or "initial_expansion",
        "feedback": feedback,
        "expansion_error": expansion_error,
        "payload": payload or {},
    }
    return json.dumps(task_context, ensure_ascii=False, indent=2)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    """构造带中文步骤说明、节点注解、输入输出说明的工作流节点。"""
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    """构造 chapter_agent 初始扩充 Prompt，使用结构化模板明确章节任务。"""
    task_context = _build_chapter_task_context(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    return f"""【角色设定】
你是一名小说章节扩写编辑。

你的任务是：
保留原有章节全部内容，
并在此基础上扩写细节、动作、情绪、对话、因果链和场景过程。

────────────────────────

【输入说明】
用户提供：
* outline_id
* novel_id
* world_id
* worldview_id
* 章节草稿（content 或 summary）

其中：
content / summary 为原始章节正文。

【任务上下文】
{task_context}

────────────────────────

【扩写规则】
本任务是：扩写（Expand）
不是：
* 总结
* 概括
* 提炼
* 压缩
* 重写

必须保留：
* 所有已写出的场景
* 所有已写出的事件
* 所有已写出的人物行为
* 所有已写出的冲突
* 所有已写出的对话
* 所有已写出的伏笔

不得删除。
不得跳过。
不得把多个场景合并成概述。

────────────────────────

【扩写内容】
优先扩写已有正文。
增加：
* 动作过程
* 冲突升级过程
* 人物行为逻辑
* 情绪变化
* 对话展开
* 场景衔接
* 因果链
* 阶段结果
* 后续影响

禁止只改措辞。
禁止同义改写。

────────────────────────

【因果链规则】
重要段落尽量补充：
起因
→ 触发
→ 发展
→ 结果
→ 影响

────────────────────────

【世界观规则】
扩写内容必须遵守父级 outline / novel / world / worldview 的全部约束。
不得新增违反约束的设定。

────────────────────────

【长度规则】
扩写后内容长度：
不得低于原文。
优先达到原文 150% 以上。

如果无法扩写：
必须保留原文。

禁止输出比输入更短。

────────────────────────

【输出规则】
payload.content：
写扩写后的完整章节正文。
不是摘要。
不是概述。
不是总结。
必须保留全部原剧情并增加细节。

expanded_input.content_seed：
写用户提交的原始章节全文。
原文不得修改。

────────────────────────

【执行顺序】
Review：
检查父级约束与正文完整性。

Expand：
保留原文并扩写。

Validate：
检查是否遗漏原剧情，是否出现摘要化。

【输入信息】
【说明】
请严格按照上面的执行顺序完成任务。

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留 outline_id、novel_id、worldview_id、world_id、chapter_id、id、target_id 和用户指定片段。
3. `payload.content` 必须是扩写后的完整章节正文，不得摘要化、概述化、压缩化。
4. `expanded_input.content_seed` 必须保留用户提交的原始章节全文，不得改写。
5. 输出必须聚焦章节正文，不得漂移到世界规则、小说主线重设或大纲结局改写。
6. 输出 JSON 示例里的占位符只是结构说明，不是让你原样输出这些方括号文字。

只返回合法 JSON：
{{
  "metadata": {{"agent": "chapter_agent", "node": "initial_expansion", "entity_type": "chapter", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "id": "[保留输入中的 id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[章节标题]",
    "content": "[扩写后的完整章节正文]"
  }},
  "expanded_input": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "id": "[保留输入中的 id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[章节标题]",
    "content_seed": "[用户提交的原始章节全文]",
    "scene_goal": "[本章扩写目标]",
    "character_states": ["[人物状态]"],
    "narrative_viewpoint": "[叙事视角]",
    "target_segments": ["[重点扩写的段落或场景]"],
    "continuity_constraints": ["[必须遵守的上下文承接和设定约束]"]
  }},
  "expansion_notes": "[本轮章节扩写补强了哪些内容]"
}}
"""


def generate_initial_expansion(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "") -> Dict[str, Any]:
    """调用 LLM 生成章节初始扩充结果，确保第二节点真实使用 chapter_agent LLM。"""
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
    _assert_chapter_not_simplified(_chapter_seed(payload), initial_payload.get("content", ""), stage="initial_expansion")
    if payload.get("name") and revision_mode != "full_rewrite":
        initial_payload["name"] = payload["name"]
    return {"payload": initial_payload, "expanded_input": expanded_input, "llm_invoked": True, "agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    """构造 chapter_agent 修改内容 Prompt，使用结构化模板明确局部修正任务。"""
    rag_context = get_unified_context(
        f"{message}\n{payload.get('name', '')}\n{payload.get('content', '') or payload.get('summary', '')}",
        outline_id=str(payload.get("outline_id") or "default"),
        worldview_id=str(payload.get("worldview_id") or "default_wv"),
    )
    task_context = _build_chapter_task_context(
        action,
        payload or {},
        message,
        revision_mode=revision_mode,
        feedback=feedback,
        expansion_error=expansion_error,
    )
    return f"""【角色设定】
你是一名小说章节修订编辑。

你的任务是：
保留原有章节全部内容，
根据修改意见修正指定问题，
并在必要时补强细节、动作、情绪、对话和因果链。

────────────────────────

【输入说明】
用户提供：
* outline_id
* novel_id
* world_id
* worldview_id
* 原始章节（content 或 summary）
* 修改意见

其中：
content / summary 为当前完整章节正文。

【任务上下文】
{task_context}

【RAG 上下文】
{rag_context}

────────────────────────

【修改规则】
本任务是：修改并扩写（Modify + Expand）
不是：
* 总结
* 概括
* 提炼
* 压缩
* 全盘重写

必须保留：
* 所有未被要求删除的场景
* 所有未被要求删除的事件
* 所有未被要求删除的人物行为
* 所有未被要求删除的冲突
* 所有未被要求删除的对话
* 所有未被要求删除的伏笔

不得删除未被点名修改的内容。
不得跳过原剧情。
不得把多个场景合并成概述。

────────────────────────

【修改内容】
优先处理修改意见直接点名的问题。
然后在相关位置补强：
* 动作过程
* 冲突升级过程
* 人物行为逻辑
* 情绪变化
* 对话展开
* 场景衔接
* 因果链
* 阶段结果
* 后续影响

禁止只修改措辞。
禁止同义改写。

────────────────────────

【因果链规则】
重要段落尽量补充：
起因
→ 触发
→ 发展
→ 结果
→ 影响

────────────────────────

【世界观规则】
修改后的内容必须遵守父级 outline / novel / world / worldview 的全部约束。
不得新增违反约束的设定。

────────────────────────

【长度规则】
修改后内容长度：
不得低于原文。

如果修改范围很小，
至少保留原文总量不缩短。

禁止输出比输入更短。

────────────────────────

【输出规则】
payload.content：
写修改并补强后的完整章节正文。
不是摘要。
不是概述。
不是总结。
必须保留全部未被要求删除的原剧情。

────────────────────────

【执行顺序】
Review：
检查父级约束与修改意见。

Modify：
先按修改意见修正。

Expand：
只在相关位置补强细节。

Validate：
检查是否遗漏原剧情，是否误删未被点名修改内容。

如果发现输出比原文更短，
或出现摘要化倾向，
重新生成。

────────────────────────

【输入信息】
【说明】
请严格按照上面的执行顺序完成任务。

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留 outline_id、novel_id、worldview_id、world_id、chapter_id、id、target_id 和 name。
3. `payload.content` 必须是修改并补强后的完整章节正文，不得摘要化、概述化、压缩化。
4. 必须严格继承前文场景、人物状态、大纲任务和既定设定。
5. 输出 JSON 示例里的占位符只是结构说明，不是让你原样输出这些方括号文字。

只返回合法 JSON：
{{
  "metadata": {{"agent": "chapter_agent", "node": "modify_content", "entity_type": "chapter", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "id": "[保留输入中的 id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[章节标题]",
    "content": "[按修改意见修正并补强后的完整章节正文]"
  }},
  "modification_notes": "chapter_agent 本轮修正的正文范围",
  "change_summary": "相对输入 payload 的变化摘要"
}}
"""


def generate_content_modification(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "", expansion_error: str = "") -> Dict[str, Any]:
    """调用 LLM 根据审查意见或人工反馈修改章节正文内容。"""
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback, expansion_error=expansion_error)
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=MODIFY_CONTENT_AGENT_NAME)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} modification returned non-object JSON: {raw_content[:500]}")
    modified_payload = parsed.get("payload")
    if not isinstance(modified_payload, dict):
        raise ValueError(f"{AGENT_NAME} modification missing payload object: {raw_content[:500]}")
    _assert_chapter_not_simplified(_chapter_seed(payload), modified_payload.get("content", ""), stage="modify_content")
    if payload.get("name") and revision_mode != "full_rewrite":
        modified_payload["name"] = payload["name"]
    return {"payload": modified_payload, "llm_invoked": True, "agent_name": MODIFY_CONTENT_AGENT_NAME, "llm_agent_name": MODIFY_CONTENT_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "modification_notes": parsed.get("modification_notes", ""), "change_summary": parsed.get("change_summary", "")}


def input_node(state: ChapterAgentState) -> ChapterAgentState:
    """输入节点：记录章节 payload、父级 outline_id、novel_id、worldview_id、world_id 和用户消息。"""
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: ChapterAgentState) -> ChapterAgentState:
    """初始扩充节点：调用 chapter_agent LLM 扩充章节正文并提交世界审查。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=state.get("feedback", ""))
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    return {"initial_expansion": expansion, "expanded_input": expansion["expanded_input"], "pending_payload": expansion["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_review", "status": "reviewing_world"}


def modify_content_node(state: ChapterAgentState) -> ChapterAgentState:
    """修改内容节点：按审查意见或人工反馈调用 chapter_agent LLM 修改章节正文。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    feedback = state.get("world_review_feedback") or state.get("worldview_review_feedback") or state.get("novel_review_feedback") or state.get("outline_review_feedback") or state.get("chapter_review_feedback") or state.get("review_feedback") or state.get("feedback", "")
    modification = generate_content_modification(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=feedback, expansion_error=feedback)
    iteration = int(state.get("iterations") or 0) + 1
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("modify_content", "completed", {"payload": payload, "feedback": feedback, "revision_mode": state.get("revision_mode")}, {**modification, "iteration": iteration}))
    return {"modification": modification, "pending_payload": modification["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_review", "status": "reviewing_world"}


_chapter_review_nodes = build_chapter_review_nodes(
    node_factory=_node,
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)
world_review_node = _chapter_review_nodes["world_review_node"]
route_after_world_review = _chapter_review_nodes["route_after_world_review"]
worldview_review_node = _chapter_review_nodes["worldview_review_node"]
route_after_worldview_review = _chapter_review_nodes["route_after_worldview_review"]
novel_review_node = _chapter_review_nodes["novel_review_node"]
route_after_novel_review = _chapter_review_nodes["route_after_novel_review"]
outline_review_node = _chapter_review_nodes["outline_review_node"]
route_after_outline_review = _chapter_review_nodes["route_after_outline_review"]
chapter_review_node = _chapter_review_nodes["chapter_review_node"]
route_after_chapter_review = _chapter_review_nodes["route_after_chapter_review"]


def human_node(state: ChapterAgentState) -> ChapterAgentState:
    """人工节点：等待批准、局部重写或中止，并记录反馈与修改模式。"""
    decision = state.get("decision")
    feedback = state.get("feedback", "")
    revision_mode = state.get("revision_mode") or "partial_rewrite"
    if not decision:
        user_input = interrupt({
            "agent": AGENT_NAME,
            "status": "waiting_human",
            "payload": state.get("pending_payload"),
            "review_errors": state.get("review_errors", []),
            "world_review_errors": state.get("world_review_errors", []),
            "worldview_review_errors": state.get("worldview_review_errors", []),
            "novel_review_errors": state.get("novel_review_errors", []),
            "outline_review_errors": state.get("outline_review_errors", []),
            "chapter_review_errors": state.get("chapter_review_errors", []),
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


def route_after_human(state: ChapterAgentState) -> str:
    """人工节点路由：批准进入写库，要求修改进入修改内容节点，中止结束。"""
    if state.get("decision") == "approve":
        return "commit"
    if state.get("decision") == "reject":
        return "end"
    return "modify_content"


def commit_node(state: ChapterAgentState) -> ChapterAgentState:
    """写库节点：人工批准后真实创建或更新 MongoDB prose 集合。"""
    db = get_mongodb_db()
    action = state.get("action", "create")
    payload = dict(state.get("pending_payload") or {})
    if action == "create":
        chapter_id = payload.get("chapter_id") or payload.get("id") or f"chapter_{uuid.uuid4().hex[:8]}"
        doc = {
            "id": chapter_id,
            "scene_id": chapter_id,
            "type": "prose",
            "title": payload["name"],
            "content": payload.get("content", ""),
            "outline_id": payload["outline_id"],
            "chapter_outline_id": payload.get("chapter_outline_id"),
            "novel_id": payload.get("novel_id"),
            "worldview_id": payload.get("worldview_id"),
            "world_id": payload.get("world_id"),
        }
        db["prose"].insert_one(doc)
        result = doc
    elif action == "update":
        target_id = payload["target_id"]
        update = {}
        if "name" in payload:
            update["title"] = payload["name"]
        if "content" in payload:
            update["content"] = payload["content"]
        if "chapter_outline_id" in payload:
            update["chapter_outline_id"] = payload.get("chapter_outline_id")
        db["prose"].update_one({"$or": [{"id": target_id}, {"scene_id": target_id}, {"prose_id": target_id}]}, {"$set": update})
        result = {"id": target_id, **update}
    else:
        raise ValueError(f"{AGENT_NAME} does not handle delete operations")
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": result}))
    return {"commit_result": result, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}


workflow = StateGraph(ChapterAgentState)
workflow.add_node("input", input_node)
workflow.add_node("initial_expansion", initial_expansion_node)
workflow.add_node("world_review", world_review_node)
workflow.add_node("worldview_review", worldview_review_node)
workflow.add_node("novel_review", novel_review_node)
workflow.add_node("outline_review", outline_review_node)
workflow.add_node("chapter_review", chapter_review_node)
workflow.add_node("human", human_node)
workflow.add_node("modify_content", modify_content_node)
workflow.add_node("commit", commit_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "initial_expansion")
workflow.add_edge("initial_expansion", "world_review")
workflow.add_conditional_edges("world_review", route_after_world_review, {"worldview_review": "worldview_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("worldview_review", route_after_worldview_review, {"novel_review": "novel_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("novel_review", route_after_novel_review, {"outline_review": "outline_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("outline_review", route_after_outline_review, {"chapter_review": "chapter_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("chapter_review", route_after_chapter_review, {"modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("human", route_after_human, {"modify_content": "modify_content", "commit": "commit", "end": END})
workflow.add_edge("modify_content", "world_review")
workflow.add_edge("commit", END)

app = workflow.compile(checkpointer=MemorySaver())
