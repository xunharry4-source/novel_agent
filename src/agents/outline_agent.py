"""Outline Agent - 独立大纲工作流。

流程：Input -> Initial Expansion -> World Review -> Worldview Review -> Novel Review -> Human -> Commit；
人工不同意或任一审查失败进入 Modify Content，再回到 World Review。
"""

import json
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.review_nodes.novel_review import make_novel_review_node, make_novel_review_route
from src.agents.review_nodes.world_review import make_world_review_node, make_world_review_route
from src.agents.review_nodes.worldview_review import make_worldview_review_node, make_worldview_review_route
from src.common.config_utils import get_config
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db, get_unified_context, parse_json_safely
from src.common.revision_mode_prompt import build_revision_mode_instruction


AGENT_NAME = "outline_agent"
INITIAL_EXPANSION_AGENT_NAME = "outline_agent_initial_expansion"
MODIFY_CONTENT_AGENT_NAME = "outline_agent_modify_content"
HUMAN_FEEDBACK_AGENT_NAME = "outline_agent_human_feedback_modify_content"
ENTITY_TYPE = "outline"
PRIMARY_FIELD = "summary"
MAX_AUTO_REVIEW_ITERATIONS = 3
WORKFLOW_DESCRIPTION = "大纲 Agent 流程：接收大纲输入 -> 初始扩充大纲内容 -> 世界审查 -> 世界观审查 -> 小说审查 -> 人工确认 -> 批准后写入 outlines；人工不同意或任一审查失败进入修改内容节点，再从世界审查重新开始。"
WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收大纲输入",
        "function": "接收大纲 payload 与父级 novel_id",
        "description": "记录大纲名称、摘要、novel_id、world_id、worldview_id、outline_id 或 target_id，确保大纲归属小说而不是世界观。",
    },
    "initial_expansion": {
        "step_index": 2,
        "step_title": "步骤 2：初始扩充",
        "function": "调用 outline_agent 专属 LLM 整理大纲输入",
        "description": "对小说简介、大纲目标和局部重写范围进行初步拆解，直接生成可审查的大纲 payload，明确章节层级、关键冲突、受影响路径、父级 novel_id/world_id/worldview_id 和必须继承的设定约束；不得写库，不得跳过 LLM，不得使用通用 Prompt。",
    },
    "world_review": {
        "step_index": 3,
        "step_title": "步骤 3：世界审查",
        "function": "检查大纲是否违反世界禁止规则与基本设定",
        "description": "基于所属世界的 forbidden_rules 与 basic_settings 审查大纲是否违反世界根禁令、时代边界、力量体系、地理边界、组织结构或资源机制；失败时写入 world_review_feedback 并进入修改内容节点。",
    },
    "worldview_review": {
        "step_index": 4,
        "step_title": "步骤 4：世界观审查",
        "function": "检查大纲是否违反关联世界观设定",
        "description": "基于 worldview_id 与同一 world_id 下已有世界观 Canon 审查大纲是否出现设定冲突、规则冲突、历史地理矛盾或 Lore 使用错误；失败时写入 worldview_review_feedback 并进入修改内容节点。",
    },
    "novel_review": {
        "step_index": 5,
        "step_title": "步骤 5：小说审查",
        "function": "检查大纲是否违反小说禁止规则、基本设定和主线约束",
        "description": "基于 novel_id 对应小说的 forbidden_rules、basic_settings、主角底线、主线冲突、叙事基调、时间线和人物关系规则审查大纲是否偏离小说设计；失败时写入 novel_review_feedback 并进入修改内容节点。",
    },
    "human": {
        "step_index": 6,
        "step_title": "步骤 6：人工确认",
        "function": "等待用户批准、局部重写或中止",
        "description": "世界审查、世界观审查、小说审查均通过后等待用户批准写库；用户不同意则选择 partial_rewrite、content_rewrite 或 full_rewrite 提交反馈进入修改内容节点，修改后必须重新通过三个审查节点。",
    },
    "modify_content": {
        "step_index": 7,
        "step_title": "步骤 7：修改内容",
        "function": "根据审查意见或人工反馈调用 outline_agent 专属 LLM 修改大纲内容",
        "description": "仅在世界审查失败、世界观审查失败、小说审查失败或人工不同意时执行。根据审查反馈或用户反馈修正大纲 payload，保留未要求修改的内容和业务 ID；不得写库，不得使用通用 Prompt。",
    },
    "commit": {
        "step_index": 8,
        "step_title": "步骤 8：写库固化",
        "function": "执行 outlines 集合写入",
        "description": "人工批准后创建或更新 MongoDB outlines 集合，写入 outline_id、novel_id、worldview_id、world_id 与最终大纲摘要。",
    },
}
NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入必须包含 novel_id，并可包含 world_id、worldview_id、outline_id 或 target_id；大纲父级是小说。",
        "output_annotation": "输出 accepted=true，并锁定大纲 payload 与父级关系。",
        "next_step_annotation": "下一步进入初始扩充节点，先拆解大纲目标、层级和局部重写范围。",
    },
    "initial_expansion": {
        "input_annotation": "输入是用户消息、大纲 payload、人工反馈和修改模式；必须保留 novel_id、world_id、worldview_id 和受影响路径。",
        "output_annotation": "输出可审查的大纲 payload、expanded_input、llm_call、raw_response 和 parsed_response。",
        "next_step_annotation": "下一步进入世界审查，先检查是否违反世界禁止规则与基本设定。",
    },
    "world_review": {
        "input_annotation": "输入是当前大纲 payload、world_id 和世界 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 world_review_feedback。",
        "next_step_annotation": "通过则进入世界观审查；失败且未超出上限则进入修改内容节点。",
    },
    "worldview_review": {
        "input_annotation": "输入是通过世界审查后的大纲 payload、worldview_id 和已有世界观 Canon 上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 worldview_review_feedback。",
        "next_step_annotation": "通过则进入小说审查；失败且未超出上限则进入修改内容节点。",
    },
    "novel_review": {
        "input_annotation": "输入是通过世界观审查后的大纲 payload、novel_id 和小说 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；失败原因会写入 novel_review_feedback。",
        "next_step_annotation": "通过则进入人工确认；失败且未超出上限则进入修改内容节点。",
    },
    "human": {
        "input_annotation": "输入是三个审查节点通过后的大纲内容、审查意见、用户反馈、修改模式和手动编辑标记。",
        "output_annotation": "输出记录用户决策，尤其是 partial_rewrite 时用户指定的局部修改范围。",
        "next_step_annotation": "批准则写库；要求修改则进入修改内容节点并再次审查；中止则结束。",
    },
    "modify_content": {
        "input_annotation": "输入是当前大纲 payload、world_review_feedback、worldview_review_feedback、novel_review_feedback、人工反馈、修改模式和局部路径。",
        "output_annotation": "输出修改后的大纲 payload、llm_call、raw_response、parsed_response 和 change_summary。",
        "next_step_annotation": "下一步回到世界审查，必须连续通过世界、世界观、小说三个审查节点后才能进入人工确认。",
    },
    "commit": {
        "input_annotation": "输入是人工批准后的最终 outline payload。",
        "output_annotation": "输出是真实 MongoDB outlines 写入结果，包含 outline_id、novel_id、worldview_id 和 world_id。",
        "next_step_annotation": "写库完成后工作流结束。",
    },
}


class OutlineAgentState(TypedDict, total=False):
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
    """生成本次 outline_agent LLM 调用的中文可审计元数据。"""
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
    """真实调用 outline_agent 对应 LLM；空响应直接报错，禁止伪成功。"""
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


def _normalize_outline_text(text: Any) -> str:
    return "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())


def _extract_outline_markers(text: str) -> List[str]:
    markers: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith("第")
            or stripped.startswith("卷名")
            or stripped.startswith("本卷简介")
            or stripped.startswith("开篇")
            or stripped.startswith("主要人物")
            or stripped.startswith("主要矛盾")
            or stripped.startswith("主要事件")
            or stripped.startswith("转折")
            or stripped.startswith("高潮")
            or stripped.startswith("结局")
            or stripped.startswith("尾声")
            or stripped.startswith("主题")
            or stripped.startswith("核心矛盾")
        ):
            markers.append(stripped)
    return markers


def _assert_outline_not_simplified(source_summary: Any, candidate_summary: Any, *, stage: str) -> None:
    source = _normalize_outline_text(source_summary)
    candidate = _normalize_outline_text(candidate_summary)
    if not source or not candidate:
        return

    source_len = len(source)
    candidate_len = len(candidate)
    if source_len >= 800 and candidate_len < int(source_len * 0.85):
        raise ValueError(
            f"{AGENT_NAME} {stage} compressed long outline too aggressively: "
            f"input_chars={source_len}, output_chars={candidate_len}"
        )

    source_markers = _extract_outline_markers(source)
    if len(source_markers) >= 3:
        missing = [marker for marker in source_markers if marker not in candidate]
        if len(missing) > max(1, len(source_markers) // 3):
            missing_preview = " | ".join(missing[:5])
            raise ValueError(
                f"{AGENT_NAME} {stage} dropped outline structure markers: {missing_preview}"
            )


def _build_outline_task_context(
    action: str,
    payload: Dict[str, Any],
    message: str,
    *,
    revision_mode: Optional[str],
    feedback: str,
    expansion_error: str = "",
    user_feedback: str = "",
    review_feedback: str = "",
) -> str:
    task_context = {
        "action": action,
        "message": message,
        "revision_mode": revision_mode or "initial_expansion",
        "payload": payload or {},
    }
    if user_feedback or feedback:
        task_context["user_feedback"] = user_feedback or feedback
    if review_feedback or expansion_error:
        task_context["review_feedback"] = review_feedback or expansion_error
    return json.dumps(task_context, ensure_ascii=False, indent=2)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    """构造带中文步骤说明、节点注解、输入输出说明的工作流节点。"""
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    """构造 outline_agent 初始扩充 Prompt，使用结构化模板明确大纲任务。"""
    task_context = _build_outline_task_context(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    return f"""【角色设定】
你是一名小说大纲扩写编辑。

你的任务是：
保留原有大纲全部内容，
并在此基础上扩写细节、冲突、因果链和章节内容。

────────────────────────

【输入说明】
用户提供：
* novel_id
* world_id
* worldview_id
* 大纲草稿（summary）

其中：
summary 为原始大纲正文。

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
* 所有章节
* 所有事件
* 所有人物
* 所有冲突
* 所有势力
* 所有伏笔

不得删除。
不得跳过。
不得合并多个事件。

────────────────────────

【扩写内容】
优先扩写已有剧情。
增加：
* 事件过程
* 冲突升级过程
* 人物行为逻辑
* 势力互动
* 因果链
* 阶段结果
* 后续影响

禁止只修改措辞。
禁止同义改写。

────────────────────────

【因果链规则】
重要事件尽量补充：
起因
→ 触发
→ 发展
→ 结果
→ 影响

────────────────────────

【世界观规则】
扩写内容必须遵守父级 world / worldview / novel 的全部约束。
不得新增违反约束的设定。
尤其不得出现：
1. 魔法
2. 超维

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
payload.summary：
写扩写后的完整大纲。
不是摘要。
不是概述。
不是总结。
必须保留全部原剧情并增加细节。

expanded_input.summary_seed：
写用户提交的原始大纲内容。
原文不得修改。

────────────────────────

【执行顺序】
Review：
检查世界观与父级约束。

Expand：
保留原文并扩写。

Validate：
检查是否遗漏原剧情。

如果发现输出比原文更短，
或出现摘要化倾向，
重新生成。

────────────────────────

【输入信息】
【说明】
请严格按照上面的执行顺序完成任务。

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留输入中的 novel_id、world_id、worldview_id、outline_id、target_id 和 name。
3. `payload.summary` 必须是扩写后的完整大纲正文，不得摘要化、概述化、压缩化。
4. `expanded_input.summary_seed` 必须保留用户提交的原始大纲全文，不得改写。
5. 输出内容必须聚焦大纲结构扩写，不得漂移到正式章节正文、小说项目或世界观条目。
6. 输出 JSON 示例里的占位符只是结构说明，不是让你原样输出这些方括号文字。

只返回合法 JSON：
{{
  "metadata": {{"agent": "outline_agent", "node": "initial_expansion", "entity_type": "outline", "action": "{action}"}},
  "payload": {{
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "outline_id": "[保留输入中的 outline_id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[大纲名称]",
    "summary": "[扩写后的完整大纲]"
  }},
  "expanded_input": {{
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "outline_id": "[保留输入中的 outline_id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[大纲名称]",
    "summary_seed": "[用户提交的原始大纲全文]",
    "structure_goal": "[本轮扩写的结构目标]",
    "affected_paths": ["[本轮重点扩写的章节或结构路径]"],
    "key_conflicts": ["[需要重点展开的冲突]"],
    "parent_constraints": ["[必须保留的父级约束]"]
  }},
  "expansion_notes": "[本轮扩写补强了哪些结构与冲突]"
}}
"""


def generate_initial_expansion(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "") -> Dict[str, Any]:
    """调用 LLM 生成大纲初始扩充结果，确保第二节点真实使用 outline_agent LLM。"""
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
    _assert_outline_not_simplified(payload.get("summary", ""), initial_payload.get("summary", ""), stage="initial_expansion")
    if payload.get("name") and revision_mode != "full_rewrite":
        initial_payload["name"] = payload["name"]
    return {"payload": initial_payload, "expanded_input": expanded_input, "llm_invoked": True, "agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    """构造 outline_agent 修改内容 Prompt，使用结构化模板明确局部修正任务。"""
    user_feedback = str(feedback or "").strip()
    review_feedback = str(expansion_error or "").strip()
    rag_context = get_unified_context(
        f"{message}\n{payload.get('name', '')}\n{payload.get('summary', '')}",
        outline_id=str(payload.get("outline_id") or payload.get("target_id") or "default"),
        worldview_id=str(payload.get("worldview_id") or "default_wv"),
    )
    task_context = _build_outline_task_context(
        action,
        payload or {},
        message,
        revision_mode=revision_mode,
        feedback=user_feedback,
        expansion_error=review_feedback,
        user_feedback=user_feedback,
        review_feedback=review_feedback,
    )
    mode_instruction = build_revision_mode_instruction(
        revision_mode,
        entity_label="分卷大纲",
        primary_field_label="payload.summary",
        content_rewrite_scope="允许围绕大纲正文 summary 做大段或全文级重写；如标题 name 未被明确要求修改，默认保留原值。",
        full_rewrite_scope="允许整体重写 name、summary 以及为结构一致性所必需的剧情编排说明。",
    )
    return f"""【角色设定】
你是一名小说大纲修订编辑。

你的任务是：
保留原有大纲全部内容，
根据用户反馈和系统审查问题定点修订当前大纲，
只在完成修正所必需时补充缺失的因果或约束说明。

────────────────────────

【输入说明】
用户提供：
* novel_id
* world_id
* worldview_id
* 原始大纲（summary）
* 修改意见

其中：
summary 为当前完整大纲正文。

【任务上下文】
{task_context}

【用户反馈】
{user_feedback or "无"}

【系统审查问题】
{review_feedback or "无"}

【RAG 上下文】
{rag_context}

【修改模式说明】
{mode_instruction}

────────────────────────

【通用修订规则】
必须保留：
* 所有未被要求删除的章节
* 所有未被要求删除的事件
* 所有未被要求删除的人物
* 所有未被要求删除的冲突
* 所有未被要求删除的势力
* 所有未被要求删除的伏笔

不得删除未被允许删除的内容。
不得跳过原剧情。
不得脱离父级约束另起设定。
不得合并多个关键事件。

────────────────────────

【修改内容】
优先处理用户反馈直接点名的问题。
如果存在系统审查问题，必须逐条修正。
只允许补充为完成修正所必需的内容，例如：
* 缺失的因果链
* 必要的世界观约束说明
* 被用户点名要求补充的局部逻辑

禁止只修改措辞。
禁止同义改写。
禁止为了“补长”而新增无关剧情。
禁止把修改任务改写成重新创作。

────────────────────────

【因果链规则】
重要事件尽量补充：
起因
→ 触发
→ 发展
→ 结果
→ 影响

────────────────────────

【世界观规则】
修改后的内容必须遵守父级 world / worldview / novel 的全部约束。
不得新增违反约束的设定。
尤其不得出现：
1. 魔法
2. 超维

────────────────────────

【长度规则】
修改后内容长度：
不得低于原文，除非用户明确要求删除内容。

如果修改范围很小，
应保持原文主体结构与篇幅基本不变。

禁止输出比输入更短。

────────────────────────

【输出规则】
payload.summary：
写按意见修正后的完整大纲。
不是摘要。
不是概述。
不是总结。
必须保留全部未被要求删除的原剧情。

────────────────────────

【执行顺序】
Review：
逐条读取用户反馈与系统审查问题。

Modify：
只修正被指出的问题。

Validate：
检查是否遗漏原剧情，是否误删未被点名修改内容，是否引入新的设定冲突。

如果发现输出比原文更短，
或出现摘要化倾向，
重新生成。

────────────────────────

【输入信息】
【说明】
请严格按照上面的执行顺序完成任务。

【输出要求】
1. 只返回合法 JSON，不得返回解释文字，不得写库。
2. 必须保留输入中的 novel_id、world_id、worldview_id、outline_id、target_id 和 name。
3. `payload.summary` 必须是按意见修正后的完整大纲正文，不得摘要化、概述化、压缩化。
4. 输出内容必须聚焦大纲结构修改，不得漂移到正式章节正文、小说项目或世界观条目。
5. 输出 JSON 示例里的占位符只是结构说明，不是让你原样输出这些方括号文字。

只返回合法 JSON：
{{
  "metadata": {{"agent": "outline_agent", "node": "modify_content", "entity_type": "outline", "action": "{action}"}},
  "payload": {{
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "outline_id": "[保留输入中的 outline_id]",
    "target_id": "[保留输入中的 target_id]",
    "name": "[大纲名称]",
    "summary": "[按意见修正后的完整大纲]"
  }},
  "modification_notes": "[本轮具体修正了哪些章节、冲突或结构]",
  "change_summary": "[相对输入大纲的修改摘要]"
}}
"""


def generate_content_modification(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str] = None, feedback: str = "", expansion_error: str = "", llm_agent_name: Optional[str] = None) -> Dict[str, Any]:
    """调用 LLM 根据审查意见或人工反馈修改大纲内容。"""
    prompt = build_modification_prompt(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback, expansion_error=expansion_error)
    agent_name = llm_agent_name or MODIFY_CONTENT_AGENT_NAME
    raw_content, llm_call = _invoke_llm(prompt, llm_agent_name=agent_name)
    parsed = parse_json_safely(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{AGENT_NAME} modification returned non-object JSON: {raw_content[:500]}")
    modified_payload = parsed.get("payload")
    if not isinstance(modified_payload, dict):
        raise ValueError(f"{AGENT_NAME} modification missing payload object: {raw_content[:500]}")
    _assert_outline_not_simplified(payload.get("summary", ""), modified_payload.get("summary", ""), stage="modify_content")
    if payload.get("name") and revision_mode != "full_rewrite":
        modified_payload["name"] = payload["name"]
    return {"payload": modified_payload, "llm_invoked": True, "agent_name": agent_name, "llm_agent_name": agent_name, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "modification_notes": parsed.get("modification_notes", ""), "change_summary": parsed.get("change_summary", "")}


def input_node(state: OutlineAgentState) -> OutlineAgentState:
    """输入节点：记录大纲 payload、父级 novel_id、world_id、worldview_id 和用户消息。"""
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {"nodes": nodes, "pending_payload": payload, "current_node": "initial_expansion", "status": "running"}


def initial_expansion_node(state: OutlineAgentState) -> OutlineAgentState:
    """初始扩充节点：调用 outline_agent LLM 扩充大纲内容并提交世界审查。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    expansion = generate_initial_expansion(state.get("action", "create"), payload, state.get("message", ""), revision_mode=state.get("revision_mode"), feedback=state.get("feedback", ""))
    nodes = list(state.get("nodes") or [])
    iteration = int(state.get("iterations") or 0) + 1
    nodes.append(_node("initial_expansion", "completed", {"payload": payload, "feedback": state.get("feedback", "")}, {**expansion, "iteration": iteration}))
    return {"initial_expansion": expansion, "expanded_input": expansion["expanded_input"], "pending_payload": expansion["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_review", "status": "reviewing_world"}


def modify_content_node(state: OutlineAgentState) -> OutlineAgentState:
    """修改内容节点：按审查意见或人工反馈调用 outline_agent LLM 修改大纲内容。"""
    payload = dict(state.get("pending_payload") or state.get("payload") or {})
    manual_edit = bool(state.get("manual_edit"))
    user_feedback = str(state.get("feedback") or "")
    review_feedback = (
        state.get("world_review_feedback")
        or state.get("worldview_review_feedback")
        or state.get("novel_review_feedback")
        or state.get("review_feedback")
        or ""
    )
    feedback = user_feedback if manual_edit and user_feedback else str(review_feedback or "") or user_feedback
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
    nodes.append(
        _node(
            "modify_content",
            "completed",
            {
                "payload": payload,
                "feedback": feedback,
                "review_feedback": review_feedback,
                "revision_mode": state.get("revision_mode"),
                "manual_edit": manual_edit,
            },
            {**modification, "iteration": iteration},
        )
    )
    return {"modification": modification, "pending_payload": modification["payload"], "nodes": nodes, "iterations": iteration, "current_node": "world_review", "status": "reviewing_world"}


world_review_node = make_world_review_node(
    node_id="world_review",
    entity_type="outline_world_rules",
    reviewer="outline_world_rules_review_agent",
    passed_key="world_review_passed",
    errors_key="world_review_errors",
    feedback_key="world_review_feedback",
    next_node="worldview_review",
    next_status="reviewing_worldview",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_world_review = make_world_review_route(
    passed_key="world_review_passed",
    next_node="worldview_review",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)
worldview_review_node = make_worldview_review_node(
    node_id="worldview_review",
    entity_type="outline_worldview_rules",
    reviewer="outline_worldview_rules_review_agent",
    passed_key="worldview_review_passed",
    errors_key="worldview_review_errors",
    feedback_key="worldview_review_feedback",
    next_node="novel_review",
    next_status="reviewing_novel",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_worldview_review = make_worldview_review_route(
    passed_key="worldview_review_passed",
    next_node="novel_review",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)
novel_review_node = make_novel_review_node(
    node_id="novel_review",
    entity_type="outline_novel_rules",
    reviewer="outline_novel_rules_review_agent",
    passed_key="novel_review_passed",
    errors_key="novel_review_errors",
    feedback_key="novel_review_feedback",
    next_node="human",
    next_status="waiting_human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
    node_factory=_node,
)
route_after_novel_review = make_novel_review_route(
    passed_key="novel_review_passed",
    next_node="human",
    max_auto_review_iterations=MAX_AUTO_REVIEW_ITERATIONS,
)


def human_node(state: OutlineAgentState) -> OutlineAgentState:
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


def route_after_human(state: OutlineAgentState) -> str:
    """人工节点路由：批准进入写库，要求修改进入修改内容节点，中止结束。"""
    if state.get("decision") == "approve":
        return "commit"
    if state.get("decision") == "reject":
        return "end"
    return "modify_content"


def commit_node(state: OutlineAgentState) -> OutlineAgentState:
    """写库节点：人工批准后真实创建或更新 MongoDB outlines 集合。"""
    db = get_mongodb_db()
    action = state.get("action", "create")
    payload = dict(state.get("pending_payload") or {})
    if action == "create":
        outline_id = payload.get("outline_id") or f"outline_{uuid.uuid4().hex[:8]}"
        doc = {"outline_id": outline_id, "id": outline_id, "novel_id": payload["novel_id"], "world_id": payload.get("world_id"), "worldview_id": payload.get("worldview_id"), "name": payload["name"], "summary": payload.get("summary", "")}
        db["outlines"].insert_one(doc)
        result = doc
    elif action == "update":
        target_id = payload["target_id"]
        update = {k: payload[k] for k in ("name", "summary", "worldview_id") if k in payload}
        db["outlines"].update_one({"$or": [{"outline_id": target_id}, {"id": target_id}]}, {"$set": update})
        result = {"outline_id": target_id, **update}
    else:
        raise ValueError(f"{AGENT_NAME} does not handle delete operations")
    nodes = list(state.get("nodes") or [])
    nodes.append(_node("commit", "completed", {"payload": payload}, {"result": result}))
    return {"commit_result": result, "committed": True, "nodes": nodes, "current_node": "commit", "status": "completed"}


workflow = StateGraph(OutlineAgentState)
workflow.add_node("input", input_node)
workflow.add_node("initial_expansion", initial_expansion_node)
workflow.add_node("world_review", world_review_node)
workflow.add_node("worldview_review", worldview_review_node)
workflow.add_node("novel_review", novel_review_node)
workflow.add_node("human", human_node)
workflow.add_node("modify_content", modify_content_node)
workflow.add_node("commit", commit_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "initial_expansion")
workflow.add_edge("initial_expansion", "world_review")
workflow.add_conditional_edges("world_review", route_after_world_review, {"worldview_review": "worldview_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("worldview_review", route_after_worldview_review, {"novel_review": "novel_review", "modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("novel_review", route_after_novel_review, {"modify_content": "modify_content", "human": "human"})
workflow.add_conditional_edges("human", route_after_human, {"modify_content": "modify_content", "commit": "commit", "end": END})
workflow.add_edge("modify_content", "world_review")
workflow.add_edge("commit", END)

app = workflow.compile(checkpointer=MemorySaver())
