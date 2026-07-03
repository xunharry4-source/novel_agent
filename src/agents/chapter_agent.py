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


def _truncate_text(value: Any, *, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _find_one_by_candidates(collection: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    for query in candidates:
        if not query:
            continue
        doc = collection.find_one(query)
        if doc:
            return doc
    return {}


def _resolve_world_doc(db: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    world_id = payload.get("world_id")
    if not world_id and payload.get("novel_id"):
        novel_doc = db["novels"].find_one({"novel_id": payload["novel_id"]}) or {}
        world_id = novel_doc.get("world_id")
    if not world_id and payload.get("outline_id"):
        outline_doc = _find_one_by_candidates(
            db["outlines"],
            [{"outline_id": payload["outline_id"]}, {"id": payload["outline_id"]}],
        )
        world_id = outline_doc.get("world_id")
    if not world_id and payload.get("worldview_id"):
        worldview_doc = db["worldviews"].find_one({"worldview_id": payload["worldview_id"]}) or {}
        world_id = worldview_doc.get("world_id")
    if not world_id:
        return {}
    return db["worlds"].find_one({"world_id": world_id}) or {}


def _resolve_novel_doc(db: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    novel_id = payload.get("novel_id")
    if not novel_id and payload.get("outline_id"):
        outline_doc = _find_one_by_candidates(
            db["outlines"],
            [{"outline_id": payload["outline_id"]}, {"id": payload["outline_id"]}],
        )
        novel_id = outline_doc.get("novel_id")
    if not novel_id:
        return {}
    return db["novels"].find_one({"novel_id": novel_id}) or {}


def _resolve_outline_doc(db: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    outline_id = payload.get("outline_id")
    if not outline_id:
        return {}
    return _find_one_by_candidates(
        db["outlines"],
        [{"outline_id": outline_id}, {"id": outline_id}],
    )


def _resolve_chapter_outline_doc(db: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    chapter_outline_id = payload.get("chapter_outline_id")
    if not chapter_outline_id:
        return {}
    return _find_one_by_candidates(
        db["prose"],
        [
            {"id": chapter_outline_id},
            {"scene_id": chapter_outline_id},
            {"prose_id": chapter_outline_id},
            {"chapter_id": chapter_outline_id},
        ],
    )


def _resolve_worldview_context(db: Any, payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    worldview_id = payload.get("worldview_id")
    worldview_doc: Dict[str, Any] = {}
    if worldview_id:
        worldview_doc = db["worldviews"].find_one({"worldview_id": worldview_id}) or {}
    if not worldview_doc:
        world_doc = _resolve_world_doc(db, payload)
        world_id = world_doc.get("world_id")
        if world_id:
            worldview_doc = db["worldviews"].find_one({"world_id": world_id}) or {}
            worldview_id = worldview_doc.get("worldview_id")
    else:
        worldview_id = worldview_doc.get("worldview_id") or worldview_id

    entries: List[Dict[str, Any]] = []
    if worldview_id:
        cursor = db["lore"].find({"worldview_id": worldview_id, "type": "worldview"})
        if hasattr(cursor, "sort"):
            try:
                cursor = cursor.sort([("updated_at", -1), ("created_at", -1)])
            except Exception:
                pass
        if hasattr(cursor, "limit"):
            try:
                cursor = cursor.limit(8)
            except Exception:
                pass
        for doc in cursor:
            entries.append(
                {
                    "id": doc.get("id"),
                    "name": doc.get("name") or doc.get("title"),
                    "path": doc.get("path") or doc.get("category"),
                    "content": _truncate_text(doc.get("content"), limit=1200),
                }
            )
            if len(entries) >= 8:
                break
    return worldview_doc, entries


def _build_parent_constraint_context(payload: Dict[str, Any]) -> str:
    try:
        db = get_mongodb_db()
    except Exception as exc:
        return f"读取父级强约束失败：{exc}"

    world_doc = _resolve_world_doc(db, payload)
    novel_doc = _resolve_novel_doc(db, payload)
    outline_doc = _resolve_outline_doc(db, payload)
    chapter_outline_doc = _resolve_chapter_outline_doc(db, payload)
    worldview_doc, worldview_entries = _resolve_worldview_context(db, payload)

    blocks = [
        "【父级世界设定】\n"
        + json.dumps(
            {
                "world_id": world_doc.get("world_id") or payload.get("world_id"),
                "name": world_doc.get("name"),
                "summary": _truncate_text(world_doc.get("summary"), limit=800),
                "forbidden_rules": world_doc.get("forbidden_rules") or [],
                "basic_settings": world_doc.get("basic_settings") or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        "【父级小说设定】\n"
        + json.dumps(
            {
                "novel_id": novel_doc.get("novel_id") or payload.get("novel_id"),
                "name": novel_doc.get("name"),
                "summary": _truncate_text(novel_doc.get("summary"), limit=800),
                "forbidden_rules": novel_doc.get("forbidden_rules") or [],
                "basic_settings": novel_doc.get("basic_settings") or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        "【父级分卷大纲】\n"
        + json.dumps(
            {
                "outline_id": outline_doc.get("outline_id") or outline_doc.get("id") or payload.get("outline_id"),
                "name": outline_doc.get("name") or outline_doc.get("title"),
                "summary": _truncate_text(outline_doc.get("summary"), limit=1200),
                "content": _truncate_text(outline_doc.get("content"), limit=1800),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "【父级世界观总设】\n"
        + json.dumps(
            {
                "worldview_id": worldview_doc.get("worldview_id") or payload.get("worldview_id"),
                "name": worldview_doc.get("name"),
                "summary": _truncate_text(worldview_doc.get("summary"), limit=1000),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "【父级世界观 Lore 条目】\n"
        + json.dumps(worldview_entries or [], ensure_ascii=False, indent=2),
    ]
    if chapter_outline_doc:
        blocks.append(
            "【当前章节内容对应的章节大纲】\n"
            + json.dumps(
                {
                    "chapter_outline_id": (
                        chapter_outline_doc.get("id")
                        or chapter_outline_doc.get("scene_id")
                        or chapter_outline_doc.get("prose_id")
                        or payload.get("chapter_outline_id")
                    ),
                    "name": chapter_outline_doc.get("name") or chapter_outline_doc.get("title"),
                    "content": _truncate_text(chapter_outline_doc.get("content"), limit=1800),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    # Inject chapter outline template as output format constraint
    template_id = payload.get("template_id")
    if template_id:
        template_doc = db["chapter_outline_templates"].find_one({"template_id": template_id}) or {}
        if template_doc:
            blocks.append(
                "【章节大纲输出格式模板】\n"
                "以下是用户为本次章节大纲选定的输出格式模板。"
                "你生成的章节大纲内容（payload.content）必须严格遵循该模板的结构、章节分段、标签和格式，"
                "同时填入符合本章节剧情的实际内容。"
                "模板中各占位符或示例文字应替换为本章实际内容，但整体结构不得改变。\n"
                + json.dumps(
                    {
                        "template_id": template_doc.get("template_id"),
                        "name": template_doc.get("name"),
                        "content": template_doc.get("content", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    return "\n\n".join(blocks)


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    """构造带中文步骤说明、节点注解、输入输出说明的工作流节点。"""
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {"node_id": node_id, "label": step["step_title"], **step, "node_annotation": f"{step['step_title']}：{step['description']}", **annotations, "status": status, "input": node_input, "output": output}


def build_initial_expansion_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str) -> str:
    """构造 chapter_agent 初始扩充 Prompt，使用结构化模板明确章节任务。"""
    task_context = _build_chapter_task_context(action, payload or {}, message, revision_mode=revision_mode, feedback=feedback)
    parent_context = _build_parent_constraint_context(payload or {})
    
    template_id = payload.get("template_id")
    if template_id:
        try:
            db = get_mongodb_db()
            template_doc = db["chapter_outline_templates"].find_one({"template_id": template_id}) or {}
            template_content = template_doc.get("content", "")
        except Exception:
            template_content = ""
            
        if template_content:
            return f"""【角色设定】
你是一名专业的小说章节大纲规划编辑。你的任务是根据用户输入的章节草稿（原始信息）、创作指示（给 Agent 的消息）以及已有的世界/小说设定，将下方的【章节大纲输出格式模板】完整填充为一份详尽、硬核的单章大纲。

────────────────────────

【任务目标】
1. 仔细阅读【章节大纲输出格式模板】，它定义了输出的完整结构。
2. 提取用户输入的草稿或摘要中的关键事实与情节。
3. 将模板中所有的 `[填空]`、`[如：...]` 等占位符，替换为符合本章剧情的实际设定和物理事实。
4. 严禁把表格结构、列表结构、复选框结构改写为纯散文段落。必须百分之百保留模板的所有 Markdown 标题、表格行、列表以及 `- [ ]` 复选框。
5. 表格中每一列的 `[填空]` 或 `[数值]` 都必须被填充为具体的文字或数值，不得残留 `[填空]` 占位符本身。

────────────────────────

【输入说明】
用户提供：
* outline_id
* novel_id
* world_id
* worldview_id
* 原始章节大纲草稿或摘要（content 或 summary）

其中：
content / summary 为原始章节大纲种子信息。

【任务上下文】
{task_context}

【父级强约束与设定】
在生成前，请阅读以下设定并确保情节逻辑、人物性格、世界物理法则与之完全吻合：
{parent_context}

────────────────────────

【章节大纲输出格式模板】
请严格按照此格式生成 payload.content 的最终内容。
{template_content}

────────────────────────

【输出要求】
1. 只返回合法 JSON，不得返回任何解释文字或 markdown 外部包装。
2. 必须保留 outline_id、novel_id、worldview_id、world_id、chapter_id、chapter_outline_id、id、target_id 和 template_id。
3. `payload.content` 必须是**严格遵循上述模板格式填充后**的完整 Markdown 格式大纲，必须保留模板中所有标题（如“📑 零、 世界动态看板”、“🔄 上游输入校验”等）、表格、复选框和分段结构，仅替换占位符。
4. `expanded_input.content_seed` 必须保留用户提交的原始大纲草稿全文。
5. 禁止输出纯散文。如果输出没有保持表格、复选框等 Markdown 模板结构，则视为任务失败。

只返回合法 JSON：
{{
  "metadata": {{"agent": "chapter_agent", "node": "initial_expansion", "entity_type": "chapter", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "chapter_outline_id": "[保留输入中的 chapter_outline_id]",
    "id": "[保留输入中的 id]",
    "target_id": "[保留输入中的 target_id]",
    "template_id": "{template_id}",
    "name": "[章节大纲标题，如 第X章：章名]",
    "content": "[严格按照【章节大纲输出格式模板】填充本章内容后的 Markdown 文本]"
  }},
  "expanded_input": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "chapter_outline_id": "[保留输入中的 chapter_outline_id]",
    "id": "[保留输入中的 id]",
    "target_id": "[保留输入中的 target_id]",
    "template_id": "{template_id}",
    "name": "[章节大纲标题]",
    "content_seed": "[用户提交的原始大纲草稿全文]"
  }},
  "expansion_notes": "[本轮大纲填充补充了哪些内容]"
}}
"""

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

【父级强约束】
在开始扩写前，必须先逐条读取并执行以下父级约束。
任何新增内容都必须同时满足世界、世界观、小说、分卷大纲，以及当前章节大纲（如果有）的要求。
不得忽略、替换、淡化或另起设定。
{parent_context}

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
2. 必须保留 outline_id、novel_id、worldview_id、world_id、chapter_id、chapter_outline_id、id、target_id 和用户指定片段。
3. `payload.content` 必须是扩写后的完整章节正文，不得摘要化、概述化、压缩化。
4. `expanded_input.content_seed` 必须保留用户提交的原始章节全文，不得改写。
5. 输出必须聚焦章节正文，不得漂移到世界规则、小说主线重设或大纲结局改写。
6. 输出 JSON 示例里的占位符只是结构说明，不是让你原样输出这些方括号文字。
7. 【格式约束】如果父级强约束中包含【章节大纲输出格式模板】，payload.content 的结构、分段方式、标签格式必须与该模板一致，仅将内容替换为本章剧情，不得改变模板的整体结构。

只返回合法 JSON：
{{
  "metadata": {{"agent": "chapter_agent", "node": "initial_expansion", "entity_type": "chapter", "action": "{action}"}},
  "payload": {{
    "outline_id": "[保留输入中的 outline_id]",
    "novel_id": "[保留输入中的 novel_id]",
    "world_id": "[保留输入中的 world_id]",
    "worldview_id": "[保留输入中的 worldview_id]",
    "chapter_id": "[保留输入中的 chapter_id]",
    "chapter_outline_id": "[保留输入中的 chapter_outline_id]",
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
    "chapter_outline_id": "[保留输入中的 chapter_outline_id]",
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
        
    # Skip the simplification validation if we are working with structured outline templates
    if not payload.get("template_id"):
        _assert_chapter_not_simplified(_chapter_seed(payload), initial_payload.get("content", ""), stage="initial_expansion")
        
    # Ensure template_id and chapter_outline_id are strictly aligned with input payload to override any LLM hallucinations
    if payload.get("template_id"):
        initial_payload["template_id"] = payload["template_id"]
    else:
        initial_payload.pop("template_id", None)

    if payload.get("chapter_outline_id"):
        initial_payload["chapter_outline_id"] = payload["chapter_outline_id"]
        expanded_input["chapter_outline_id"] = payload["chapter_outline_id"]
    else:
        initial_payload.pop("chapter_outline_id", None)
        expanded_input.pop("chapter_outline_id", None)
        
    if payload.get("name") and revision_mode != "full_rewrite":
        initial_payload["name"] = payload["name"]
    return {"payload": initial_payload, "expanded_input": expanded_input, "llm_invoked": True, "agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_agent_name": INITIAL_EXPANSION_AGENT_NAME, "llm_call": llm_call, "raw_response": raw_content, "parsed_response": parsed, "expansion_notes": parsed.get("expansion_notes", "")}


def build_modification_prompt(action: str, payload: Dict[str, Any], message: str, *, revision_mode: Optional[str], feedback: str, expansion_error: str = "") -> str:
    """\u6784\u9020 chapter_agent \u4fee\u6539\u5185\u5bb9 Prompt\uff0c\u4f7f\u7528\u7ed3\u6784\u5316\u6a21\u677f\u660e\u786e\u5c40\u90e8\u4fee\u6b63\u4efb\u52a1\u3002"""
    try:
        rag_context = get_unified_context(
            f"{message}\n{payload.get('name', '')}\n{payload.get('content', '') or payload.get('summary', '')}",
            outline_id=str(payload.get("outline_id") or "default"),
            worldview_id=str(payload.get("worldview_id") or "default_wv"),
        )
    except Exception:
        rag_context = ""
    task_context = _build_chapter_task_context(
        action,
        payload or {},
        message,
        revision_mode=revision_mode,
        feedback=feedback,
        expansion_error=expansion_error,
    )
    
    template_id = payload.get("template_id")
    if template_id:
        try:
            db = get_mongodb_db()
            template_doc = db["chapter_outline_templates"].find_one({"template_id": template_id}) or {}
            template_content = template_doc.get("content", "")
        except Exception:
            template_content = ""
            
        if template_content:
            return f"""\u3010\u89d2\u8272\u8bbe\u5b9a\u3011
\u4f60\u662f\u4e00\u540d\u4e13\u4e1a\u7684\u5c0f\u8bf4\u7ae0\u8282\u5927\u7eb2\u89c4\u5212\u7f16\u8f91\u3002\u4f60\u7684\u4efb\u52a1\u662f\u6839\u636e\u4fee\u6539\u610f\u89c1\uff08\u4eba\u5de5\u53cd\u9988/\u5ba1\u67e5\u9519\u8bef\uff09\uff0c\u5bf9\u73b0\u6709\u7684\u7ed3\u6784\u5316\u7ae0\u8282\u5927\u7eb2\u8fdb\u884c\u4fee\u6539\u548c\u4fee\u8ba2\uff0c\u540c\u65f6\u786e\u4fdd\u5927\u7eb2\u4e25\u683c\u4fdd\u6301\u539f\u6709\u7684\u6a21\u677f\u683c\u5f0f\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

            \u3010\u4efb\u52a1\u76ee\u6807\u3011
            1. \u4ed4\u7ec6\u9605\u8bfb\u4fee\u6539\u610f\u89c1\uff08\u5728\u4e0b\u65b9\u4efb\u52a1\u4e0a\u4e0b\u6587\u4e2d\uff09\u3002
            2. \u5728\u4fdd\u7559\u5927\u7eb2\u539f\u6709\u7ed3\u6784\uff08Markdown \u8868\u683c\u3001\u590d\u9009\u6846\u3001\u591a\u7ea7\u6807\u9898\uff09\u7684\u524d\u63d0\u4e0b\uff0c\u4fee\u6b63\u6307\u5b9a\u95ee\u9898\uff0c\u8c03\u6574\u5bf9\u5e94\u90e8\u5206\u3002
            3. \u4e25\u683c\u7981\u6b62\u628a\u8868\u683c\u7ed3\u6784\u3001\u5217\u8868\u7ed3\u6784\u3001\u590d\u9009\u6846\u7ed3\u6784\u6539\u5199\u6216\u5408\u5e76\u4e3a\u7eaf\u6563\u6587\u6bb5\u843d\u3002\u5fc5\u987b\u4fdd\u7559\u6a21\u677f\u7684\u6240\u6709 Markdown \u6807\u9898\u3001\u8868\u683c\u884c\u3001\u5217\u8868\u4ee5\u53ca `- [ ]` \u590d\u9009\u6846\u3002
            4. \u8868\u683c\u4e2d\u4fee\u6539\u548c\u586b\u5199\u540e\u4f9d\u7136\u8981\u4fdd\u6301\u6bcf\u5217\u6709\u5bf9\u5e94\u5185\u5bb9\uff0c\u4e0d\u53ef\u7559\u7a7a\u6216\u8fd8\u539f\u4e3a `[\u586b\u7a7a]` \u5360\u4f4d\u7b26\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

            \u3010\u8f93\u5165\u8bf4\u660e\u3011
            \u7528\u6237\u63d0\u4f9b\uff1a
            * outline_id
            * novel_id
            * world_id
            * worldview_id
            * \u539f\u59cb\u7ae0\u8282\u5927\u7eb2\uff08content \u6216 summary\uff0c\u5f5d\u524d\u4e3a\u7ed3\u6784\u5316\u5927\u7eb2\u5f62\u5f0f\uff09
            * \u4fee\u6539\u610f\u89c1

            \u3010\u4efb\u52a1\u4e0a\u4e0b\u6587\u3011
            {task_context}

            \u3010\u7236\u7ea7\u5f3a\u7ea6\u675f\u4e0e\u8bbe\u5b9a\u3011
            \u5728\u4fee\u6539\u524d\uff0c\u8bf7\u786e\u4fdd\u7b26\u5408\u4ee5\u4e0b\u5f3a\u7ea6\u675f\uff1a
            {_build_parent_constraint_context(payload or {})}

            \u3010RAG \u4e0a\u4e0b\u6587\u3011
            {rag_context}

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

            \u3010\u539f\u672c\u9075\u5faa\u7684\u7ae0\u8282\u5927\u7eb2\u8f33\u51fa\u683c\u5f0f\u6a21\u677f\u3011
            \u8fd9\u662f\u672c\u7ae0\u8282\u5927\u7eb2\u6240\u57fa\u4e8e\u7684\u6a21\u677f\uff0c\u4fee\u6539\u548c\u8f33\u51fa\u5fc5\u987b\u5728\u6b64\u6a21\u677f\u683c\u5f0f\u6846\u67b6\u4e0b\u8fdb\u884c\uff0c\u4e0d\u5f97\u7834\u574f\u683c\u5f0f\uff1a
            {template_content}

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

            \u3010\u8f33\u51fa\u8981\u6c42\u3011
            1. \u53ea\u8fd4\u56de\u5408\u6cd5 JSON\uff0c\u4e0d\u5f97\u8fd4\u56de\u4efb\u4f55\u89e3\u91ca\u6587\u5b57\u6216 markdown \u5916\u90e8\u5305\u88c5\u3002
            2. \u5fc5\u987b\u4fdd\u7559 outline_id\u3001novel_id\u3001worldview_id\u3001world_id\u3001chapter_id\u3001chapter_outline_id\u3001id\u3001target_id\u3001template_id \u548c name\u3002
            3. `payload.content` \u5fc5\u987b\u662f\u201c\u4fee\u6539\u548c\u586b\u5199\u540e\u4e14\u4fdd\u6301\u4e0a\u8ff0\u6a21\u677f\u683c\u5f0f\u201d\u7684\u5b8c\u6574 Markdown \u683c\u5f0f\u5927\u7eb2\uff0c\u5fc5\u987b\u4fdd\u7559\u6a21\u677f\u4e2d\u6240\u6709\u6807\u9898\u3001\u8868\u683c\u3001\u590d\u9009\u6846\u548c\u5206\u6bb5\u7ed3\u6784\uff0c\u4ec5\u5728\u7ed3\u6784\u5185\u4fee\u6539\u5177\u4f53\u6587\u672c\u3002
            4. \u5fc5\u987b\u4e25\u683c\u7ee7\u627f\u524d\u6587\u573a\u666f\u3001\u4eba\u7269\u72b6\u6001\u548c\u5927\u7eb2\u65e2\u5b9a\u8bbe\u5b9a\u3002
            5. \u5982\u679c\u8f33\u51fa\u7834\u574f\u4e86\u6a21\u677f\u683c\u5f0f\u6216\u5c06\u8868\u683c\u8f6c\u6362\u6210\u4e86\u6563\u6587\uff0c\u5219\u4efb\u52a1\u5931\u8d25\u3002

            \u53ea\u8fd4\u56de\u5408\u6cd5 JSON\uff1a
            {{
              \"metadata\": {{\"agent\": \"chapter_agent\", \"node\": \"modify_content\", \"entity_type\": \"chapter\", \"action\": \"{action}\"}},
              \"payload\": {{
                \"outline_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 outline_id]\",
                \"novel_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 novel_id]\",
                \"world_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 world_id]\",
                \"worldview_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 worldview_id]\",
                \"chapter_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 chapter_id]\",
                \"chapter_outline_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 chapter_outline_id]\",
                \"id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 id]\",
                \"target_id\": \"[\u4fdd\u7559\u8f93\u5165\u4e2d\u7684 target_id]\",
                \"template_id\": \"{template_id}\",
                \"name\": \"[\u7ae0\u8282\u5927\u7eb2\u6807\u9898]\",
                \"content\": \"[\u6309\u4fee\u6539\u610f\u89c1\u4fee\u6b63\u4e14\u4fdd\u6301\u6a21\u677f\u7ed3\u6784\u7684\u6700\u7ec8\u5927\u7eb2 Markdown \u6587\u672c]\"
              }},
              \"modification_notes\": \"[\u672c\u8f6e\u4fee\u6539\u4fee\u6539\u4e86\u54ea\u4e9b\u6a21\u677f\u9879]\",
              \"change_summary\": \"[\u5927\u7eb2\u4fee\u6539\u524d\u540e\u5dee\u5f02\u6458\u8981]\"
            }}
            """
    
    return f"""\u3010\u89d2\u8272\u8bbe\u5b9a\u3011
\u4f60\u662f\u4e00\u540d\u5c0f\u8bf4\u7ae0\u8282\u4fee\u8ba2\u7f16\u8f91\u3002

\u4f60\u7684\u4efb\u52a1\u662f\uff1a
\u4fdd\u7559\u539f\u6709\u7ae0\u8282\u5168\u90e8\u5185\u5bb9\uff0c
\u6839\u636e\u4fee\u6539\u610f\u89c1\u4fee\u6b63\u6307\u5b9a\u95ee\u9898\uff0c
\u5e76\u5728\u5fc5\u8981\u65f6\u8865\u5f3a\u7ed3\u8282\u3001\u52a8\u4f5c\u3001\u60c5\u7eea\u3001\u5bf9\u8bdd\u548c\u56e0\u679c\u94fe\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u8f93\u5165\u8bf4\u660e\u3011
\u7528\u6237\u63d0\u4f9b\uff1a
* outline_id
* novel_id
* world_id
* worldview_id
* \u539f\u59cb\u7ae0\u8282\uff08content \u6216 summary\uff09
* \u4fee\u6539\u610f\u89c1

\u5176\u4e2d\uff1a
content / summary \u4e3a\u5f5d\u524d\u5b8c\u6574\u7ae0\u8282\u6b63\u6587\u3002

\u3010\u4efb\u52a1\u4e0a\u4e0b\u6587\u3011
{task_context}

\u3010\u7236\u7ea7\u5f3a\u7ea6\u675f\u3011
\u5728\u5f00\u59cb\u4fee\u6539\u524d\uff0c\u5fc5\u987b\u5148\u9010\u6761\u9605\u8bfb\u5e76\u6267\u884c\u4ee5\u4e0b\u7236\u7ea7\u7ea6\u675f\u3002
\u4efb\u4f55\u4fee\u6b63\u3001\u8865\u5199\u3001\u6269\u5199\u90fd\u4e0d\u5f97\u8fdd\u80cc\u8fd9\u4e9b\u7ea6\u675f\uff1b\u5982\u679c\u4eba\u5de5\u610f\u89c1\u4e0e\u7236\u7ea7\u8bbe\u5b9a\u5bc5\u7a81\uff0c\u4f18\u5148\u4fdd\u7559\u7236\u7ea7\u8bbe\u5b9a\u5e76\u5728\u6b63\u6587\u5185\u4fee\u6b63\u5b9e\u73b0\u65b9\u5f0f\u3002
{_build_parent_constraint_context(payload or {})}

\u3010RAG \u4e0a\u4e0b\u6587\u3011
{rag_context}

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u4fee\u6539\u89c4\u5219\u3011
\u672c\u4efb\u52a1\u662f\uff1a\u4fee\u6539\u5e76\u6269\u5199\uff08Modify + Expand\uff09
\u4e0d\u662f\uff1a
* \u603b\u7ed3
* \u6982\u62ec
* \u63d0\u70bc
* \u538b\u7f29
* \u5168\u76d8\u91cd\u5199

\u5fc5\u987b\u4fdd\u7559\uff1a
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u573a\u666f
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u4e8b\u4ef6
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u4eba\u7269\u884c\u4e3a
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u5bc5\u7a81
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u5bf9\u8bdd
* \u6240\u6709\u672a\u88ab\u8981\u6c42\u5220\u9664\u7684\u4f0f\u7b14

\u4e0d\u5f97\u5220\u9664\u672a\u88ab\u70b9\u540d\u4fee\u6539\u7684\u5185\u5bb9\u3002
\u4e0d\u5f97\u8df3\u8fc7\u539f\u5267\u60c5\u3002
\u4e0d\u5f97\u628a\u591a\u4e2a\u573a\u666f\u5408\u5e76\u6210\u6982\u8ff0\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u4fee\u6539\u5185\u5bb9\u3011
\u4f18\u5148\u5904\u7406\u4fee\u6539\u610f\u89c1\u76f4\u63a5\u70b9\u540d\u7684\u95ee\u9898\u3002
\u7136\u540e\u5728\u76f8\u5173\u4f4d\u7f6e\u8885\u5f3a\uff1a
* \u52a8\u4f5c\u8fc7\u7a0b
* \u5bc5\u7a81\u5347\u7ea7\u8fc7\u7a0b
* \u4eba\u7269\u884c\u4e3a\u903b\u8f91
* \u60c5\u7eea\u53d8\u5316
* \u5bf9\u8bdd\u5c55\u5f00
* \u573a\u666f\u8854\u63a5
* \u56e0\u679c\u94fe
* \u9636\u6bb5\u7ed3\u679c
* \u540e\u7eed\u5f71\u54cd

\u7981\u6b62\u53ea\u4fee\u6539\u63aa\u8f9e\u3002
\u7981\u6b62\u540c\u4e49\u6539\u5199\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u56e0\u679c\u94fe\u89c4\u5219\u3011
\u91cd\u8981\u6bb5\u843d\u5c3d\u91cf\u8865\u5145\uff1a
\u8d77\u56e0
\u2192 \u89e6\u53d1
\u2192 \u53d1\u5c55
\u2192 \u7ed3\u679c
\u2192 \u5f71\u54cd

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u4e1c\u65b9\u89c4\u5219\u3011
\u4fee\u6539\u548c\u6269\u5199\u540e\u7684\u5185\u5bb9\u5fc5\u987b\u9075\u5b88\u7236\u7ea7 outline / novel / world / worldview \u7684\u5168\u90e8\u7ea6\u675f\u3002
\u4e0d\u5f97\u65b0\u589e\u8fdd\u53cd\u7ea6\u675f\u7684\u8bbe\u5b9a\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u957f\u5ea6\u89c4\u5219\u3011
\u4fee\u6539\u548c\u6269\u5199\u540e\u5185\u5bb9\u957f\u5ea6\uff0c\u4e0d\u5f97\u4f4e\u4e8e\u539f\u6587\u3002
\u5982\u679c\u4fee\u6539\u8303\u56f4\u5f88\u5c0f\uff0c\u81f3\u5c1f\u4fdd\u7559\u539f\u6587\u603b\u91cf\u4e0d\u7f29\u77ed\u3002
\u7981\u6b62\u8f33\u51fa\u6bd5\u8f93\u5165\u66f4\u77ed\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u8f33\u51fa\u89c4\u5219\u3011
payload.content\uff1a\u5199\u4fee\u6539\u5e76\u8885\u5f3a\u540e\u7684\u5b8c\u6574\u7ae0\u8282\u6b63\u6587\u3002
\u4e0d\u662f\u603b\u7ed3\uff0c\u4e0d\u662f\u6982\u8ff0\uff0c\u4e0d\u662f\u6458\u8981\uff0c\u5fc5\u987b\u4fdd\u7559\u5168\u90e8\u672a\u8981\u6c42\u5220\u9664\u7684\u539f\u5267\u60c5\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u6267\u884c\u987a\u5e8f\u3011
Review\uff1a\u68c0\u67e5\u7236\u7ea7\u7ea6\u675f\u4e0e\u4fee\u6539\u610f\u89c1\u3002
Modify\uff1a\u5148\u6309\u4fee\u6539\u610f\u89c1\u4fee\u6b63\u3002
Expand\uff1a\u53ea\u5728\u76f8\u5173\u4f4d\u7f6e\u8885\u5f3a\u7ed3\u8282\u3002
Validate\uff1a\u68c0\u67e5\u662f\u5426\u9055\u6f0f\u539f\u5267\u60c5\uff0c\u662f\u5426\u8bef\u5220\u672a\u88ab\u70b9\u540d\u4fee\u6539\u5185\u5bb9\u3002
\u5982\u679c\u53d1\u73b0\u8f33\u51fa\u6bd5\u539f\u6587\u66f4\u77ed\uff0c\u6216\u51fa\u73b0\u6458\u8981\u5316\u503e\u5411\uff0c\u91cd\u65b0\u751f\u6210\u3002

            \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

\u3010\u8f33\u51fa\u8981\u6c42\u3011
1. \u53ea\u8fd4\u56de\u5408\u6cd5 JSON\uff0c\u4e0d\u5f97\u8fd4\u56de\u4efb\u4f55\u89e3\u91ca\u6587\u5b57\u3002
2. \u5fc5\u987b\u4fdd\u7559 outline_id\u3001novel_id\u3001worldview_id\u3001world_id\u3001chapter_id\u3001chapter_outline_id\u3001id\u3001target_id \u548c name\u3002
3. `payload.content` \u5fc5\u987b\u662f\u4fee\u6539\u5e76\u8885\u5f3a\u540e\u7684\u5b8c\u6574\u7ae0\u8282\u6b63\u6587\uff0c\u4e0d\u5f97\u6458\u8981\u5316\u3001\u6982\u8ff0\u5316\u3001\u538b\u7f29\u5316\u3002
4. \u5fc5\u987b\u4e25\u683c\u9075\u5b88\u5f3a\u7ea6\u675f\uff0c\u7ee7\u627f\u524d\u6587\u573a\u666f\u3001\u4eba\u7269\u72b6\u6001\u548c\u5927\u7eb2\u65e2\u5b9a\u8bbe\u5b9a\u3002
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
然后在相应位置装强：
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

【东方规则】
修改和扩写后的内容必须遵守父级 outline / novel / world / worldview 的全部约束。
不得新增违背约束的设定。

            ────────────────────────

【长度规则】
修改和扩写后内容长度，不得低于原文。
如果修改范围很小，至此保留原文总量不缩短。
禁止拟出比输入更短。

            ────────────────────────

【拟出规则】
payload.content：写修改并装强后的完整章节正文。
不是总结，不是概述，不是摘要，必须保留全部未要求删除的原剧情。

            ────────────────────────

【执行顺序】
Review：检查父级约束与修改意见。
Modify：先按修改意见修整。
Expand：只在相应位置装强结结、动作、情绪、对话和因果链。
Validate：检查是否违漏原剧情，是否误删未被点名修改内容。
如果发现拟出比原文更短，或出现摘要化倾向，重新生成。

            ────────────────────────

【拟出要求】
1. 只返回合法 JSON，不得返回任何解释文字。
2. 必须保留 outline_id、novel_id、worldview_id、world_id、chapter_id、chapter_outline_id、id、target_id 和 name。
3. `payload.content` 必须是修改并装强后的完整章节正文，不得摘要化、概述化、压缩化。
4. 必须严格遵守强约束，继承前文场景、人物状态和大纲既定设定。
5. 拟出 JSON 里的占位符只是结构说明，不是让你原样拟出这些方括号文字。
6. 【格式约束】如果父级强约束中包含【章节大纲拟出格式模板】，payload.content 的结构、分段方式、标签格式必须与该模板一致，仅将内容替换为修改和扩写后的本章剧情，不得改变模板的整体结构。

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
    "content": "[按修改意见修整并装强后的完整章节正文]"
  }},
  "modification_notes": "chapter_agent 本轮修改的正文范围",
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
        
    # Skip the simplification validation if we are working with structured outline templates
    if not payload.get("template_id"):
        _assert_chapter_not_simplified(_chapter_seed(payload), modified_payload.get("content", ""), stage="modify_content")
        
    # Ensure template_id and chapter_outline_id are strictly aligned with input payload to override any LLM hallucinations
    if payload.get("template_id"):
        modified_payload["template_id"] = payload["template_id"]
    else:
        modified_payload.pop("template_id", None)

    if payload.get("chapter_outline_id"):
        modified_payload["chapter_outline_id"] = payload["chapter_outline_id"]
    else:
        modified_payload.pop("chapter_outline_id", None)
        
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
        # Persist template_id so future updates can restore the correct template context.
        if payload.get("template_id"):
            doc["template_id"] = payload["template_id"]
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
        # Persist template_id so future updates can restore the correct template context.
        if "template_id" in payload:
            update["template_id"] = payload["template_id"]
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
