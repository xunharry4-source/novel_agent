import json
import logging
from typing import Callable, Dict, Any, Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage
from openai import APIConnectionError, APITimeoutError

from src.common.llm_factory import get_llm, get_provider_info
from src.common.lore_utils import get_unified_context, parse_json_safely

logger = logging.getLogger("novel_agent.review_agent")


def _review_llm_metadata(raw_content: str, system_prompt: str, user_message: str, entity_type: str) -> Dict[str, Any]:
    return {
        "llm_invoked": True,
        "llm_agent_name": f"{entity_type}_review_agent",
        "prompt": f"[System]\n{system_prompt}\n\n[User]\n{user_message}",
        "prompt_chars": len(system_prompt) + len(user_message) + 18,
        "raw_response_chars": len(raw_content),
    }

def _get_context_for_review(entity_type: str, payload: Dict[str, Any]) -> str:
    """获取审核所需的上下文信息"""
    try:
        query = f"{payload.get('name', '')} {payload.get('title', '')} {payload.get('summary', '')} {payload.get('content', '')}"
        outline_id = payload.get("outline_id") or "default"
        worldview_id = payload.get("worldview_id") or "default_wv"

        # 只在有具体内容时进行检索，且如果报错则静默失败，避免阻塞审核
        if len(query.strip()) > 5:
            return get_unified_context(query, outline_id=outline_id, worldview_id=worldview_id)
    except Exception as e:
        logger.warning(f"Failed to get context for review: {e}")
    return "未能获取到有效的背景上下文设定。"


def _truncate_review_text(value: Any, *, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _get_world_policy_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """读取世界根实体中的禁止规则与基本设定，供审查节点强制校验。"""
    world_id = payload.get("world_id")
    target_id = payload.get("target_id")
    novel_id = payload.get("novel_id")
    outline_id = payload.get("outline_id")
    inline_forbidden = payload.get("forbidden_rules")
    inline_basic = payload.get("basic_settings")
    world_doc: Dict[str, Any] = {}
    if not world_id and novel_id and db is not None:
        try:
            source_doc = db["novels"].find_one({"novel_id": novel_id}) or {}
            world_id = source_doc.get("world_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id from novel for review: {e}")
    if not world_id and outline_id and db is not None:
        try:
            source_doc = db["outlines"].find_one({"outline_id": outline_id}) or db["outlines"].find_one({"id": outline_id}) or {}
            world_id = source_doc.get("world_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id from outline for review: {e}")
    if not world_id and target_id and db is not None:
        try:
            if entity_type.startswith("worldview"):
                source_doc = db["worldviews"].find_one({"worldview_id": target_id}) or {}
                world_id = source_doc.get("world_id")
            elif entity_type.startswith("novel"):
                source_doc = db["novels"].find_one({"novel_id": target_id}) or {}
                world_id = source_doc.get("world_id")
            elif entity_type.startswith("outline"):
                source_doc = db["outlines"].find_one({"outline_id": target_id}) or db["outlines"].find_one({"id": target_id}) or {}
                world_id = source_doc.get("world_id")
            elif entity_type.startswith("chapter"):
                source_doc = db["prose"].find_one({"id": target_id}) or db["prose"].find_one({"scene_id": target_id}) or {}
                world_id = source_doc.get("world_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id for review: {e}")
    if world_id and db is not None:
        try:
            world_doc = db["worlds"].find_one({"world_id": world_id}) or {}
        except Exception as e:
            logger.warning(f"Failed to load world policy for review: {e}")

    forbidden_rules = world_doc.get("forbidden_rules", inline_forbidden)
    basic_settings = world_doc.get("basic_settings", inline_basic)
    return (
        "【世界禁止规则】\n"
        f"{json.dumps(forbidden_rules or [], ensure_ascii=False, indent=2)}\n\n"
        "【世界基本设定】\n"
        f"{json.dumps(basic_settings or {}, ensure_ascii=False, indent=2)}"
    )


def _get_novel_policy_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """读取小说项目中的禁止规则与基本设定，供大纲和章节审查强制校验。"""
    novel_id = payload.get("novel_id")
    target_id = payload.get("target_id")
    outline_id = payload.get("outline_id")
    inline_forbidden = payload.get("novel_forbidden_rules") or payload.get("forbidden_rules")
    inline_basic = payload.get("novel_basic_settings") or payload.get("basic_settings")
    novel_doc: Dict[str, Any] = {}
    if not novel_id and db is not None:
        try:
            if entity_type.startswith("outline") and target_id:
                source_doc = db["outlines"].find_one({"outline_id": target_id}) or {}
                novel_id = source_doc.get("novel_id")
            elif entity_type.startswith("chapter") and outline_id:
                source_doc = db["outlines"].find_one({"outline_id": outline_id}) or {}
                novel_id = source_doc.get("novel_id")
            elif entity_type.startswith("chapter") and target_id:
                source_doc = db["prose"].find_one({"id": target_id}) or {}
                novel_id = source_doc.get("novel_id")
        except Exception as e:
            logger.warning(f"Failed to resolve novel_id for review: {e}")
    if novel_id and db is not None:
        try:
            novel_doc = db["novels"].find_one({"novel_id": novel_id}) or {}
        except Exception as e:
            logger.warning(f"Failed to load novel policy for review: {e}")

    forbidden_rules = novel_doc.get("forbidden_rules", inline_forbidden)
    basic_settings = novel_doc.get("basic_settings", inline_basic)
    return (
        "【小说禁止规则】\n"
        f"{json.dumps(forbidden_rules or [], ensure_ascii=False, indent=2)}\n\n"
        "【小说基本设定】\n"
        f"{json.dumps(basic_settings or {}, ensure_ascii=False, indent=2)}"
    )


def _get_worldview_policy_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """直接读取 worldview 根摘要和 lore 条目，禁止只依赖模糊检索。"""
    if db is None:
        return _get_context_for_review(entity_type, payload)

    worldview_id = payload.get("worldview_id")
    world_id = payload.get("world_id")
    target_id = payload.get("target_id")
    outline_id = payload.get("outline_id")
    novel_id = payload.get("novel_id")
    worldview_doc: Dict[str, Any] = {}

    if not world_id and novel_id:
        try:
            novel_doc = db["novels"].find_one({"novel_id": novel_id}) or {}
            world_id = novel_doc.get("world_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id from novel for worldview review: {e}")
    if not world_id and outline_id:
        try:
            outline_doc = db["outlines"].find_one({"outline_id": outline_id}) or db["outlines"].find_one({"id": outline_id}) or {}
            world_id = outline_doc.get("world_id")
            worldview_id = worldview_id or outline_doc.get("worldview_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id from outline for worldview review: {e}")
    if not world_id and entity_type.startswith("chapter") and target_id:
        try:
            prose_doc = (
                db["prose"].find_one({"id": target_id})
                or db["prose"].find_one({"scene_id": target_id})
                or db["prose"].find_one({"prose_id": target_id})
                or {}
            )
            world_id = prose_doc.get("world_id")
            worldview_id = worldview_id or prose_doc.get("worldview_id")
        except Exception as e:
            logger.warning(f"Failed to resolve world_id from prose for worldview review: {e}")

    try:
        if worldview_id:
            worldview_doc = db["worldviews"].find_one({"worldview_id": worldview_id}) or {}
        if not worldview_doc and world_id:
            worldview_doc = db["worldviews"].find_one({"world_id": world_id}) or {}
            worldview_id = worldview_doc.get("worldview_id")
    except Exception as e:
        logger.warning(f"Failed to load worldview policy for review: {e}")

    entries: List[Dict[str, Any]] = []
    if worldview_id:
        try:
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
                        "content": _truncate_review_text(doc.get("content"), limit=1200),
                    }
                )
                if len(entries) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Failed to load worldview lore entries for review: {e}")

    root_block = {
        "worldview_id": worldview_doc.get("worldview_id") or worldview_id,
        "world_id": worldview_doc.get("world_id") or world_id,
        "name": worldview_doc.get("name"),
        "summary": _truncate_review_text(worldview_doc.get("summary"), limit=1000),
    }
    return (
        "【世界观总设】\n"
        f"{json.dumps(root_block, ensure_ascii=False, indent=2)}\n\n"
        "【世界观 Lore 条目】\n"
        f"{json.dumps(entries or [], ensure_ascii=False, indent=2)}"
    )


def _get_outline_policy_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """读取父级大纲内容，供章节大纲审查节点强制校验。"""
    outline_id = payload.get("outline_id")
    target_id = payload.get("target_id")
    outline_doc: Dict[str, Any] = {}
    if not outline_id and entity_type.startswith("chapter") and target_id and db is not None:
        try:
            prose_doc = db["prose"].find_one({"id": target_id}) or db["prose"].find_one({"scene_id": target_id}) or {}
            outline_id = prose_doc.get("outline_id")
        except Exception as e:
            logger.warning(f"Failed to resolve outline_id for review: {e}")
    if outline_id and db is not None:
        try:
            outline_doc = db["outlines"].find_one({"outline_id": outline_id}) or db["outlines"].find_one({"id": outline_id}) or {}
        except Exception as e:
            logger.warning(f"Failed to load outline policy for review: {e}")

    return (
        "【父级大纲】\n"
        f"{json.dumps({k: outline_doc.get(k) for k in ('outline_id', 'id', 'name', 'title', 'summary', 'content') if outline_doc.get(k) is not None}, ensure_ascii=False, indent=2)}"
    )


def _get_chapter_outline_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """读取章节大纲内容，供章节直接内容检查节点强制校验。"""
    if not entity_type.startswith("chapter"):
        return "非章节审查，不需要章节大纲。"

    inline_outline = payload.get("chapter_outline") or payload.get("chapter_outline_content") or payload.get("chapter_summary")
    inline_name = payload.get("chapter_outline_name") or payload.get("name") or payload.get("title")
    if inline_outline:
        return (
            "【章节大纲】\n"
            f"{json.dumps({'name': inline_name, 'content': inline_outline}, ensure_ascii=False, indent=2)}"
        )

    target_id = (
        payload.get("chapter_outline_id")
        or payload.get("target_id")
        or payload.get("chapter_id")
        or payload.get("scene_id")
        or payload.get("prose_id")
        or payload.get("id")
    )
    if not target_id or db is None:
        return "未提供章节大纲；如果当前流程要求检查章节大纲，请补充 chapter_outline 或指定可读取的目标章节。"

    try:
        doc = (
            db["prose"].find_one({"id": target_id})
            or db["prose"].find_one({"scene_id": target_id})
            or db["prose"].find_one({"prose_id": target_id})
            or {}
        )
    except Exception as e:
        logger.warning(f"Failed to load chapter outline context for review: {e}")
        return f"读取章节大纲失败：{e}"

    if not doc:
        return f"未找到 target_id={target_id} 对应的章节大纲。"

    chapter_outline = {
        "id": doc.get("id") or doc.get("scene_id") or doc.get("prose_id"),
        "name": doc.get("name") or doc.get("title"),
        "outline_id": doc.get("outline_id"),
        "content": doc.get("content"),
    }
    return "【章节大纲】\n" + json.dumps(chapter_outline, ensure_ascii=False, indent=2)


def _get_previous_chapter_context(db, entity_type: str, payload: Dict[str, Any]) -> str:
    """读取当前章节之前已入库章节，供章节一致性审查强制校验。"""
    if not entity_type.startswith("chapter") or db is None:
        return "非章节审查，不需要读取前置章节。"

    inline_previous = payload.get("previous_chapters") or payload.get("previous_chapter_context")
    if inline_previous:
        return "【用户提供的前置章节上下文】\n" + json.dumps(inline_previous, ensure_ascii=False, indent=2)

    outline_id = payload.get("outline_id")
    novel_id = payload.get("novel_id")
    world_id = payload.get("world_id")
    target_ids = {
        str(value)
        for value in (
            payload.get("target_id"),
            payload.get("chapter_id"),
            payload.get("id"),
            payload.get("scene_id"),
            payload.get("prose_id"),
        )
        if value
    }

    query: Dict[str, Any] = {}
    if outline_id:
        query["outline_id"] = outline_id
    elif novel_id:
        query["novel_id"] = novel_id
    elif world_id:
        query["world_id"] = world_id
    else:
        return "当前 payload 缺少 outline_id/novel_id/world_id，无法可靠读取前置章节。"

    try:
        cursor = (
            db["prose"]
            .find(query)
            .sort([("chapter_index", 1), ("order", 1), ("sequence", 1), ("created_at", 1), ("timestamp", 1)])
            .limit(12)
        )
        chapters = []
        for doc in cursor:
            doc_ids = {str(doc.get(key)) for key in ("id", "scene_id", "prose_id", "chapter_id") if doc.get(key)}
            if target_ids and target_ids.intersection(doc_ids):
                continue
            chapters.append({
                "id": doc.get("id") or doc.get("scene_id") or doc.get("prose_id") or doc.get("chapter_id"),
                "title": doc.get("title") or doc.get("name"),
                "chapter_index": doc.get("chapter_index") or doc.get("order") or doc.get("sequence"),
                "outline_id": doc.get("outline_id"),
                "content": str(doc.get("content") or "")[:3000],
            })
        if not chapters:
            return "未检索到同一 outline/novel/world 下已入库的前置章节；若这是第一章，可通过审查。"
        return "【前置章节内容】\n" + json.dumps(chapters, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to load previous chapters for review: {e}")
        return f"读取前置章节失败：{e}"


REVIEW_SECTION_BUILDERS: Dict[str, Tuple[str, Callable[..., str]]] = {
    "world_policy": (
        "以下是世界禁止规则与基本设定（必须优先遵守）：",
        _get_world_policy_context,
    ),
    "novel_policy": (
        "以下是小说禁止规则与基本设定（大纲和章节必须遵守）：",
        _get_novel_policy_context,
    ),
    "worldview_policy": (
        "以下是世界观 Canon 设定（必须优先遵守）：",
        _get_worldview_policy_context,
    ),
    "outline_policy": (
        "以下是父级大纲约束（章节必须遵守）：",
        _get_outline_policy_context,
    ),
    "chapter_outline": (
        "以下是章节大纲约束（直接内容检查必须遵守）：",
        _get_chapter_outline_context,
    ),
    "previous_chapter": (
        "以下是前置章节上下文（章节一致性审查必须遵守）：",
        _get_previous_chapter_context,
    ),
    "context_reference": (
        "以下是世界观和背景设定参考（如果有）：",
        lambda _db, entity_type, payload: _get_context_for_review(entity_type, payload),
    ),
}


ENTITY_REVIEW_SECTIONS: Dict[str, List[str]] = {
    "worldview_world_rules": ["world_policy"],
    "worldview_consistency": ["context_reference"],
    "novel_world_rules": ["world_policy", "worldview_policy", "context_reference"],
    "outline_world_rules": ["world_policy"],
    "outline_worldview_rules": ["worldview_policy", "context_reference"],
    "outline_novel_rules": ["novel_policy"],
    "chapter_world_rules": ["world_policy"],
    "chapter_worldview_rules": ["worldview_policy", "context_reference"],
    "chapter_novel_rules": ["novel_policy"],
    "chapter_outline_rules": ["outline_policy"],
    "chapter_chapter_outline_rules": ["chapter_outline"],
    "chapter_consistency": ["previous_chapter", "chapter_outline"],
    "chapter_plot_errors": ["outline_policy", "chapter_outline", "previous_chapter"],
    "worldview": ["world_policy", "worldview_policy", "context_reference"],
    "novel": ["world_policy", "novel_policy", "worldview_policy", "context_reference"],
    "outline": ["world_policy", "novel_policy", "worldview_policy", "context_reference"],
    "chapter": ["world_policy", "novel_policy", "worldview_policy", "outline_policy", "chapter_outline", "previous_chapter", "context_reference"],
}


def _review_sections_for_entity_type(entity_type: str) -> List[str]:
    return ENTITY_REVIEW_SECTIONS.get(
        entity_type,
        ["world_policy", "novel_policy", "worldview_policy", "outline_policy", "chapter_outline", "previous_chapter", "context_reference"],
    )


def _build_review_context_blocks(db, entity_type: str, payload: Dict[str, Any]) -> List[str]:
    blocks: List[str] = []
    for section_key in _review_sections_for_entity_type(entity_type):
        title, builder = REVIEW_SECTION_BUILDERS[section_key]
        content = builder(db, entity_type, payload)
        blocks.append(f"{title}\n{content}")
    return blocks


def _classify_review_exception(exc: Exception) -> Tuple[str, str]:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, APITimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "timeout", f"大模型审查超时：{message}"
    if isinstance(exc, APIConnectionError) or "connection error" in lowered or "connection refused" in lowered:
        return "connection_error", f"大模型连接失败：{message}"
    return "llm_error", f"执行大模型审查时发生异常: {message}"


def _review_error_metadata(system_prompt: str, user_message: str, entity_type: str, error_kind: str, error_message: str) -> Dict[str, Any]:
    provider_info = get_provider_info()
    return {
        "llm_invoked": False,
        "llm_attempted": True,
        "llm_agent_name": f"{entity_type}_review_agent",
        "prompt": f"[System]\n{system_prompt}\n\n[User]\n{user_message}",
        "prompt_chars": len(system_prompt) + len(user_message) + 18,
        "error_kind": error_kind,
        "error_message": error_message,
        "provider": provider_info.get("provider"),
        "model": provider_info.get("model"),
        "base_url": provider_info.get("base_url"),
    }


def get_review_prompt(entity_type: str) -> str:
    """根据实体类型返回专门的系统提示词"""
    base_prompt = (
        "【角色设定】\n"
        "你是一名极其严格的设定审查专家（Review Agent）。你的唯一职责是审查输入的 JSON 业务内容，找出逻辑漏洞、设定冲突和规范问题。\n\n"
        "【操作流程 (Mandatory Workflow)】\n"
        "1. 审查（Review）：逐条检查当前业务内容是否违反父级规则、既有设定、结构要求和显式约束。\n"
        "2. 判定（Decide）：如果发现任何实质性冲突、缺失或违规，必须判定为不通过，并明确指出问题位置与修正方向。\n"
        "3. 输出（Output）：只返回合法 JSON，不得输出解释文字、代码块或额外前后缀。\n\n"
        "【输出要求】\n"
        "你必须返回如下 JSON：\n"
        "{\n"
        '  "passed": true 或 false,\n'
        '  "errors": ["错误描述1", "错误建议2"]\n'
        "}\n\n"
    )

    if entity_type == "worldview_world_rules":
        base_prompt += (
            "【Worldview World Rules Review (世界观-世界规则审查) 审查重点】\n"
            "1. 必须逐条检查当前世界观是否违反所属世界的 forbidden_rules（世界禁止规则）。\n"
            "2. 必须检查当前世界观是否破坏 basic_settings（世界基本设定），包括时代、力量体系、地理边界、组织结构、资源机制和基础禁令。\n"
            "3. 只判断世界根规则冲突；如果违反禁止规则或基本设定，passed 必须为 false，并指出具体冲突字段与修改方向。\n"
        )
    elif entity_type == "worldview_consistency":
        base_prompt += (
            "【Worldview Consistency Review (世界观-既有设定一致性审查) 审查重点】\n"
            "1. 检查新增或修改后的世界观条目是否与同一 world_id 下已有世界观设定冲突。\n"
            "2. 重点审查历史、地理、规则、势力、人物常识、资源机制和前后 Canon 是否自洽。\n"
            "3. 若与已存在世界观设定冲突，passed 必须为 false，并说明冲突对象、冲突原因和修正建议。\n"
        )
    elif entity_type == "novel_world_rules":
        base_prompt += (
            "【Novel World Rules Review (小说-世界规则审查) 审查重点】\n"
            "1. 必须检查小说项目是否违反所属世界 forbidden_rules（世界禁止规则）。\n"
            "2. 必须检查小说项目是否偏离 basic_settings（世界基本设定），包括时代背景、力量体系、世界边界、资源约束和基础禁令。\n"
            "3. 必须检查小说自身 forbidden_rules 与 basic_settings 是否自洽，且不能与父级世界规则冲突。\n"
            "4. 如关联 worldview_id，还要检查小说设定是否与该世界观约束矛盾。\n"
            "5. 发现反吃设定、绕开禁令、破坏基本设定时，passed 必须为 false。\n"
        )
    elif entity_type == "outline_world_rules":
        base_prompt += (
            "【Outline World Review (大纲-世界审查) 审查重点】\n"
            "1. 必须检查大纲是否违反所属世界 forbidden_rules（世界禁止规则）。\n"
            "2. 必须检查大纲是否偏离所属世界 basic_settings（世界基本设定），包括时代、力量体系、地理边界、组织结构、资源机制和基础禁令。\n"
            "3. 发现大纲绕开世界禁令、改变世界底层规则、引入不属于该世界的能力或资源时，passed 必须为 false。\n"
        )
    elif entity_type == "outline_worldview_rules":
        base_prompt += (
            "【Outline Worldview Review (大纲-世界观审查) 审查重点】\n"
            "1. 必须检查大纲是否违反关联 worldview_id 的世界观设定。\n"
            "2. 必须检查大纲是否与同一 world_id 下已有 Canon 设定冲突，包括历史、地理、规则、势力、人物常识和资源机制。\n"
            "3. 发现大纲误用 Lore、改写已有世界观规则或制造 Canon 前后矛盾时，passed 必须为 false。\n"
        )
    elif entity_type == "outline_novel_rules":
        base_prompt += (
            "【Outline Novel Review (大纲-小说审查) 审查重点】\n"
            "1. 必须检查大纲是否违反所属小说 forbidden_rules（小说禁止规则）。\n"
            "2. 必须检查大纲是否偏离所属小说 basic_settings（小说基本设定），包括主角底线、主线冲突、叙事基调、时间线、人物关系规则和剧情约束。\n"
            "3. 必须检查剧情推进、冲突升级、高潮安排是否服务于小说主线，不能喧宾夺主或重写小说核心方向。\n"
            "4. 发现大纲偏离小说主旨、破坏主角设定、违背时间线或人物关系规则时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_world_rules":
        base_prompt += (
            "【Chapter World Review (章节-世界审查) 审查重点】\n"
            "1. 必须检查章节正文是否违反所属世界 forbidden_rules（世界禁止规则）。\n"
            "2. 必须检查章节正文是否偏离所属世界 basic_settings（世界基本设定），包括时代、力量体系、地理边界、组织结构、资源机制和基础禁令。\n"
            "3. 发现正文绕开世界禁令、改变世界底层规则、引入不属于该世界的能力或资源时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_worldview_rules":
        base_prompt += (
            "【Chapter Worldview Review (章节-世界观审查) 审查重点】\n"
            "1. 必须检查章节正文是否违反关联 worldview_id 的世界观设定。\n"
            "2. 必须检查章节正文是否与同一 world_id 下已有 Canon 设定冲突，包括历史、地理、规则、势力、人物常识和资源机制。\n"
            "3. 发现正文误用 Lore、改写已有世界观规则或制造 Canon 前后矛盾时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_novel_rules":
        base_prompt += (
            "【Chapter Novel Review (章节-小说审查) 审查重点】\n"
            "1. 必须检查章节正文是否违反所属小说 forbidden_rules（小说禁止规则）。\n"
            "2. 必须检查章节正文是否偏离所属小说 basic_settings（小说基本设定），包括主角底线、主线冲突、叙事基调、时间线、人物关系规则和剧情约束。\n"
            "3. 必须检查人物行为、对话、动机是否符合小说主线与角色状态，不能破坏小说核心方向。\n"
            "4. 发现正文偏离小说主旨、破坏主角设定、违背时间线或人物关系规则时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_outline_rules":
        base_prompt += (
            "【Chapter Outline Review (章节-大纲审查) 审查重点】\n"
            "1. 必须检查章节正文是否严格执行父级 outline_id 对应大纲的剧情任务。\n"
            "2. 必须检查正文是否擅自删除、提前、延后或改写大纲安排的关键事件、转折、冲突升级和结尾任务。\n"
            "3. 必须检查章节收尾是否服务于父级大纲节点，不能自行扩展到未批准的大纲之外。\n"
            "4. 发现正文偏离大纲、删改大纲目标或越权推进后续剧情时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_chapter_outline_rules":
        base_prompt += (
            "【Chapter Chapter-Outline Review (章节-章节大纲审查) 审查重点】\n"
            "1. 必须检查直接内容是否严格执行章节大纲中的场景任务、关键事件、冲突顺序、人物动作和收尾目标。\n"
            "2. 必须检查直接内容是否擅自删除、提前、延后或改写章节大纲中已明确的关键桥段、信息揭示和情绪推进。\n"
            "3. 必须检查直接内容是否越权补写未在章节大纲中批准的重大设定、重大剧情跳跃或结局变化。\n"
            "4. 发现直接内容偏离章节大纲、删改章节任务或错置关键事件时，passed 必须为 false。\n"
        )
    elif entity_type == "chapter_consistency":
        base_prompt += (
            "【Chapter Consistency Review (章节-前文一致性审查) 审查重点】\n"
            "1. 必须检查当前章节与此前已入库章节在剧情承接、时间线、地点变化、人物状态、人物关系、伤势/装备/资源和伏笔回收上是否一致。\n"
            "2. 必须检查是否出现前一章结尾尚未解决、当前章却跳过解释的断裂；是否出现人物突然知道未知信息、道具凭空出现、情绪状态无过渡改变。\n"
            "3. 必须检查叙事视角、语气、章节标题和正文内容是否延续同一作品的连续性。\n"
            "4. 如果当前章节是第一章或没有可用前置章节，可通过审查，但必须只基于当前 payload 判断是否自洽。\n"
            "5. 发现与前置章节冲突或承接断裂时，passed 必须为 false，并指出冲突章节、冲突点和修正方向。\n"
        )
    elif entity_type == "chapter_plot_errors":
        base_prompt += (
            "【Chapter Plot Error Review (章节-剧情错误审查) 审查重点】\n"
            "1. 必须检查直接内容内部是否存在因果断裂、时间线跳变、人物动机突变、信息来源不明、道具/伤势/资源凭空变化等剧情错误。\n"
            "2. 必须检查场景切换是否交代清楚，事件推进是否存在缺失步骤、逻辑黑箱或前后自相矛盾。\n"
            "3. 必须检查直接内容与章节大纲、前置章节和当前父级上下文之间是否存在知识泄漏、伏笔丢失、角色突然知晓未知事实等问题。\n"
            "4. 发现任何实质性剧情错误时，passed 必须为 false，并说明错误位置、错误类型和修正方向。\n"
        )
    elif entity_type == "worldview":
        base_prompt += (
            "【Worldview (世界观) 审查重点】\n"
            "1. 内容格式是否规范：名称与摘要/详情是否匹配。\n"
            "2. 逻辑漏洞：设定的内部机制能否自洽（例如说‘人人平等’但又设定了‘天生贵族’）。\n"
            "3. 设定冲突：新增设定是否与背景上下文中已有核心设定（历史、地理、规则等）存在直接矛盾。\n"
        )
    elif entity_type == "novel":
        base_prompt += (
            "【Novel (小说项目) 审查重点】\n"
            "1. 故事背景是否契合世界观约束：不能出现超出该世界当前科技/魔法水平的事物。\n"
            "2. 主角设定与核心主线是否符合逻辑：动机是否明确，故事目标是否清晰。\n"
            "3. 是否存在破坏世界基础规则的设定（反吃设定）。\n"
        )
    elif entity_type == "outline":
        base_prompt += (
            "【Outline (大纲节点) 审查重点】\n"
            "1. 上下文节点剧情逻辑是否连贯：没有突兀的转折或未交代的跳跃。\n"
            "2. 故事发展是否偏离小说主旨：支线是否喧宾夺主。\n"
            "3. 核心冲突与高潮安排是否合理。\n"
            "4. 是否违反小说禁止规则与小说基本设定，包括主角底线、主线冲突、叙事基调、时间线和人物关系规则。\n"
            "5. 设定冲突：是否出现与前面剧情或已有世界观设定矛盾的情节。\n"
        )
    elif entity_type == "chapter":
        base_prompt += (
            "【Chapter (正文章节) 审查重点】\n"
            "1. 人物行为动机与对话是否符合已有的人设模板（OOC检查）。\n"
            "2. 场景与道具描写是否符合世界观物理法则。\n"
            "3. 剧情推进与章节收尾是否严格遵循大纲约束，不能自行删改大纲布置的任务。\n"
            "4. 是否违反小说禁止规则与小说基本设定，包括主角底线、主线冲突、叙事基调、时间线和人物关系规则。\n"
            "5. 文字风格与视角是否存在突兀变化。\n"
        )
    else:
        base_prompt += "【综合审查】请检查逻辑连贯性和设定冲突。"

    return base_prompt


def build_review_messages(db, entity_type: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """构造审查节点实际发送给 LLM 的 system/user 消息。"""
    system_prompt = get_review_prompt(entity_type)
    context_blocks = _build_review_context_blocks(db, entity_type, payload)
    message_parts = list(context_blocks)
    message_parts.append(
        f"以下是需要你审查的当前 {entity_type} 业务内容（JSON格式）：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    message_parts.append("请严格按照要求审查，并仅输出符合要求的 JSON 格式结果。")
    user_message = "\n\n".join(message_parts)
    return system_prompt, user_message

def execute_llm_review(db, entity_type: str, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    调用大模型对 payload 进行深度逻辑与设定审查。

    Returns:
        (passed: bool, errors: list[str])
    """
    logger.info(f"Executing LLM review for {entity_type}")

    try:
        system_prompt, user_message = build_review_messages(db, entity_type, payload)

        llm = get_llm(json_mode=True, agent_name=f"{entity_type}_review_agent")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = llm.invoke(messages)
        content = response.content
        logger.debug(f"Review Agent Raw Response: {content}")

        result = parse_json_safely(content)
        if isinstance(result, dict) and "passed" in result:
            passed = bool(result.get("passed", False))
            errors = result.get("errors", [])
            if not isinstance(errors, list):
                errors = [str(errors)]

            # 如果判定为 failed 但没有给理由，强制补充
            if not passed and not errors:
                errors = ["LLM 审查未通过，但未提供具体原因。"]

            return passed, errors
        else:
            logger.warning(f"Review Agent returned malformed JSON: {content}")
            return False, ["审查模型返回了无效的格式，无法解析判定结果。"]

    except Exception as e:
        logger.error(f"Error in execute_llm_review for {entity_type}: {e}", exc_info=True)
        return False, [f"执行大模型审查时发生异常: {str(e)}"]


def execute_llm_review_detail(db, entity_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """返回审查结论及可审计的 LLM 调用明细。"""
    logger.info(f"Executing LLM review detail for {entity_type}")
    system_prompt = ""
    user_message = ""

    try:
        system_prompt, user_message = build_review_messages(db, entity_type, payload)

        llm = get_llm(json_mode=True, agent_name=f"{entity_type}_review_agent")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        content = response.content
        logger.debug(f"Review Agent Raw Response: {content}")
        llm_call = _review_llm_metadata(content, system_prompt, user_message, entity_type)

        result = parse_json_safely(content)
        if isinstance(result, dict) and "passed" in result:
            passed = bool(result.get("passed", False))
            errors = result.get("errors", [])
            if not isinstance(errors, list):
                errors = [str(errors)]
            if not passed and not errors:
                errors = ["LLM 审查未通过，但未提供具体原因。"]
            return {
                "passed": passed,
                "errors": errors,
                "llm_invoked": True,
                "llm_call": llm_call,
                "raw_response": content,
            }

        logger.warning(f"Review Agent returned malformed JSON: {content}")
        return {
            "passed": False,
            "errors": ["审查模型返回了无效的格式，无法解析判定结果。"],
            "llm_invoked": True,
            "llm_call": llm_call,
            "raw_response": content,
        }
    except Exception as e:
        logger.error(f"Error in execute_llm_review_detail for {entity_type}: {e}", exc_info=True)
        error_kind, error_text = _classify_review_exception(e)
        return {
            "passed": False,
            "errors": [error_text],
            "llm_invoked": False,
            "llm_call": _review_error_metadata(system_prompt, user_message, entity_type, error_kind, str(e)),
            "raw_response": "",
        }
