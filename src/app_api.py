"""Flask API backend for Novel Agent.

This file is the real HTTP entrypoint expected by the Makefile, README, and
generated API docs. It uses MongoDB through the project helper and calls the
existing agent node functions directly for hierarchy workflows.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import tempfile
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Callable

from bson import ObjectId
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from src.agents import (
    chapter_agent,
    chapter_check_agent,
    chapter_content_summary_create_agent,
    chapter_content_summary_update_agent,
    chapter_outline_summary_create_agent,
    chapter_outline_summary_update_agent,
    novel_agent,
    outline_agent,
    outline_summary_create_agent,
    outline_summary_update_agent,
    review_agent,
    worldview_agent,
    world_agent,
)
from src.common.config_utils import get_config
from src.common.llm_identity_registry import LEGACY_SHARED_LLM_AGENT_NAMES, expected_llm_agent_name, validate_llm_identity_registry
from src.common.lore_utils import get_mongodb_db


app = Flask(__name__)
CORS(app)
validate_llm_identity_registry((get_config() or {}).get("AGENT_MODELS", {}))


AGENT_MODULES = {
    "world": world_agent,
    "worldview": worldview_agent,
    "novel": novel_agent,
    "outline": outline_agent,
    "chapter": chapter_agent,
    "outline_summary_create": outline_summary_create_agent,
    "outline_summary_update": outline_summary_update_agent,
    "chapter_outline_summary_create": chapter_outline_summary_create_agent,
    "chapter_outline_summary_update": chapter_outline_summary_update_agent,
    "chapter_intro_summary_create": chapter_content_summary_create_agent,
    "chapter_intro_summary_update": chapter_content_summary_update_agent,
    "chapter_content_summary_create": chapter_content_summary_create_agent,
    "chapter_content_summary_update": chapter_content_summary_update_agent,
}

AGENT_CAPABILITIES = {
    "world": {
        "label": "世界 Agent",
        "description": "创建或修改顶层世界、世界禁止规则与世界基础设定。",
        "required_context": [],
        "id_fields": ["world_id", "target_id"],
        "content_fields": ["name", "summary", "forbidden_rules", "basic_settings"],
    },
    "worldview": {
        "label": "世界观 Agent",
        "description": "创建或修改当前世界唯一设定库下的世界观设定条目。",
        "required_context": ["world_id"],
        "id_fields": ["id", "worldview_id", "target_id"],
        "content_fields": ["name", "summary", "forbidden_rules", "basic_settings"],
    },
    "novel": {
        "label": "小说 Agent",
        "description": "创建或修改小说项目、小说介绍、小说简介与小说级规则。",
        "required_context": ["world_id"],
        "id_fields": ["novel_id", "target_id"],
        "content_fields": ["name", "introduction", "summary", "forbidden_rules", "basic_settings"],
    },
    "outline": {
        "label": "大纲 Agent",
        "description": "创建或修改小说下的大纲，并执行世界、世界观和小说约束审查。",
        "required_context": ["novel_id"],
        "id_fields": ["outline_id", "id", "target_id"],
        "content_fields": ["name", "summary", "worldview_id"],
    },
    "chapter": {
        "label": "章节 Agent",
        "description": "创建或修改大纲下的章节正文，并执行全链路约束审查。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "scene_id", "prose_id", "id", "target_id"],
        "content_fields": ["name", "content", "chapter_outline"],
    },
    "outline_summary_create": {
        "label": "新增分卷大纲总结工作流",
        "description": "把新增分卷大纲压缩成下游摘要，单独写入 downstream_summaries。",
        "required_context": ["novel_id"],
        "id_fields": ["outline_id", "target_id"],
        "content_fields": ["name", "summary", "downstream_summary"],
    },
    "outline_summary_update": {
        "label": "修改分卷大纲总结工作流",
        "description": "把修改后的分卷大纲压缩成下游摘要，单独写入 downstream_summaries。",
        "required_context": ["novel_id"],
        "id_fields": ["outline_id", "target_id"],
        "content_fields": ["name", "summary", "downstream_summary"],
    },
    "chapter_outline_summary_create": {
        "label": "新增章节大纲总结工作流",
        "description": "把章节大纲压缩成下游摘要，单独写入 downstream_summaries。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "downstream_summary"],
    },
    "chapter_outline_summary_update": {
        "label": "修改章节大纲总结工作流",
        "description": "把修改后的章节大纲压缩成下游摘要，单独写入 downstream_summaries。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "downstream_summary"],
    },
    "chapter_intro_summary_create": {
        "label": "新增章节简介与总结工作流",
        "description": "基于章节正文生成章节简介与章节总结，并兼容写入 downstream_summary。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "chapter_intro", "chapter_summary", "downstream_summary"],
    },
    "chapter_intro_summary_update": {
        "label": "修改章节简介与总结工作流",
        "description": "基于修改后的章节正文重新生成章节简介与章节总结，并兼容写入 downstream_summary。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "chapter_intro", "chapter_summary", "downstream_summary"],
    },
    "chapter_content_summary_create": {
        "label": "新增章节简介与总结工作流（兼容旧类型）",
        "description": "兼容旧 chapter_content_summary_create 类型，内部按章节简介与总结工作流处理。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "chapter_intro", "chapter_summary", "downstream_summary"],
    },
    "chapter_content_summary_update": {
        "label": "修改章节简介与总结工作流（兼容旧类型）",
        "description": "兼容旧 chapter_content_summary_update 类型，内部按章节简介与总结工作流处理。",
        "required_context": ["outline_id"],
        "id_fields": ["chapter_id", "id", "scene_id", "target_id"],
        "content_fields": ["name", "content", "chapter_intro", "chapter_summary", "downstream_summary"],
    },
}

AGENT_ALIASES = {
    "world": "world",
    "world_agent": "world",
    "世界": "world",
    "worldview": "worldview",
    "worldview_agent": "worldview",
    "世界观": "worldview",
    "设定": "worldview",
    "novel": "novel",
    "novel_agent": "novel",
    "小说": "novel",
    "outline": "outline",
    "outline_agent": "outline",
    "大纲": "outline",
    "chapter": "chapter",
    "chapter_agent": "chapter",
    "章节": "chapter",
    "正文": "chapter",
    "outline_summary_create": "outline_summary_create",
    "outline_summary_update": "outline_summary_update",
    "chapter_outline_summary_create": "chapter_outline_summary_create",
    "chapter_outline_summary_update": "chapter_outline_summary_update",
    "chapter_intro_summary_create": "chapter_intro_summary_create",
    "chapter_intro_summary_update": "chapter_intro_summary_update",
    "chapter_content_summary_create": "chapter_intro_summary_create",
    "chapter_content_summary_update": "chapter_intro_summary_update",
}

ACTION_ALIASES = {
    "create": "create",
    "new": "create",
    "add": "create",
    "新增": "create",
    "创建": "create",
    "update": "update",
    "modify": "update",
    "edit": "update",
    "修改": "update",
    "更新": "update",
    "check": "check",
    "review": "check",
    "inspect": "check",
    "检查": "check",
    "审查": "check",
}

AGENT_KEYWORDS = [
    ("chapter", ["chapter", "章节", "正文", "scene", "prose"]),
    ("outline", ["outline", "大纲"]),
    ("novel", ["novel", "小说"]),
    ("worldview", ["worldview", "世界观", "设定"]),
    ("world", ["world", "世界"]),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key != "_id"}
    return value


def _json(data: Any, status: int = 200):
    return jsonify(_clean(data)), status


def _body() -> dict[str, Any]:
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _db():
    return get_mongodb_db()


def _find_one(collection: str, query: dict[str, Any]) -> dict[str, Any] | None:
    return _db()[collection].find_one(query)


def _find_worldview_by_world(world_id: str, *, exclude_worldview_id: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"world_id": world_id}
    if exclude_worldview_id:
        query["worldview_id"] = {"$ne": exclude_worldview_id}
    return _db()["worldviews"].find_one(query)


def _ensure_worldview_library(world_id: str) -> dict[str, Any]:
    worldview = _find_worldview_by_world(world_id)
    if worldview:
        return worldview
    world = _find_one("worlds", {"world_id": world_id})
    if not world:
        raise ValueError(f"Parent world not found for worldview library: {world_id}")
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
    _db()["worldviews"].insert_one(worldview)
    return worldview


def _require(value: Any, message: str) -> Any:
    if value in (None, ""):
        raise ValueError(message)
    return value


def _new_api_key() -> str:
    return f"na_{secrets.token_urlsafe(32)}"


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "display_name": user.get("display_name") or user.get("username"),
        "email": user.get("email", ""),
        "api_key": user.get("api_key"),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
        "last_login_at": user.get("last_login_at"),
    }


def _auth_token() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return request.args.get("token", "").strip()


def _request_api_key() -> str:
    header = request.headers.get("X-API-Key") or request.headers.get("X-Api-Key") or ""
    if header:
        return header.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("apikey "):
        return authorization.split(" ", 1)[1].strip()
    return request.args.get("api_key", "").strip()


def _current_user() -> dict[str, Any]:
    token = _auth_token()
    api_key = _request_api_key()
    if token:
        session = _find_one("auth_sessions", {"token": token})
        if session:
            user = _find_one("users", {"user_id": session.get("user_id")})
            if not user:
                raise PermissionError("User not found")
            return user

        if not api_key:
            raise PermissionError("Invalid auth token")

    if api_key:
        user = _find_one("users", {"api_key": api_key})
        if not user:
            raise PermissionError("Invalid API key")
        return user

    raise PermissionError("Missing auth token or API key")


def _list_collection(collection: str, query: dict[str, Any], *, sort_field: str = "created_at") -> list[dict[str, Any]]:
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 100)), 1), 200)
    cursor = _db()[collection].find(query).sort(sort_field, -1).skip((page - 1) * page_size).limit(page_size)
    items = [_clean(doc) for doc in cursor]
    for item in items:
        if "name" not in item and item.get("title"):
            item["name"] = item["title"]
        if "title" not in item and item.get("name"):
            item["title"] = item["name"]
    return items


def _normalize_hierarchy_path(path: str) -> list[str]:
    parts = [part.strip() for part in str(path).replace("/", ">").split(">") if part.strip()]
    return parts


def _parse_markdown_worldview_entries(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    stack: list[str] = []
    current_title: str | None = None
    current_level = 0
    current_lines: list[str] = []
    entries: list[dict[str, str]] = []

    def flush_current() -> None:
        nonlocal current_title, current_level, current_lines
        if not current_title:
            return
        content = "\n".join(current_lines).strip()
        if content:
            path = " > ".join(stack[:current_level])
            entries.append({"name": current_title, "path": path, "content": content})
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = pattern.match(line)
        if match:
            flush_current()
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            current_title = title
            current_level = level
            current_lines = []
            continue
        if current_title:
            current_lines.append(line)
    flush_current()
    return entries


def _parse_json_worldview_entries(payload: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def walk(node: Any, path_stack: list[str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, path_stack)
            return
        if isinstance(node, dict):
            label = str(node.get("name") or node.get("title") or node.get("label") or node.get("text") or "").strip()
            next_stack = path_stack + ([label] if label else [])
            content = node.get("content")
            if content is None:
                content = node.get("summary")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False, indent=2)
            if label and isinstance(content, str) and content.strip():
                entries.append({"name": label, "path": " > ".join(next_stack), "content": content.strip()})
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child, next_stack)
                return
            for key, value in node.items():
                if key in {"name", "title", "label", "text", "content", "summary", "children"}:
                    continue
                if isinstance(value, (dict, list)):
                    key_stack = next_stack + ([str(key)] if not label else [])
                    walk(value, key_stack)
            return
        if path_stack and isinstance(node, str) and node.strip():
            entries.append({"name": path_stack[-1], "path": " > ".join(path_stack), "content": node.strip()})

    walk(payload, [])
    return entries


def _parse_opml_worldview_entries(root: ET.Element) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    body = root.find("body")
    if body is None:
        return entries

    def walk(node: ET.Element, path_stack: list[str]) -> None:
        label = (node.get("text") or node.get("title") or node.get("name") or "").strip()
        next_stack = path_stack + ([label] if label else [])
        children = [child for child in list(node) if child.tag.lower().endswith("outline")]

        note_parts = []
        for attr_name in ("_note", "note", "content", "summary", "description"):
            attr_value = node.get(attr_name)
            if attr_value and attr_value.strip():
                note_parts.append(attr_value.strip())

        leaf_child_texts = []
        branch_children: list[ET.Element] = []
        for child in children:
            child_label = (child.get("text") or child.get("title") or child.get("name") or "").strip()
            grand_children = [grand for grand in list(child) if grand.tag.lower().endswith("outline")]
            child_note = ""
            for attr_name in ("_note", "note", "content", "summary", "description"):
                attr_value = child.get(attr_name)
                if attr_value and attr_value.strip():
                    child_note = attr_value.strip()
                    break
            if grand_children:
                branch_children.append(child)
                continue
            if child_note:
                if child_label:
                    child_path = " > ".join(next_stack + [child_label])
                    entries.append({"name": child_label, "path": child_path, "content": child_note})
                else:
                    leaf_child_texts.append(child_note)
            elif child_label:
                leaf_child_texts.append(child_label)

        content_parts = note_parts + leaf_child_texts
        if label and content_parts:
            entries.append({"name": label, "path": " > ".join(next_stack), "content": "\n".join(content_parts).strip()})

        if label and not children and not content_parts:
            entries.append({"name": label, "path": " > ".join(next_stack), "content": label})
        for child in branch_children:
            walk(child, next_stack)

    for outline in body:
        walk(outline, [])
    return entries


def _parse_xml_worldview_entries(root: ET.Element) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def walk(node: ET.Element, path_stack: list[str]) -> None:
        label = (node.get("name") or node.get("title") or node.get("label") or node.get("text") or node.tag).strip()
        next_stack = path_stack + [label]
        text_parts = []
        if node.text and node.text.strip():
            text_parts.append(node.text.strip())
        for attr_name in ("content", "summary", "note", "description", "value"):
            attr_value = node.get(attr_name)
            if attr_value and attr_value.strip():
                text_parts.append(attr_value.strip())
        text_content = "\n".join(text_parts).strip()
        children = list(node)
        if text_content:
            entries.append({"name": label, "path": " > ".join(next_stack), "content": text_content})
        for child in children:
            walk(child, next_stack)

    walk(root, [])
    return entries


def _parse_worldview_import_file(file_path: str, filename: str) -> list[dict[str, str]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".md", ".markdown"}:
        with open(file_path, "r", encoding="utf-8") as handle:
            return _parse_markdown_worldview_entries(handle.read())
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as handle:
            return _parse_json_worldview_entries(json.load(handle))
    if ext == ".opml":
        tree = ET.parse(file_path)
        return _parse_opml_worldview_entries(tree.getroot())
    if ext == ".xml":
        tree = ET.parse(file_path)
        root = tree.getroot()
        if root.tag.lower().endswith("opml"):
            return _parse_opml_worldview_entries(root)
        return _parse_xml_worldview_entries(root)
    raise ValueError(f"Unsupported worldview import format: {ext}")


def _resolve_novel_context(payload: dict[str, Any]) -> dict[str, Any]:
    novel_id = payload.get("novel_id")
    if novel_id:
        novel = _find_one("novels", {"novel_id": novel_id}) or {}
        payload.setdefault("world_id", novel.get("world_id"))
    return payload


def _resolve_outline_context(payload: dict[str, Any]) -> dict[str, Any]:
    outline_id = payload.get("outline_id")
    if outline_id:
        outline = _find_one("outlines", {"$or": [{"outline_id": outline_id}, {"id": outline_id}]}) or {}
        payload.setdefault("novel_id", outline.get("novel_id"))
        payload.setdefault("world_id", outline.get("world_id"))
        payload.setdefault("worldview_id", outline.get("worldview_id"))
    return payload


def _enrich_payload(agent_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    if agent_type in {"outline", "outline_summary_create", "outline_summary_update"}:
        _resolve_novel_context(payload)
    if agent_type in {
        "chapter",
        "chapter_outline_summary_create",
        "chapter_outline_summary_update",
        "chapter_intro_summary_create",
        "chapter_intro_summary_update",
        "chapter_content_summary_create",
        "chapter_content_summary_update",
    }:
        _resolve_outline_context(payload)
        _resolve_novel_context(payload)
    if agent_type == "worldview" and payload.get("target_id"):
        doc = _find_one("worldviews", {"worldview_id": payload["target_id"]}) or {}
        payload.setdefault("world_id", doc.get("world_id"))
    if agent_type == "novel" and payload.get("target_id"):
        doc = _find_one("novels", {"novel_id": payload["target_id"]}) or {}
        payload.setdefault("world_id", doc.get("world_id"))
    return payload


def _run_node(module: Any, node_name: str, state: dict[str, Any]) -> dict[str, Any]:
    result = getattr(module, node_name)(state)
    if result:
        state.update(result)
    return state


SUMMARY_AUTO_RETRY_AGENT_TYPES = {
    "outline_summary_create",
    "outline_summary_update",
    "chapter_outline_summary_create",
    "chapter_outline_summary_update",
    "chapter_intro_summary_create",
    "chapter_intro_summary_update",
    "chapter_content_summary_create",
    "chapter_content_summary_update",
}
SUMMARY_AUTO_RETRY_LIMIT = 2
SUMMARY_REVISION_AGENT_TYPES = {
    "outline_summary_create",
    "outline_summary_update",
    "chapter_outline_summary_create",
    "chapter_outline_summary_update",
    "chapter_intro_summary_create",
    "chapter_intro_summary_update",
    "chapter_content_summary_create",
    "chapter_content_summary_update",
}


def _allowed_revision_modes(agent_type: str) -> set[str]:
    if agent_type in SUMMARY_REVISION_AGENT_TYPES:
        return {"summary_rewrite"}
    return {"partial_rewrite", "content_rewrite", "full_rewrite"}


def _rename_last_node(state: dict[str, Any], expected_node_id: str, next_node_id: str) -> None:
    nodes = list(state.get("nodes") or [])
    if not nodes or nodes[-1].get("node_id") != expected_node_id:
        return
    renamed = copy.deepcopy(nodes[-1])
    renamed["node_id"] = next_node_id
    nodes[-1] = renamed
    state["nodes"] = nodes


def _run_review_sequence_with_auto_retry(module: Any, agent_type: str, state: dict[str, Any]) -> dict[str, Any]:
    max_auto_retries = SUMMARY_AUTO_RETRY_LIMIT if agent_type in SUMMARY_AUTO_RETRY_AGENT_TYPES else 0
    retry_count = 0

    while True:
        review_failed = False
        for node_name in _review_sequence(agent_type):
            _run_node(module, node_name, state)
            if retry_count and node_name == "review_node":
                _rename_last_node(state, "review", f"review_retry_{retry_count}")
            if state.get("current_node") == "modify_content":
                review_failed = True
                break

        if not review_failed:
            state["auto_retry_count"] = retry_count
            return state

        if retry_count >= max_auto_retries:
            state["status"] = "review_failed"
            state["auto_retry_count"] = retry_count
            return state

        retry_count += 1
        state["feedback"] = state.get("review_feedback") or state.get("feedback", "")
        state["revision_mode"] = state.get("revision_mode") or "summary_rewrite"
        result = module.modify_content_node(state)
        if result:
            state.update(result)
        _rename_last_node(state, "modify_content", f"revision_retry_{retry_count}")


def _run_until_human(agent_type: str, action: str, payload: dict[str, Any], message: str) -> dict[str, Any]:
    module = AGENT_MODULES[agent_type]
    state: dict[str, Any] = {
        "action": action,
        "payload": _enrich_payload(agent_type, payload),
        "message": message,
        "nodes": [],
        "conversation": [{"role": "user", "content": message, "created_at": _now()}],
        "iterations": 0,
        "committed": False,
    }
    _run_node(module, "input_node", state)
    _run_node(module, "initial_expansion_node", state)
    _run_review_sequence_with_auto_retry(module, agent_type, state)

    if state.get("current_node") not in {"modify_content", "human"}:
        state["current_node"] = "human"
    state.setdefault("status", "waiting_human")
    return state


def _save_run(run: dict[str, Any]) -> dict[str, Any]:
    db = _db()
    run = copy.deepcopy(run)
    run.setdefault("run_id", f"run_{uuid.uuid4().hex[:12]}")
    run["updated_at"] = _now()
    run.setdefault("created_at", run["updated_at"])
    db["hierarchy_agent_runs"].update_one({"run_id": run["run_id"]}, {"$set": _clean(run)}, upsert=True)
    return _clean(run)


def _rebuild_node_prompt(run: dict[str, Any], node: dict[str, Any]) -> str:
    module = AGENT_MODULES.get(run.get("agent_type"))
    if not module:
        return ""

    node_id = str(node.get("node_id") or "")
    node_input = node.get("input") if isinstance(node.get("input"), dict) else {}
    payload = node_input.get("payload") if isinstance(node_input.get("payload"), dict) else {}
    message = str(run.get("message") or "")
    revision_mode = node_input.get("revision_mode")
    feedback = str(node_input.get("feedback") or "")
    review_feedback = str(node_input.get("review_feedback") or "")

    if node_id == "initial_expansion" and hasattr(module, "build_initial_expansion_prompt"):
        return module.build_initial_expansion_prompt(run.get("action", "create"), payload, message, revision_mode=revision_mode, feedback=feedback)
    if node_id == "modify_content" and hasattr(module, "build_modification_prompt"):
        return module.build_modification_prompt(
            run.get("action", "create"),
            payload,
            message,
            revision_mode=revision_mode,
            feedback=feedback,
            expansion_error=review_feedback,
        )

    review_entity_type_map = {
        "world_rule_review": "worldview_world_rules",
        "worldview_consistency_review": "worldview_consistency",
        "review": "novel_world_rules",
        "world_review": {
            "outline": "outline_world_rules",
            "chapter": "chapter_world_rules",
        },
        "worldview_review": {
            "outline": "outline_worldview_rules",
            "chapter": "chapter_worldview_rules",
        },
        "novel_review": {
            "outline": "outline_novel_rules",
            "chapter": "chapter_novel_rules",
        },
        "outline_review": "chapter_outline_rules",
        "chapter_review": "chapter_consistency",
        "chapter_outline_review": "chapter_chapter_outline_rules",
        "plot_review": "chapter_plot_errors",
    }
    entity_type = review_entity_type_map.get(node_id)
    if isinstance(entity_type, dict):
        entity_type = entity_type.get(run.get("agent_type"))
    if isinstance(entity_type, str) and payload:
        system_prompt, user_message = review_agent.build_review_messages(_db(), entity_type, payload)
        return f"[System]\n{system_prompt}\n\n[User]\n{user_message}"

    return ""


def _expected_llm_agent_name(run: dict[str, Any], node: dict[str, Any]) -> str:
    node_id = str(node.get("node_id") or "")
    node_input = node.get("input") if isinstance(node.get("input"), dict) else {}
    return expected_llm_agent_name(str(run.get("agent_type") or ""), node_id, manual_edit=bool(node_input.get("manual_edit")))


def _is_llm_node(node_id: str) -> bool:
    return node_id in {
        "initial_expansion",
        "modify_content",
        "world_rule_review",
        "worldview_consistency_review",
        "review",
        "world_review",
        "worldview_review",
        "novel_review",
        "outline_review",
        "chapter_review",
        "chapter_outline_review",
        "plot_review",
    }


def _hydrate_run_prompts(run: dict[str, Any]) -> dict[str, Any]:
    hydrated = copy.deepcopy(run)
    nodes = []
    for node in hydrated.get("nodes") or []:
        next_node = copy.deepcopy(node)
        output = copy.deepcopy(next_node.get("output")) if isinstance(next_node.get("output"), dict) else {}
        llm_call = output.get("llm_call") if isinstance(output.get("llm_call"), dict) else {}
        node_id = str(next_node.get("node_id") or "")
        expected_llm_agent_name = _expected_llm_agent_name(hydrated, next_node)
        if _is_llm_node(node_id) and "llm_invoked" not in output:
            output["llm_invoked"] = True
        if expected_llm_agent_name:
            if output.get("agent_name") in {"", None, *LEGACY_SHARED_LLM_AGENT_NAMES}:
                output["agent_name"] = expected_llm_agent_name
            if output.get("llm_agent_name") in {"", None, *LEGACY_SHARED_LLM_AGENT_NAMES}:
                output["llm_agent_name"] = expected_llm_agent_name
            if llm_call.get("llm_agent_name") in {"", None, *LEGACY_SHARED_LLM_AGENT_NAMES}:
                llm_call["llm_agent_name"] = expected_llm_agent_name
        if output.get("llm_invoked") and not llm_call.get("prompt"):
            prompt = _rebuild_node_prompt(hydrated, next_node)
            if prompt:
                llm_call["prompt"] = prompt
                llm_call["prompt_chars"] = len(prompt)
        if llm_call:
            output["llm_call"] = llm_call
        next_node["output"] = output
        nodes.append(next_node)
    hydrated["nodes"] = nodes
    return hydrated


def _load_run(run_id: str) -> dict[str, Any]:
    run = _find_one("hierarchy_agent_runs", {"run_id": run_id})
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    return _clean(_hydrate_run_prompts(run))


def _list_hierarchy_runs(query: dict[str, Any]) -> list[dict[str, Any]]:
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 100)), 1), 200)
    cursor = _db()["hierarchy_agent_runs"].find(query).sort("created_at", -1).skip((page - 1) * page_size).limit(page_size)
    return [_clean(_hydrate_run_prompts(doc)) for doc in cursor]


def _commit_run(run: dict[str, Any]) -> dict[str, Any]:
    module = AGENT_MODULES[run["agent_type"]]
    state = dict(run)
    state["decision"] = "approve"
    result = module.commit_node(state)
    state.update(result or {})
    state["status"] = "completed"
    state["committed"] = True
    return _save_run(state)


def _review_sequence(agent_type: str) -> list[str]:
    return {
        "world": [],
        "worldview": ["world_rule_review_node", "worldview_consistency_review_node"],
        "novel": ["review_node"],
        "outline": ["world_review_node", "worldview_review_node", "novel_review_node"],
        "chapter": ["world_review_node", "worldview_review_node", "novel_review_node", "outline_review_node", "chapter_review_node"],
        "outline_summary_create": ["review_node"],
        "outline_summary_update": ["review_node"],
        "chapter_outline_summary_create": ["review_node"],
        "chapter_outline_summary_update": ["review_node"],
        "chapter_intro_summary_create": ["review_node"],
        "chapter_intro_summary_update": ["review_node"],
        "chapter_content_summary_create": ["review_node"],
        "chapter_content_summary_update": ["review_node"],
    }[agent_type]


def _request_changes_run(run: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    module = AGENT_MODULES[run["agent_type"]]
    revision_mode = data.get("revision_mode")
    allowed_revision_modes = _allowed_revision_modes(run["agent_type"])
    if revision_mode not in allowed_revision_modes:
        raise ValueError(f"Invalid revision_mode: {revision_mode}")

    patch_payload = data.get("payload") or {}
    if not isinstance(patch_payload, dict):
        raise ValueError("payload must be an object when decision=request_changes")

    state = copy.deepcopy(run)
    merged_payload = dict(state.get("pending_payload") or state.get("payload") or {})
    merged_payload.update(patch_payload)
    state["pending_payload"] = _enrich_payload(run["agent_type"], merged_payload)
    state["feedback"] = str(data.get("message") or "")
    state["decision"] = "request_changes"
    state["revision_mode"] = revision_mode
    state["manual_edit"] = bool(data.get("manual_edit"))
    state["committed"] = False

    conversation = list(state.get("conversation") or [])
    conversation.append({
        "role": "user",
        "message": state["feedback"],
        "decision": "request_changes",
        "revision_mode": revision_mode,
        "manual_edit": state["manual_edit"],
        "payload": state["pending_payload"],
        "created_at": _now(),
    })
    state["conversation"] = conversation

    result = module.modify_content_node(state)
    if result:
        state.update(result)
    nodes = list(state.get("nodes") or [])
    if nodes and nodes[-1].get("node_id") == "modify_content":
        modify_input = dict(nodes[-1].get("input") or {})
        modify_input["manual_edit"] = state["manual_edit"]
        nodes[-1]["input"] = modify_input
        state["nodes"] = nodes

    _run_review_sequence_with_auto_retry(module, run["agent_type"], state)

    if state.get("status") != "review_failed":
        state["current_node"] = "human"
        state["status"] = "waiting_human"

    return _save_run(state)


def _reject_run(run: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(run)
    state["decision"] = "reject"
    state["feedback"] = str(data.get("message") or "")
    state["status"] = "rejected"
    state["current_node"] = "end"
    state["committed"] = False
    conversation = list(state.get("conversation") or [])
    conversation.append({
        "role": "user",
        "message": state["feedback"],
        "decision": "reject",
        "created_at": _now(),
    })
    state["conversation"] = conversation
    return _save_run(state)


def _normalize_agent_type(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return AGENT_ALIASES.get(str(value).strip().lower()) or AGENT_ALIASES.get(str(value).strip())


def _normalize_action(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return ACTION_ALIASES.get(str(value).strip().lower()) or ACTION_ALIASES.get(str(value).strip())


def _infer_agent_type(data: dict[str, Any], payload: dict[str, Any], message: str) -> tuple[str, str]:
    explicit = (
        data.get("agent_type")
        or data.get("agent")
        or data.get("target_agent")
        or data.get("entity")
        or data.get("entity_type")
        or data.get("type")
        or payload.get("agent_type")
        or payload.get("entity_type")
        or payload.get("type")
    )
    agent_type = _normalize_agent_type(explicit)
    if agent_type:
        return agent_type, "explicit"

    if payload.get("outline_id") or payload.get("chapter_id") or payload.get("scene_id") or payload.get("prose_id"):
        return "chapter", "payload_context"
    if payload.get("novel_id"):
        return "outline", "payload_context"
    if payload.get("worldview_id"):
        return "worldview", "payload_context"

    text = f"{data.get('title', '')} {message}".lower()
    for candidate, keywords in AGENT_KEYWORDS:
        if any(keyword.lower() in text for keyword in keywords):
            return candidate, "message_keyword"

    raise ValueError("Cannot infer agent_type. Provide one of: world, worldview, novel, outline, chapter")


def _infer_action(data: dict[str, Any], payload: dict[str, Any], agent_type: str, message: str) -> tuple[str, str]:
    explicit = data.get("action") or payload.get("action")
    action = _normalize_action(explicit)
    if action:
        return action, "explicit"

    id_fields = AGENT_CAPABILITIES[agent_type]["id_fields"]
    if payload.get("target_id") or any(payload.get(field) for field in id_fields):
        return "update", "payload_id"

    text = f"{data.get('title', '')} {message}".lower()
    for raw, normalized in ACTION_ALIASES.items():
        if raw.lower() in text:
            return normalized, "message_keyword"

    return "create", "default"


def _normalize_dispatch_payload(agent_type: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    if action == "update" and not payload.get("target_id"):
        for field in AGENT_CAPABILITIES[agent_type]["id_fields"]:
            if field != "target_id" and payload.get(field):
                payload["target_id"] = payload[field]
                break
    return payload


def _validate_dispatch(agent_type: str, action: str, payload: dict[str, Any]) -> None:
    if agent_type not in AGENT_MODULES:
        raise ValueError(f"Unsupported agent_type: {agent_type}")
    if action not in {"create", "update"}:
        raise ValueError(f"Unsupported action for router dispatch: {action}. Supported actions: create, update")
    if action == "update" and not payload.get("target_id"):
        raise ValueError(f"Update dispatch for {agent_type} requires target_id or one of {AGENT_CAPABILITIES[agent_type]['id_fields']}")
    missing = [field for field in AGENT_CAPABILITIES[agent_type]["required_context"] if not payload.get(field)]
    if missing and action == "create":
        raise ValueError(f"{agent_type} dispatch missing required context: {', '.join(missing)}")


def _save_dispatch(dispatch: dict[str, Any]) -> dict[str, Any]:
    dispatch = copy.deepcopy(dispatch)
    dispatch.setdefault("dispatch_id", f"dispatch_{uuid.uuid4().hex[:12]}")
    dispatch["updated_at"] = _now()
    dispatch.setdefault("created_at", dispatch["updated_at"])
    _db()["agent_dispatch_requests"].update_one(
        {"dispatch_id": dispatch["dispatch_id"]},
        {"$set": _clean(dispatch)},
        upsert=True,
    )
    return _clean(dispatch)


def _load_dispatch(dispatch_id: str) -> dict[str, Any]:
    dispatch = _find_one("agent_dispatch_requests", {"dispatch_id": dispatch_id})
    if not dispatch:
        raise ValueError(f"Dispatch not found: {dispatch_id}")
    return _clean(dispatch)


def _load_dispatch_by_task_ref(task_ref: str) -> dict[str, Any]:
    dispatch = _find_one("agent_dispatch_requests", {"external_task_ref": task_ref})
    if not dispatch:
        raise ValueError(f"Dispatch not found for task_ref: {task_ref}")
    return _clean(dispatch)


def _dispatch_response(dispatch: dict[str, Any], route: dict[str, Any], run: dict[str, Any] | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"status": "success", "dispatch": dispatch, "route": route}
    if dispatch.get("external_task_ref"):
        response["task_ref"] = dispatch["external_task_ref"]
        response["external_task_ref"] = dispatch["external_task_ref"]
    if run is not None:
        response["run"] = run
    return response


def _start_agent_run(agent_type: str, action: str, payload: dict[str, Any], message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if agent_type == "chapter" and action == "check":
        state = chapter_check_agent.run_check(_enrich_payload(agent_type, payload), message)
        run = {
            **state,
            "run_id": f"run_{uuid.uuid4().hex[:12]}",
            "agent_type": agent_type,
            "action": action,
            "review_required": True,
            "status": state.get("status", "completed"),
            "current_node": state.get("current_node", "output"),
            "committed": False,
        }
        if metadata:
            run["dispatch"] = metadata
        return _save_run(run)
    state = _run_until_human(agent_type, action, payload, message)
    run = {
        **state,
        "run_id": f"run_{uuid.uuid4().hex[:12]}",
        "agent_type": agent_type,
        "action": action,
        "review_required": agent_type != "world",
        "status": state.get("status", "waiting_human"),
        "current_node": state.get("current_node", "human"),
        "committed": False,
    }
    if metadata:
        run["dispatch"] = metadata
    return _save_run(run)


@app.errorhandler(Exception)
def handle_error(exc: Exception):
    status = 500
    if isinstance(exc, HTTPException):
        status = exc.code or 500
    if isinstance(exc, PermissionError):
        status = 401
    if isinstance(exc, ValueError):
        status = 400
    return _json({"status": "error", "error": str(exc)}, status)


@app.get("/")
def index():
    return _json({"status": "success", "service": "novel_agent"})


@app.post("/api/auth/register")
def register_user():
    data = _body()
    username = str(_require(data.get("username"), "Missing username")).strip()
    password = str(_require(data.get("password"), "Missing password"))
    display_name = str(data.get("display_name") or username).strip()
    email = str(data.get("email") or "").strip()
    if len(username) < 3:
        return _json({"status": "error", "error": "Username must be at least 3 characters"}, 400)
    if len(password) < 6:
        return _json({"status": "error", "error": "Password must be at least 6 characters"}, 400)
    if _find_one("users", {"username": username}):
        return _json({"status": "error", "error": f"Username already exists: {username}"}, 409)
    now = _now()
    user = {
        "user_id": f"user_{uuid.uuid4().hex[:12]}",
        "username": username,
        "display_name": display_name,
        "email": email,
        "api_key": _new_api_key(),
        "password_hash": generate_password_hash(password),
        "created_at": now,
        "updated_at": now,
        "last_login_at": now,
    }
    _db()["users"].insert_one(user)
    token = secrets.token_urlsafe(32)
    _db()["auth_sessions"].insert_one(
        {
            "token": token,
            "user_id": user["user_id"],
            "username": username,
            "created_at": now,
            "updated_at": now,
        }
    )
    return _json({"status": "success", "token": token, "api_key": user["api_key"], "user": _public_user(user)})


@app.post("/api/auth/login")
def login_user():
    data = _body()
    username = str(_require(data.get("username"), "Missing username")).strip()
    password = str(_require(data.get("password"), "Missing password"))
    user = _find_one("users", {"username": username})
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return _json({"status": "error", "error": "Invalid username or password"}, 401)
    token = secrets.token_urlsafe(32)
    now = _now()
    api_key = user.get("api_key") or _new_api_key()
    session = {
        "token": token,
        "user_id": user["user_id"],
        "username": username,
        "created_at": now,
        "updated_at": now,
    }
    _db()["auth_sessions"].insert_one(session)
    _db()["users"].update_one(
        {"user_id": user["user_id"]},
        {"$set": {"api_key": api_key, "last_login_at": now, "updated_at": now}},
    )
    user["api_key"] = api_key
    user["last_login_at"] = now
    user["updated_at"] = now
    return _json({"status": "success", "token": token, "api_key": api_key, "user": _public_user(user)})


@app.get("/api/auth/me")
def get_current_user():
    return _json({"status": "success", "user": _public_user(_current_user())})


@app.post("/api/auth/logout")
def logout_user():
    token = _auth_token()
    if token:
        _db()["auth_sessions"].delete_many({"token": token})
    return _json({"status": "success"})


@app.post("/api/worlds/create")
def create_world():
    data = _body()
    world_id = data.get("world_id") or f"world_{uuid.uuid4().hex[:8]}"
    if not data.get("name"):
        return _json({"status": "error", "error": "Missing world name"}, 400)
    if _find_one("worlds", {"world_id": world_id}):
        return _json({"status": "error", "error": f"World already exists: {world_id}"}, 409)
    worldview_id = f"wv_{world_id}"
    doc = {
        "world_id": world_id,
        "name": data["name"],
        "summary": data.get("summary", ""),
        "forbidden_rules": data.get("forbidden_rules", []),
        "basic_settings": data.get("basic_settings", {}),
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db()["worlds"].insert_one(doc)
    _db()["worldviews"].insert_one(
        {
            "worldview_id": worldview_id,
            "world_id": world_id,
            "name": f"{data['name']} 世界观设定集",
            "summary": data.get("summary", ""),
            "forbidden_rules": [],
            "basic_settings": {},
            "auto_created": True,
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    return _json({"status": "success", "world_id": world_id, "worldview_id": worldview_id})


@app.post("/api/worlds/update")
def update_world():
    data = _body()
    world_id = _require(data.get("world_id") or data.get("target_id"), "Missing world_id")
    if not _find_one("worlds", {"world_id": world_id}):
        return _json({"status": "error", "error": f"World not found: {world_id}"}, 404)
    update = {key: data[key] for key in ("name", "summary", "forbidden_rules", "basic_settings") if key in data}
    update["updated_at"] = _now()
    _db()["worlds"].update_one({"world_id": world_id}, {"$set": update})
    return _json({"status": "success", "world_id": world_id})


@app.delete("/api/worlds/delete")
def delete_world():
    data = _body()
    world_id = _require(data.get("world_id"), "Missing world_id")
    db = _db()
    if not data.get("cascade", False):
        if db["novels"].count_documents({"world_id": world_id}) > 0:
            return _json({"status": "error", "error": "Conflict: World has children. Use cascade=True to delete."}, 409)
    db["worlds"].delete_many({"world_id": world_id})
    db["worldviews"].delete_many({"world_id": world_id})
    db["lore"].delete_many({"world_id": world_id})
    if data.get("cascade", True):
        db["novels"].delete_many({"world_id": world_id})
        db["outlines"].delete_many({"world_id": world_id})
        db["prose"].delete_many({"world_id": world_id})
    return _json({"status": "success", "world_id": world_id})


@app.get("/api/worlds/list")
def list_worlds():
    return _json(_list_collection("worlds", {}))


@app.get("/api/worlds/get")
def get_world():
    world_id = _require(request.args.get("world_id"), "Missing world_id")
    world = _find_one("worlds", {"world_id": world_id})
    if not world:
        return _json({"status": "error", "error": f"World not found: {world_id}"}, 404)
    return _json({"status": "success", "world": world})


@app.post("/api/worldviews/create")
def create_worldview():
    data = _body()
    world_id = _require(data.get("world_id"), "Missing world_id")
    if not _find_one("worlds", {"world_id": world_id}):
        return _json({"status": "error", "error": f"Parent World {world_id} not found"}, 404)
    existing_worldview = _find_worldview_by_world(world_id)
    return _json(
        {
            "status": "error",
            "error": (
                "Create worldview is disabled. Each world automatically owns one independent worldview library when the world is created. "
                f"Use /workflow/worldview?action=create&world_id={world_id} to create worldview settings inside library "
                f"{existing_worldview.get('worldview_id') if existing_worldview else ''}, or use /api/worldviews/update only to maintain the library metadata."
            ).strip(),
        },
        409,
    )


@app.post("/api/worldviews/update")
def update_worldview():
    data = _body()
    worldview_id = _require(data.get("worldview_id") or data.get("target_id"), "Missing worldview_id")
    worldview = _find_one("worldviews", {"worldview_id": worldview_id})
    if not worldview:
        return _json({"status": "error", "error": f"Worldview not found: {worldview_id}"}, 404)
    requested_world_id = data.get("world_id")
    if requested_world_id and requested_world_id != worldview.get("world_id"):
        return _json(
            {
                "status": "error",
                "error": (
                    f"Worldview {worldview_id} belongs to world {worldview.get('world_id')} and cannot be moved "
                    f"to world {requested_world_id}. Each world keeps its own independent worldview set."
                ),
            },
            409,
        )
    update = {key: data[key] for key in ("name", "summary", "forbidden_rules", "basic_settings") if key in data}
    update["updated_at"] = _now()
    _db()["worldviews"].update_one({"worldview_id": worldview_id}, {"$set": update})
    return _json({"status": "success", "worldview_id": worldview_id})


@app.delete("/api/worldviews/delete")
def delete_worldview():
    data = _body()
    worldview_id = _require(data.get("worldview_id"), "Missing worldview_id")
    db = _db()
    db["worldviews"].delete_many({"worldview_id": worldview_id})
    if data.get("cascade", True):
        db["lore"].delete_many({"worldview_id": worldview_id})
    return _json({"status": "success", "worldview_id": worldview_id})


@app.get("/api/worldviews/list")
def list_worldviews():
    query = {key: request.args[key] for key in ("world_id", "worldview_id") if request.args.get(key)}
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    if "world_id" in query and "worldview_id" not in query:
        if _find_one("worlds", {"world_id": query["world_id"]}):
            _ensure_worldview_library(query["world_id"])
        else:
            return _json([])
    return _json(_list_collection("worldviews", query))


@app.post("/api/worldviews/import")
def import_worldviews():
    world_id = _require(request.form.get("world_id"), "Missing world_id")
    worldview_id = _require(request.form.get("worldview_id"), "Missing worldview_id")
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _json({"status": "error", "error": "Missing import file"}, 400)

    world = _find_one("worlds", {"world_id": world_id})
    if not world:
        return _json({"status": "error", "error": f"World not found: {world_id}"}, 404)
    worldview = _ensure_worldview_library(world_id)
    if worldview["worldview_id"] != worldview_id:
        return _json(
            {
                "status": "error",
                "error": (
                    f"World {world_id} owns unique worldview library {worldview['worldview_id']}; "
                    f"requested import target {worldview_id} is invalid."
                ),
            },
            409,
        )

    suffix = os.path.splitext(upload.filename)[1].lower()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            upload.save(temp_file)
            temp_path = temp_file.name
        parsed_entries = _parse_worldview_import_file(temp_path, upload.filename)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if not parsed_entries:
        return _json({"status": "error", "error": "No structured worldview entries parsed from file"}, 400)

    imported_entries: list[dict[str, Any]] = []
    db = _db()
    for entry in parsed_entries:
        path = " > ".join(_normalize_hierarchy_path(entry["path"]))
        doc_id = f"wvimp_{uuid.uuid5(uuid.NAMESPACE_URL, f'{world_id}:{worldview_id}:{path}').hex}"
        existing = db["lore"].find_one({"id": doc_id}) or {}
        created_at = existing.get("created_at") or _now()
        doc = {
            "id": doc_id,
            "doc_id": doc_id,
            "type": "worldview",
            "name": entry["name"],
            "content": entry["content"],
            "category": path,
            "path": path,
            "world_id": world_id,
            "worldview_id": worldview_id,
            "created_at": created_at,
            "updated_at": _now(),
            "timestamp": _now(),
            "source_filename": upload.filename,
        }
        db["lore"].update_one({"id": doc_id}, {"$set": doc}, upsert=True)
        imported_entries.append({"id": doc_id, "name": entry["name"], "path": path})

    return _json(
        {
            "status": "success",
            "world_id": world_id,
            "worldview_id": worldview_id,
            "imported_count": len(imported_entries),
            "entries": imported_entries,
        }
    )


@app.post("/api/novels/create")
def create_novel():
    data = _body()
    world_id = _require(data.get("world_id"), "Missing world_id")
    if not _find_one("worlds", {"world_id": world_id}):
        return _json({"status": "error", "error": f"Parent World {world_id} not found"}, 404)
    novel_id = data.get("novel_id") or f"novel_{uuid.uuid4().hex[:8]}"
    if _find_one("novels", {"novel_id": novel_id}):
        return _json({"status": "error", "error": f"Novel already exists: {novel_id}"}, 409)
    doc = {
        "novel_id": novel_id,
        "world_id": world_id,
        "name": _require(data.get("name"), "Missing novel name"),
        "introduction": data.get("introduction", ""),
        "summary": data.get("summary", ""),
        "forbidden_rules": data.get("forbidden_rules", []),
        "basic_settings": data.get("basic_settings", {}),
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db()["novels"].insert_one(doc)
    return _json({"status": "success", "novel_id": novel_id})


@app.post("/api/novels/update")
def update_novel():
    data = _body()
    novel_id = _require(data.get("novel_id") or data.get("target_id"), "Missing novel_id")
    novel = _find_one("novels", {"novel_id": novel_id})
    if not novel:
        return _json({"status": "error", "error": f"Novel not found: {novel_id}"}, 404)
    if data.get("world_id") and not _find_one("worlds", {"world_id": data["world_id"]}):
        return _json({"status": "error", "error": f"Parent World {data['world_id']} not found"}, 404)
    update = {key: data[key] for key in ("name", "introduction", "summary", "world_id", "worldview_id", "forbidden_rules", "basic_settings") if key in data}
    update["updated_at"] = _now()
    _db()["novels"].update_one({"novel_id": novel_id}, {"$set": update})
    return _json({"status": "success", "novel_id": novel_id})


@app.delete("/api/novels/delete")
def delete_novel():
    data = _body()
    novel_id = _require(data.get("novel_id"), "Missing novel_id")
    db = _db()
    if not _find_one("novels", {"novel_id": novel_id}):
        return _json({"status": "error", "error": f"Novel not found: {novel_id}"}, 404)
    if not data.get("cascade", False):
        if db["outlines"].count_documents({"novel_id": novel_id}) > 0:
            return _json({"status": "error", "error": "Conflict: Novel has children. Use cascade=True to delete."}, 409)
    db["novels"].delete_many({"novel_id": novel_id})
    if data.get("cascade", True):
        db["outlines"].delete_many({"novel_id": novel_id})
        db["prose"].delete_many({"novel_id": novel_id})
    return _json({"status": "success", "novel_id": novel_id})


@app.get("/api/novels/list")
def list_novels():
    query = {key: request.args[key] for key in ("world_id", "novel_id") if request.args.get(key)}
    if request.args.get("query"):
        text = request.args["query"]
        query["$or"] = [
            {"novel_id": {"$regex": text, "$options": "i"}},
            {"name": {"$regex": text, "$options": "i"}},
            {"introduction": {"$regex": text, "$options": "i"}},
            {"summary": {"$regex": text, "$options": "i"}},
        ]
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    return _json(_list_collection("novels", query))


@app.get("/api/novels/get")
def get_novel():
    novel_id = _require(request.args.get("novel_id"), "Missing novel_id")
    novel = _find_one("novels", {"novel_id": novel_id})
    if not novel:
        return _json({"status": "error", "error": f"Novel not found: {novel_id}"}, 404)
    return _json({"status": "success", "novel": novel})


@app.post("/api/novels/chapter-outline-templates/create")
def create_chapter_outline_template():
    data = _body()
    novel_id = _require(data.get("novel_id"), "Missing novel_id")
    if not _find_one("novels", {"novel_id": novel_id}):
        return _json({"status": "error", "error": f"Parent Novel {novel_id} not found"}, 404)
    template_id = f"tpl_{uuid.uuid4().hex[:8]}"
    doc = {
        "template_id": template_id,
        "novel_id": novel_id,
        "name": _require(data.get("name"), "Missing template name"),
        "content": _require(data.get("content"), "Missing template content"),
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db()["chapter_outline_templates"].insert_one(doc)
    return _json({"status": "success", "template": doc})


@app.post("/api/novels/chapter-outline-templates/update")
def update_chapter_outline_template():
    data = _body()
    template_id = _require(data.get("template_id"), "Missing template_id")
    template = _find_one("chapter_outline_templates", {"template_id": template_id})
    if not template:
        return _json({"status": "error", "error": f"Template not found: {template_id}"}, 404)
    update = {key: data[key] for key in ("name", "content") if key in data}
    update["updated_at"] = _now()
    _db()["chapter_outline_templates"].update_one({"template_id": template_id}, {"$set": update})
    return _json({"status": "success", "template_id": template_id})


@app.delete("/api/novels/chapter-outline-templates/delete")
def delete_chapter_outline_template():
    data = _body()
    template_id = _require(data.get("template_id"), "Missing template_id")
    db = _db()
    if not _find_one("chapter_outline_templates", {"template_id": template_id}):
        return _json({"status": "error", "error": f"Template not found: {template_id}"}, 404)
    db["chapter_outline_templates"].delete_many({"template_id": template_id})
    return _json({"status": "success", "template_id": template_id})


@app.get("/api/novels/chapter-outline-templates/list")
def list_chapter_outline_templates():
    novel_id = _require(request.args.get("novel_id"), "Missing novel_id")
    if not _find_one("novels", {"novel_id": novel_id}):
        return _json({"status": "error", "error": f"Novel not found: {novel_id}"}, 404)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    query = {"novel_id": novel_id}
    return _json(_list_collection("chapter_outline_templates", query))


@app.post("/api/outlines/create")
def create_outline():
    data = _body()
    payload = _resolve_novel_context(dict(data))
    novel_id = _require(payload.get("novel_id"), "Missing novel_id")
    if not _find_one("novels", {"novel_id": novel_id}):
        return _json({"status": "error", "error": f"Parent Novel {novel_id} not found"}, 404)
    outline_id = payload.get("outline_id") or payload.get("id") or f"outline_{uuid.uuid4().hex[:8]}"
    doc = {
        "outline_id": outline_id,
        "id": outline_id,
        "novel_id": novel_id,
        "world_id": payload.get("world_id"),
        "worldview_id": payload.get("worldview_id"),
        "name": _require(payload.get("name"), "Missing outline name"),
        "summary": payload.get("summary", ""),
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db()["outlines"].update_one({"outline_id": outline_id}, {"$set": doc}, upsert=True)
    return _json({"status": "success", "outline_id": outline_id})


@app.post("/api/outlines/update")
def update_outline():
    data = _body()
    outline_id = _require(data.get("outline_id") or data.get("target_id"), "Missing outline_id")
    update = {key: data[key] for key in ("name", "summary", "worldview_id") if key in data}
    update["updated_at"] = _now()
    _db()["outlines"].update_one({"$or": [{"outline_id": outline_id}, {"id": outline_id}]}, {"$set": update})
    return _json({"status": "success", "outline_id": outline_id})


@app.get("/api/outlines/list")
def list_outlines():
    query = {key: request.args[key] for key in ("world_id", "worldview_id", "novel_id", "outline_id", "id") if request.args.get(key)}
    if request.args.get("query"):
        text = request.args["query"]
        query["$or"] = [
            {"outline_id": {"$regex": text, "$options": "i"}},
            {"id": {"$regex": text, "$options": "i"}},
            {"name": {"$regex": text, "$options": "i"}},
            {"title": {"$regex": text, "$options": "i"}},
            {"summary": {"$regex": text, "$options": "i"}},
        ]
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    return _json(_list_collection("outlines", query))


@app.post("/api/archive/update")
def update_archive():
    data = _body()
    item_type = _require(data.get("type"), "Missing type")
    item_id = _require(data.get("id"), "Missing id")
    db = _db()
    
    update_fields = {"updated_at": _now()}
    if item_type == "prose":
        update_fields.update({
            "id": item_id, "scene_id": item_id, "type": "prose"
        })
        name = data.get("name") or data.get("title")
        if name: update_fields["name"] = update_fields["title"] = name
        for key in ("content", "outline_id", "novel_id", "worldview_id", "world_id", "chapter_outline_id", "template_id"):
            if key in data: update_fields[key] = data[key]
        db["prose"].update_one({"id": item_id}, {"$set": update_fields}, upsert=True)
    elif item_type == "worldview":
        update_fields.update({"id": item_id, "type": "worldview"})
        if "name" in data: update_fields["name"] = data["name"]
        if "content" in data: update_fields["content"] = data["content"]
        for key in ("category", "path", "world_id", "worldview_id"):
            if key in data: update_fields[key] = data[key]
        if "path" in update_fields and "category" not in update_fields:
            update_fields["category"] = update_fields["path"]
        if "category" in update_fields and "path" not in update_fields:
            update_fields["path"] = update_fields["category"]
        db["lore"].update_one({"id": item_id}, {"$set": update_fields}, upsert=True)
    elif item_type == "outline":
        update_fields.update({"outline_id": item_id, "id": item_id})
        if "name" in data: update_fields["name"] = data["name"]
        summary = data.get("content") or data.get("summary")
        if summary: update_fields["summary"] = summary
        if "worldview_id" in data: update_fields["worldview_id"] = data["worldview_id"]
        db["outlines"].update_one({"outline_id": item_id}, {"$set": update_fields}, upsert=True)
    else:
        raise ValueError(f"Invalid type: {item_type}")
    return _json({"status": "success", "id": item_id})


@app.delete("/api/archive/delete")
def delete_archive():
    data = _body()
    item_type = _require(data.get("type"), "Missing type")
    item_id = _require(data.get("id"), "Missing id")
    if item_type == "prose":
        _db()["prose"].delete_many({"$or": [{"id": item_id}, {"scene_id": item_id}, {"prose_id": item_id}]})
    elif item_type == "worldview":
        _db()["lore"].delete_many({"id": item_id})
    elif item_type == "outline":
        _db()["outlines"].delete_many({"$or": [{"outline_id": item_id}, {"id": item_id}]})
    else:
        raise ValueError(f"Invalid type: {item_type}")
    return _json({"status": "success", "id": item_id})


@app.get("/api/lore/list")
def list_lore():
    filters: list[dict[str, Any]] = []
    for key in ("world_id", "worldview_id", "novel_id", "outline_id", "type", "chapter_outline_id"):
        if request.args.get(key):
            filters.append({key: request.args[key]})
    chapter_outline_mode = request.args.get("chapter_outline_mode", "").strip().lower()
    if not request.args.get("chapter_outline_id"):
        if chapter_outline_mode == "root":
            filters.append({
                "$or": [
                    {"chapter_outline_id": {"$exists": False}},
                    {"chapter_outline_id": ""},
                    {"chapter_outline_id": None},
                ],
            })
        elif chapter_outline_mode == "child":
            filters.append({"chapter_outline_id": {"$exists": True, "$nin": ["", None]}})
    if not filters:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    if request.args.get("query"):
        text = request.args["query"]
        filters.append({
            "$or": [
                {"name": {"$regex": text, "$options": "i"}},
                {"title": {"$regex": text, "$options": "i"}},
                {"content": {"$regex": text, "$options": "i"}},
            ],
        })
    query: dict[str, Any]
    if len(filters) == 1:
        query = filters[0]
    else:
        query = {"$and": filters}
    items = _list_collection("prose", query) + _list_collection("lore", query)
    return _json(items)


def _entry_tree_path(entry: dict[str, Any]) -> list[str]:
    raw_path = entry.get("path") or entry.get("category") or entry.get("type") or "未分类"
    parts = [part.strip() for part in str(raw_path).replace("/", ">").split(">") if part.strip()]
    return parts or ["未分类"]


def _insert_tree_entry(node: dict[str, Any], path: list[str], entry: dict[str, Any]) -> None:
    if not path:
        node["entries"].append(entry)
        return
    child_name = path[0]
    child = next((item for item in node["children"] if item["name"] == child_name), None)
    if child is None:
        child = {"name": child_name, "children": [], "entries": []}
        node["children"].append(child)
    _insert_tree_entry(child, path[1:], entry)


@app.get("/api/lore/tree")
def get_lore_tree():
    world_id = _require(request.args.get("world_id"), "world_id is required")
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    world = _find_one("worlds", {"world_id": world_id})
    if not world:
        return _json({"status": "error", "error": "World not found"}, 404)

    query = {"world_id": world_id}
    entries = _list_collection("lore", query) + _list_collection("prose", query)
    root = {"name": world.get("name") or world_id, "children": [], "entries": []}
    for entry in entries:
        _insert_tree_entry(root, _entry_tree_path(entry), entry)
    return _json(root)


@app.get("/api/world-hierarchy/tree")
def get_world_hierarchy_tree():
    world_id = _require(request.args.get("world_id"), "world_id is required")
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    db = _db()
    world = _find_one("worlds", {"world_id": world_id})
    if not world:
        return _json({"status": "error", "error": "World not found"}, 404)

    world["worldviews"] = _list_collection("worldviews", {"world_id": world_id})
    for wv in world["worldviews"]:
        wv["lore"] = _list_collection("lore", {"worldview_id": wv["worldview_id"]})

    world["novels"] = _list_collection("novels", {"world_id": world_id})
    for novel in world["novels"]:
        novel["outlines"] = _list_collection("outlines", {"novel_id": novel.get("novel_id")})
        for outline in novel["outlines"]:
            outline["chapters"] = _list_collection("prose", {"outline_id": outline.get("outline_id")})

    return _json({"status": "success", "worlds": [world]})


@app.get("/api/workflow/outline-chapter/state")
def get_outline_chapter_state():
    if not request.args.get("page") or not request.args.get("page_size"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)

    query = {key: request.args[key] for key in ("world_id", "worldview_id", "novel_id", "outline_id") if request.args.get(key)}
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)

    world_id = query.get("world_id")
    worldview_id = query.get("worldview_id")
    novel_id = query.get("novel_id")
    outline_id = query.get("outline_id")

    if world_id and not _find_one("worlds", {"world_id": world_id}):
        return _json({"status": "error", "error": f"World not found: {world_id}"}, 404)
    if worldview_id:
        worldview_query = {"worldview_id": worldview_id}
        if world_id:
            worldview_query["world_id"] = world_id
        if not _find_one("worldviews", worldview_query):
            return _json({"status": "error", "error": f"Worldview not found: {worldview_id}"}, 404)
    if novel_id:
        novel_query = {"novel_id": novel_id}
        if world_id:
            novel_query["world_id"] = world_id
        if not _find_one("novels", novel_query):
            return _json({"status": "error", "error": f"Novel not found: {novel_id}"}, 404)
    if outline_id:
        outline_query: dict[str, Any] = {"$or": [{"outline_id": outline_id}, {"id": outline_id}]}
        outline = _find_one("outlines", outline_query)
        if not outline:
            return _json({"status": "error", "error": f"Outline not found: {outline_id}"}, 404)
        if world_id and outline.get("world_id") and outline.get("world_id") != world_id:
            return _json({"status": "error", "error": f"Outline {outline_id} does not belong to world {world_id}"}, 409)
        if worldview_id and outline.get("worldview_id") and outline.get("worldview_id") != worldview_id:
            return _json({"status": "error", "error": f"Outline {outline_id} does not belong to worldview {worldview_id}"}, 409)

    chapters = _list_collection("prose", query)
    for chapter in chapters:
        chapter.setdefault("type", "prose")
        if not chapter.get("id"):
            chapter["id"] = chapter.get("scene_id") or chapter.get("prose_id")
        if not chapter.get("name") and chapter.get("title"):
            chapter["name"] = chapter["title"]
        if not chapter.get("title") and chapter.get("name"):
            chapter["title"] = chapter["name"]

    return _json({
        "status": "success",
        "world_id": world_id,
        "worldview_id": worldview_id,
        "novel_id": novel_id,
        "outline_id": outline_id,
        "chapters": chapters,
    })


@app.get("/api/downstream-summaries/list")
def list_downstream_summaries():
    query = {
        key: request.args[key]
        for key in (
            "summary_id",
            "summary_key",
            "agent_type",
            "summary_scope",
            "summary_action",
            "world_id",
            "worldview_id",
            "novel_id",
            "outline_id",
            "chapter_id",
            "target_id",
        )
        if request.args.get(key)
    }
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    if not request.args.get("page"):
        return _json({"status": "error", "error": "Missing pagination"}, 400)
    return _json(_list_collection("downstream_summaries", query))


@app.get("/api/router/agents")
def list_router_agents():
    return _json({
        "status": "success",
        "agents": [
            {"agent_type": agent_type, **capability}
            for agent_type, capability in AGENT_CAPABILITIES.items()
        ],
        "actions": sorted(set(ACTION_ALIASES.values())),
        "dispatch_endpoint": "/api/router/dispatch",
    })


@app.post("/api/router/dispatch")
def dispatch_agent_request():
    data = _body()
    payload = data.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    invocation = data.get("invocation") or {}
    if invocation and not isinstance(invocation, dict):
        raise ValueError("invocation must be an object")

    message = data.get("message") or data.get("task") or data.get("prompt") or ""
    agent_type, agent_reason = _infer_agent_type(data, payload, str(message))
    action, action_reason = _infer_action(data, payload, agent_type, str(message))
    payload = _normalize_dispatch_payload(agent_type, action, payload)
    _validate_dispatch(agent_type, action, payload)

    dispatch_id = data.get("dispatch_id") or f"dispatch_{uuid.uuid4().hex[:12]}"
    external_task_ref = (
        data.get("external_task_ref")
        or data.get("task_ref")
        or invocation.get("external_task_ref")
        or invocation.get("task_ref")
    )
    route = {
        "agent_type": agent_type,
        "agent_reason": agent_reason,
        "action": action,
        "action_reason": action_reason,
        "requires_human_approval": not bool(data.get("auto_approve", False)),
    }
    dispatch = {
        "dispatch_id": dispatch_id,
        "status": "planned" if data.get("dry_run") else "started",
        "message": message,
        "payload": payload,
        "route": route,
        "source": data.get("source", "external"),
        "external_request_id": data.get("external_request_id"),
        "external_task_ref": external_task_ref,
        "zentex_task_id": data.get("zentex_task_id") or invocation.get("zentex_task_id"),
        "callback_url": data.get("callback_url") or invocation.get("callback_url"),
    }

    if data.get("dry_run"):
        saved_dispatch = _save_dispatch(dispatch)
        return _json(_dispatch_response(saved_dispatch, route))

    run = _start_agent_run(
        agent_type,
        action,
        payload,
        str(message),
        metadata={
            "dispatch_id": dispatch_id,
            "source": dispatch["source"],
            "external_request_id": dispatch.get("external_request_id"),
            "external_task_ref": dispatch.get("external_task_ref"),
            "zentex_task_id": dispatch.get("zentex_task_id"),
        },
    )
    dispatch["run_id"] = run["run_id"]
    dispatch["status"] = "waiting_human" if run.get("status") == "waiting_human" else run.get("status", "started")

    if data.get("auto_approve") and run.get("status") == "waiting_human":
        run = _commit_run(run)
        dispatch["status"] = "completed"
        dispatch["auto_approved"] = True
    elif data.get("auto_approve"):
        dispatch["auto_approved"] = False
        dispatch["auto_approve_blocked_reason"] = f"Run status is {run.get('status')}; only waiting_human runs can be auto-approved."

    saved_dispatch = _save_dispatch(dispatch)
    return _json(_dispatch_response(saved_dispatch, route, run))


@app.get("/api/router/dispatch/get")
def get_dispatch_request():
    dispatch_id = request.args.get("dispatch_id")
    task_ref = request.args.get("external_task_ref") or request.args.get("task_ref")
    if dispatch_id:
        dispatch = _load_dispatch(dispatch_id)
    elif task_ref:
        dispatch = _load_dispatch_by_task_ref(task_ref)
    else:
        raise ValueError("Missing dispatch_id or task_ref")
    response: dict[str, Any] = {"status": "success", "dispatch": dispatch}
    if dispatch.get("external_task_ref"):
        response["task_ref"] = dispatch["external_task_ref"]
        response["external_task_ref"] = dispatch["external_task_ref"]
    if dispatch.get("run_id"):
        response["run"] = _load_run(dispatch["run_id"])
    return _json(response)


@app.get("/api/router/dispatch/list")
def list_dispatch_requests():
    query = {
        key: request.args[key]
        for key in ("dispatch_id", "status", "source", "external_request_id", "external_task_ref", "run_id")
        if request.args.get(key)
    }
    if request.args.get("agent_type"):
        query["route.agent_type"] = request.args["agent_type"]
    if request.args.get("action"):
        query["route.action"] = request.args["action"]
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    return _json({"status": "success", "dispatches": _list_collection("agent_dispatch_requests", query)})


@app.post("/api/hierarchy-agent/start")
def start_hierarchy_agent():
    data = _body()
    agent_type = _require(data.get("agent_type"), "Missing agent_type")
    action = data.get("action", "create")
    if agent_type not in AGENT_MODULES:
        raise ValueError(f"Unsupported agent_type: {agent_type}")
    if action == "check" and agent_type != "chapter":
        raise ValueError("Only chapter supports action=check")
    if action not in {"create", "update", "check"}:
        raise ValueError(f"Unsupported action: {action}")
    payload = data.get("payload") or {}
    if action == "check":
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if not payload.get("outline_id"):
            raise ValueError("章节检查工作流必须提供 outline_id")
        if not str(payload.get("content") or "").strip():
            raise ValueError("章节检查工作流必须提供直接内容")
        if not str(payload.get("chapter_outline") or "").strip():
            # Allow target chapter readback only when an explicit target is supplied.
            has_target = any(payload.get(key) for key in ("target_id", "chapter_outline_id", "chapter_id", "scene_id", "prose_id", "id"))
            if not has_target:
                raise ValueError("章节检查工作流必须提供章节大纲，或指定可读取的目标章节")
    run = _start_agent_run(agent_type, action, data.get("payload") or {}, data.get("message", ""))
    return _json({"status": "success", "run": run})


@app.post("/api/hierarchy-agent/respond")
def respond_hierarchy_agent():
    data = _body()
    run_id = _require(data.get("run_id"), "Missing run_id")
    decision = data.get("decision")
    run = _load_run(run_id)
    if run.get("status") not in {"waiting_human", "review_failed"}:
        raise ValueError(f"Run status does not accept human decisions: {run.get('status')}")
    if decision == "approve":
        run = _commit_run(run)
    elif decision == "request_changes":
        run = _request_changes_run(run, data)
    elif decision == "reject":
        run = _reject_run(run, data)
    else:
        raise ValueError(f"Unsupported decision in minimal backend: {decision}")
    return _json({"status": "success", "run": run})


@app.get("/api/hierarchy-agent/list")
def list_hierarchy_agents():
    query = {key: request.args[key] for key in ("agent_type", "run_id", "world_id") if request.args.get(key)}
    if not query:
        return _json({"status": "error", "error": "Missing required query condition"}, 400)
    return _json({"status": "success", "runs": _list_hierarchy_runs(query)})


@app.get("/api/hierarchy-agent/get")
def get_hierarchy_agent():
    run_id = _require(request.args.get("run_id"), "Missing run_id")
    return _json({"status": "success", "run": _load_run(run_id)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", "5006")))
    app.run(host="127.0.0.1", port=port, debug=False)
