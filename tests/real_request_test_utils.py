"""真实 requests 测试公共工具。

本文件不是测试文件，只提供 HTTP 调用、业务查询和严格断言工具。
禁止 MOCK，禁止假库，禁止吞 HTTP 错误。
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5006").rstrip("/")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))
API_PREFIX = f"{BASE_URL}/api"


def unique_suffix(prefix: str) -> str:
    """生成真实测试用唯一后缀，避免污染已有业务数据。"""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def request_json(method: str, path: str, expected_status: int = 200, **kwargs: Any) -> Any:
    """通过 requests 调用真实 API；状态码或 JSON 解析失败时直接抛出完整错误。"""
    response = requests.request(method, f"{BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"{method} {path} returned non-JSON response\n"
            f"HTTP: {response.status_code}\n"
            f"Body: {response.text}"
        ) from exc
    if response.status_code != expected_status:
        raise AssertionError(
            f"{method} {path} expected HTTP {expected_status}, got {response.status_code}\n"
            f"Response: {payload}"
        )
    return payload


def assert_success(payload: Any) -> dict[str, Any]:
    """断言 API 返回业务成功状态。"""
    assert isinstance(payload, dict), payload
    assert payload.get("status") == "success", payload
    return payload


def list_worlds() -> list[dict[str, Any]]:
    """查询真实世界列表。"""
    data = request_json("GET", "/api/worlds/list")
    assert isinstance(data, list), data
    return data


def list_worldviews(**params: Any) -> list[dict[str, Any]]:
    """按条件查询真实世界观列表。"""
    data = request_json("GET", "/api/worldviews/list", params=params)
    assert isinstance(data, list), data
    return data


def list_novels(**params: Any) -> list[dict[str, Any]]:
    """按条件查询真实小说列表。"""
    data = request_json("GET", "/api/novels/list", params=params)
    assert isinstance(data, list), data
    return data


def list_outlines(**params: Any) -> list[dict[str, Any]]:
    """按条件查询真实大纲列表。"""
    data = request_json("GET", "/api/outlines/list", params=params)
    assert isinstance(data, list), data
    return data


def list_lore(**params: Any) -> list[dict[str, Any]]:
    """按条件查询真实 lore/prose 列表。"""
    data = request_json("GET", "/api/lore/list", params=params)
    assert isinstance(data, list), data
    return data


def find_one(items: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    """从业务查询返回中查找单条记录。"""
    return next((item for item in items if item.get(key) == value or item.get("id") == value), None)


def get_world(world_id: str) -> dict[str, Any] | None:
    """查询单个世界。"""
    return find_one(list_worlds(), "world_id", world_id)


def get_worldview(worldview_id: str) -> dict[str, Any] | None:
    """查询单个世界观。"""
    return find_one(list_worldviews(worldview_id=worldview_id, page=1, page_size=20), "worldview_id", worldview_id)


def get_worldview_for_world(world_id: str) -> dict[str, Any]:
    """查询某个世界唯一的 worldview 库。"""
    items = list_worldviews(world_id=world_id, page=1, page_size=20)
    assert len(items) == 1, {"world_id": world_id, "worldviews": items}
    return items[0]


def get_novel(novel_id: str) -> dict[str, Any] | None:
    """查询单个小说。"""
    return find_one(list_novels(novel_id=novel_id, page=1, page_size=20), "novel_id", novel_id)


def get_outline(outline_id: str) -> dict[str, Any] | None:
    """查询单个大纲。"""
    return find_one(list_outlines(outline_id=outline_id, page=1, page_size=20), "outline_id", outline_id)


def get_chapter(chapter_id: str, *, outline_id: str, worldview_id: str | None = None) -> dict[str, Any] | None:
    """查询单个章节正文。"""
    params: dict[str, Any] = {"outline_id": outline_id, "page": 1, "page_size": 50}
    if worldview_id:
        params["worldview_id"] = worldview_id
    return find_one(list_lore(**params), "id", chapter_id)


def create_world(*, world_id: str, name: str, summary: str, forbidden_rules: list[str] | None = None, basic_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """通过真实 API 创建世界，并查询确认真实入库。"""
    payload: dict[str, Any] = {"world_id": world_id, "name": name, "summary": summary}
    if forbidden_rules is not None:
        payload["forbidden_rules"] = forbidden_rules
    if basic_settings is not None:
        payload["basic_settings"] = basic_settings
    data = assert_success(request_json("POST", "/api/worlds/create", json=payload))
    created_id = data.get("world_id", world_id)
    created = get_world(created_id)
    assert created is not None, {"created_id": created_id, "response": data}
    assert created["name"] == name, created
    worldview = get_worldview_for_world(created_id)
    created["worldview_id"] = worldview["worldview_id"]
    return created


def create_worldview(*, world_id: str, worldview_id: str, name: str, summary: str) -> dict[str, Any]:
    """配置世界创建时自动生成的唯一 worldview 库，并查询确认真实入库。"""
    existing = get_worldview_for_world(world_id)
    data = assert_success(
        request_json(
            "POST",
            "/api/worldviews/update",
            json={"worldview_id": existing["worldview_id"], "name": name, "summary": summary},
        )
    )
    created_id = data.get("worldview_id", existing["worldview_id"])
    created = get_worldview(created_id)
    assert created is not None, {"created_id": created_id, "response": data}
    assert created["world_id"] == world_id, created
    assert created["name"] == name, created
    return created


def create_novel(*, world_id: str, novel_id: str, name: str, summary: str, forbidden_rules: list[str] | None = None, basic_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """通过真实 API 创建小说，并查询确认真实入库。"""
    payload: dict[str, Any] = {"world_id": world_id, "novel_id": novel_id, "name": name, "summary": summary}
    if forbidden_rules is not None:
        payload["forbidden_rules"] = forbidden_rules
    if basic_settings is not None:
        payload["basic_settings"] = basic_settings
    data = assert_success(request_json("POST", "/api/novels/create", json=payload))
    created_id = data.get("novel_id", novel_id)
    created = get_novel(created_id)
    assert created is not None, {"created_id": created_id, "response": data}
    assert created["world_id"] == world_id, created
    return created


def create_outline(*, world_id: str, worldview_id: str, novel_id: str, outline_id: str, name: str, summary: str) -> dict[str, Any]:
    """通过真实 API 创建大纲，并查询确认真实入库。"""
    data = assert_success(request_json("POST", "/api/outlines/create", json={"world_id": world_id, "worldview_id": worldview_id, "novel_id": novel_id, "outline_id": outline_id, "name": name, "summary": summary}))
    created_id = data.get("outline_id", outline_id)
    created = get_outline(created_id)
    assert created is not None, {"created_id": created_id, "response": data}
    assert created["novel_id"] == novel_id, created
    return created


def create_chapter(*, world_id: str, worldview_id: str, novel_id: str, outline_id: str, chapter_id: str, name: str, content: str) -> dict[str, Any]:
    """通过真实 API 创建章节，并查询确认真实入库。"""
    data = assert_success(request_json("POST", "/api/archive/update", json={"id": chapter_id, "type": "prose", "world_id": world_id, "worldview_id": worldview_id, "novel_id": novel_id, "outline_id": outline_id, "name": name, "content": content}))
    created = get_chapter(chapter_id, outline_id=outline_id, worldview_id=worldview_id)
    assert created is not None, {"chapter_id": chapter_id, "response": data}
    assert created["content"] == content, created
    return created


def cleanup_world(world_id: str | None) -> None:
    """清理测试世界；只用于测试末尾资源回收。"""
    if world_id:
        request_json("DELETE", "/api/worlds/delete", json={"world_id": world_id, "cascade": True})


def start_agent(agent_type: str, action: str, payload: dict[str, Any], message: str) -> dict[str, Any]:
    """启动真实 hierarchy-agent 工作流。"""
    data = assert_success(request_json("POST", "/api/hierarchy-agent/start", json={"agent_type": agent_type, "action": action, "payload": payload, "message": message}))
    run = data["run"]
    assert run["agent_type"] == agent_type, run
    assert run["action"] == action, run
    assert run["nodes"], run
    assert all(node["node_id"] != "draft" for node in run["nodes"]), run
    assert any(node["node_id"] == "initial_expansion" and node["output"].get("llm_invoked") is True for node in run["nodes"]), run
    return run


def approve_agent(run_id: str) -> dict[str, Any]:
    """批准真实 hierarchy-agent 工作流，并断言已写库。"""
    data = assert_success(request_json("POST", "/api/hierarchy-agent/respond", json={"run_id": run_id, "decision": "approve", "message": "批准写入"}))
    run = data["run"]
    assert run["status"] == "completed", run
    assert run["committed"] is True, run
    assert run.get("commit_result"), run
    return run


def assert_review_node(run: dict[str, Any], node_id: str, reviewer: str) -> dict[str, Any]:
    """断言工作流中存在指定真实审核节点，并检查审核输出结构。"""
    node = next((item for item in run["nodes"] if item.get("node_id") == node_id), None)
    assert node is not None, run
    assert node["output"].get("reviewer") == reviewer, node
    assert isinstance(node["output"].get("passed"), bool), node
    assert isinstance(node["output"].get("errors"), list), node
    assert node["input"].get("payload"), node
    return node


def make_full_chain(prefix: str) -> dict[str, str]:
    """创建世界、世界观、小说、大纲完整父级链路，并逐项查询确认。"""
    suffix = unique_suffix(prefix)
    world_id = f"world_{suffix}"
    worldview_id = f"wv_{suffix}"
    novel_id = f"novel_{suffix}"
    outline_id = f"outline_{suffix}"
    create_world(
        world_id=world_id,
        name=f"World {suffix}",
        summary="真实测试世界：低魔蒸汽群岛。",
        forbidden_rules=["禁止凭空出现现代枪械", "禁止无代价复活"],
        basic_settings={"era": "蒸汽航海", "power_system": "潮汐水晶", "boundary": "群岛航线"},
    )
    create_worldview(world_id=world_id, worldview_id=worldview_id, name=f"Worldview {suffix}", summary="灯塔公会负责航线和潮汐水晶登记。")
    create_novel(
        world_id=world_id,
        novel_id=novel_id,
        name=f"Novel {suffix}",
        summary="航图师追查灯塔异常。",
        forbidden_rules=["主角不能无证操控禁航舰队", "不能跳过灯塔登记制度"],
        basic_settings={"protagonist_rule": "主角必须依靠航图和登记簿破局", "tone": "克制悬疑", "timeline": "北港事件后一周内"},
    )
    create_outline(world_id=world_id, worldview_id=worldview_id, novel_id=novel_id, outline_id=outline_id, name=f"Outline {suffix}", summary="第一章发现潮汐刻度异常。")
    return {"suffix": suffix, "world_id": world_id, "worldview_id": worldview_id, "novel_id": novel_id, "outline_id": outline_id}
