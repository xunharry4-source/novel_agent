#!/usr/bin/env python3
"""Real requests test for the five hierarchy workflow create/update paths.

No mocks are used. The test talks to an already running Flask API service and
verifies each approved workflow through a follow-up business query.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests


BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5006").rstrip("/")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))


def request_json(method: str, path: str, expected: int = 200, **kwargs: Any) -> Any:
    response = requests.request(method, f"{BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if response.status_code != expected:
        raise AssertionError(f"{method} {path} expected {expected}, got {response.status_code}: {payload}")
    return payload


def find_one(path: str, id_key: str, expected_id: str, params: dict[str, Any]) -> dict[str, Any]:
    items = request_json("GET", path, params=params)
    if not isinstance(items, list):
        raise AssertionError(f"{path} did not return a list: {items}")
    for item in items:
        if item.get(id_key) == expected_id or item.get("id") == expected_id:
            return item
    raise AssertionError(f"{expected_id} not found in {path}: {items}")


def start_workflow(agent_type: str, action: str, payload: dict[str, Any], message: str) -> dict[str, Any]:
    data = request_json(
        "POST",
        "/api/hierarchy-agent/start",
        json={"agent_type": agent_type, "action": action, "payload": payload, "message": message},
    )
    run = data["run"]
    assert run["agent_type"] == agent_type
    assert run["action"] == action
    assert run["nodes"], run
    assert run.get("conversation"), run
    assert run.get("current_node") in {"human", "review", "world_rule_review", "worldview_consistency_review", "world_review", "worldview_review", "novel_review", "outline_review", "chapter_review"}, run
    generation_node_id = "initial_expansion"
    generation_node = next(node for node in run["nodes"] if node["node_id"] == generation_node_id)
    assert generation_node["output"]["llm_invoked"] is True, generation_node
    assert generation_node["output"]["raw_response"], generation_node
    assert all(node["node_id"] != "draft" for node in run["nodes"]), run
    if agent_type == "world":
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
    if agent_type == "worldview":
        assert any(node["node_id"] == "world_rule_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "worldview_consistency_review" for node in run["nodes"]), run
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
    if agent_type == "outline":
        assert any(node["node_id"] == "world_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "worldview_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "novel_review" for node in run["nodes"]), run
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
    if agent_type == "chapter":
        assert any(node["node_id"] == "world_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "worldview_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "novel_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "outline_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "chapter_review" for node in run["nodes"]), run
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
    for key, value in generation_node["output"]["payload"].items():
        if key in {"name", "summary", "content", "world_id", "worldview_id", "novel_id", "outline_id", "target_id"}:
            assert run["pending_payload"].get(key) == value, {
                "field": key,
                "generation_payload": generation_node["output"]["payload"],
                "pending_payload": run["pending_payload"],
            }
    return run


def request_changes(run_id: str, payload: dict[str, Any], message: str = "人工小范围修改") -> dict[str, Any]:
    data = request_json(
        "POST",
        "/api/hierarchy-agent/respond",
        json={
            "run_id": run_id,
            "decision": "request_changes",
            "message": message,
            "revision_mode": "partial_rewrite",
            "manual_edit": True,
            "payload": payload,
        },
    )
    run = data["run"]
    assert run["iterations"] >= 2, run
    assert run.get("conversation"), run
    revision_nodes = [node for node in run["nodes"] if node["node_id"] == "modify_content"]
    assert revision_nodes, run
    assert revision_nodes[-1]["input"]["manual_edit"] is True
    assert revision_nodes[-1]["input"]["revision_mode"] == "partial_rewrite"
    return run


def approve(run_id: str) -> dict[str, Any]:
    data = request_json("POST", "/api/hierarchy-agent/respond", json={"run_id": run_id, "decision": "approve", "message": "批准写库"})
    run = data["run"]
    assert run["status"] == "completed", run
    assert run["committed"] is True, run
    assert run["current_node"] in {"apply", "commit"}, run
    assert run.get("commit_result"), run
    assert any(node["node_id"] in {"apply", "commit"} for node in run["nodes"]), run
    return run


def assert_llm_expanded(run: dict[str, Any], field: str, original_value: str) -> None:
    generation_node_id = "initial_expansion"
    generation_node = next(node for node in run["nodes"] if node["node_id"] == generation_node_id)
    generation_payload = generation_node["output"]["payload"]
    assert generation_node["output"]["llm_invoked"] is True, generation_node
    assert generation_node["output"]["raw_response"], generation_node
    assert field in generation_payload, generation_payload
    assert generation_payload[field] != original_value, {
        "reason": "LLM generation did not change the requested business field",
        "field": field,
        "original": original_value,
        "generation_payload": generation_payload,
    }
    assert len(str(generation_payload[field])) > len(original_value) + 20, {
        "reason": "LLM generation did not materially expand the requested business field",
        "field": field,
        "original": original_value,
        "generation_payload": generation_payload,
    }
    assert run["pending_payload"][field] == generation_payload[field], {
        "reason": "LLM generation was not propagated into pending_payload",
        "field": field,
        "pending_payload": run["pending_payload"],
        "generation_payload": generation_payload,
    }


def run_change(agent_type: str, action: str, payload: dict[str, Any], message: str, revised_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    run = start_workflow(agent_type, action, payload, message)
    if revised_payload is not None:
        run = request_changes(run["run_id"], revised_payload)
    return approve(run["run_id"])


def main() -> None:
    suffix = str(int(time.time()))
    world_id = f"world_agent_workflow_{suffix}"
    worldview_id = ""
    worldview_entry_id = ""
    novel_id = ""
    outline_id = ""
    chapter_id = ""

    try:
      # world create/update
        worldview_summary = "低魔蒸汽群岛：潮汐水晶供能，灯塔公会管理航线，贵族不能垄断矿权。"
        worldview_summary_final = (
            "这是一个低魔蒸汽群岛世界观：潮汐水晶只能在满月海沟稳定充能，灯塔公会负责记录航线、矿权和灵术消耗。"
            "贵族、商会和船长都必须服从公开登记制度，违规者会失去港口停泊权。"
        )
        novel_summary = "航图师林澈追查失踪灯塔，阻止商会伪造潮汐刻度并挑起航线战争。"
        novel_summary_final = (
            "故事讲述航图师林澈与前灯塔守卫合作，追查被篡改的潮汐刻度，阻止商会利用假航线挑起群岛战争。"
            "人物目标、能源规则和港口制度均与既有世界观保持一致。"
        )
        outline_summary = "第一卷：北港灯塔误报潮汐，主角查登记簿、追伪造水晶，并在满月海沟揭露阴谋。"
        outline_summary_final = (
            "第一卷大纲：北港灯塔误报潮汐后，主角先核对灯塔公会登记簿，再调查被替换的充能水晶，"
            "最后在满月海沟公开证据并阻止舰队误入禁航区。"
        )
        chapter_content = "北港起雾，林澈发现登记的潮汐刻度比水晶余辉高出两格，决定封锁离港信号。"
        chapter_content_final = (
            "北港的雾压在灯塔窗外，林澈重新核对铜桌上的航图和潮汐刻度。登记簿写着水晶仍有七格余辉，"
            "但灯芯只剩五格微光；按照公会规则，他必须立刻封锁离港信号。"
        )
        world_original_summary = "短草案：潮汐城市。"
        run = run_change(
            "world",
            "create",
            {"world_id": world_id, "name": f"Workflow World {suffix}", "summary": world_original_summary},
            "创建测试世界",
        )
        assert_llm_expanded(run, "summary", world_original_summary)
        assert run["commit_result"]["world_id"] == world_id
        world = find_one("/api/worlds/list", "world_id", world_id, params={})
        assert world["name"] == f"Workflow World {suffix}"
        assert world["summary"] == run["commit_result"]["summary"]
        assert world["summary"] == run["pending_payload"]["summary"]
        assert world["summary"] != world_original_summary
        worldview_libraries = request_json("GET", "/api/worldviews/list", params={"world_id": world_id, "page": 1, "page_size": 10})
        assert len(worldview_libraries) == 1, worldview_libraries
        worldview_id = worldview_libraries[0]["worldview_id"]

        updated_world_name = f"Workflow World Updated {suffix}"
        run = run_change(
            "world",
            "update",
            {"target_id": world_id, "name": updated_world_name, "summary": "world update initial"},
            "修改测试世界",
            {"target_id": world_id, "name": updated_world_name, "summary": "world update final"},
        )
        world = find_one("/api/worlds/list", "world_id", world_id, params={})
        assert world["name"] == updated_world_name
        assert world["summary"] == run["pending_payload"]["summary"]

        # worldview create/update
        run = run_change(
            "worldview",
            "create",
            {"world_id": world_id, "worldview_id": worldview_id, "name": f"Workflow Worldview {suffix}", "summary": worldview_summary},
            "创建测试世界观",
        )
        assert_llm_expanded(run, "summary", worldview_summary)
        worldview_entry_id = run["commit_result"]["id"]
        worldview_entry = find_one("/api/lore/list", "id", worldview_entry_id, {"world_id": world_id, "worldview_id": worldview_id, "page": 1, "page_size": 50})
        assert worldview_entry["world_id"] == world_id
        assert worldview_entry["worldview_id"] == worldview_id
        assert worldview_entry["content"] == run["pending_payload"]["summary"]

        updated_worldview_name = f"Workflow Worldview Updated {suffix}"
        run = run_change(
            "worldview",
            "update",
            {"target_id": worldview_entry_id, "world_id": world_id, "worldview_id": worldview_id, "name": updated_worldview_name, "summary": worldview_summary},
            "修改测试世界观",
            {"target_id": worldview_entry_id, "world_id": world_id, "worldview_id": worldview_id, "name": updated_worldview_name, "summary": worldview_summary_final},
        )
        worldview_entry = find_one("/api/lore/list", "id", worldview_entry_id, {"world_id": world_id, "worldview_id": worldview_id, "page": 1, "page_size": 50})
        assert worldview_entry["name"] == updated_worldview_name
        assert worldview_entry["content"] == run["pending_payload"]["summary"]

        # novel create/update
        run = run_change(
            "novel",
            "create",
            {"world_id": world_id, "name": f"Workflow Novel {suffix}", "summary": novel_summary},
            "创建测试小说",
        )
        assert_llm_expanded(run, "summary", novel_summary)
        novel_id = run["commit_result"]["novel_id"]
        novel = find_one("/api/novels/list", "novel_id", novel_id, {"world_id": world_id, "page": 1, "page_size": 50})
        assert novel["world_id"] == world_id
        assert novel["summary"] == run["pending_payload"]["summary"]

        updated_novel_name = f"Workflow Novel Updated {suffix}"
        run = run_change(
            "novel",
            "update",
            {"target_id": novel_id, "name": updated_novel_name, "summary": novel_summary},
            "修改测试小说",
            {"target_id": novel_id, "name": updated_novel_name, "summary": novel_summary_final},
        )
        novel = find_one("/api/novels/list", "novel_id", novel_id, {"novel_id": novel_id, "page": 1, "page_size": 10})
        assert novel["name"] == updated_novel_name
        assert novel["summary"] == run["pending_payload"]["summary"]

        # outline create/update
        run = run_change(
            "outline",
            "create",
            {
                "world_id": world_id,
                "worldview_id": worldview_id,
                "novel_id": novel_id,
                "name": f"Workflow Outline {suffix}",
                "summary": outline_summary,
            },
            "创建测试大纲",
        )
        assert_llm_expanded(run, "summary", outline_summary)
        outline_id = run["commit_result"]["outline_id"]
        outline = find_one("/api/outlines/list", "outline_id", outline_id, {"world_id": world_id, "outline_id": outline_id, "page": 1, "page_size": 10})
        assert outline["novel_id"] == novel_id
        assert outline["summary"] == run["pending_payload"]["summary"]

        updated_outline_name = f"Workflow Outline Updated {suffix}"
        run = run_change(
            "outline",
            "update",
            {"target_id": outline_id, "name": updated_outline_name, "summary": outline_summary},
            "修改测试大纲",
            {"target_id": outline_id, "name": updated_outline_name, "summary": outline_summary_final},
        )
        outline = find_one("/api/outlines/list", "outline_id", outline_id, {"outline_id": outline_id, "page": 1, "page_size": 10})
        assert outline["title"] == updated_outline_name
        assert outline["summary"] == run["pending_payload"]["summary"]

        # chapter create/update
        run = run_change(
            "chapter",
            "create",
            {
                "world_id": world_id,
                "worldview_id": worldview_id,
                "novel_id": novel_id,
                "outline_id": outline_id,
                "name": f"Workflow Chapter {suffix}",
                "content": chapter_content,
            },
            "创建测试章节",
        )
        assert_llm_expanded(run, "content", chapter_content)
        chapter_id = run["commit_result"]["id"]
        chapter = find_one("/api/lore/list", "id", chapter_id, {"world_id": world_id, "outline_id": outline_id, "page": 1, "page_size": 50})
        assert chapter["outline_id"] == outline_id
        assert chapter["content"] == run["pending_payload"]["content"]

        updated_chapter_name = f"Workflow Chapter Updated {suffix}"
        run = run_change(
            "chapter",
            "update",
            {"target_id": chapter_id, "name": updated_chapter_name, "content": chapter_content},
            "修改测试章节",
            {"target_id": chapter_id, "name": updated_chapter_name, "content": chapter_content_final},
        )
        chapter = find_one("/api/lore/list", "id", chapter_id, {"world_id": world_id, "outline_id": outline_id, "page": 1, "page_size": 50})
        assert chapter["name"] == updated_chapter_name
        assert chapter["content"] == run["pending_payload"]["content"]

        print("five module workflow create/update real requests test passed")
    finally:
        if world_id:
            try:
                requests.delete(f"{BASE_URL}/api/worlds/delete", json={"world_id": world_id, "cascade": True}, timeout=30)
            except requests.RequestException as exc:
                print(f"cleanup warning: {exc}")


if __name__ == "__main__":
    main()
