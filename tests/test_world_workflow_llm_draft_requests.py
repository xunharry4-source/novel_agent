#!/usr/bin/env python3
"""Real requests test: world workflow must use dedicated node-level LLM identities.

The test talks to a running Flask API service. It does not mock the LLM, DB, or
HTTP layer. A passing run proves that `/api/hierarchy-agent/start` creates a
real LLM-backed initial expansion node for `world + create`, and that
`/api/hierarchy-agent/respond` with `request_changes` creates a distinct
LLM-backed modify node. The real world Agent graph is:
input -> initial expansion -> human -> modify content -> human -> saver.
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


def main() -> None:
    suffix = str(int(time.time()))
    world_id = f"world_llm_initial_{suffix}"
    world_name = f"LLM Initial World {suffix}"
    original_summary = "短草案：海上城市。"
    try:
        started = request_json(
            "POST",
            "/api/hierarchy-agent/start",
            json={
                "agent_type": "world",
                "action": "create",
                "message": "创建世界，并生成真实初始扩充内容",
                "payload": {
                    "world_id": world_id,
                    "name": world_name,
                    "summary": original_summary,
                },
            },
        )
        run = started["run"]
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        improved_payload = initial_expansion["output"]["payload"]

        assert run["agent_type"] == "world", run
        assert run["action"] == "create", run
        assert run["status"] == "waiting_human", run
        assert run["review_required"] is False, run
        assert all(node["node_id"] != "draft" for node in run["nodes"]), run
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
        assert initial_expansion["output"]["llm_invoked"] is True, initial_expansion
        assert initial_expansion["output"]["agent_name"] == "world_agent_initial_expansion", initial_expansion
        assert initial_expansion["output"]["llm_agent_name"] == "world_agent_initial_expansion", initial_expansion
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "world_agent_initial_expansion", initial_expansion
        assert initial_expansion["output"]["raw_response"], initial_expansion
        assert improved_payload["world_id"] == world_id, initial_expansion
        assert improved_payload["name"] == world_name, initial_expansion
        assert improved_payload["summary"] != original_summary, initial_expansion
        assert len(improved_payload["summary"]) > len(original_summary) + 20, initial_expansion
        assert run["pending_payload"]["summary"] == improved_payload["summary"], run

        revised = request_json(
            "POST",
            "/api/hierarchy-agent/respond",
            json={
                "run_id": run["run_id"],
                "decision": "request_changes",
                "message": "请补充更明确的世界禁止规则与基础设定",
                "feedback": "请补充更明确的世界禁止规则与基础设定",
                "revision_mode": "partial_rewrite",
            },
        )
        revised_run = revised["run"]
        modify_content = next(node for node in revised_run["nodes"] if node["node_id"] == "modify_content")
        assert modify_content["output"]["llm_invoked"] is True, modify_content
        assert modify_content["output"]["agent_name"] == "world_agent_modify_content", modify_content
        assert modify_content["output"]["llm_agent_name"] == "world_agent_modify_content", modify_content
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "world_agent_modify_content", modify_content
        assert modify_content["output"]["llm_call"]["llm_agent_name"] != initial_expansion["output"]["llm_call"]["llm_agent_name"], {
            "initial_expansion": initial_expansion,
            "modify_content": modify_content,
        }

        approved = request_json(
            "POST",
            "/api/hierarchy-agent/respond",
            json={"run_id": run["run_id"], "decision": "approve", "message": "批准写库"},
        )
        committed = approved["run"]
        assert committed["status"] == "completed", committed
        assert committed["committed"] is True, committed
        assert committed["commit_result"]["world_id"] == world_id, committed

        worlds = request_json("GET", "/api/worlds/list")
        found = next((item for item in worlds if item.get("world_id") == world_id), None)
        assert found is not None, worlds
        assert found["name"] == world_name, found
        assert found["summary"] == improved_payload["summary"], found
        assert found["summary"] != original_summary, found
        print("world workflow LLM initial expansion requests test passed")
    finally:
        if world_id:
            try:
                requests.delete(
                    f"{BASE_URL}/api/worlds/delete",
                    json={"world_id": world_id, "cascade": True},
                    timeout=30,
                )
            except requests.RequestException as exc:
                print(f"cleanup warning: {exc}")


if __name__ == "__main__":
    main()
