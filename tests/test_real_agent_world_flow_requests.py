import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    cleanup_world,
    get_world,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_world_flow_requests():
    suffix = unique_suffix("agent_world")
    world_id = f"world_{suffix}"
    try:
        run = start_agent("world", "create", {"world_id": world_id, "name": f"Agent World {suffix}", "summary": "短草案：潮汐群岛。"}, "真实测试世界 Agent 流程")
        assert run["review_required"] is False, run
        assert all(node["node_id"] != "review" for node in run["nodes"]), run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "world_agent_initial_expansion", initial_expansion

        revised = request_json(
            "POST",
            "/api/hierarchy-agent/respond",
            json={
                "run_id": run["run_id"],
                "decision": "request_changes",
                "message": "请补充更明确的世界禁止规则",
                "feedback": "请补充更明确的世界禁止规则",
                "revision_mode": "partial_rewrite",
            },
        )["run"]
        modify_content = next(node for node in revised["nodes"] if node["node_id"] == "modify_content")
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "world_agent_modify_content", modify_content
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] != modify_content["output"]["llm_call"]["llm_agent_name"], {
            "initial_expansion": initial_expansion,
            "modify_content": modify_content,
        }

        approved = approve_agent(run["run_id"])
        queried = get_world(world_id)
        assert queried is not None, approved
        assert queried["world_id"] == world_id, queried
        assert queried["name"] == approved["pending_payload"]["name"], queried
        assert queried["summary"] == approved["pending_payload"]["summary"], queried
        assert "forbidden_rules" in queried, queried
        assert "basic_settings" in queried, queried
    finally:
        cleanup_world(world_id)
