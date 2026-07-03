import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_world,
    get_novel,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_novel_flow_requests():
    suffix = unique_suffix("agent_novel")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="小说 Agent 父级世界")
        run = start_agent("novel", "create", {"world_id": world_id, "name": f"Agent Novel {suffix}", "summary": "航图师追查灯塔异常。"}, "真实测试小说 Agent 流程")
        assert run["review_required"] is True, run
        assert any(node["node_id"] == "review" for node in run["nodes"]), run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        review = next(node for node in run["nodes"] if node["node_id"] == "review")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "novel_agent_initial_expansion", initial_expansion
        assert review["output"]["llm_call"]["llm_agent_name"] == "novel_world_rules_review_agent", review

        modified = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/respond",
                json={
                    "run_id": run["run_id"],
                    "decision": "request_changes",
                    "message": "请把主线冲突写得更明确，并强调世界规则约束。",
                    "revision_mode": "partial_rewrite",
                    "manual_edit": True,
                    "payload": {"world_id": world_id, "name": f"Agent Novel {suffix}", "summary": "航图师追查灯塔异常。"},
                },
            )
        )["run"]
        modify_content = next(node for node in modified["nodes"] if node["node_id"] == "modify_content")
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "novel_agent_human_feedback_modify_content", modify_content
        assert len(
            {
                initial_expansion["output"]["llm_call"]["llm_agent_name"],
                review["output"]["llm_call"]["llm_agent_name"],
                modify_content["output"]["llm_call"]["llm_agent_name"],
            }
        ) == 3, {
            "initial_expansion": initial_expansion,
            "review": review,
            "modify_content": modify_content,
        }
        assert modified["status"] == "waiting_human", modified

        approved = approve_agent(run["run_id"])
        novel_id = approved["commit_result"]["novel_id"]
        queried = get_novel(novel_id)
        assert queried is not None, approved
        assert queried["world_id"] == world_id, queried
        assert queried["name"] == approved["pending_payload"]["name"], queried
        assert queried["summary"] == approved["pending_payload"]["summary"], queried
        assert "forbidden_rules" in queried, queried
        assert "basic_settings" in queried, queried
    finally:
        cleanup_world(world_id)
