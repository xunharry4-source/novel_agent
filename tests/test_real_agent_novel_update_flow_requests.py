import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_novel,
    create_world,
    get_novel,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_novel_update_flow_requests():
    suffix = unique_suffix("agent_novel_update")
    world_id = f"world_{suffix}"
    novel_id = f"novel_{suffix}"
    original_summary = "原始小说：航图师在北港追查灯塔异常。"
    updated_name = f"Agent Novel Updated {suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="小说 Agent 更新父级世界")
        create_novel(world_id=world_id, novel_id=novel_id, name=f"Agent Novel Base {suffix}", summary=original_summary)

        run = start_agent(
            "novel",
            "update",
            {
                "target_id": novel_id,
                "world_id": world_id,
                "name": updated_name,
                "summary": "更新草案：强调主线冲突、登记制度和潮汐水晶约束。",
            },
            "真实测试小说 Agent 修改流程",
        )
        assert run["review_required"] is True, run
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
                    "message": "请把主线冲突写得更明确，并保持世界规则和小说规则一致。",
                    "revision_mode": "partial_rewrite",
                    "manual_edit": True,
                    "payload": {
                        "target_id": novel_id,
                        "world_id": world_id,
                        "name": updated_name,
                        "summary": "更新草案：强调主线冲突、登记制度和潮汐水晶约束。",
                    },
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
        queried = get_novel(novel_id)
        assert queried is not None, approved
        assert queried["novel_id"] == novel_id, queried
        assert queried["world_id"] == world_id, queried
        assert queried["name"] == updated_name, queried
        assert queried["summary"] == approved["pending_payload"]["summary"], queried
        assert queried["summary"] != original_summary, queried
        assert "forbidden_rules" in queried, queried
        assert "basic_settings" in queried, queried
    finally:
        cleanup_world(world_id)
