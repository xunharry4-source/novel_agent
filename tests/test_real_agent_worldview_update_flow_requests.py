import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_world,
    get_worldview_for_world,
    list_lore,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_worldview_update_flow_requests():
    suffix = unique_suffix("agent_worldview_update")
    world_id = f"world_{suffix}"
    original_summary = "原始设定：灯塔公会登记潮汐水晶。"
    updated_name = f"Agent Worldview Updated {suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="世界观 Agent 更新父级世界")
        worldview_library = get_worldview_for_world(world_id)

        created = start_agent(
            "worldview",
            "create",
            {
                "world_id": world_id,
                "worldview_id": worldview_library["worldview_id"],
                "name": f"Agent Worldview Base {suffix}",
                "summary": original_summary,
            },
            "先创建一个可用于更新的世界观条目",
        )
        created_approved = approve_agent(created["run_id"])
        entry_id = created_approved["commit_result"]["id"]

        run = start_agent(
            "worldview",
            "update",
            {
                "target_id": entry_id,
                "world_id": world_id,
                "worldview_id": worldview_library["worldview_id"],
                "name": updated_name,
                "summary": "更新草案：强调潮汐刻度、资源规则和登记制度。",
            },
            "真实测试世界观 Agent 修改流程",
        )
        assert run["review_required"] is True, run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        world_rule_review = next(node for node in run["nodes"] if node["node_id"] == "world_rule_review")
        worldview_consistency_review = next(node for node in run["nodes"] if node["node_id"] == "worldview_consistency_review")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "worldview_agent_initial_expansion", initial_expansion
        assert world_rule_review["output"]["llm_call"]["llm_agent_name"] == "worldview_world_rules_review_agent", world_rule_review
        assert worldview_consistency_review["output"]["llm_call"]["llm_agent_name"] == "worldview_consistency_review_agent", worldview_consistency_review

        modified = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/respond",
                json={
                    "run_id": run["run_id"],
                    "decision": "request_changes",
                    "message": "请把资源规则写得更明确，并保持世界规则一致。",
                    "revision_mode": "partial_rewrite",
                    "manual_edit": True,
                    "payload": {
                        "target_id": entry_id,
                        "world_id": world_id,
                        "worldview_id": worldview_library["worldview_id"],
                        "name": updated_name,
                        "summary": "更新草案：强调潮汐刻度、资源规则和登记制度。",
                    },
                },
            )
        )["run"]
        modify_content = next(node for node in modified["nodes"] if node["node_id"] == "modify_content")
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "worldview_agent_human_feedback_modify_content", modify_content
        assert len(
            {
                initial_expansion["output"]["llm_call"]["llm_agent_name"],
                world_rule_review["output"]["llm_call"]["llm_agent_name"],
                worldview_consistency_review["output"]["llm_call"]["llm_agent_name"],
                modify_content["output"]["llm_call"]["llm_agent_name"],
            }
        ) == 4, {
            "initial_expansion": initial_expansion,
            "world_rule_review": world_rule_review,
            "worldview_consistency_review": worldview_consistency_review,
            "modify_content": modify_content,
        }
        assert modified["status"] == "waiting_human", modified

        approved = approve_agent(run["run_id"])
        queried = next(
            (item for item in list_lore(world_id=world_id, worldview_id=worldview_library["worldview_id"], page=1, page_size=50) if item.get("id") == entry_id),
            None,
        )
        assert queried is not None, approved
        assert queried["name"] == updated_name, queried
        assert queried["content"] == approved["pending_payload"]["summary"], queried
        assert queried["content"] != original_summary, queried
        assert queried["world_id"] == world_id, queried
        assert queried["worldview_id"] == worldview_library["worldview_id"], queried
    finally:
        cleanup_world(world_id)
