import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import approve_agent, assert_success, cleanup_world, create_world, get_worldview_for_world, list_lore, request_json, start_agent, unique_suffix


def test_real_agent_worldview_flow_requests():
    suffix = unique_suffix("agent_worldview")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="世界观 Agent 父级世界")
        worldview_library = get_worldview_for_world(world_id)
        run = start_agent("worldview", "create", {"world_id": world_id, "name": f"Agent Worldview {suffix}", "summary": "灯塔公会登记潮汐水晶。"}, "真实测试世界观 Agent 流程")
        assert run["review_required"] is True, run
        assert any(node["node_id"] == "world_rule_review" for node in run["nodes"]), run
        assert any(node["node_id"] == "worldview_consistency_review" for node in run["nodes"]), run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        world_rule_review = next(node for node in run["nodes"] if node["node_id"] == "world_rule_review")
        worldview_consistency_review = next(node for node in run["nodes"] if node["node_id"] == "worldview_consistency_review")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "worldview_agent_initial_expansion", initial_expansion
        assert world_rule_review["output"]["llm_call"]["llm_agent_name"] == "worldview_world_rules_review_agent", world_rule_review
        assert worldview_consistency_review["output"]["llm_call"]["llm_agent_name"] == "worldview_consistency_review_agent", worldview_consistency_review
        assert len(
            {
                initial_expansion["output"]["llm_call"]["llm_agent_name"],
                world_rule_review["output"]["llm_call"]["llm_agent_name"],
                worldview_consistency_review["output"]["llm_call"]["llm_agent_name"],
            }
        ) == 3, {
            "initial_expansion": initial_expansion,
            "world_rule_review": world_rule_review,
            "worldview_consistency_review": worldview_consistency_review,
        }
        approved = approve_agent(run["run_id"])
        entry_id = approved["commit_result"]["id"]
        queried = next(
            (item for item in list_lore(world_id=world_id, worldview_id=worldview_library["worldview_id"], page=1, page_size=50) if item.get("id") == entry_id),
            None,
        )
        assert queried is not None, approved
        assert queried["world_id"] == world_id, queried
        assert queried["worldview_id"] == worldview_library["worldview_id"], queried
        assert queried["name"] == approved["pending_payload"]["name"], queried
        assert queried["content"] == approved["pending_payload"]["summary"], queried
    finally:
        cleanup_world(world_id)


def test_real_agent_worldview_modify_content_uses_distinct_llm_identity():
    suffix = unique_suffix("agent_worldview_modify")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="世界观 Agent 修改节点父级世界")
        run = start_agent(
            "worldview",
            "create",
            {"world_id": world_id, "name": f"Agent Worldview {suffix}", "summary": "初版设定"},
            "真实测试世界观 Agent 修改链路",
        )
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "worldview_agent_initial_expansion", initial_expansion

        modified = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/respond",
                json={
                    "run_id": run["run_id"],
                    "decision": "request_changes",
                    "message": "请把设定改长一些，并强调资源规则",
                    "revision_mode": "partial_rewrite",
                    "manual_edit": True,
                    "payload": {"world_id": world_id, "name": f"Agent Worldview {suffix}", "summary": "初版设定"},
                },
            )
        )["run"]
        modify_content = next(node for node in modified["nodes"] if node["node_id"] == "modify_content")

        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "worldview_agent_human_feedback_modify_content", modify_content
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] != modify_content["output"]["llm_call"]["llm_agent_name"], {
            "initial_expansion": initial_expansion,
            "modify_content": modify_content,
        }
        assert modify_content["input"]["manual_edit"] is True, modify_content
        assert modified["status"] == "waiting_human", modified
    finally:
        cleanup_world(world_id)


def test_real_agent_worldview_create_respects_parent_path():
    suffix = unique_suffix("agent_worldview_nested")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="世界观 Agent 嵌套层级父级世界")
        worldview_library = get_worldview_for_world(world_id)
        run = start_agent(
            "worldview",
            "create",
            {
                "world_id": world_id,
                "worldview_id": worldview_library["worldview_id"],
                "name": f"新生命_{suffix}",
                "summary": "碳基生命分支下新增的生命设定。",
                "parent_path": "种族 > 碳基生命",
            },
            "真实测试世界观 Agent 层级新增",
        )
        approved = approve_agent(run["run_id"])
        entry_id = approved["commit_result"]["id"]
        queried = next(
            (item for item in list_lore(world_id=world_id, worldview_id=worldview_library["worldview_id"], page=1, page_size=50) if item.get("id") == entry_id),
            None,
        )
        assert queried is not None, approved
        assert queried["name"] == f"新生命_{suffix}", queried
        assert queried["path"] == f"种族 > 碳基生命 > 新生命_{suffix}", queried
        assert queried["category"] == queried["path"], queried
        assert queried["worldview_id"] == worldview_library["worldview_id"], queried
    finally:
        cleanup_world(world_id)
