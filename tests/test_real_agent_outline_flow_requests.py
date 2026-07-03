import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_novel,
    create_world,
    create_worldview,
    get_outline,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_outline_flow_requests():
    suffix = unique_suffix("agent_outline")
    world_id = f"world_{suffix}"
    novel_id = f"novel_{suffix}"
    try:
        world = create_world(
            world_id=world_id,
            name=f"World {suffix}",
            summary="近代港口调查世界。",
            forbidden_rules=["禁止超自然复活"],
            basic_settings={"era": "近代工业港口", "power_system": "电力与潮汐能源并存", "boundary": "沿海港区与近海航线"},
        )
        worldview_id = world["worldview_id"]
        create_worldview(
            world_id=world_id,
            worldview_id=worldview_id,
            name=f"Worldview {suffix}",
            summary="港务局、灯塔署、航运商会共同管理港区，允许无线电、发电机、潮汐能源与工业机械并存。",
        )
        create_novel(
            world_id=world_id,
            novel_id=novel_id,
            name=f"Novel {suffix}",
            summary="调查员追查港口异常事故与账册造假。",
            forbidden_rules=["主角不能无证定罪", "不能跳过官方调查程序"],
            basic_settings={"protagonist_rule": "主角必须通过调查和证据推进主线", "tone": "调查悬疑", "timeline": "一周内", "main_conflict": "事故真相与港务体系遮掩冲突"},
        )
        payload = {
            "world_id": world_id,
            "worldview_id": worldview_id,
            "novel_id": novel_id,
            "name": f"Agent Outline {suffix}",
            "summary": "调查员发现港口事故记录与潮汐能源账册不一致。",
        }
        run = start_agent(
            "outline",
            "create",
            payload,
            "真实测试大纲 Agent 流程",
        )
        for _ in range(4):
            if any(node["node_id"] == "novel_review" and node["status"] == "completed" for node in run["nodes"]):
                break
            latest_review = next((node for node in reversed(run["nodes"]) if node["node_id"] in {"world_review", "worldview_review", "novel_review"}), None)
            assert latest_review is not None, run
            modified_message = "请严格修复以下问题，并保持世界、世界观与小说规则一致：" + "; ".join(latest_review["output"].get("errors") or ["补全逻辑链"])
            run = assert_success(
                request_json(
                    "POST",
                    "/api/hierarchy-agent/respond",
                    json={
                        "run_id": run["run_id"],
                        "decision": "request_changes",
                        "message": modified_message,
                        "revision_mode": "partial_rewrite",
                        "manual_edit": True,
                        "payload": payload,
                    },
                )
            )["run"]
        assert any(node["node_id"] == "novel_review" and node["status"] == "completed" for node in run["nodes"]), run
        assert run["review_required"] is True, run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        world_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "world_review")
        worldview_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "worldview_review")
        novel_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "novel_review")
        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "outline_agent_initial_expansion", initial_expansion
        assert world_review["output"]["llm_call"]["llm_agent_name"] == "outline_world_rules_review_agent", world_review
        assert worldview_review["output"]["llm_call"]["llm_agent_name"] == "outline_worldview_rules_review_agent", worldview_review
        assert novel_review["output"]["llm_call"]["llm_agent_name"] == "outline_novel_rules_review_agent", novel_review

        modify_content = next(node for node in reversed(run["nodes"]) if node["node_id"] == "modify_content")
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "outline_agent_human_feedback_modify_content", modify_content
        assert len(
            {
                initial_expansion["output"]["llm_call"]["llm_agent_name"],
                world_review["output"]["llm_call"]["llm_agent_name"],
                worldview_review["output"]["llm_call"]["llm_agent_name"],
                novel_review["output"]["llm_call"]["llm_agent_name"],
                modify_content["output"]["llm_call"]["llm_agent_name"],
            }
        ) == 5, {
            "initial_expansion": initial_expansion,
            "world_review": world_review,
            "worldview_review": worldview_review,
            "novel_review": novel_review,
            "modify_content": modify_content,
        }
        assert run["status"] == "waiting_human", run

        approved = approve_agent(run["run_id"])
        outline_id = approved["commit_result"]["outline_id"]
        queried = get_outline(outline_id)
        assert queried is not None, approved
        assert queried["novel_id"] == novel_id, queried
        assert queried["world_id"] == world_id, queried
        assert queried["summary"] == approved["pending_payload"]["summary"], queried
    finally:
        cleanup_world(world_id)
