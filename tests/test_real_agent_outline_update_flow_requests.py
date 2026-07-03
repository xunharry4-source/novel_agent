import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_novel,
    create_outline,
    create_world,
    create_worldview,
    get_outline,
    request_json,
    start_agent,
    unique_suffix,
)


def test_real_agent_outline_update_flow_requests():
    suffix = unique_suffix("agent_outline_update")
    world_id = f"world_{suffix}"
    novel_id = f"novel_{suffix}"
    outline_id = f"outline_{suffix}"
    original_summary = "调查员在港口账册中发现异常条目。"
    updated_name = f"Agent Outline Updated {suffix}"
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
        create_outline(
            world_id=world_id,
            worldview_id=worldview_id,
            novel_id=novel_id,
            outline_id=outline_id,
            name=f"Outline Base {suffix}",
            summary=original_summary,
        )

        payload = {
            "target_id": outline_id,
            "world_id": world_id,
            "worldview_id": worldview_id,
            "novel_id": novel_id,
            "name": updated_name,
            "summary": "调查员发现港口事故记录与潮汐能源账册不一致。",
        }
        run = start_agent("outline", "update", payload, "真实测试分卷大纲 Agent 修改流程")

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
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        world_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "world_review")
        worldview_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "worldview_review")
        novel_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "novel_review")
        if any(node["node_id"] == "modify_content" for node in run["nodes"]):
            modify_run = run
        else:
            modify_run = assert_success(
                request_json(
                    "POST",
                    "/api/hierarchy-agent/respond",
                    json={
                        "run_id": run["run_id"],
                        "decision": "request_changes",
                        "message": "请进一步强化卷内主要矛盾与高潮，同时保持世界、世界观和小说规则一致。",
                        "revision_mode": "partial_rewrite",
                        "manual_edit": True,
                        "payload": payload,
                    },
                )
            )["run"]
        modify_content = next(node for node in reversed(modify_run["nodes"]) if node["node_id"] == "modify_content")

        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "outline_agent_initial_expansion", initial_expansion
        assert world_review["output"]["llm_call"]["llm_agent_name"] == "outline_world_rules_review_agent", world_review
        assert worldview_review["output"]["llm_call"]["llm_agent_name"] == "outline_worldview_rules_review_agent", worldview_review
        assert novel_review["output"]["llm_call"]["llm_agent_name"] == "outline_novel_rules_review_agent", novel_review
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
        assert modify_run["status"] == "waiting_human", modify_run

        approved = approve_agent(run["run_id"])
        queried = get_outline(outline_id)
        assert queried is not None, approved
        assert queried["novel_id"] == novel_id, queried
        assert queried["world_id"] == world_id, queried
        assert queried["title"] == updated_name, queried
        assert queried["summary"] == approved["pending_payload"]["summary"], queried
        assert queried["summary"] != original_summary, queried
    finally:
        cleanup_world(world_id)
