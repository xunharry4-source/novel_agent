import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    approve_agent,
    assert_success,
    cleanup_world,
    create_chapter,
    get_chapter,
    make_full_chain,
    request_json,
    start_agent,
)


def test_real_agent_chapter_flow_requests():
    chain = make_full_chain("agent_chapter")
    payload = {
        "world_id": chain["world_id"],
        "worldview_id": chain["worldview_id"],
        "novel_id": chain["novel_id"],
        "outline_id": chain["outline_id"],
        "name": f"Agent Chapter {chain['suffix']}",
        "content": "北港灯塔的蒸汽钟再次鸣响后，林澈按灯塔公会登记簿复核潮汐水晶刻度，发现异常从外海群岛航线延伸到内港。他记录异常编号，维持禁航信号，并准备沿登记簿追查潮汐刻度异常的来源。",
    }
    try:
        create_chapter(
            world_id=chain["world_id"],
            worldview_id=chain["worldview_id"],
            novel_id=chain["novel_id"],
            outline_id=chain["outline_id"],
            chapter_id=f"chapter_prev_{chain['suffix']}",
            name="前置章节",
            content="林澈在北港灯塔发现潮汐刻度异常，并决定封锁离港信号。",
        )
        run = start_agent(
            "chapter",
            "create",
            payload,
            "真实测试章节 Agent 流程",
        )
        assert run["review_required"] is True, run
        for _ in range(6):
            if any(node["node_id"] == "chapter_review" and node["status"] == "completed" for node in run["nodes"]):
                break
            latest_review = next(
                (
                    node
                    for node in reversed(run["nodes"])
                    if node["node_id"] in {"world_review", "worldview_review", "novel_review", "outline_review", "chapter_review"}
                ),
                None,
            )
            assert latest_review is not None, run
            modified_message = "请严格修复以下问题，并保持世界、世界观、小说、大纲和章节承接一致：" + "; ".join(
                latest_review["output"].get("errors") or ["补全逻辑链"]
            )
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

        assert any(node["node_id"] == "chapter_review" and node["status"] == "completed" for node in run["nodes"]), run
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        world_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "world_review")
        worldview_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "worldview_review")
        novel_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "novel_review")
        outline_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "outline_review")
        chapter_review = next(node for node in reversed(run["nodes"]) if node["node_id"] == "chapter_review")

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
                        "message": "请进一步强化章节承接、保留灯塔封港主线，并保持世界、世界观、小说与大纲一致。",
                        "revision_mode": "partial_rewrite",
                        "manual_edit": True,
                        "payload": payload,
                    },
                )
            )["run"]
        modify_content = next(node for node in reversed(modify_run["nodes"]) if node["node_id"] == "modify_content")

        assert initial_expansion["output"]["llm_call"]["llm_agent_name"] == "chapter_agent_initial_expansion", initial_expansion
        assert world_review["output"]["reviewer"] == "chapter_world_rules_review_agent", world_review
        assert world_review["output"]["llm_call"]["llm_agent_name"] == "chapter_world_rules_review_agent", world_review
        assert worldview_review["output"]["reviewer"] == "chapter_worldview_rules_review_agent", worldview_review
        assert worldview_review["output"]["llm_call"]["llm_agent_name"] == "chapter_worldview_rules_review_agent", worldview_review
        assert novel_review["output"]["reviewer"] == "chapter_novel_rules_review_agent", novel_review
        assert novel_review["output"]["llm_call"]["llm_agent_name"] == "chapter_novel_rules_review_agent", novel_review
        assert outline_review["output"]["reviewer"] == "chapter_outline_rules_review_agent", outline_review
        assert outline_review["output"]["llm_call"]["llm_agent_name"] == "chapter_outline_rules_review_agent", outline_review
        assert chapter_review["output"]["reviewer"] == "chapter_consistency_review_agent", chapter_review
        assert chapter_review["output"]["llm_call"]["llm_agent_name"] == "chapter_consistency_review_agent", chapter_review
        assert modify_content["output"]["llm_call"]["llm_agent_name"] == "chapter_agent_human_feedback_modify_content", modify_content
        assert len(
            {
                initial_expansion["output"]["llm_call"]["llm_agent_name"],
                world_review["output"]["llm_call"]["llm_agent_name"],
                worldview_review["output"]["llm_call"]["llm_agent_name"],
                novel_review["output"]["llm_call"]["llm_agent_name"],
                outline_review["output"]["llm_call"]["llm_agent_name"],
                chapter_review["output"]["llm_call"]["llm_agent_name"],
                modify_content["output"]["llm_call"]["llm_agent_name"],
            }
        ) == 7, {
            "initial_expansion": initial_expansion,
            "world_review": world_review,
            "worldview_review": worldview_review,
            "novel_review": novel_review,
            "outline_review": outline_review,
            "chapter_review": chapter_review,
            "modify_content": modify_content,
        }
        assert modify_run["status"] == "waiting_human", modify_run

        approved = approve_agent(run["run_id"])
        chapter_id = approved["commit_result"]["id"]
        queried = get_chapter(chapter_id, outline_id=chain["outline_id"], worldview_id=chain["worldview_id"])
        assert queried is not None, approved
        assert queried["outline_id"] == chain["outline_id"], queried
        assert queried["novel_id"] == chain["novel_id"], queried
        assert queried["content"] == approved["pending_payload"]["content"], queried
    finally:
        cleanup_world(chain["world_id"])
