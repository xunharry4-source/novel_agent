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


def test_real_agent_chapter_content_create_flow_requests():
    chain = make_full_chain("agent_chapter_content")
    chapter_outline_id = f"chapter_outline_{chain['suffix']}"
    payload = {
        "world_id": chain["world_id"],
        "worldview_id": chain["worldview_id"],
        "novel_id": chain["novel_id"],
        "outline_id": chain["outline_id"],
        "chapter_outline_id": chapter_outline_id,
        "name": f"Agent Chapter Content {chain['suffix']}",
        "content": "林澈在北港灯塔封锁离港信号后，沿着章节大纲要求继续核对登记簿、追查外海群岛异常航线，并确认下一步调查目标。",
    }
    try:
        create_chapter(
            world_id=chain["world_id"],
            worldview_id=chain["worldview_id"],
            novel_id=chain["novel_id"],
            outline_id=chain["outline_id"],
            chapter_id=chapter_outline_id,
            name="父级章节大纲",
            content="章节大纲：林澈在封锁离港信号后核对登记簿，确认异常航线来自外海群岛，并锁定下一步调查目标。",
        )
        create_chapter(
            world_id=chain["world_id"],
            worldview_id=chain["worldview_id"],
            novel_id=chain["novel_id"],
            outline_id=chain["outline_id"],
            chapter_id=f"chapter_prev_{chain['suffix']}",
            name="前置章节",
            content="林澈在北港灯塔发现潮汐刻度异常，并决定封锁离港信号。",
        )

        run = start_agent("chapter", "create", payload, "真实测试章节内容 Agent 流程")
        assert run["review_required"] is True, run

        for _ in range(6):
            if run["status"] == "waiting_human" and any(node["node_id"] == "chapter_review" for node in run["nodes"]):
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
            modified_message = "请严格修复以下问题，并保持世界、世界观、小说、大纲、章节大纲和章节承接一致：" + "; ".join(
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

        assert any(node["node_id"] == "chapter_review" for node in run["nodes"]), run
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
                        "message": "请进一步强化章节承接，严格执行父级章节大纲，并保持世界、世界观、小说与分卷大纲一致。",
                        "revision_mode": "partial_rewrite",
                        "manual_edit": True,
                        "payload": payload,
                    },
                )
            )["run"]
        modify_content = next(node for node in reversed(modify_run["nodes"]) if node["node_id"] == "modify_content")
        initial_prompt = initial_expansion["output"]["llm_call"]["prompt"]
        modify_prompt = modify_content["output"]["llm_call"]["prompt"]
        modify_feedback = str(modify_content["input"].get("feedback") or "")

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
        assert "【父级强约束】" in initial_prompt, initial_prompt
        assert "禁止凭空出现现代枪械" in initial_prompt, initial_prompt
        assert "主角必须依靠航图和登记簿破局" in initial_prompt, initial_prompt
        assert "第一章发现潮汐刻度异常" in initial_prompt, initial_prompt
        assert "章节大纲：林澈在封锁离港信号后核对登记簿" in initial_prompt, initial_prompt
        assert "【父级强约束】" in modify_prompt, modify_prompt
        assert "章节大纲：林澈在封锁离港信号后核对登记簿" in modify_prompt, modify_prompt
        assert modify_feedback in modify_prompt, {"feedback": modify_feedback, "prompt": modify_prompt}
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

        approved = approve_agent(modify_run["run_id"])
        content_id = approved["commit_result"]["id"]
        queried = get_chapter(content_id, outline_id=chain["outline_id"], worldview_id=chain["worldview_id"])
        assert queried is not None, approved
        assert queried["outline_id"] == chain["outline_id"], queried
        assert queried["novel_id"] == chain["novel_id"], queried
        assert queried["chapter_outline_id"] == chapter_outline_id, queried
        assert queried["content"] == approved["pending_payload"]["content"], queried
    finally:
        cleanup_world(chain["world_id"])
