import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import (
    assert_success,
    cleanup_world,
    create_chapter,
    make_full_chain,
    request_json,
)


def test_real_chapter_check_workflow_requests():
    chain = make_full_chain("chapter_check")
    chapter_id = f"chapter_outline_{chain['suffix']}"
    chapter_outline = (
        "第一节：林澈在北港灯塔复核潮汐刻度，确认登记簿与水晶余辉不一致；"
        "第二节：他依据公会规则暂时封锁离港信号；"
        "第三节：他决定连夜核查登记簿来源。"
    )
    direct_content = (
        "北港的雾贴在灯塔玻璃上。林澈翻开登记簿，对照潮汐刻度与水晶余辉，"
        "确认两者不一致后，依照灯塔公会规则暂停离港信号，并准备彻查这份登记。"
    )
    try:
        create_chapter(
            world_id=chain["world_id"],
            worldview_id=chain["worldview_id"],
            novel_id=chain["novel_id"],
            outline_id=chain["outline_id"],
            chapter_id=chapter_id,
            name="第一章：灯塔复核",
            content=chapter_outline,
        )
        payload = {
            "world_id": chain["world_id"],
            "worldview_id": chain["worldview_id"],
            "novel_id": chain["novel_id"],
            "outline_id": chain["outline_id"],
            "target_id": chapter_id,
            "chapter_outline_id": chapter_id,
            "name": "第一章：灯塔复核",
            "chapter_outline": chapter_outline,
            "content": direct_content,
        }
        data = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/start",
                json={
                    "agent_type": "chapter",
                    "action": "check",
                    "payload": payload,
                    "message": "真实测试章节直接内容检查工作流",
                },
            )
        )
        run = data["run"]
        assert run["agent_type"] == "chapter", run
        assert run["action"] == "check", run
        assert run["status"] == "completed", run
        assert run["current_node"] == "output", run
        assert run["committed"] is False, run
        assert isinstance(run.get("check_passed"), bool), run
        assert isinstance(run.get("failed_checks"), list), run
        assert isinstance(run.get("check_results"), list), run

        node_ids = [node["node_id"] for node in run["nodes"]]
        assert node_ids == [
            "input",
            "world_review",
            "novel_review",
            "worldview_review",
            "outline_review",
            "chapter_outline_review",
            "plot_review",
            "output",
        ], run

        output_node = next(node for node in run["nodes"] if node["node_id"] == "output")
        assert isinstance(output_node["output"].get("overall_passed"), bool), output_node
        assert isinstance(output_node["output"].get("failed_checks"), list), output_node
        assert isinstance(output_node["output"].get("check_results"), list), output_node

        fetched = assert_success(
            request_json("GET", "/api/hierarchy-agent/get", params={"run_id": run["run_id"]})
        )["run"]
        chapter_outline_review = next(node for node in fetched["nodes"] if node["node_id"] == "chapter_outline_review")
        plot_review = next(node for node in fetched["nodes"] if node["node_id"] == "plot_review")
        assert chapter_outline_review["output"].get("reviewer") == "chapter_chapter_outline_review_agent", chapter_outline_review
        assert plot_review["output"].get("reviewer") == "chapter_plot_review_agent", plot_review
        assert chapter_outline_review["output"].get("llm_invoked") is True, chapter_outline_review
        assert plot_review["output"].get("llm_invoked") is True, plot_review
        assert "章节大纲" in chapter_outline_review["output"]["llm_call"]["prompt"], chapter_outline_review
        assert "剧情错误" in plot_review["output"]["llm_call"]["prompt"], plot_review
    finally:
        cleanup_world(chain["world_id"])


def test_real_chapter_check_workflow_requires_chapter_outline():
    chain = make_full_chain("chapter_check_missing_outline")
    try:
        payload = {
            "world_id": chain["world_id"],
            "worldview_id": chain["worldview_id"],
            "novel_id": chain["novel_id"],
            "outline_id": chain["outline_id"],
            "content": "林澈直接封锁离港信号。",
        }
        data = request_json(
            "POST",
            "/api/hierarchy-agent/start",
            expected_status=400,
            json={
                "agent_type": "chapter",
                "action": "check",
                "payload": payload,
                "message": "缺少章节大纲时必须拒绝启动检查工作流",
            },
        )
        assert data["status"] == "error", data
        assert "章节大纲" in data["error"], data
    finally:
        cleanup_world(chain["world_id"])
