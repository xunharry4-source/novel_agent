import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import assert_success, request_json, start_agent, unique_suffix


def test_real_outline_summary_request_changes_uses_summary_rewrite_prompt_guidance():
    suffix = unique_suffix("outline_summary_mode")
    run = start_agent(
        "outline_summary_create",
        "create",
        {
            "world_id": "world_test",
            "worldview_id": "wv_test",
            "novel_id": "novel_test",
            "outline_id": f"outline_{suffix}",
            "name": f"测试分卷 {suffix}",
            "summary": (
                "第一卷：北港疑云\n"
                "开篇：灯塔误报引发封港。\n"
                "主要矛盾：登记簿失真与航线造假交织。\n"
                "主要事件：林澈复核登记簿、追查外海群岛异常航线。\n"
                "高潮：禁航令引发港口势力冲突。\n"
                "结局：确定幕后势力与下一步追查方向。"
            ),
        },
        "请生成新增分卷大纲总结",
    )
    modified = assert_success(
        request_json(
            "POST",
            "/api/hierarchy-agent/respond",
            json={
                "run_id": run["run_id"],
                "decision": "request_changes",
                "message": "请保留主冲突、高潮和结局，重做摘要。",
                "revision_mode": "summary_rewrite",
                "manual_edit": True,
                "payload": {
                    "summary": (
                        "第一卷：北港疑云\n"
                        "开篇：灯塔误报引发封港。\n"
                        "主要矛盾：登记簿失真与航线造假交织。\n"
                        "主要事件：林澈复核登记簿、追查外海群岛异常航线。\n"
                        "高潮：禁航令引发港口势力冲突。\n"
                        "结局：确定幕后势力与下一步追查方向。"
                    ),
                },
            },
        )
    )["run"]
    modify_node = next(node for node in reversed(modified["nodes"]) if node["node_id"] == "modify_content")
    prompt = modify_node["output"]["llm_call"]["prompt"]

    assert "当前模式：summary_rewrite（摘要重做）" in prompt, prompt
    assert "只允许重写 payload.downstream_summary。" in prompt, prompt
    assert "必须保留 `payload.summary` 原文" in prompt, prompt
