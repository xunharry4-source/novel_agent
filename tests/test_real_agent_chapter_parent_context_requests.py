import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import cleanup_world, make_full_chain, request_json, start_agent


def test_real_agent_chapter_parent_context_requests():
    chain = make_full_chain("chapter_parent_context")
    worldview_rule_id = f"worldview_rule_{chain['suffix']}"
    payload = {
        "world_id": chain["world_id"],
        "worldview_id": chain["worldview_id"],
        "novel_id": chain["novel_id"],
        "outline_id": chain["outline_id"],
        "name": f"Parent Context Chapter {chain['suffix']}",
        "content": (
            "林澈在北港灯塔复核离港登记簿，发现潮汐水晶刻度异常后，"
            "按照灯塔公会制度暂停离港信号，并准备追查异常来源。"
        ),
    }
    try:
        request_json(
            "POST",
            "/api/archive/update",
            json={
                "id": worldview_rule_id,
                "type": "worldview",
                "world_id": chain["world_id"],
                "worldview_id": chain["worldview_id"],
                "name": "灯塔公会制度",
                "category": "组织 > 灯塔公会",
                "path": "组织 > 灯塔公会",
                "content": "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。",
            },
        )

        run = start_agent("chapter", "create", payload, "真实测试章节创建必须遵守父级强约束")
        initial_expansion = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")
        worldview_review = next(node for node in run["nodes"] if node["node_id"] == "worldview_review")

        initial_prompt = initial_expansion["output"]["llm_call"]["prompt"]
        worldview_prompt = worldview_review["output"]["llm_call"]["prompt"]

        assert "【父级强约束】" in initial_prompt, initial_prompt
        assert "禁止凭空出现现代枪械" in initial_prompt, initial_prompt
        assert "主角必须依靠航图和登记簿破局" in initial_prompt, initial_prompt
        assert "第一章发现潮汐刻度异常" in initial_prompt, initial_prompt
        assert "灯塔公会制度" in initial_prompt, initial_prompt
        assert "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。" in initial_prompt, initial_prompt

        assert "以下是世界观 Canon 设定（必须优先遵守）" in worldview_prompt, worldview_prompt
        assert "灯塔公会制度" in worldview_prompt, worldview_prompt
        assert "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。" in worldview_prompt, worldview_prompt
    finally:
        cleanup_world(chain["world_id"])
