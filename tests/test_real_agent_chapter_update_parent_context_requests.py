import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import assert_success, cleanup_world, create_chapter, make_full_chain, request_json, start_agent


def test_real_agent_chapter_update_parent_context_requests():
    chain = make_full_chain("chapter_update_parent_context")
    chapter_id = f"chapter_target_{chain['suffix']}"
    try:
        request_json(
            "POST",
            "/api/archive/update",
            json={
                "id": f"worldview_rule_{chain['suffix']}",
                "type": "worldview",
                "world_id": chain["world_id"],
                "worldview_id": chain["worldview_id"],
                "name": "灯塔公会制度",
                "category": "组织 > 灯塔公会",
                "path": "组织 > 灯塔公会",
                "content": "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。",
            },
        )
        create_chapter(
            world_id=chain["world_id"],
            worldview_id=chain["worldview_id"],
            novel_id=chain["novel_id"],
            outline_id=chain["outline_id"],
            chapter_id=chapter_id,
            name="原始章节大纲",
            content="林澈在北港灯塔封锁离港信号后，开始核对潮汐刻度异常。",
        )

        run = start_agent(
            "chapter",
            "update",
            {
                "target_id": chapter_id,
                "world_id": chain["world_id"],
                "worldview_id": chain["worldview_id"],
                "novel_id": chain["novel_id"],
                "outline_id": chain["outline_id"],
                "name": "修订后的章节大纲",
                "content": "林澈在北港灯塔复核登记簿、追踪外海群岛异常航线，并在维持禁航信号的同时确认下一步调查目标。",
            },
            "真实测试章节修改必须遵守父级强约束",
        )
        initial_prompt = next(node for node in run["nodes"] if node["node_id"] == "initial_expansion")["output"]["llm_call"]["prompt"]
        assert "【父级强约束】" in initial_prompt, initial_prompt
        assert "禁止凭空出现现代枪械" in initial_prompt, initial_prompt
        assert "主角必须依靠航图和登记簿破局" in initial_prompt, initial_prompt
        assert "第一章发现潮汐刻度异常" in initial_prompt, initial_prompt
        assert "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。" in initial_prompt, initial_prompt

        run = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/respond",
                json={
                    "run_id": run["run_id"],
                    "decision": "request_changes",
                    "message": "请只按父级分卷大纲和灯塔公会制度修订，不要扩展到未批准剧情。",
                    "revision_mode": "partial_rewrite",
                    "manual_edit": True,
                    "payload": {
                        "target_id": chapter_id,
                        "world_id": chain["world_id"],
                        "worldview_id": chain["worldview_id"],
                        "novel_id": chain["novel_id"],
                        "outline_id": chain["outline_id"],
                        "name": "修订后的章节大纲",
                        "content": "林澈在北港灯塔复核登记簿、追踪外海群岛异常航线，并在维持禁航信号的同时确认下一步调查目标。",
                    },
                },
            )
        )["run"]
        modify_prompt = next(node for node in reversed(run["nodes"]) if node["node_id"] == "modify_content")["output"]["llm_call"]["prompt"]
        assert "【父级强约束】" in modify_prompt, modify_prompt
        assert "禁止凭空出现现代枪械" in modify_prompt, modify_prompt
        assert "主角必须依靠航图和登记簿破局" in modify_prompt, modify_prompt
        assert "第一章发现潮汐刻度异常" in modify_prompt, modify_prompt
        assert "所有离港船只必须登记潮汐水晶刻度，未登记不得离港。" in modify_prompt, modify_prompt
        assert "请只按父级分卷大纲和灯塔公会制度修订，不要扩展到未批准剧情。" in modify_prompt, modify_prompt
    finally:
        cleanup_world(chain["world_id"])
