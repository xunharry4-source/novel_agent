import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import (  # noqa: E402
    chapter_content_summary_create_agent,
    chapter_content_summary_update_agent,
    chapter_outline_summary_create_agent,
    chapter_outline_summary_update_agent,
    outline_summary_create_agent,
    outline_summary_update_agent,
)
from src import app_api  # noqa: E402


class FakeSummaryLLM:
    def __init__(self, agent_name: str, calls: list[dict]):
        self.agent_name = agent_name
        self.calls = calls

    def invoke(self, prompt, config=None):
        self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
        source_field = "summary" if "outline_summary" in self.agent_name else "content"
        source_text = "第一卷：失踪者\n开篇：运输舰坠落。\n主要事件：三方分裂。\n高潮：营地暴乱。"
        payload = {
            "world_id": "world_test",
            "worldview_id": "wv_test",
            "novel_id": "novel_test",
            "outline_id": "outline_test",
            "chapter_id": "chapter_test",
            "target_id": "target_test",
            "name": "测试对象",
            "downstream_summary": f"{self.agent_name} 下游摘要：保留核心冲突、关键事件和结果。",
        }
        if "chapter_content_summary" not in self.agent_name:
            payload[source_field] = source_text
        if "chapter_content_summary" in self.agent_name:
            payload["chapter_intro"] = "林澈在坠机后迅速稳住局面，但营地权力冲突开始浮出水面。"
            payload["chapter_summary"] = (
                "运输舰坠落后，林澈先封锁航道并组织伤员转移，随后与委员会争抢药品和电池。"
                "救援迟迟未到，营地内部围绕配给与指挥权迅速分裂，为后续暴乱埋下导火索。"
            )
            payload["downstream_summary"] = (
                "章节简介：林澈在坠机后迅速稳住局面，但营地权力冲突开始浮出水面。\n\n"
                "章节总结：\n运输舰坠落后，林澈先封锁航道并组织伤员转移，随后与委员会争抢药品和电池。"
                "救援迟迟未到，营地内部围绕配给与指挥权迅速分裂，为后续暴乱埋下导火索。"
            )
        body = {
            "metadata": {"agent": self.agent_name, "node": "initial_expansion", "entity_type": "summary", "action": "create"},
            "payload": payload,
            "expanded_input": {
                f"{source_field}_seed": source_text,
                "summary_goal": "保留核心推进链",
                "must_keep": ["核心冲突", "关键事件", "结果"],
            },
            "expansion_notes": f"{self.agent_name} expansion notes",
            "modification_notes": f"{self.agent_name} modification notes",
            "change_summary": f"{self.agent_name} change summary",
        }
        return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))


def test_summary_workflow_modules_use_dedicated_llm_names():
    cases = [
        (
            outline_summary_create_agent,
            outline_summary_create_agent.INITIAL_EXPANSION_AGENT_NAME,
            outline_summary_create_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试分卷", "summary": "第一卷：失踪者\n主要事件：坠落与分裂。"},
            "summary",
        ),
        (
            outline_summary_update_agent,
            outline_summary_update_agent.INITIAL_EXPANSION_AGENT_NAME,
            outline_summary_update_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "target_id": "outline_test", "name": "测试分卷", "summary": "第一卷：失踪者\n主要事件：坠落与分裂。"},
            "summary",
        ),
        (
            chapter_outline_summary_create_agent,
            chapter_outline_summary_create_agent.INITIAL_EXPANSION_AGENT_NAME,
            chapter_outline_summary_create_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节大纲", "content": "开篇：坠落。\n主要事件：配给冲突。"},
            "content",
        ),
        (
            chapter_outline_summary_update_agent,
            chapter_outline_summary_update_agent.INITIAL_EXPANSION_AGENT_NAME,
            chapter_outline_summary_update_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "target_id": "chapter_test", "name": "测试章节大纲", "content": "开篇：坠落。\n主要事件：配给冲突。"},
            "content",
        ),
        (
            chapter_content_summary_create_agent,
            chapter_content_summary_create_agent.INITIAL_EXPANSION_AGENT_NAME,
            chapter_content_summary_create_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节正文", "content": "运输舰坠落后，林澈先封锁航道，再组织伤员转移。"},
            "content",
        ),
        (
            chapter_content_summary_update_agent,
            chapter_content_summary_update_agent.INITIAL_EXPANSION_AGENT_NAME,
            chapter_content_summary_update_agent.MODIFY_CONTENT_AGENT_NAME,
            {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "target_id": "chapter_test", "name": "测试章节正文", "content": "运输舰坠落后，林澈先封锁航道，再组织伤员转移。"},
            "content",
        ),
    ]

    for module, initial_llm_name, modify_llm_name, payload, source_field in cases:
        calls = []

        def fake_get_llm(json_mode=False, agent_name="unknown"):
            assert json_mode is True
            return FakeSummaryLLM(agent_name, calls)

        with (
            patch.object(module, "get_llm", fake_get_llm),
            patch.object(module, "get_langfuse_callback", lambda: None),
        ):
            initial = module.generate_initial_expansion("create", payload, "请生成下游摘要")
            assert calls[-1]["agent_name"] == initial_llm_name
            assert initial["payload"][source_field] == payload[source_field]
            assert initial["payload"]["downstream_summary"]
            if "chapter_content_summary" in initial_llm_name:
                assert initial["payload"]["chapter_intro"]
                assert initial["payload"]["chapter_summary"]

            modified = module.generate_content_modification("update", payload, "请按意见重做摘要", revision_mode="summary_rewrite", feedback="保留更多冲突链")
            assert calls[-1]["agent_name"] == modify_llm_name
            assert modified["payload"][source_field] == payload[source_field]
            assert modified["payload"]["downstream_summary"]
            if "chapter_content_summary" in modify_llm_name:
                assert modified["payload"]["chapter_intro"]
                assert modified["payload"]["chapter_summary"]


def test_summary_manual_edit_uses_human_feedback_llm_names():
    cases = [
        (
            outline_summary_create_agent,
            outline_summary_create_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "create",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试分卷", "summary": "第一卷：失踪者\n主要事件：坠落与分裂。"},
                "feedback": "用户要求保留高潮",
                "review_feedback": "审查要求补足结构",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
        (
            outline_summary_update_agent,
            outline_summary_update_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "update",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "target_id": "outline_test", "name": "测试分卷", "summary": "第一卷：失踪者\n主要事件：坠落与分裂。"},
                "feedback": "用户要求保留结局",
                "review_feedback": "审查要求补足结构",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
        (
            chapter_outline_summary_create_agent,
            chapter_outline_summary_create_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "create",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节大纲", "content": "开篇：坠落。\n主要事件：配给冲突。"},
                "feedback": "用户要求保留冲突升级",
                "review_feedback": "审查要求补足结构",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
        (
            chapter_outline_summary_update_agent,
            chapter_outline_summary_update_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "update",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "target_id": "chapter_test", "name": "测试章节大纲", "content": "开篇：坠落。\n主要事件：配给冲突。"},
                "feedback": "用户要求保留高潮",
                "review_feedback": "审查要求补足结构",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
        (
            chapter_content_summary_create_agent,
            chapter_content_summary_create_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "create",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节正文", "content": "运输舰坠落后，林澈先封锁航道，再组织伤员转移。"},
                "feedback": "用户要求保留结果",
                "review_feedback": "审查要求补足因果",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
        (
            chapter_content_summary_update_agent,
            chapter_content_summary_update_agent.HUMAN_FEEDBACK_AGENT_NAME,
            {
                "action": "update",
                "message": "请返工摘要",
                "pending_payload": {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "target_id": "chapter_test", "name": "测试章节正文", "content": "运输舰坠落后，林澈先封锁航道，再组织伤员转移。"},
                "feedback": "用户要求保留后续影响",
                "review_feedback": "审查要求补足因果",
                "manual_edit": True,
                "revision_mode": "summary_rewrite",
                "nodes": [],
                "iterations": 1,
            },
        ),
    ]

    for module, expected_agent_name, state in cases:
        calls = []

        def fake_get_llm(json_mode=False, agent_name="unknown"):
            assert json_mode is True
            return FakeSummaryLLM(agent_name, calls)

        with (
            patch.object(module, "get_llm", fake_get_llm),
            patch.object(module, "get_langfuse_callback", lambda: None),
        ):
            result = module.modify_content_node(state)

        assert calls[-1]["agent_name"] == expected_agent_name
        assert result["nodes"][-1]["input"]["feedback"] == state["feedback"]
        assert result["nodes"][-1]["input"]["manual_edit"] is True
        assert result["nodes"][-1]["output"]["llm_call"]["llm_agent_name"] == expected_agent_name


def test_summary_review_nodes_block_oversimplified_output():
    state = {
        "pending_payload": {
            "name": "测试对象",
            "summary": "第一卷：失踪者\n开篇：运输舰坠落。\n主要人物：林澈、议员艾岚。\n主要矛盾：旧秩序是否继续有效。\n主要事件：等待救援失败后营地爆发分裂。\n高潮：暴乱导致三方彻底分家。\n结局：发现余烬痕迹。",
            "downstream_summary": "坠毁，分裂。",
        },
        "nodes": [],
    }
    result = outline_summary_create_agent.review_node(state)
    assert result["review_passed"] is False
    assert result["current_node"] == "modify_content"
    assert result["review_errors"]

    chapter_state = {
        "pending_payload": {
            "name": "测试章节",
            "content": ("运输舰坠落后，林澈先封锁航道，再组织伤员转移，并和委员会争抢电池与药品。\n" * 20),
            "downstream_summary": "坠毁。",
        },
        "nodes": [],
    }
    result = chapter_content_summary_create_agent.review_node(chapter_state)
    assert result["review_passed"] is False
    assert result["current_node"] == "modify_content"
    assert result["review_errors"]


def test_chapter_intro_summary_review_accepts_dense_single_paragraph_summary():
    chapter_state = {
        "pending_payload": {
            "name": "测试章节",
            "content": (
                "序幕：星海尽头\n\n"
                "天际号载着两万多名乘客前往苍穹星，船上秩序井然，主角一家讨论抵达后的安排。"
                "与此同时，深层导航系统闪过一丝极其微弱却未触发警报的异常数据波动，预示飞船即将失踪。\n\n"
            )
            * 18,
            "chapter_intro": "天际号载着两万多名乘客照常启航，在看似宁静的旅途中，一次未被察觉的导航异常预示着整艘船即将失踪。",
            "chapter_summary": (
                "星海历3842年，天际号搭载两万多名乘客前往苍穹星。船上乘客沉浸在帝国和平繁荣的日常中，"
                "主角一家也只把这次航行视为普通旅程。观景大厅里孩子嬉闹、老人闲谈、商人交换市场消息，"
                "所有人都默认这只是一次普通的跨星系民航航行。与此同时，飞船深层导航系统闪过一丝极其微弱且未触发警报的异常波动，"
                "既没有引起船员注意，也没有被乘客察觉。天际号继续驶向目的地，但几小时后它将从帝国航道记录中彻底消失，"
                "这场看似平静的起航也会因此成为第七殖民星域历史上最著名的失踪事件序幕，船上所有人的命运会被就此改写。"
            ),
            "downstream_summary": (
                "章节简介：天际号载着两万多名乘客照常启航，在看似宁静的旅途中，一次未被察觉的导航异常预示着整艘船即将失踪。\n\n"
                "章节总结：\n星海历3842年，天际号搭载两万多名乘客前往苍穹星。船上乘客沉浸在帝国和平繁荣的日常中，"
                "主角一家也只把这次航行视为普通旅程。观景大厅里孩子嬉闹、老人闲谈、商人交换市场消息，"
                "所有人都默认这只是一次普通的跨星系民航航行。与此同时，飞船深层导航系统闪过一丝极其微弱且未触发警报的异常波动，"
                "既没有引起船员注意，也没有被乘客察觉。天际号继续驶向目的地，但几小时后它将从帝国航道记录中彻底消失，"
                "这场看似平静的起航也会因此成为第七殖民星域历史上最著名的失踪事件序幕，船上所有人的命运会被就此改写。"
            ),
        },
        "nodes": [],
    }

    result = chapter_content_summary_create_agent.review_node(chapter_state)
    assert result["review_passed"] is True
    assert result["current_node"] == "human"
    assert result["review_errors"] == []


def test_summary_workflow_prompts_are_separate_and_preserve_source_text():
    outline_create_prompt = outline_summary_create_agent.build_initial_expansion_prompt(
        "create",
        {"novel_id": "novel_test", "outline_id": "outline_test", "name": "测试分卷", "summary": "原始大纲"},
        "请总结",
        revision_mode=None,
        feedback="",
    )
    outline_update_prompt = outline_summary_update_agent.build_modification_prompt(
        "update",
        {"novel_id": "novel_test", "outline_id": "outline_test", "target_id": "outline_test", "name": "测试分卷", "summary": "原始大纲"},
        "请返工总结",
        revision_mode="summary_rewrite",
        feedback="保留转折",
        expansion_error="摘要过短",
    )
    chapter_outline_create_prompt = chapter_outline_summary_create_agent.build_initial_expansion_prompt(
        "create",
        {"outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节大纲", "content": "原始章节大纲"},
        "请总结",
        revision_mode=None,
        feedback="",
    )
    chapter_content_update_prompt = chapter_content_summary_update_agent.build_modification_prompt(
        "update",
        {"outline_id": "outline_test", "chapter_id": "chapter_test", "target_id": "chapter_test", "name": "测试章节正文", "content": "原始章节正文"},
        "请返工总结",
        revision_mode="summary_rewrite",
        feedback="保留结果",
        expansion_error="缺少结果",
    )

    assert "分卷大纲下游摘要编辑" in outline_create_prompt
    assert "payload.summary 必须保留输入原文" in outline_create_prompt
    assert "修改分卷大纲摘要返工编辑" in outline_update_prompt
    assert "当前模式：summary_rewrite（摘要重做）" in outline_update_prompt
    assert "payload.summary 保留原文" in outline_update_prompt
    assert "章节大纲下游摘要编辑" in chapter_outline_create_prompt
    assert "payload.content 保留输入原文" in chapter_outline_create_prompt
    assert "修改章节简介与总结返工编辑" in chapter_content_update_prompt
    assert "当前模式：summary_rewrite（摘要重做）" in chapter_content_update_prompt
    assert "payload 里不要返回 content" in chapter_content_update_prompt


def test_chapter_intro_summary_workflow_auto_retries_after_failed_review():
    calls = []

    class RetryThenPassLLM:
        def __init__(self, agent_name: str, calls: list[dict]):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            is_modify = "modify" in self.agent_name
            body = {
                "metadata": {"agent": self.agent_name, "node": "modify_content" if is_modify else "initial_expansion", "entity_type": "summary", "action": "create"},
                "payload": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "chapter_id": "chapter_test",
                    "target_id": "chapter_test",
                    "name": "测试章节正文",
                    "chapter_intro": "坠毁。",
                    "chapter_summary": "坠毁。",
                    "downstream_summary": "章节简介：坠毁。\n\n章节总结：\n坠毁。",
                },
            }
            if is_modify:
                body["payload"]["chapter_intro"] = "林澈在坠毁后接管现场，但营地权力冲突迅速浮现。"
                body["payload"]["chapter_summary"] = (
                    "运输舰坠毁后，林澈先封锁危险舱段并组织伤员转移，随后与委员会争抢药品和能源。"
                    "等待救援失败后，营地围绕配给和指挥权发生分裂，后续暴乱风险被彻底点燃。"
                )
                body["payload"]["downstream_summary"] = (
                    "章节简介：林澈在坠毁后接管现场，但营地权力冲突迅速浮现。\n\n"
                    "章节总结：\n运输舰坠毁后，林澈先封锁危险舱段并组织伤员转移，随后与委员会争抢药品和能源。"
                    "等待救援失败后，营地围绕配给和指挥权发生分裂，后续暴乱风险被彻底点燃。"
                )
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return RetryThenPassLLM(agent_name, calls)

    with (
        patch.object(chapter_content_summary_create_agent, "get_llm", fake_get_llm),
        patch.object(chapter_content_summary_create_agent, "get_langfuse_callback", lambda: None),
    ):
        state = app_api._run_until_human(
            "chapter_intro_summary_create",
            "create",
            {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "name": "测试章节正文",
                "content": ("运输舰坠毁后，林澈先封锁危险舱段，再组织伤员转移，并与委员会争抢药品和能源。\n" * 20),
            },
            "请生成章节简介与总结",
        )

    node_ids = [node["node_id"] for node in state["nodes"]]
    assert state["status"] == "waiting_human"
    assert state["current_node"] == "human"
    assert state["auto_retry_count"] == 1
    assert "revision_retry_1" in node_ids
    assert "review_retry_1" in node_ids
    assert calls[-1]["agent_name"] == chapter_content_summary_create_agent.MODIFY_CONTENT_AGENT_NAME
    assert state["pending_payload"]["content"].startswith("运输舰坠毁后")


def test_chapter_intro_summary_workflow_stops_after_retry_limit():
    calls = []

    class AlwaysFailLLM:
        def __init__(self, agent_name: str, calls: list[dict]):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {"agent": self.agent_name, "node": "modify_content" if "modify" in self.agent_name else "initial_expansion", "entity_type": "summary", "action": "create"},
                "payload": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "chapter_id": "chapter_test",
                    "target_id": "chapter_test",
                    "name": "测试章节正文",
                    "chapter_intro": "太短。",
                    "chapter_summary": "太短。",
                    "downstream_summary": "章节简介：太短。\n\n章节总结：\n太短。",
                },
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return AlwaysFailLLM(agent_name, calls)

    with (
        patch.object(chapter_content_summary_create_agent, "get_llm", fake_get_llm),
        patch.object(chapter_content_summary_create_agent, "get_langfuse_callback", lambda: None),
    ):
        state = app_api._run_until_human(
            "chapter_intro_summary_create",
            "create",
            {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "name": "测试章节正文",
                "content": ("运输舰坠毁后，林澈先封锁危险舱段，再组织伤员转移，并与委员会争抢药品和能源。\n" * 20),
            },
            "请生成章节简介与总结",
        )

    node_ids = [node["node_id"] for node in state["nodes"]]
    assert state["status"] == "review_failed"
    assert state["current_node"] == "modify_content"
    assert state["auto_retry_count"] == app_api.SUMMARY_AUTO_RETRY_LIMIT
    assert "revision_retry_1" in node_ids
    assert "review_retry_1" in node_ids
    assert f"revision_retry_{app_api.SUMMARY_AUTO_RETRY_LIMIT}" in node_ids
    assert f"review_retry_{app_api.SUMMARY_AUTO_RETRY_LIMIT}" in node_ids


def test_app_api_registers_all_summary_workflow_types():
    expected_types = {
        "outline_summary_create",
        "outline_summary_update",
        "chapter_outline_summary_create",
        "chapter_outline_summary_update",
        "chapter_intro_summary_create",
        "chapter_intro_summary_update",
        "chapter_content_summary_create",
        "chapter_content_summary_update",
    }
    assert expected_types.issubset(set(app_api.AGENT_MODULES.keys()))
    for agent_type in expected_types:
        assert app_api._review_sequence(agent_type) == ["review_node"]
        assert agent_type in app_api.AGENT_CAPABILITIES


def test_app_api_normalizes_legacy_chapter_content_summary_types():
    assert app_api._normalize_agent_type("chapter_content_summary_create") == "chapter_intro_summary_create"
    assert app_api._normalize_agent_type("chapter_content_summary_update") == "chapter_intro_summary_update"
    assert app_api._normalize_agent_type("chapter_intro_summary_create") == "chapter_intro_summary_create"
    assert app_api._normalize_agent_type("chapter_intro_summary_update") == "chapter_intro_summary_update"


def test_app_api_can_run_outline_summary_workflow_until_human():
    calls = []

    class PassableOutlineSummaryLLM:
        def __init__(self, agent_name: str, calls: list[dict]):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {"agent": self.agent_name, "node": "initial_expansion", "entity_type": "summary", "action": "create"},
                "payload": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "target_id": "outline_test",
                    "name": "测试分卷",
                    "summary": "第一卷：失踪者\n开篇：坠落。\n主要矛盾：旧秩序是否继续有效。\n主要事件：营地分裂并发现余烬痕迹。\n高潮：暴乱导致三方对立。\n结局：新威胁浮现。",
                    "downstream_summary": "第一卷：失踪者\n开篇：运输舰坠落。\n主要矛盾：旧秩序与生存现实冲突。\n主要事件：营地分裂并发现余烬痕迹。\n高潮：暴乱导致三方对立。\n结局：新威胁浮现。",
                },
                "expanded_input": {"summary_seed": "原文", "summary_goal": "保留主冲突", "must_keep": ["主要矛盾", "高潮", "结局"]},
                "expansion_notes": "保留了主要结构。",
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return PassableOutlineSummaryLLM(agent_name, calls)

    with (
        patch.object(outline_summary_create_agent, "get_llm", fake_get_llm),
        patch.object(outline_summary_create_agent, "get_langfuse_callback", lambda: None),
    ):
        state = app_api._run_until_human(
            "outline_summary_create",
            "create",
            {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "name": "测试分卷",
                "summary": "第一卷：失踪者\n开篇：坠落。\n主要矛盾：旧秩序是否继续有效。\n主要事件：营地分裂并发现余烬痕迹。\n高潮：暴乱导致三方对立。\n结局：新威胁浮现。",
            },
            "请生成新增分卷大纲总结",
        )

    assert calls
    assert state["current_node"] == "human"
    assert state["status"] == "waiting_human"
    assert any(node["node_id"] == "review" for node in state["nodes"])


def test_app_api_uses_summary_only_revision_mode_for_summary_workflows():
    assert app_api._allowed_revision_modes("outline_summary_create") == {"summary_rewrite"}
    assert app_api._allowed_revision_modes("chapter_content_summary_update") == {"summary_rewrite"}
    assert app_api._allowed_revision_modes("world") == {"partial_rewrite", "content_rewrite", "full_rewrite"}
