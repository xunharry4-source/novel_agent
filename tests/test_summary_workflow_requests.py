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
        body = {
            "metadata": {"agent": self.agent_name, "node": "initial_expansion", "entity_type": "summary", "action": "create"},
            "payload": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "target_id": "target_test",
                "name": "测试对象",
                source_field: source_text,
                "downstream_summary": f"{self.agent_name} 下游摘要：保留核心冲突、关键事件和结果。",
            },
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

            modified = module.generate_content_modification("update", payload, "请按意见重做摘要", revision_mode="summary_rewrite", feedback="保留更多冲突链")
            assert calls[-1]["agent_name"] == modify_llm_name
            assert modified["payload"][source_field] == payload[source_field]
            assert modified["payload"]["downstream_summary"]


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
    assert "payload.summary 保留原文" in outline_update_prompt
    assert "章节大纲下游摘要编辑" in chapter_outline_create_prompt
    assert "payload.content 保留输入原文" in chapter_outline_create_prompt
    assert "修改章节正文摘要返工编辑" in chapter_content_update_prompt
    assert "payload.content 保留原文" in chapter_content_update_prompt


def test_app_api_registers_all_summary_workflow_types():
    expected_types = {
        "outline_summary_create",
        "outline_summary_update",
        "chapter_outline_summary_create",
        "chapter_outline_summary_update",
        "chapter_content_summary_create",
        "chapter_content_summary_update",
    }
    assert expected_types.issubset(set(app_api.AGENT_MODULES.keys()))
    for agent_type in expected_types:
        assert app_api._review_sequence(agent_type) == ["review_node"]
        assert agent_type in app_api.AGENT_CAPABILITIES


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
