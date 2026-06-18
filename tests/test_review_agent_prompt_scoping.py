import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agents.review_agent import build_review_messages, _classify_review_exception
from src.common.llm_factory import _default_timeout_for, _resolve_timeout


def _chapter_payload():
    return {
        "world_id": "world_test",
        "novel_id": "novel_test",
        "outline_id": "outline_test",
        "worldview_id": "worldview_test",
        "name": "第一章：测试",
        "content": "林澈按灯塔规则核对登记簿。",
        "forbidden_rules": ["不得出现现代互联网"],
        "basic_settings": {"era": "蒸汽航海时代"},
        "novel_forbidden_rules": ["主角不得滥杀无辜"],
        "novel_basic_settings": {"tone": "冷静调查"},
        "chapter_outline": "林澈检查灯塔登记簿，暂停离港信号。",
        "previous_chapters": [{"title": "序章", "content": "北港进入戒严前夜。"}],
    }


def test_chapter_world_review_prompt_only_keeps_world_context():
    _, user_message = build_review_messages(None, "chapter_world_rules", _chapter_payload())

    assert "以下是世界禁止规则与基本设定（必须优先遵守）" in user_message, user_message
    assert "以下是小说禁止规则与基本设定（大纲和章节必须遵守）" not in user_message, user_message
    assert "以下是父级大纲约束（章节必须遵守）" not in user_message, user_message
    assert "以下是章节大纲约束（直接内容检查必须遵守）" not in user_message, user_message
    assert "以下是前置章节上下文（章节一致性审查必须遵守）" not in user_message, user_message
    assert "以下是世界观和背景设定参考（如果有）" not in user_message, user_message


def test_chapter_plot_review_prompt_keeps_outline_and_previous_chapter_context():
    _, user_message = build_review_messages(None, "chapter_plot_errors", _chapter_payload())

    assert "以下是父级大纲约束（章节必须遵守）" in user_message, user_message
    assert "以下是章节大纲约束（直接内容检查必须遵守）" in user_message, user_message
    assert "以下是前置章节上下文（章节一致性审查必须遵守）" in user_message, user_message
    assert "以下是世界禁止规则与基本设定（必须优先遵守）" not in user_message, user_message
    assert "以下是小说禁止规则与基本设定（大纲和章节必须遵守）" not in user_message, user_message
    assert "以下是世界观和背景设定参考（如果有）" not in user_message, user_message


def test_review_exception_classification_and_timeout_defaults():
    error_kind, error_text = _classify_review_exception(Exception("Connection error."))
    assert error_kind == "connection_error", (error_kind, error_text)
    assert "连接失败" in error_text, error_text

    timeout_kind, timeout_text = _classify_review_exception(Exception("Request timed out."))
    assert timeout_kind == "timeout", (timeout_kind, timeout_text)
    assert "审查超时" in timeout_text, timeout_text

    assert _default_timeout_for("ollama", "chapter_world_rules_review_agent") == 180
    assert _default_timeout_for("ollama", "chapter_agent") == 60
    assert _resolve_timeout({"REVIEW_AGENT_TIMEOUT": 240}, "ollama", "chapter_world_rules_review_agent", {}) == 240
    assert _resolve_timeout({}, "gemini", "novel_agent", {}) == 120
