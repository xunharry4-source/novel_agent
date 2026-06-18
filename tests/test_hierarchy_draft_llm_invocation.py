import json
import os
import sys
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import chapter_agent, novel_agent, outline_agent, review_agent, worldview_agent, world_agent


class FakeLLM:
    def __init__(self, agent_name, calls):
        self.agent_name = agent_name
        self.calls = calls

    def invoke(self, prompt, config=None):
        self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
        payloads = {
            "world_agent": {
                "world_id": "world_test",
                "name": "测试世界",
                "summary": "world_agent LLM 已扩充：底层规则、资源机制、组织结构、核心冲突、地理边界、风险约束都已形成可执行设定。",
            },
            "worldview_agent": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "name": "测试世界观",
                "summary": "worldview_agent LLM 已扩充：设定边界、核心规则、冲突风险、引用约束都已形成可检索 Canon。",
            },
            "worldview_agent_initial_expansion": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "name": "测试世界观",
                "summary": "worldview_agent_initial_expansion LLM 已扩充：设定边界、核心规则、冲突风险、引用约束都已形成可检索 Canon。",
            },
            "worldview_agent_modify_content": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "name": "测试世界观",
                "summary": "worldview_agent_modify_content LLM 已扩充：设定边界、核心规则、冲突风险、引用约束都已形成可检索 Canon。",
            },
            "novel_agent": {
                "world_id": "world_test",
                "novel_id": "novel_test",
                "name": "测试小说",
                "summary": "novel_agent LLM 已扩充：故事定位、主角视角、核心冲突、世界规则契合方式、后续大纲约束都已明确。",
            },
            "outline_agent": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "name": "测试大纲",
                "summary": "outline_agent LLM 已扩充：卷章结构、关键转折、冲突升级、高潮收束、设定一致性约束都已明确。",
            },
            "outline_agent_initial_expansion": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "name": "测试大纲",
                "summary": "outline_agent_initial_expansion LLM 已扩充：卷章结构、关键转折、冲突升级、高潮收束、设定一致性约束都已明确。",
            },
            "outline_agent_modify_content": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "name": "测试大纲",
                "summary": "outline_agent_modify_content LLM 已扩充：卷章结构、关键转折、冲突升级、高潮收束、设定一致性约束都已明确。",
            },
            "chapter_agent": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "id": "chapter_test",
                "name": "测试章节",
                "content": "chapter_agent LLM 已生成：正文场景、人物行动、冲突推进、设定执行和段落节奏都已写成可入库正文。",
            },
            "chapter_agent_initial_expansion": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "id": "chapter_test",
                "name": "测试章节",
                "content": "chapter_agent_initial_expansion LLM 已生成：正文场景、人物行动、冲突推进、设定执行和段落节奏都已写成可入库正文。",
            },
            "chapter_agent_modify_content": {
                "world_id": "world_test",
                "worldview_id": "wv_test",
                "novel_id": "novel_test",
                "outline_id": "outline_test",
                "chapter_id": "chapter_test",
                "id": "chapter_test",
                "name": "测试章节",
                "content": "chapter_agent_modify_content LLM 已生成：正文场景、人物行动、冲突推进、设定执行和段落节奏都已写成可入库正文。",
            },
        }
        body = {
            "payload": payloads[self.agent_name],
            "modification_notes": f"{self.agent_name} modification notes",
            "change_summary": f"{self.agent_name} change summary",
        }
        return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))


def test_all_hierarchy_modules_content_modification_calls_dedicated_llm():
    calls = []

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return FakeLLM(agent_name, calls)

    cases = [
        (world_agent, "world_agent", "summary", {"world_id": "world_test", "name": "测试世界", "summary": "短"}),
        (worldview_agent, "worldview_agent_modify_content", "summary", {"world_id": "world_test", "worldview_id": "wv_test", "name": "测试世界观", "summary": "短"}),
        (novel_agent, "novel_agent", "summary", {"world_id": "world_test", "novel_id": "novel_test", "name": "测试小说", "summary": "短"}),
        (outline_agent, "outline_agent_modify_content", "summary", {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试大纲", "summary": "短"}),
        (chapter_agent, "chapter_agent_modify_content", "content", {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "id": "chapter_test", "name": "测试章节", "content": "短"}),
    ]

    for module, expected_agent_name, primary_field, payload in cases:
        with (
            patch.object(module, "get_llm", fake_get_llm),
            patch.object(module, "get_langfuse_callback", lambda: None),
            patch.object(module, "get_unified_context", lambda *args, **kwargs: "测试检索上下文"),
        ):
            calls.clear()
            result = module.generate_content_modification(
                "create",
                payload,
                "用户不同意，请修改内容",
            )
            assert calls, expected_agent_name
            assert calls[0]["agent_name"] == expected_agent_name
            assert result["llm_invoked"] is True
            assert result["agent_name"] == expected_agent_name
            assert result["llm_agent_name"] == expected_agent_name
            assert result["llm_call"]["llm_agent_name"] == expected_agent_name
            assert result["llm_call"]["raw_response_chars"] > 0
            assert expected_agent_name in result["payload"][primary_field]
            assert result["raw_response"]
            assert result["parsed_response"]["payload"]


def test_all_hierarchy_modules_initial_expansion_calls_dedicated_llm():
    calls = []

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return FakeLLM(agent_name, calls)

    cases = [
        (world_agent, "world_agent", {"world_id": "world_test", "name": "测试世界", "summary": "短"}),
        (worldview_agent, "worldview_agent_initial_expansion", {"world_id": "world_test", "worldview_id": "wv_test", "name": "测试世界观", "summary": "短"}),
        (novel_agent, "novel_agent", {"world_id": "world_test", "novel_id": "novel_test", "name": "测试小说", "summary": "短"}),
        (outline_agent, "outline_agent_initial_expansion", {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试大纲", "summary": "短"}),
        (chapter_agent, "chapter_agent_initial_expansion", {"world_id": "world_test", "worldview_id": "wv_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "id": "chapter_test", "name": "测试章节", "content": "短"}),
    ]

    for module, expected_agent_name, payload in cases:
        with (
            patch.object(module, "get_llm", fake_get_llm),
            patch.object(module, "get_langfuse_callback", lambda: None),
        ):
            calls.clear()
            result = module.generate_initial_expansion(
                "create",
                payload,
                "请先做初始扩充",
            )
            assert calls, expected_agent_name
            assert calls[0]["agent_name"] == expected_agent_name
            assert result["llm_invoked"] is True
            assert result["agent_name"] == expected_agent_name
            assert result["llm_agent_name"] == expected_agent_name
            assert result["llm_call"]["llm_agent_name"] == expected_agent_name
            assert result["llm_call"]["raw_response_chars"] > 0
            assert isinstance(result["expanded_input"], dict)
            assert isinstance(result["payload"], dict)
            assert result["raw_response"]


def test_worldview_modify_content_accepts_chinese_output_contract():
    calls = []

    class ChineseContractLLM:
        def __init__(self, agent_name, calls):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {
                    "agent": "worldview_agent",
                    "node": "modify_content",
                    "entity_type": "worldview",
                    "action": "create",
                },
                "有效载荷": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "target_id": "",
                    "name": "测试世界观",
                    "summary": "晶钜石是一种稀有矿物，用于高密度储能与工业能源供应，不涉及物理法则或创造物质。",
                },
                "expanded_input": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "target_id": "",
                    "name": "测试世界观",
                    "summary_seed": "晶钜石是一种稀有矿物",
                    "category": "能源矿物",
                    "canon_keywords": ["晶钜石", "储能矿物"],
                    "must_keep": ["必须是稀有矿物", "不得超越科学"],
                    "review_focus": "检查是否仍有超自然或超科学描述",
                },
                "expansion_notes": "已按修改意见删除超自然与超科学描述，仅保留稀有矿物设定。",
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return ChineseContractLLM(agent_name, calls)

    with (
        patch.object(worldview_agent, "get_llm", fake_get_llm),
        patch.object(worldview_agent, "get_langfuse_callback", lambda: None),
        patch.object(worldview_agent, "get_unified_context", lambda *args, **kwargs: "测试检索上下文"),
    ):
        result = worldview_agent.generate_content_modification(
            "create",
            {"world_id": "world_test", "worldview_id": "wv_test", "name": "测试世界观", "summary": "初版设定"},
            "请修改世界观",
            revision_mode="partial_rewrite",
            feedback="禁止出现物理法则，它只是稀有矿物",
        )

    assert calls, "worldview_agent_modify_content"
    assert calls[0]["agent_name"] == "worldview_agent_modify_content"
    assert "操作流程" in calls[0]["prompt"]
    assert "【世界禁止规则】" in calls[0]["prompt"]
    assert "【修改意见】" in calls[0]["prompt"]
    assert '"payload"' in calls[0]["prompt"]
    assert result["payload"]["summary"].startswith("晶钜石是一种稀有矿物")
    assert result["expanded_input"]["category"] == "能源矿物"
    assert result["modification_notes"] == "已按修改意见删除超自然与超科学描述，仅保留稀有矿物设定。"
    assert result["change_summary"] == "已按修改意见删除超自然与超科学描述，仅保留稀有矿物设定。"


def test_worldview_initial_expansion_accepts_chinese_output_contract():
    calls = []

    class ChineseInitialContractLLM:
        def __init__(self, agent_name, calls):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {
                    "agent": "worldview_agent",
                    "node": "initial_expansion",
                    "entity_type": "worldview",
                    "action": "create",
                },
                "有效载荷": {
                    "world_id": "world_test",
                    "worldview_id": "",
                    "target_id": "",
                    "name": "晶钜石",
                    "summary": "晶钜石是一种高能工业矿物，因储能密度高、提纯难度大而成为关键能源资源。",
                },
                "expanded_input": {
                    "world_id": "world_test",
                    "worldview_id": "",
                    "target_id": "",
                    "name": "晶钜石",
                    "summary_seed": "晶钜石超强能量石，丰富能量",
                    "category": "自然资源/能源矿物",
                    "canon_keywords": ["晶钜石", "能源矿物", "工业储能"],
                    "must_keep": ["晶钜石是高能量矿物", "不得出现魔法", "不得出现神"],
                    "review_focus": "检查是否出现超自然解释，是否仍符合矿物与工业能源设定",
                },
                "expansion_notes": "已在不违反世界禁止规则的前提下补充能源用途、开采价值与潜在冲突风险。",
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return ChineseInitialContractLLM(agent_name, calls)

    with (
        patch.object(worldview_agent, "get_llm", fake_get_llm),
        patch.object(worldview_agent, "get_langfuse_callback", lambda: None),
    ):
        result = worldview_agent.generate_initial_expansion(
            "create",
            {"world_id": "world_test", "name": "晶钜石", "summary": "晶钜石超强能量石，丰富能量"},
            "请扩充这个世界观",
        )

    assert calls, "worldview_agent_initial_expansion"
    assert calls[0]["agent_name"] == "worldview_agent_initial_expansion"
    assert "操作流程" in calls[0]["prompt"]
    assert "【世界禁止规则】" in calls[0]["prompt"]
    assert "【世界观草稿】" in calls[0]["prompt"]
    assert '"payload"' in calls[0]["prompt"]
    assert result["payload"]["summary"].startswith("晶钜石是一种高能工业矿物")
    assert result["expanded_input"]["category"] == "自然资源/能源矿物"
    assert result["expansion_notes"] == "已在不违反世界禁止规则的前提下补充能源用途、开采价值与潜在冲突风险。"


def test_all_prompt_templates_use_structured_sections():
    with (
        patch.object(world_agent, "get_unified_context", lambda *args, **kwargs: "世界测试上下文"),
        patch.object(novel_agent, "get_unified_context", lambda *args, **kwargs: "小说测试上下文"),
        patch.object(outline_agent, "get_unified_context", lambda *args, **kwargs: "大纲测试上下文"),
        patch.object(chapter_agent, "get_unified_context", lambda *args, **kwargs: "章节测试上下文"),
        patch.object(worldview_agent, "get_unified_context", lambda *args, **kwargs: "世界观测试上下文"),
        patch.object(worldview_agent, "_load_world_forbidden_rules", lambda world_id: ["不得出现魔法", "不得出现神"]),
    ):
        prompts = {
            "world_initial": world_agent.build_initial_expansion_prompt("create", {"world_id": "world_test", "name": "测试世界", "summary": "摘要"}, "消息", revision_mode=None, feedback=""),
            "world_modify": world_agent.build_modification_prompt("update", {"world_id": "world_test", "target_id": "world_test", "name": "测试世界", "summary": "摘要"}, "消息", revision_mode="partial_rewrite", feedback="修改摘要"),
            "worldview_initial": worldview_agent.build_initial_expansion_prompt("create", {"world_id": "world_test", "name": "测试世界观", "summary": "摘要"}, "消息", revision_mode=None, feedback=""),
            "worldview_modify": worldview_agent.build_modification_prompt("update", {"world_id": "world_test", "worldview_id": "wv_test", "name": "测试世界观", "summary": "摘要"}, "消息", revision_mode="partial_rewrite", feedback="修改摘要"),
            "novel_initial": novel_agent.build_initial_expansion_prompt("create", {"world_id": "world_test", "novel_id": "novel_test", "name": "测试小说", "summary": "摘要"}, "消息", revision_mode=None, feedback=""),
            "novel_modify": novel_agent.build_modification_prompt("update", {"world_id": "world_test", "novel_id": "novel_test", "name": "测试小说", "summary": "摘要"}, "消息", revision_mode="partial_rewrite", feedback="修改摘要"),
            "outline_initial": outline_agent.build_initial_expansion_prompt("create", {"world_id": "world_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试大纲", "summary": "摘要"}, "消息", revision_mode=None, feedback=""),
            "outline_modify": outline_agent.build_modification_prompt("update", {"world_id": "world_test", "novel_id": "novel_test", "outline_id": "outline_test", "name": "测试大纲", "summary": "摘要"}, "消息", revision_mode="partial_rewrite", feedback="修改摘要"),
            "chapter_initial": chapter_agent.build_initial_expansion_prompt("create", {"world_id": "world_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节", "content": "摘要"}, "消息", revision_mode=None, feedback=""),
            "chapter_modify": chapter_agent.build_modification_prompt("update", {"world_id": "world_test", "novel_id": "novel_test", "outline_id": "outline_test", "chapter_id": "chapter_test", "name": "测试章节", "content": "摘要"}, "消息", revision_mode="partial_rewrite", feedback="修改摘要"),
        }

    for name, prompt in prompts.items():
        assert "【角色设定】" in prompt, name
        assert "【操作流程 (Mandatory Workflow)】" in prompt, name
        assert "【输入信息】" in prompt, name
        assert "【输出要求】" in prompt, name
        assert '"payload"' in prompt, name

    review_prompt = review_agent.get_review_prompt("worldview_world_rules")
    assert "【角色设定】" in review_prompt
    assert "【操作流程 (Mandatory Workflow)】" in review_prompt
    assert "【输出要求】" in review_prompt
    assert '"passed"' in review_prompt
    assert '"errors"' in review_prompt


def test_outline_prompts_require_preserving_long_form_outline_content():
    with patch.object(outline_agent, "get_unified_context", lambda *args, **kwargs: "大纲测试上下文"):
        payload = {
            "world_id": "world_test",
            "worldview_id": "wv_test",
            "novel_id": "novel_test",
            "outline_id": "outline_test",
            "name": "测试大纲",
            "summary": "第一卷：北港异象\\n第一章：灯塔误报\\n第二章：登记簿缺页\\n第三章：海沟追查",
        }
        initial_prompt = outline_agent.build_initial_expansion_prompt("create", payload, "请扩充大纲", revision_mode=None, feedback="")
        modify_prompt = outline_agent.build_modification_prompt("update", payload, "请按意见修改", revision_mode="partial_rewrite", feedback="补强第二章调查链路")
        initial_contract = initial_prompt.split("只返回合法 JSON：", 1)[1]
        modify_contract = modify_prompt.split("只返回合法 JSON：", 1)[1]

    assert "你是一名小说大纲扩写编辑" in initial_prompt
    assert "保留原有大纲全部内容" in initial_prompt
    assert "本任务是：扩写（Expand）" in initial_prompt
    assert "不是：" in initial_prompt
    assert "不得删除。" in initial_prompt
    assert "不得合并多个事件。" in initial_prompt
    assert "优先达到原文 150% 以上。" in initial_prompt
    assert "禁止输出比输入更短。" in initial_prompt
    assert "expanded_input.summary_seed" in initial_prompt
    assert "Review：" in initial_prompt
    assert "Expand：" in initial_prompt
    assert "Validate：" in initial_prompt
    assert "[保留输入中的 novel_id]" in initial_prompt
    assert "[扩写后的完整大纲]" in initial_prompt
    assert "[用户提交的原始大纲全文]" in initial_prompt
    assert payload["summary"] not in initial_contract

    assert "你是一名小说大纲修订编辑" in modify_prompt
    assert "本任务是：修改并扩写（Modify + Expand）" in modify_prompt
    assert "不得删除未被点名修改的内容。" in modify_prompt
    assert "如果修改范围很小，" in modify_prompt
    assert "至少保留原文总量不缩短。" in modify_prompt
    assert "Modify：" in modify_prompt
    assert "Expand：" in modify_prompt
    assert "Validate：" in modify_prompt
    assert "【任务上下文】" in modify_prompt
    assert "[按修改意见修正并补强后的完整大纲]" in modify_prompt
    assert payload["summary"] not in modify_contract


def test_chapter_prompts_preserve_long_form_content_and_keep_modify_separate():
    with patch.object(chapter_agent, "get_unified_context", lambda *args, **kwargs: "章节测试上下文"):
        payload = {
            "world_id": "world_test",
            "worldview_id": "wv_test",
            "novel_id": "novel_test",
            "outline_id": "outline_test",
            "chapter_id": "chapter_test",
            "id": "chapter_test",
            "name": "第一章：坠落",
            "content": "第一章：坠落\\n运输舰进入折跃航道。\\n导航开始偏移。\\n乘客最初没有察觉异常。",
        }
        initial_prompt = chapter_agent.build_initial_expansion_prompt("create", payload, "请扩充章节", revision_mode=None, feedback="")
        modify_prompt = chapter_agent.build_modification_prompt("update", payload, "请按意见修改章节", revision_mode="partial_rewrite", feedback="补强驾驶舱混乱过程")
        initial_contract = initial_prompt.split("只返回合法 JSON：", 1)[1]
        modify_contract = modify_prompt.split("只返回合法 JSON：", 1)[1]

    assert "你是一名小说章节扩写编辑" in initial_prompt
    assert "保留原有章节全部内容" in initial_prompt
    assert "本任务是：扩写（Expand）" in initial_prompt
    assert "不得把多个场景合并成概述。" in initial_prompt
    assert "expanded_input.content_seed" in initial_prompt
    assert "Review：" in initial_prompt
    assert "Expand：" in initial_prompt
    assert "Validate：" in initial_prompt
    assert "[扩写后的完整章节正文]" in initial_prompt
    assert "[用户提交的原始章节全文]" in initial_prompt
    assert payload["content"] not in initial_contract

    assert "你是一名小说章节修订编辑" in modify_prompt
    assert "本任务是：修改并扩写（Modify + Expand）" in modify_prompt
    assert "不得删除未被点名修改的内容。" in modify_prompt
    assert "至少保留原文总量不缩短。" in modify_prompt
    assert "Modify：" in modify_prompt
    assert "Expand：" in modify_prompt
    assert "Validate：" in modify_prompt
    assert "[按修改意见修正并补强后的完整章节正文]" in modify_prompt
    assert payload["content"] not in modify_contract


def test_outline_generation_rejects_over_simplified_long_form_output():
    calls = []

    class OversimplifiedOutlineLLM:
        def __init__(self, agent_name, calls):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {
                    "agent": "outline_agent",
                    "node": "initial_expansion",
                    "entity_type": "outline",
                    "action": "create",
                },
                "payload": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "name": "第一卷：失踪者",
                    "summary": "第一卷讲坠毁求生，最后建立营地。",
                },
                "expanded_input": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "name": "第一卷：失踪者",
                    "summary_seed": "长篇大纲输入",
                    "structure_goal": "扩写",
                    "affected_paths": ["第一卷"],
                    "key_conflicts": ["求生"],
                    "parent_constraints": ["保持世界观一致"],
                },
                "expansion_notes": "错误地压缩了长纲。",
                "modification_notes": "错误地压缩了长纲。",
                "change_summary": "错误地压缩了长纲。",
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return OversimplifiedOutlineLLM(agent_name, calls)

    payload = {
        "world_id": "world_test",
        "worldview_id": "wv_test",
        "novel_id": "novel_test",
        "outline_id": "outline_test",
        "name": "第一卷：失踪者",
        "summary": "\n".join(
            [
                "第一卷：失踪者",
                "第一章：坠落",
                "第二章：等待",
                "第三章：裂痕",
                "第四章：第三势力",
                "第五章：暴乱",
                "第六章：分裂",
                "第七章：痕迹",
                "尾声：余烬",
            ]
            + ["详细冲突说明与事件链条。" * 90]
        ),
    }

    with (
        patch.object(outline_agent, "get_llm", fake_get_llm),
        patch.object(outline_agent, "get_langfuse_callback", lambda: None),
        patch.object(outline_agent, "get_unified_context", lambda *args, **kwargs: "大纲测试上下文"),
    ):
        try:
            outline_agent.generate_initial_expansion("create", payload, "请扩写")
            raise AssertionError("generate_initial_expansion should reject oversimplified long-form outline output")
        except ValueError as exc:
            assert "compressed long outline too aggressively" in str(exc)

        try:
            outline_agent.generate_content_modification("update", payload, "请按意见修改", revision_mode="partial_rewrite", feedback="不要压缩")
            raise AssertionError("generate_content_modification should reject oversimplified long-form outline output")
        except ValueError as exc:
            assert "compressed long outline too aggressively" in str(exc)


def test_chapter_generation_rejects_over_simplified_long_form_output():
    calls = []

    class OversimplifiedChapterLLM:
        def __init__(self, agent_name, calls):
            self.agent_name = agent_name
            self.calls = calls

        def invoke(self, prompt, config=None):
            self.calls.append({"agent_name": self.agent_name, "prompt": prompt, "config": config})
            body = {
                "metadata": {
                    "agent": "chapter_agent",
                    "node": "initial_expansion",
                    "entity_type": "chapter",
                    "action": "create",
                },
                "payload": {
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "novel_id": "novel_test",
                    "outline_id": "outline_test",
                    "chapter_id": "chapter_test",
                    "id": "chapter_test",
                    "name": "第一章：坠落",
                    "content": "飞船坠毁，众人陷入混乱。",
                },
                "expanded_input": {
                    "outline_id": "outline_test",
                    "novel_id": "novel_test",
                    "world_id": "world_test",
                    "worldview_id": "wv_test",
                    "chapter_id": "chapter_test",
                    "id": "chapter_test",
                    "name": "第一章：坠落",
                    "content_seed": "长篇章节输入",
                    "scene_goal": "扩写",
                    "character_states": ["慌乱"],
                    "narrative_viewpoint": "第三人称",
                    "target_segments": ["开场"],
                    "continuity_constraints": ["保持设定一致"],
                },
                "expansion_notes": "错误地压缩了长章节。",
                "modification_notes": "错误地压缩了长章节。",
                "change_summary": "错误地压缩了长章节。",
            }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    def fake_get_llm(json_mode=False, agent_name="unknown"):
        assert json_mode is True
        return OversimplifiedChapterLLM(agent_name, calls)

    payload = {
        "world_id": "world_test",
        "worldview_id": "wv_test",
        "novel_id": "novel_test",
        "outline_id": "outline_test",
        "chapter_id": "chapter_test",
        "id": "chapter_test",
        "name": "第一章：坠落",
        "content": "第一章：坠落\n" + ("运输舰开始震动，警报不断升级，船员与乘客的反应层层叠加。\n" * 80),
    }

    with (
        patch.object(chapter_agent, "get_llm", fake_get_llm),
        patch.object(chapter_agent, "get_langfuse_callback", lambda: None),
        patch.object(chapter_agent, "get_unified_context", lambda *args, **kwargs: "章节测试上下文"),
    ):
        try:
            chapter_agent.generate_initial_expansion("create", payload, "请扩写章节")
            raise AssertionError("generate_initial_expansion should reject oversimplified long-form chapter output")
        except ValueError as exc:
            assert "compressed long chapter too aggressively" in str(exc)

        try:
            chapter_agent.generate_content_modification("update", payload, "请按意见修改章节", revision_mode="partial_rewrite", feedback="不要压缩")
            raise AssertionError("generate_content_modification should reject oversimplified long-form chapter output")
        except ValueError as exc:
            assert "compressed long chapter too aggressively" in str(exc)


def test_five_agents_are_independent_state_graph_instances():
    cases = [
        (world_agent, "world_agent", []),
        (worldview_agent, "worldview_agent", ["world_rule_review", "worldview_consistency_review"]),
        (novel_agent, "novel_agent", ["review"]),
        (outline_agent, "outline_agent", ["world_review", "worldview_review", "novel_review"]),
        (chapter_agent, "chapter_agent", ["world_review", "worldview_review", "novel_review", "outline_review", "chapter_review"]),
    ]

    apps = []
    for module, expected_agent_name, review_nodes in cases:
        assert module.AGENT_NAME == expected_agent_name
        assert module.WORKFLOW_DESCRIPTION
        assert module.WORKFLOW_STEPS
        assert module.NODE_ANNOTATIONS
        assert hasattr(module, "app")
        assert hasattr(module, "workflow")
        apps.append(module.app)
        graph = module.app.get_graph()
        node_names = set(graph.nodes.keys())
        assert {"input", "initial_expansion", "human", "modify_content", "commit"}.issubset(node_names)
        assert "draft" not in node_names
        assert module.WORKFLOW_STEPS["initial_expansion"]["step_index"] == 2
        assert module.NODE_ANNOTATIONS["initial_expansion"]["input_annotation"]
        assert module.NODE_ANNOTATIONS["initial_expansion"]["output_annotation"]
        assert module.NODE_ANNOTATIONS["initial_expansion"]["next_step_annotation"]
        for review_node in review_nodes:
            assert review_node in node_names
        if module in {worldview_agent, outline_agent, chapter_agent}:
            assert "review" not in node_names
        elif not review_nodes:
            assert "review" not in node_names
        for node_id, step in module.WORKFLOW_STEPS.items():
            assert step["step_index"]
            assert step["step_title"].startswith("步骤")
            assert step["function"]
            assert step["description"]
            rendered = module._node(node_id, "completed", {}, {})
            assert rendered["step_index"] == step["step_index"]
            assert rendered["step_title"] == step["step_title"]
            assert rendered["function"] == step["function"]
            assert rendered["description"] == step["description"]
            assert rendered["node_annotation"].startswith(rendered["step_title"])
            assert rendered["input_annotation"]
            assert rendered["output_annotation"]
            assert rendered["next_step_annotation"]

    assert len({id(app) for app in apps}) == 5


def test_agent_methods_have_chinese_annotations():
    cases = [world_agent, worldview_agent, novel_agent, outline_agent, chapter_agent]
    required_methods = [
        "_extract_llm_content",
        "_llm_metadata",
        "_invoke_llm",
        "_node",
        "build_initial_expansion_prompt",
        "generate_initial_expansion",
        "input_node",
        "initial_expansion_node",
        "human_node",
        "route_after_human",
        "commit_node",
    ]
    modify_methods = ["build_modification_prompt", "generate_content_modification", "modify_content_node"]
    review_methods = ["review_node", "route_after_review"]
    worldview_review_methods = [
        "world_rule_review_node",
        "route_after_world_rule_review",
        "worldview_consistency_review_node",
        "route_after_worldview_consistency_review",
    ]
    outline_review_methods = [
        "world_review_node",
        "route_after_world_review",
        "worldview_review_node",
        "route_after_worldview_review",
        "novel_review_node",
        "route_after_novel_review",
    ]
    chapter_review_methods = [
        "world_review_node",
        "route_after_world_review",
        "worldview_review_node",
        "route_after_worldview_review",
        "novel_review_node",
        "route_after_novel_review",
        "outline_review_node",
        "route_after_outline_review",
        "chapter_review_node",
        "route_after_chapter_review",
    ]

    for module in cases:
        names = list(required_methods)
        names.extend(modify_methods)
        if module is worldview_agent:
            names.extend(worldview_review_methods)
        elif module is outline_agent:
            names.extend(outline_review_methods)
        elif module is chapter_agent:
            names.extend(chapter_review_methods)
        elif module is not world_agent:
            names.extend(review_methods)
        for name in names:
            method = getattr(module, name)
            doc = inspect.getdoc(method)
            assert doc, f"{module.AGENT_NAME}.{name} 缺少中文方法注解"
            assert any("\u4e00" <= char <= "\u9fff" for char in doc), f"{module.AGENT_NAME}.{name} 注解不是中文"


def test_review_nodes_are_split_into_dedicated_files():
    root = Path(__file__).resolve().parents[1]
    review_dir = root / "src" / "agents" / "review_nodes"
    expected_review_files = {
        "world_review.py",
        "worldview_review.py",
        "novel_review.py",
        "outline_review.py",
        "chapter_review.py",
    }
    actual_review_files = {path.name for path in review_dir.glob("*.py")}
    assert expected_review_files <= actual_review_files

    for filename in expected_review_files:
        source = (review_dir / filename).read_text(encoding="utf-8")
        assert "execute_llm_review" in source or "build_chapter_review_nodes" in source, f"{filename} 必须承载独立审核调用或章节审核组装"
        assert any("\u4e00" <= char <= "\u9fff" for char in source), f"{filename} 必须包含中文审核说明"

    for relative_path in [
        "src/agents/worldview_agent.py",
        "src/agents/novel_agent.py",
        "src/agents/outline_agent.py",
        "src/agents/chapter_agent.py",
    ]:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "from src.agents.review_agent import execute_llm_review" not in source
        assert "execute_llm_review(" not in source


if __name__ == "__main__":
    test_all_hierarchy_modules_content_modification_calls_dedicated_llm()
    test_all_hierarchy_modules_initial_expansion_calls_dedicated_llm()
    test_five_agents_are_independent_state_graph_instances()
    test_agent_methods_have_chinese_annotations()
    test_review_nodes_are_split_into_dedicated_files()
