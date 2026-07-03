import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import assert_success, request_json, start_agent, unique_suffix


def test_real_world_request_changes_uses_mode_specific_prompt_guidance():
    prompts: dict[str, str] = {}

    for revision_mode in ("partial_rewrite", "content_rewrite", "full_rewrite"):
        suffix = unique_suffix(f"revision_mode_{revision_mode}")
        run = start_agent(
            "world",
            "create",
            {
                "world_id": f"world_{suffix}",
                "name": f"Revision Mode World {suffix}",
                "summary": "一个群岛蒸汽航海世界，灯塔与航图制度严格控制离港路线。",
                "forbidden_rules": ["禁止现代枪械直接出现"],
                "basic_settings": {"era": "蒸汽航海", "power_system": "航图与潮汐灯塔规则"},
            },
            "创建一个用于验证修改模式提示词的世界",
        )
        modified = assert_success(
            request_json(
                "POST",
                "/api/hierarchy-agent/respond",
                json={
                    "run_id": run["run_id"],
                    "decision": "request_changes",
                    "message": f"请按 {revision_mode} 修正当前世界设定",
                    "revision_mode": revision_mode,
                    "manual_edit": True,
                    "payload": {
                        "summary": "修正后的世界摘要：强调灯塔禁航、群岛贸易与潮汐登记制度。",
                    },
                },
            )
        )["run"]
        modify_node = next(node for node in reversed(modified["nodes"]) if node["node_id"] == "modify_content")
        prompts[revision_mode] = modify_node["output"]["llm_call"]["prompt"]

    assert "当前模式：partial_rewrite（指定局部重写）" in prompts["partial_rewrite"], prompts["partial_rewrite"]
    assert "禁止整篇重写或整体改写结构。" in prompts["partial_rewrite"], prompts["partial_rewrite"]

    assert "当前模式：content_rewrite（指定内容重写）" in prompts["content_rewrite"], prompts["content_rewrite"]
    assert "允许围绕世界摘要 summary 做较大幅重写" in prompts["content_rewrite"], prompts["content_rewrite"]

    assert "当前模式：full_rewrite（完全重写）" in prompts["full_rewrite"], prompts["full_rewrite"]
    assert "允许整体重写 name、summary、forbidden_rules 和 basic_settings 等世界级业务内容。" in prompts["full_rewrite"], prompts["full_rewrite"]
    assert "禁止整篇重写或整体改写结构。" not in prompts["full_rewrite"], prompts["full_rewrite"]
