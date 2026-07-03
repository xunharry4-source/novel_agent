import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import assert_review_node, cleanup_world, make_full_chain, start_agent


def test_real_review_rule_world_review_requests():
    chain = make_full_chain("review_world")
    try:
        run = start_agent(
            "outline",
            "create",
            {"world_id": chain["world_id"], "worldview_id": chain["worldview_id"], "novel_id": chain["novel_id"], "name": f"Outline Review {chain['suffix']}", "summary": "世界审核真实测试大纲"},
            "真实测试世界审核节点",
        )
        node = assert_review_node(run, "world_review", "outline_world_rules_review_agent")
        assert node["input"]["payload"].get("world_id") == chain["world_id"], node
    finally:
        cleanup_world(chain["world_id"])
