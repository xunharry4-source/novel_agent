import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import assert_review_node, cleanup_world, make_full_chain, start_agent


def test_real_review_rule_novel_review_requests():
    chain = make_full_chain("review_novel")
    try:
        run = start_agent(
            "outline",
            "create",
            {"world_id": chain["world_id"], "worldview_id": chain["worldview_id"], "novel_id": chain["novel_id"], "name": f"Outline Review {chain['suffix']}", "summary": "小说审核真实测试大纲"},
            "真实测试小说审核节点",
        )
        node = assert_review_node(run, "novel_review", "outline_novel_rules_review_agent")
        assert node["input"]["payload"].get("novel_id") == chain["novel_id"], node
    finally:
        cleanup_world(chain["world_id"])
