"""章节审查链路文件。

本文件负责章节 Agent 的章节一致性审查，并组装章节审核链路。
章节一致性审查用于比较当前章节与此前已入库章节的内容承接是否一致。
"""

from typing import Any, Callable, Dict

from src.agents.review_nodes.novel_review import make_novel_review_node, make_novel_review_route
from src.agents.review_nodes.outline_review import make_outline_review_node, make_outline_review_route
from src.agents.review_nodes.world_review import make_world_review_node, make_world_review_route
from src.agents.review_nodes.worldview_review import make_worldview_review_node, make_worldview_review_route
from src.agents.review_agent import execute_llm_review_detail
from src.common.lore_utils import get_mongodb_db


NodeFactory = Callable[[str, str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def make_chapter_review_node(
    *,
    node_factory: NodeFactory,
    max_auto_review_iterations: int,
):
    """创建章节审查节点：检查当前章节与此前章节内容是否一致。"""

    def chapter_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """章节审查节点：调用 LLM 比对当前章节和此前章节的剧情、人物、时间线与承接关系。"""
        db = get_mongodb_db()
        payload = dict(state.get("pending_payload") or {})
        review_detail = execute_llm_review_detail(db, "chapter_consistency", payload)
        passed = bool(review_detail.get("passed"))
        errors = list(review_detail.get("errors") or [])
        nodes = list(state.get("nodes") or [])
        nodes.append(node_factory("chapter_review", "completed" if passed else "failed", {"payload": payload}, {"passed": passed, "errors": errors, "reviewer": "chapter_consistency_review_agent", "llm_invoked": review_detail.get("llm_invoked"), "llm_call": review_detail.get("llm_call"), "raw_response": review_detail.get("raw_response", "")}))
        waiting_human = int(state.get("iterations") or 0) >= max_auto_review_iterations
        return {
            "chapter_review_passed": passed,
            "chapter_review_errors": errors,
            "chapter_review_feedback": "; ".join(errors),
            "review_passed": passed,
            "review_errors": errors,
            "review_feedback": "; ".join(errors),
            "nodes": nodes,
            "current_node": "human" if passed or waiting_human else "modify_content",
            "status": "waiting_human" if passed or waiting_human else "review_failed",
        }

    return chapter_review_node


def make_chapter_review_route(*, max_auto_review_iterations: int):
    """创建章节审查路由：通过进入人工节点，失败未超限进入修改内容节点。"""

    def route_after_chapter_review(state: Dict[str, Any]) -> str:
        """章节审查路由：根据章节一致性审查结果决定进入人工节点或修改内容节点。"""
        if state.get("chapter_review_passed"):
            return "human"
        if int(state.get("iterations") or 0) >= max_auto_review_iterations:
            return "human"
        return "modify_content"

    return route_after_chapter_review


def build_chapter_review_nodes(*, node_factory: NodeFactory, max_auto_review_iterations: int) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """组装章节审查节点：返回章节五审节点与对应路由方法。"""
    return {
        "world_review_node": make_world_review_node(
            node_id="world_review",
            entity_type="chapter_world_rules",
            reviewer="chapter_world_review_agent",
            passed_key="world_review_passed",
            errors_key="world_review_errors",
            feedback_key="world_review_feedback",
            next_node="worldview_review",
            next_status="reviewing_worldview",
            max_auto_review_iterations=max_auto_review_iterations,
            node_factory=node_factory,
        ),
        "route_after_world_review": make_world_review_route(
            passed_key="world_review_passed",
            next_node="worldview_review",
            max_auto_review_iterations=max_auto_review_iterations,
        ),
        "worldview_review_node": make_worldview_review_node(
            node_id="worldview_review",
            entity_type="chapter_worldview_rules",
            reviewer="chapter_worldview_review_agent",
            passed_key="worldview_review_passed",
            errors_key="worldview_review_errors",
            feedback_key="worldview_review_feedback",
            next_node="novel_review",
            next_status="reviewing_novel",
            max_auto_review_iterations=max_auto_review_iterations,
            node_factory=node_factory,
        ),
        "route_after_worldview_review": make_worldview_review_route(
            passed_key="worldview_review_passed",
            next_node="novel_review",
            max_auto_review_iterations=max_auto_review_iterations,
        ),
        "novel_review_node": make_novel_review_node(
            node_id="novel_review",
            entity_type="chapter_novel_rules",
            reviewer="chapter_novel_review_agent",
            passed_key="novel_review_passed",
            errors_key="novel_review_errors",
            feedback_key="novel_review_feedback",
            next_node="outline_review",
            next_status="reviewing_outline",
            max_auto_review_iterations=max_auto_review_iterations,
            node_factory=node_factory,
        ),
        "route_after_novel_review": make_novel_review_route(
            passed_key="novel_review_passed",
            next_node="outline_review",
            max_auto_review_iterations=max_auto_review_iterations,
        ),
        "outline_review_node": make_outline_review_node(
            node_id="outline_review",
            entity_type="chapter_outline_rules",
            reviewer="chapter_outline_review_agent",
            passed_key="outline_review_passed",
            errors_key="outline_review_errors",
            feedback_key="outline_review_feedback",
            next_node="chapter_review",
            next_status="reviewing_chapter_consistency",
            max_auto_review_iterations=max_auto_review_iterations,
            node_factory=node_factory,
        ),
        "route_after_outline_review": make_outline_review_route(
            passed_key="outline_review_passed",
            next_node="chapter_review",
            max_auto_review_iterations=max_auto_review_iterations,
        ),
        "chapter_review_node": make_chapter_review_node(
            node_factory=node_factory,
            max_auto_review_iterations=max_auto_review_iterations,
        ),
        "route_after_chapter_review": make_chapter_review_route(
            max_auto_review_iterations=max_auto_review_iterations,
        ),
    }
