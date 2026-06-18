"""大纲审核节点文件。

本文件只承载“父级大纲任务审核”节点逻辑，主要用于章节正文提交前校验是否
偏离父级大纲安排。
"""

from typing import Any, Callable, Dict

from src.agents.review_agent import execute_llm_review_detail
from src.common.lore_utils import get_mongodb_db


NodeFactory = Callable[[str, str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def make_outline_review_node(
    *,
    node_id: str,
    entity_type: str,
    reviewer: str,
    passed_key: str,
    errors_key: str,
    feedback_key: str,
    next_node: str,
    next_status: str,
    max_auto_review_iterations: int,
    node_factory: NodeFactory,
):
    """创建大纲审核节点：检查章节正文是否严格执行父级大纲任务。"""

    def outline_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """大纲审核节点：读取父级大纲，调用专属 LLM 审核当前章节 payload。"""
        db = get_mongodb_db()
        payload = dict(state.get("pending_payload") or {})
        review_detail = execute_llm_review_detail(db, entity_type, payload)
        passed = bool(review_detail.get("passed"))
        errors = list(review_detail.get("errors") or [])
        nodes = list(state.get("nodes") or [])
        nodes.append(node_factory(node_id, "completed" if passed else "failed", {"payload": payload}, {"passed": passed, "errors": errors, "reviewer": reviewer, "llm_invoked": review_detail.get("llm_invoked"), "llm_call": review_detail.get("llm_call"), "raw_response": review_detail.get("raw_response", "")}))
        waiting_human = int(state.get("iterations") or 0) >= max_auto_review_iterations
        return {
            passed_key: passed,
            errors_key: errors,
            feedback_key: "; ".join(errors),
            "review_passed": passed,
            "review_errors": errors,
            "review_feedback": "; ".join(errors),
            "nodes": nodes,
            "current_node": next_node if passed else ("human" if waiting_human else "modify_content"),
            "status": next_status if passed else ("waiting_human" if waiting_human else "review_failed"),
        }

    return outline_review_node


def make_outline_review_route(*, passed_key: str, next_node: str, max_auto_review_iterations: int):
    """创建大纲审核路由：通过进入下一节点，失败未超限进入修改内容节点。"""

    def route_after_outline_review(state: Dict[str, Any]) -> str:
        """大纲审核路由：按审核结果决定进入下一节点、修改内容节点或人工节点。"""
        if state.get(passed_key):
            return next_node
        if int(state.get("iterations") or 0) >= max_auto_review_iterations:
            return "human"
        return "modify_content"

    return route_after_outline_review
