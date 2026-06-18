"""小说审查节点文件。

本文件只承载“小说规则/小说主线约束审核”节点逻辑，禁止混入 Agent 主流程、
初始扩充、人工确认或写库逻辑。
"""

from typing import Any, Callable, Dict

from src.agents.review_agent import execute_llm_review_detail
from src.common.lore_utils import get_mongodb_db


NodeFactory = Callable[[str, str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def make_novel_review_node(
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
    """创建小说审查节点：检查业务内容是否违反小说禁止规则、基本设定和主线约束。"""

    def novel_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """小说审查节点：读取小说规则，调用专属 LLM 审核当前 payload。"""
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

    return novel_review_node


def make_novel_review_route(*, passed_key: str, next_node: str, max_auto_review_iterations: int):
    """创建小说审查路由：通过进入下一节点，失败未超限进入修改内容节点。"""

    def route_after_novel_review(state: Dict[str, Any]) -> str:
        """小说审查路由：按审核结果决定进入下一节点、修改内容节点或人工节点。"""
        if state.get(passed_key):
            return next_node
        if int(state.get("iterations") or 0) >= max_auto_review_iterations:
            return "human"
        return "modify_content"

    return route_after_novel_review
