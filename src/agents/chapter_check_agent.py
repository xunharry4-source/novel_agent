"""章节内容检查工作流。

本模块只负责“检查直接内容是否违规”的检查链：
输入 -> 世界规则 -> 小说规则 -> 世界观 -> 分卷大纲 -> 章节大纲 -> 剧情错误 -> 输出结果。

禁止在此模块中混入扩写、人工批准或写库逻辑。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, TypedDict

from src.agents.review_agent import execute_llm_review_detail
from src.common.lore_utils import get_mongodb_db


WORKFLOW_DESCRIPTION = (
    "章节检查工作流：接收直接内容 -> 检查世界规则 -> 检查小说规则 -> 检查世界观规则 -> "
    "检查分卷大纲 -> 检查章节大纲 -> 检查剧情错误 -> 输出检查结果。"
)

WORKFLOW_STEPS = {
    "input": {
        "step_index": 1,
        "step_title": "步骤 1：接收输入",
        "function": "接收直接内容、章节大纲和父级上下文",
        "description": "记录当前要检查的直接内容、章节大纲、outline_id、novel_id、worldview_id、world_id 和目标章节上下文。",
    },
    "world_review": {
        "step_index": 2,
        "step_title": "步骤 2：检查世界规则",
        "function": "检查是否违反世界禁止规则与基本设定",
        "description": "基于所属世界的 forbidden_rules 与 basic_settings 检查直接内容是否越过世界根约束。",
    },
    "novel_review": {
        "step_index": 3,
        "step_title": "步骤 3：检查小说规则",
        "function": "检查是否违反小说禁止规则与主线约束",
        "description": "基于小说 forbidden_rules、basic_settings、主角底线、主线冲突和人物关系规则检查直接内容。",
    },
    "worldview_review": {
        "step_index": 4,
        "step_title": "步骤 4：检查世界观",
        "function": "检查是否违反世界观 Canon",
        "description": "基于 worldview_id 对应设定和同一世界已入库 Lore 检查直接内容是否出现世界观冲突。",
    },
    "outline_review": {
        "step_index": 5,
        "step_title": "步骤 5：检查分卷大纲",
        "function": "检查是否偏离分卷大纲任务",
        "description": "基于 outline_id 对应分卷大纲检查直接内容是否提前、延后、删改或绕开分卷任务。",
    },
    "chapter_outline_review": {
        "step_index": 6,
        "step_title": "步骤 6：检查章节大纲",
        "function": "检查是否偏离章节大纲",
        "description": "基于用户提供或目标章节预填的章节大纲，检查直接内容是否偏离本章任务、场景顺序和关键事件。",
    },
    "plot_review": {
        "step_index": 7,
        "step_title": "步骤 7：检查剧情错误",
        "function": "检查剧情逻辑、因果链与前后承接错误",
        "description": "检查直接内容是否存在因果断裂、人物动机跳变、知识泄漏、时间线错误或场景承接问题。",
    },
    "output": {
        "step_index": 8,
        "step_title": "步骤 8：输出检查结果",
        "function": "汇总所有检查结论",
        "description": "输出逐项检查结果、失败项和总判定，不写库，不进入人工批准。",
    },
}

NODE_ANNOTATIONS = {
    "input": {
        "input_annotation": "输入必须包含 outline_id、直接内容；如需检查章节大纲，必须提供 chapter_outline 或可解析的目标章节。",
        "output_annotation": "输出 accepted=true，并锁定本次检查使用的直接内容与父级上下文。",
        "next_step_annotation": "下一步进入世界规则检查。",
    },
    "world_review": {
        "input_annotation": "输入是直接内容 payload、world_id 和世界 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都继续进入下一项检查。",
        "next_step_annotation": "下一步进入小说规则检查。",
    },
    "novel_review": {
        "input_annotation": "输入是直接内容 payload、novel_id 和小说 forbidden_rules/basic_settings。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都继续进入下一项检查。",
        "next_step_annotation": "下一步进入世界观检查。",
    },
    "worldview_review": {
        "input_annotation": "输入是直接内容 payload、worldview_id 和世界观 Canon 上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都继续进入下一项检查。",
        "next_step_annotation": "下一步进入分卷大纲检查。",
    },
    "outline_review": {
        "input_annotation": "输入是直接内容 payload、outline_id 和父级分卷大纲。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都继续进入下一项检查。",
        "next_step_annotation": "下一步进入章节大纲检查。",
    },
    "chapter_outline_review": {
        "input_annotation": "输入是直接内容 payload 和章节大纲上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都继续进入下一项检查。",
        "next_step_annotation": "下一步进入剧情错误检查。",
    },
    "plot_review": {
        "input_annotation": "输入是直接内容 payload、章节大纲和前置章节上下文。",
        "output_annotation": "输出包含 passed、errors 和 reviewer；无论通过或失败都进入汇总输出。",
        "next_step_annotation": "下一步进入结果汇总。",
    },
    "output": {
        "input_annotation": "输入是前面六个检查节点的通过/失败结论和错误列表。",
        "output_annotation": "输出包含 overall_passed、failed_checks、check_results 和 summary。",
        "next_step_annotation": "检查完成，不写库。",
    },
}

REVIEW_SEQUENCE = [
    {
        "node_id": "world_review",
        "entity_type": "chapter_world_rules",
        "reviewer": "chapter_world_review_agent",
        "passed_key": "world_review_passed",
        "errors_key": "world_review_errors",
        "feedback_key": "world_review_feedback",
    },
    {
        "node_id": "novel_review",
        "entity_type": "chapter_novel_rules",
        "reviewer": "chapter_novel_review_agent",
        "passed_key": "novel_review_passed",
        "errors_key": "novel_review_errors",
        "feedback_key": "novel_review_feedback",
    },
    {
        "node_id": "worldview_review",
        "entity_type": "chapter_worldview_rules",
        "reviewer": "chapter_worldview_review_agent",
        "passed_key": "worldview_review_passed",
        "errors_key": "worldview_review_errors",
        "feedback_key": "worldview_review_feedback",
    },
    {
        "node_id": "outline_review",
        "entity_type": "chapter_outline_rules",
        "reviewer": "chapter_outline_review_agent",
        "passed_key": "outline_review_passed",
        "errors_key": "outline_review_errors",
        "feedback_key": "outline_review_feedback",
    },
    {
        "node_id": "chapter_outline_review",
        "entity_type": "chapter_chapter_outline_rules",
        "reviewer": "chapter_chapter_outline_review_agent",
        "passed_key": "chapter_outline_review_passed",
        "errors_key": "chapter_outline_review_errors",
        "feedback_key": "chapter_outline_review_feedback",
    },
    {
        "node_id": "plot_review",
        "entity_type": "chapter_plot_errors",
        "reviewer": "chapter_plot_review_agent",
        "passed_key": "plot_review_passed",
        "errors_key": "plot_review_errors",
        "feedback_key": "plot_review_feedback",
    },
]


class ChapterCheckState(TypedDict, total=False):
    action: str
    message: str
    payload: Dict[str, Any]
    pending_payload: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    conversation: List[Dict[str, Any]]
    status: str
    current_node: str
    committed: bool
    review_required: bool
    check_passed: bool
    check_results: List[Dict[str, Any]]
    failed_checks: List[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node(node_id: str, status: str, node_input: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    step = WORKFLOW_STEPS[node_id]
    annotations = NODE_ANNOTATIONS[node_id]
    return {
        "node_id": node_id,
        "label": step["step_title"],
        **step,
        "node_annotation": f"{step['step_title']}：{step['description']}",
        **annotations,
        "status": status,
        "input": node_input,
        "output": output,
    }


def _append_review_node(state: ChapterCheckState, spec: Dict[str, str]) -> ChapterCheckState:
    db = get_mongodb_db()
    payload = dict(state.get("pending_payload") or {})
    review_detail = execute_llm_review_detail(db, spec["entity_type"], payload)
    passed = bool(review_detail.get("passed"))
    errors = list(review_detail.get("errors") or [])
    nodes = list(state.get("nodes") or [])
    nodes.append(
        _node(
            spec["node_id"],
            "completed" if passed else "failed",
            {"payload": payload},
            {
                "passed": passed,
                "errors": errors,
                "reviewer": spec["reviewer"],
                "llm_invoked": review_detail.get("llm_invoked"),
                "llm_call": review_detail.get("llm_call"),
                "raw_response": review_detail.get("raw_response", ""),
            },
        )
    )
    return {
        spec["passed_key"]: passed,
        spec["errors_key"]: errors,
        spec["feedback_key"]: "; ".join(errors),
        "nodes": nodes,
        "current_node": spec["node_id"],
        "status": f"completed_{spec['node_id']}",
    }


def input_node(state: ChapterCheckState) -> ChapterCheckState:
    nodes = list(state.get("nodes") or [])
    payload = dict(state.get("payload") or {})
    nodes.append(_node("input", "completed", {"message": state.get("message", ""), "payload": payload}, {"accepted": True}))
    return {
        "nodes": nodes,
        "pending_payload": payload,
        "current_node": "input",
        "status": "running",
    }


def output_node(state: ChapterCheckState) -> ChapterCheckState:
    check_results = []
    for spec in REVIEW_SEQUENCE:
        check_results.append(
            {
                "node_id": spec["node_id"],
                "reviewer": spec["reviewer"],
                "passed": bool(state.get(spec["passed_key"])),
                "errors": list(state.get(spec["errors_key"]) or []),
            }
        )
    failed_checks = [item["node_id"] for item in check_results if not item["passed"]]
    overall_passed = len(failed_checks) == 0
    summary = (
        "所有检查均通过。"
        if overall_passed
        else f"存在 {len(failed_checks)} 项未通过：{', '.join(failed_checks)}。"
    )
    nodes = list(state.get("nodes") or [])
    nodes.append(
        _node(
            "output",
            "completed",
            {"check_results": check_results},
            {
                "overall_passed": overall_passed,
                "failed_checks": failed_checks,
                "check_results": check_results,
                "summary": summary,
            },
        )
    )
    conversation = list(state.get("conversation") or [])
    conversation.append(
        {
            "role": "assistant",
            "message": summary,
            "review_errors": [
                {"node_id": item["node_id"], "errors": item["errors"]}
                for item in check_results
                if item["errors"]
            ],
            "created_at": _now(),
        }
    )
    return {
        "nodes": nodes,
        "conversation": conversation,
        "current_node": "output",
        "status": "completed",
        "check_passed": overall_passed,
        "check_results": check_results,
        "failed_checks": failed_checks,
        "review_passed": overall_passed,
        "review_errors": failed_checks,
        "review_feedback": summary,
    }


def run_check(payload: Dict[str, Any], message: str) -> ChapterCheckState:
    state: ChapterCheckState = {
        "action": "check",
        "message": message,
        "payload": dict(payload or {}),
        "pending_payload": dict(payload or {}),
        "nodes": [],
        "conversation": [{"role": "user", "message": message, "payload": payload, "created_at": _now()}],
        "committed": False,
        "review_required": True,
        "status": "running",
        "current_node": "input",
    }
    state.update(input_node(state))
    for spec in REVIEW_SEQUENCE:
        state.update(_append_review_node(state, spec))
    state.update(output_node(state))
    return state
