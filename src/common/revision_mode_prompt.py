"""Shared prompt guidance for human-selected revision modes.

The workflow exposes three distinct revision modes. Their prompt instructions
must stay mode-specific; collapsing them into one generic modify prompt is
forbidden because it breaks the user-selected scope contract.
"""

from __future__ import annotations

from typing import Optional


REVISION_MODE_LABELS = {
    "partial_rewrite": "指定局部重写",
    "content_rewrite": "指定内容重写",
    "full_rewrite": "完全重写",
    "summary_rewrite": "摘要重做",
}


def normalize_revision_mode(revision_mode: Optional[str]) -> str:
    value = str(revision_mode or "").strip()
    if value in REVISION_MODE_LABELS:
        return value
    return "partial_rewrite"


def revision_mode_label(revision_mode: Optional[str]) -> str:
    normalized = normalize_revision_mode(revision_mode)
    return REVISION_MODE_LABELS[normalized]


def build_revision_mode_instruction(
    revision_mode: Optional[str],
    *,
    entity_label: str,
    primary_field_label: str,
    content_rewrite_scope: str,
    full_rewrite_scope: str,
) -> str:
    normalized = normalize_revision_mode(revision_mode)
    label = revision_mode_label(normalized)

    if normalized == "partial_rewrite":
        return f"""当前模式：{normalized}（{label}）
本轮目标：只修正用户反馈或系统审查明确点名的局部问题。
范围边界：
* 只能修改与点名问题直接相关的局部片段或字段。
* 必须保留未点名的 {entity_label} 内容、结构和业务字段。
* `{primary_field_label}` 只允许在受影响位置做定点修正。
执行要求：
* 先定位问题，再在原位置修正，不得把局部问题扩大成整体重写。
* 如果必须补充说明，只能补充完成该局部修正所必需的最小内容。
禁止事项：
* 禁止整篇重写或整体改写结构。
* 禁止为了“顺手优化”修改未被点名的内容。
* 禁止改动任何业务 ID、父级绑定和未被允许的字段。"""

    if normalized == "content_rewrite":
        return f"""当前模式：{normalized}（{label}）
本轮目标：围绕主要内容字段做成段或全文级重写，但范围只限于用户指定的内容主体。
范围边界：
* 允许重点重写 `{primary_field_label}`。
* {content_rewrite_scope}
* 除 `{primary_field_label}` 及保持一致性所必需的最小联动字段外，其他字段不得改动。
执行要求：
* 必须完整落实用户反馈和系统审查问题，但仍然遵守父级约束和既有业务边界。
* 如 `name`、标题或其他辅助字段未被明确要求修改，默认保留原值。
禁止事项：
* 禁止把内容重写扩大成整个实体的重新创作。
* 禁止改动任何业务 ID、父级绑定和无关字段。"""

    return f"""当前模式：{normalized}（{label}）
本轮目标：允许对当前 {entity_label} 的业务内容整体重写，但仍必须服从用户意见和父级约束。
范围边界：
* 允许整体重组 `{primary_field_label}` 以及与之直接相关的业务内容。
* {full_rewrite_scope}
* 无论怎么重写，都必须保留业务 ID、父级绑定、路由字段和必要主键。
执行要求：
* 必须完整覆盖用户反馈和系统审查问题，不能只改措辞。
* 整体重写后仍要保持逻辑自洽、约束一致、字段结构合法。
禁止事项：
* 禁止脱离父级约束另起设定。
* 禁止删除必要业务字段或伪造不存在的绑定关系。"""


def build_summary_revision_mode_instruction(
    revision_mode: Optional[str],
    *,
    entity_label: str,
    source_field_label: str,
    output_field_labels: str,
) -> str:
    normalized = str(revision_mode or "").strip() or "summary_rewrite"
    if normalized != "summary_rewrite":
        normalized = "summary_rewrite"
    label = revision_mode_label(normalized)
    return f"""当前模式：{normalized}（{label}）
本轮目标：保留原始来源文本不动，重做下游摘要结果。
范围边界：
* 只允许重写 {output_field_labels}。
* 必须保留 `{source_field_label}` 原文，不得改写、压缩、删减或回传伪造原文。
* 业务 ID、父级绑定、name 和目标实体标识必须保留。
执行要求：
* 必须逐条落实人工反馈和审查问题，重点修复过度简化、遗漏关键冲突、遗漏结果、没有真正压缩等问题。
* 新摘要必须比原文更短，但不能靠丢主线、丢因果、丢结果来换取长度。
禁止事项：
* 禁止把返工任务扩大成原文重写。
* 禁止回传未经要求的原始正文或大纲全文。
* 禁止为凑字数添加原文中不存在的剧情或设定。"""
