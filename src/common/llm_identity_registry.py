"""Central registry for node-level LLM identities.

Every LLM-calling workflow node must own a unique `llm_agent_name`.
Shared identities across different nodes are forbidden.
"""

from __future__ import annotations

from typing import Dict, List


WORKFLOW_NODE_LLM_IDENTITIES: Dict[str, Dict[str, str]] = {
    "world": {
        "initial_expansion": "world_agent_initial_expansion",
        "modify_content": "world_agent_modify_content",
    },
    "worldview": {
        "initial_expansion": "worldview_agent_initial_expansion",
        "modify_content": "worldview_agent_modify_content",
        "modify_content_manual_edit": "worldview_agent_human_feedback_modify_content",
    },
    "novel": {
        "initial_expansion": "novel_agent_initial_expansion",
        "modify_content": "novel_agent_modify_content",
        "modify_content_manual_edit": "novel_agent_human_feedback_modify_content",
    },
    "outline": {
        "initial_expansion": "outline_agent_initial_expansion",
        "modify_content": "outline_agent_modify_content",
        "modify_content_manual_edit": "outline_agent_human_feedback_modify_content",
    },
    "chapter": {
        "initial_expansion": "chapter_agent_initial_expansion",
        "modify_content": "chapter_agent_modify_content",
        "modify_content_manual_edit": "chapter_agent_human_feedback_modify_content",
    },
    "outline_summary_create": {
        "initial_expansion": "outline_summary_create_llm",
        "modify_content": "outline_summary_create_modify_llm",
        "modify_content_manual_edit": "outline_summary_create_human_feedback_modify_llm",
    },
    "outline_summary_update": {
        "initial_expansion": "outline_summary_update_llm",
        "modify_content": "outline_summary_update_modify_llm",
        "modify_content_manual_edit": "outline_summary_update_human_feedback_modify_llm",
    },
    "chapter_outline_summary_create": {
        "initial_expansion": "chapter_outline_summary_create_llm",
        "modify_content": "chapter_outline_summary_create_modify_llm",
        "modify_content_manual_edit": "chapter_outline_summary_create_human_feedback_modify_llm",
    },
    "chapter_outline_summary_update": {
        "initial_expansion": "chapter_outline_summary_update_llm",
        "modify_content": "chapter_outline_summary_update_modify_llm",
        "modify_content_manual_edit": "chapter_outline_summary_update_human_feedback_modify_llm",
    },
    "chapter_content_summary_create": {
        "initial_expansion": "chapter_content_summary_create_llm",
        "modify_content": "chapter_content_summary_create_modify_llm",
        "modify_content_manual_edit": "chapter_content_summary_create_human_feedback_modify_llm",
    },
    "chapter_content_summary_update": {
        "initial_expansion": "chapter_content_summary_update_llm",
        "modify_content": "chapter_content_summary_update_modify_llm",
        "modify_content_manual_edit": "chapter_content_summary_update_human_feedback_modify_llm",
    },
}

AGENT_TYPE_ALIASES: Dict[str, str] = {
    "chapter_intro_summary_create": "chapter_content_summary_create",
    "chapter_intro_summary_update": "chapter_content_summary_update",
}

REVIEW_NODE_LLM_IDENTITIES: Dict[str, str] = {
    "worldview_world_rules": "worldview_world_rules_review_agent",
    "worldview_consistency": "worldview_consistency_review_agent",
    "novel_world_rules": "novel_world_rules_review_agent",
    "outline_world_rules": "outline_world_rules_review_agent",
    "outline_worldview_rules": "outline_worldview_rules_review_agent",
    "outline_novel_rules": "outline_novel_rules_review_agent",
    "chapter_world_rules": "chapter_world_rules_review_agent",
    "chapter_worldview_rules": "chapter_worldview_rules_review_agent",
    "chapter_novel_rules": "chapter_novel_rules_review_agent",
    "chapter_outline_rules": "chapter_outline_rules_review_agent",
    "chapter_chapter_outline_rules": "chapter_chapter_outline_rules_review_agent",
    "chapter_consistency": "chapter_consistency_review_agent",
    "chapter_plot_errors": "chapter_plot_errors_review_agent",
}

LEGACY_SHARED_LLM_AGENT_NAMES = {
    "world_agent",
    "worldview_agent",
    "novel_agent",
    "outline_agent",
    "chapter_agent",
    "outline_summary_create_agent",
    "outline_summary_update_agent",
    "chapter_outline_summary_create_agent",
    "chapter_outline_summary_update_agent",
    "chapter_content_summary_create_agent",
    "chapter_content_summary_update_agent",
}


def expected_llm_agent_name(agent_type: str, node_id: str, *, manual_edit: bool = False) -> str:
    canonical_agent_type = AGENT_TYPE_ALIASES.get(agent_type, agent_type)
    node_map = WORKFLOW_NODE_LLM_IDENTITIES.get(canonical_agent_type, {})
    if manual_edit and node_id == "modify_content":
        return node_map.get("modify_content_manual_edit") or node_map.get(node_id, "")
    return node_map.get(node_id, "")


def all_node_llm_agent_names() -> List[str]:
    names: List[str] = []
    for node_map in WORKFLOW_NODE_LLM_IDENTITIES.values():
        names.extend(node_map.values())
    names.extend(REVIEW_NODE_LLM_IDENTITIES.values())
    return names


def validate_llm_identity_registry(agent_models: dict | None = None) -> dict:
    names = all_node_llm_agent_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing_agent_model_keys: List[str] = []
    if agent_models is not None:
        missing_agent_model_keys = sorted(name for name in names if name not in agent_models)
    result = {
        "all_names": names,
        "duplicates": duplicates,
        "missing_agent_model_keys": missing_agent_model_keys,
    }
    if duplicates:
        raise ValueError(f"Duplicate llm_agent_name detected across workflow nodes: {duplicates}")
    if missing_agent_model_keys:
        raise ValueError(
            "AGENT_MODELS missing dedicated llm_agent_name entries: "
            f"{missing_agent_model_keys}"
        )
    return result
