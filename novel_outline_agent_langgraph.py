"""
Novel Outline Agent (小说大纲 Agent) - PGA 小说创作引擎核心组件

本模块负责根据用户需求和已有的世界观设定（Lore），生成结构化的小说大纲。
设计思路:
1. 节拍控制: 强制生成符合标准文学结构的大纲（如三幕式或英雄之旅）。
2. 设定对齐: 自动从 RAG 文献库中提取世界观禁令与规则，确保剧情不“吃书”。
3. 冲突驱动: 分析设定中的矛盾点（如不同势力的能量争端），将其转化为核心剧情冲突。
"""
import os
import json
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import datetime

# Import shared utilities
from lore_utils import (
    get_llm, 
    get_vector_store, 
    get_prohibited_rules, 
    get_worldview_context_by_category, 
    get_unified_context,
    parse_json_safely
)

# ==========================================
# 0. State Definition
# ==========================================
class OutlineState(TypedDict):
    """
    大纲 Agent 运行时的状态机上下文。
    
    Attributes:
        query: 用户对小说的初步设想或修改要求。
        context: 检索到的相关世界观背景知识。
        proposal: Agent 生成的当前大纲提案。
        review_log: 逻辑审计日志。
        user_feedback: 用户对大纲的反馈。
        is_approved: 是否通过逻辑审核或人类确认。
        status_message: 进度描述。
    """
    query: str             # 用户的小说想法/需求
    context: str           # 检索到的世界观背景
    proposal: str          # 当前生成的大纲草案
    review_log: str        # 逻辑审计日志
    user_feedback: str     # 用户的调整意见
    iterations: int        # 总生成次数
    audit_count: int       # 当前自审重试次数
    is_approved: bool      # 是否通过审核/用户批准
    status_message: str    # 执行进度描述

# ==========================================
# Nodes Implementation
# ==========================================

def outline_planner(state: OutlineState):
    """大纲策划节点"""
    print(f"\n[DEBUG] outline_planner entry. State keys: {list(state.keys())}")
    query = state.get('query', '')
    if not query:
        print("[WARNING] 'query' is missing in state at outline_planner!")
    
    # 检索世界观
    rag_context = get_unified_context(query)
    
    feedback_section = ""
    user_feedback = state.get('user_feedback', '')
    if user_feedback:
        feedback_section = f"""
【！！！当前核心修改需求 - 必须首先满足！！！】
用户提出以下问题或要求：
>>> {user_feedback} <<<
你必须在本次生成中深刻结合该反馈调整大纲。
"""

    prompt = f"""你是一个专精于 PGA 世界观的小说策划专家。
你的任务是根据用户的需求编写详尽的小说大纲。必须输出为 JSON。

{feedback_section}

【JSON Schema】
{{
  "meta_info": {{"title": "...", "genre": [...], "tone": "...", "writing_style": "..."}},
  "core_hook": {{"logline": "...", "inciting_incident": "...", "core_conflict": "..."}},
  "world_building_ref": {{"base_rules": "...", "key_locations": [...], "power_system": "..."}},
  "character_roster": [{{"name": "...", "role": "...", "motivation": "...", "internal_flaw": "...", "character_arc": "..."}}],
  "plot_beats": {{"act_1": "...", "midpoint": "...", "climax": "...", "resolution": "..."}},
  "themes": [...]
}}

背景参考: {rag_context}
需求描述: {query}
审计建议: {state.get('review_log', '无')}

请输出符合 Schema 的 JSON。
"""
    res = get_llm(json_mode=True).invoke(prompt)
    curr_iterations = state.get('iterations', 0)
    
    return {
        "proposal": res.content,
        "iterations": int(curr_iterations) + 1,
        "status_message": "大纲提案已生成，进入审计流程..."
    }

def outline_critic(state: OutlineState):
    """大纲审计节点"""
    print(f"\n[DEBUG] outline_critic entry. State keys: {list(state.keys())}")
    query = state.get('query', '')
    proposal = state.get('proposal', '')
    if not proposal:
        print("[WARNING] proposal is missing at outline_critic!")
        
    prohibited_items = get_prohibited_rules()
    worldview_rules = get_worldview_context_by_category(query)
    
    prompt = f"""你是一个 PGA 世界观与故事逻辑审核官。
必须输出 JSON。

最高禁令: {prohibited_items}
官方定义: {worldview_rules}

待审核大纲：
{proposal}

请审核一致性、完整性与动机，输出 JSON: {{"status": "合理/不合理", "audit_log": "..."}}
"""
    res = get_llm(json_mode=True).invoke(prompt)
    try:
        audit_data = parse_json_safely(res.content)
        is_ok = audit_data.get("status") == "合理"
        count = state.get('audit_count', 0)
        return {
            "review_log": audit_data.get("audit_log", ""),
            "is_approved": is_ok,
            "audit_count": int(count) + 1,
            "status_message": "剧本审计完成。"
        }
    except Exception:
        return {"is_approved": False, "audit_count": int(state.get('audit_count', 0)) + 1, "status_message": "审计解析异常"}

def human_gate(state: OutlineState):
    """人工核准节点"""
    print(f"\n[DEBUG] human_gate entry. State keys: {list(state.keys())}")
    proposal = state.get('proposal', '')
    user_input = interrupt({"status_message": "大纲已就绪，等待您的调整或批准...", "proposal": proposal})
    print(f"[DEBUG] human_gate: Resumed. Received feedback: '{user_input}'")
    return {"user_feedback": user_input, "is_approved": user_input == "批准", "status_message": "正在处理反馈..."}

def outline_saver(state: OutlineState):
    """存档节点"""
    print(f"\n[DEBUG] outline_saver entry. State keys: {list(state.keys())}")
    proposal = state.get('proposal', '')
    try:
        outline_data = parse_json_safely(proposal)
        record = {
            "id": f"outline_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now().isoformat(),
            "query": state.get('query', ''),
            "outline": outline_data,
            "iterations": state.get('iterations', 0)
        }
        with open('outlines_db.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status_message": "大纲已存档至 outlines_db.json。"}
    except Exception as e:
        return {"status_message": f"存档失败: {str(e)}"}

# ==========================================
# Graph Definition
# ==========================================
workflow = StateGraph(OutlineState)
workflow.add_node("planner", outline_planner)
workflow.add_node("critic", outline_critic)
workflow.add_node("human", human_gate)
workflow.add_node("saver", outline_saver)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "critic")

def route_after_critic(state: OutlineState):
    if state.get("is_approved") or int(state.get("audit_count", 0)) >= 3: return "human"
    return "planner"

workflow.add_conditional_edges("critic", route_after_critic, {"human": "human", "planner": "planner"})

def route_after_human(state: OutlineState):
    fb = state.get("user_feedback", "").strip()
    if fb == "批准": return "saver"
    if fb == "终止": return END
    if fb: return "planner"
    return END

workflow.add_conditional_edges("human", route_after_human, {"saver": "saver", "planner": "planner", END: END})
workflow.add_edge("saver", END)

app = workflow.compile(checkpointer=MemorySaver())
