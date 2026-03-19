"""
Writing Execution Agent (正文执行 Agent) - PGA 小说创作引擎核心组件

本模块负责根据大纲和世界观设定，生成具体的小说章节正文。
设计思路:
1. 场次拆解 (Scene Breaking): 将大纲中的单一章节进一步细化为一系列具体场次，实现更精准的节奏控制。
2. 逻辑快照 (Logic Snapshot): 
   - 每次编写场次前，自动检索相关设定（RAG）。
   - 编写完成后，生成“人物状态”和“场景环境”的逻辑快照，确保下一场次写作时设定不偏航。
3. 视觉引导: 自动生成视觉快照描述，帮助创作者通过画面感进行微调。
4. 闭环审计: 对生成的正文进行逻辑矛盾审计（如人物突然出现在不可能出现的地方）。
"""
import os
import json
import datetime
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

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
class WritingState(TypedDict):
    """
    正文执行 Agent 运行时的状态机上下文。
    
    Attributes:
        outline_id: 关联的大纲 ID。
        scene_list: 本章节拆解后的场次列表。
        active_scene_index: 当前工作场次的索引。
        draft_content: 生成的正文初稿。
        char_status_summary: 场次结束后的角色状态快照。
        scene_status_summary: 场次结束后的场景环境快照。
        is_audit_passed: 逻辑审计是否通过。
    """
    # 输入信息
    outline_id: str             # 已保存的大纲 ID
    outline_content: str        # 大纲全文内容 (用于参考)
    current_act: str            # 当前编写的小说章节/幕
    
    # 过程数据
    scene_list: List[dict]      # 拆解后的场次清单 [{'title': '...', 'description': '...'}]
    active_scene_index: int     # 当前正在写的场次索引
    context_data: str           # 从 ChromaDB/MongoDB 检索出的背景知识
    
    # 输出数据
    draft_content: str          # 生成的初稿正文
    audit_feedback: str         # 审计意见/逻辑漏洞报告
    user_feedback: str          # 用户的人工意见
    is_audit_passed: bool       # 审计是否通过
    is_approved: bool           # 用户是否批准
    status_message: str         # 执行进度描述
    
    # 快照数据
    char_status_summary: str    # 人物逻辑快照 (位置、状态、动机)
    scene_status_summary: str   # 场景逻辑快照 (天气、物品毁损)
    visual_snapshot_path: str   # 视觉快照图片路径
    visual_description_summary: str # 视觉描述摘要

# ==========================================
# Nodes Implementation
# ==========================================

def plan_scenes_func(state: WritingState):
    """场次拆解节点 (Scene Planner)"""
    print(f"\n[DEBUG] plan_scenes_func entry. State keys: {list(state.keys())}")
    outline_content = state.get('outline_content', '')
    current_act = state.get('current_act', '')
    
    prompt = f"""你是一个专业的小说场次策划师。
你的任务是将大纲拆解为具体的“原子场次”。必须输出 JSON。

【大纲内容】
{outline_content}

【当前编写部分】
{current_act}

请输出 JSON：
{{
  "scene_list": [
    {{ "id": 1, "title": "标题", "description": "场次核心内容描述" }},
    ...
  ]
}}
"""
    res = get_llm(json_mode=True).invoke(prompt)
    data = parse_json_safely(res.content)
    if not data:
        return {"status_message": "场次拆解失败：JSON 解析异常"}
        
    scenes = data.get("scene_list", [])
    return {
        "scene_list": scenes,
        "active_scene_index": 0,
        "status_message": f"已拆解为 {len(scenes)} 个场次。准备编写第一场..."
    }

def load_context_func(state: WritingState):
    """语境加载节点"""
    print(f"\n[DEBUG] load_context_func entry. State keys: {list(state.keys())}")
    idx = state.get('active_scene_index', 0)
    scene_list = state.get('scene_list', [])
    
    if not scene_list or idx >= len(scene_list):
        return {"status_message": "语境加载异常：索引越界"}
        
    scene = scene_list[idx]
    query = f"{scene.get('title', '')} {scene.get('description', '')}"
    rag_context = get_unified_context(query)
    
    return {
        "context_data": rag_context,
        "status_message": f"正在处理第 {idx+1} 场：{scene.get('title', '')}。已加载背景语境。"
    }

def write_draft_func(state: WritingState):
    """正文生成节点"""
    print(f"\n[DEBUG] write_draft_func entry. State keys: {list(state.keys())}")
    idx = state.get('active_scene_index', 0)
    scene_list = state.get('scene_list', [])
    if not scene_list or idx >= len(scene_list):
        return {"status_message": "正文生成失败：索引越界"}
        
    scene = scene_list[idx]
    user_feedback = state.get('user_feedback', '')
    feedback_section = ""
    if user_feedback:
        feedback_section = f"\n【！！！当前核心修改需求！！！】\n要求：{user_feedback}\n"

    prompt = f"""你是一个创作专家。撰写具体的小说正文。

{feedback_section}

【大纲】{state.get('outline_content', '')}
【场次计划】{scene.get('title', '')}: {scene.get('description', '')}
【语境】{state.get('context_data', '')}

要求：文采斐然，符合 PGA 世界观，落实修改建议。直接输出正文。
"""
    res = get_llm().invoke(prompt)
    return {
        "draft_content": res.content,
        "user_feedback": "", # 清空已处理的反馈
        "status_message": f"第 {idx+1} 场初稿已完成，提交审计..."
    }

def audit_logic_func(state: WritingState):
    """逻辑审计节点"""
    print(f"\n[DEBUG] audit_logic_func entry. State keys: {list(state.keys())}")
    prohibited_items = get_prohibited_rules()
    
    idx = state.get('active_scene_index', 0)
    scene_list = state.get('scene_list', [])
    if not scene_list or idx >= len(scene_list):
        return {"is_audit_passed": False, "status_message": "审计信息缺失"}
        
    scene = scene_list[idx]
    worldview_rules = get_worldview_context_by_category(f"{scene.get('title', '')} {scene.get('description', '')}")
    
    char_status = state.get("char_status_summary", "无")
    
    prompt = f"""小说逻辑审计员。检查冲突。
禁令: {prohibited_items}
官方规则: {worldview_rules}
上场快照: {char_status}

待审计正文:
{state.get('draft_content', '')}

输出 JSON: {{"is_consistent": true/false, "audit_log": "..."}}
"""
    res = get_llm(json_mode=True).invoke(prompt)
    data = parse_json_safely(res.content)
    is_ok = data.get("is_consistent", False) if data else False
    
    return {
        "is_audit_passed": is_ok,
        "audit_feedback": data.get("audit_log", "解析异常") if data else "解析异常",
        "status_message": "审计通过。" if is_ok else "审计发现逻辑冲突。"
    }

def human_review_node(state: WritingState):
    """人工核准节点"""
    print(f"\n[DEBUG] human_review_node entry. State keys: {list(state.keys())}")
    draft = state.get('draft_content', '')
    user_input = interrupt({
        "status_message": "正文已生成，请核准或提供修改建议。",
        "proposal": draft
    })
    print(f"[DEBUG] human_review_node: Resumed. Received feedback: '{user_input}'")
    return {"user_feedback": user_input, "is_approved": user_input == "批准"}

def prose_saver_func(state: WritingState):
    """存档节点"""
    print(f"\n[DEBUG] prose_saver_func entry. State keys: {list(state.keys())}")
    idx = state.get('active_scene_index', 0)
    scene_list = state.get('scene_list', [])
    if not scene_list or idx >= len(scene_list):
        return {"status_message": "存档越界"}
        
    scene = scene_list[idx]
    record = {
        "id": f"prose_{state.get('outline_id', 'unknown')}_{idx}",
        "outline_id": state.get('outline_id', 'unknown'),
        "scene_title": scene.get('title', 'unknown'),
        "content": state.get('draft_content', ''),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open('prose_db.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    return {
        "active_scene_index": idx + 1,
        "is_approved": False, 
        "status_message": f"第 {idx+1} 场已保存，生成快照中..."
    }

def snapshot_node_func(state: WritingState):
    """快照生成节点"""
    print(f"\n[DEBUG] snapshot_node_func entry. State keys: {list(state.keys())}")
    idx = state.get('active_scene_index', 0) - 1 
    
    prompt = f"""提取快照及视觉描述。JSON。
正文: {state.get('draft_content', '')}
输出 JSON: {{"char_status": "...", "scene_status": "...", "visual_description": "..."}}
"""
    res = get_llm(json_mode=True).invoke(prompt)
    data = parse_json_safely(res.content)
    
    char_status = data.get("char_status", "正常") if data else "解析异常"
    visual_desc = data.get("visual_description", "") if data else ""
    
    return {
        "char_status_summary": char_status,
        "scene_status_summary": data.get("scene_status", "正常") if data else "解析异常",
        "visual_description_summary": visual_desc,
        "status_message": f"快照已生成。下一阶段准备中。"
    }

# ==========================================
# Graph Definition
# ==========================================
workflow = StateGraph(WritingState)

workflow.add_node("plan_scenes", plan_scenes_func)
workflow.add_node("load_context", load_context_func)
workflow.add_node("write_draft", write_draft_func)
workflow.add_node("audit_logic", audit_logic_func)
workflow.add_node("human_review", human_review_node)
workflow.add_node("prose_saver", prose_saver_func)
workflow.add_node("snapshot_node", snapshot_node_func)

workflow.add_edge(START, "plan_scenes")
workflow.add_edge("plan_scenes", "load_context")
workflow.add_edge("load_context", "write_draft")
workflow.add_edge("write_draft", "audit_logic")

def route_after_audit(state: WritingState):
    if state.get("is_audit_passed"): return "human_review"
    return "write_draft"

workflow.add_conditional_edges("audit_logic", route_after_audit, {"human_review": "human_review", "write_draft": "write_draft"})

def route_after_human(state: WritingState):
    fb = (state.get("user_feedback") or "").strip()
    if fb == "批准": return "prose_saver"
    if fb == "终止": return END
    if fb: return "write_draft" 
    return END

workflow.add_conditional_edges("human_review", route_after_human, {"prose_saver": "prose_saver", "write_draft": "write_draft", END: END})

workflow.add_edge("prose_saver", "snapshot_node")

def route_next_scene(state: WritingState):
    if state.get('active_scene_index', 0) < len(state.get('scene_list', [])): return "load_context"
    return END

workflow.add_conditional_edges("snapshot_node", route_next_scene, {"load_context": "load_context", END: END})

app = workflow.compile(checkpointer=MemorySaver())
