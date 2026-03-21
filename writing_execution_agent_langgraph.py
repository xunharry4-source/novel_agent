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
    get_grounded_context,
    format_grounded_context_for_prompt,
    parse_json_safely,
    get_entity_registry, 
    format_entity_registry_for_prompt,
    register_draft_entity, 
    get_category_template
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
    grounding_sources: List[dict] # 绑定的源素材索引
    
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
    """语境加载节点 - 融合分布式 SKILL"""
    print(f"\n[DEBUG] load_context_func entry. State keys: {list(state.keys())}")
    
    # 显式类型防护以修复 Pyre2 报警
    val = state.get('active_scene_index')
    idx = int(val) if isinstance(val, (int, str)) else 0
    
    scene_list = state.get('scene_list')
    if not isinstance(scene_list, list):
        scene_list = []
    
    if idx >= len(scene_list):
        return {"status_message": "语境加载异常：索引越界"}
        
    scene = scene_list[idx]
    if not isinstance(scene, dict):
        return {"status_message": "语境加载异常：场次格式错误"}
        
    query = f"{str(scene.get('title') or '')} {str(scene.get('description') or '')}"
    
    # 1. 基础 RAG 检索 (带索引的 Grounding 模式)
    sources = get_grounded_context(query)
    grounded_context_str = format_grounded_context_for_prompt(sources)
    
    # 2. 加载分布式 SKILL (高优控制)
    skills_context = ""
    try:
        # 加载剧情锚点 (宪法)
        with open('.gemini/skills/lore/ANCHORS.md', 'r', encoding='utf-8') as f:
            skills_context += f"\n【剧情锚点 (不可违背)】\n{f.read()}\n"
        # 加载活跃窗口 (施工图)
        with open('.gemini/skills/catalog/ACTIVE_WINDOW.md', 'r', encoding='utf-8') as f:
            skills_context += f"\n【活跃章节窗口 (当前目标)】\n{f.read()}\n"
    except Exception as e:
        print(f"[WARNING] Loading skills failed: {e}")
        skills_context = "\n[警告] 未能加载分布式 SKILL 约束，仅依靠 RAG 语境。\n"

    # 3. 加载实体注册表 (A 层约束)
    entity_registry = get_entity_registry()
    entity_constraint = format_entity_registry_for_prompt(entity_registry)

    return {
        "context_data": f"{skills_context}\n{entity_constraint}\n{grounded_context_str}",
        "grounding_sources": sources,
        "status_message": f"正在处理第 {idx+1} 场：{scene.get('title', '')}。已加载 {len(sources)} 条素材锚定。"
    }

def write_draft_func(state: WritingState):
    """正文生成节点"""
    print(f"\n[DEBUG] write_draft_func entry. State keys: {list(state.keys())}")
    
    val = state.get('active_scene_index')
    idx = int(val) if isinstance(val, (int, str)) else 0
    
    scene_list = state.get('scene_list')
    if not isinstance(scene_list, list) or idx >= len(scene_list):
        return {"status_message": "正文生成失败：索引越界"}
        
    scene = scene_list[idx]
    if not isinstance(scene, dict):
        return {"status_message": "正文生成失败：场次格式错误"}
        
    user_feedback = str(state.get('user_feedback') or '')
    feedback_section = ""
    if user_feedback:
        feedback_section = f"\n【！！！当前核心修改需求！！！】\n要求：{user_feedback}\n"

    prompt = f"""你是一个创作专家。撰写具体的小说正文。

{feedback_section}

【大纲】{state.get('outline_content', '')}
【场次计划】{str(scene.get('title') or '')}: {str(scene.get('description') or '')}
【语境】{state.get('context_data', '')}

要求：文采斐然，落实修改建议。
直接输出正文。

【重要：素材锚定规则】
1. 当你描写涉及世界观背景、科技原理、历史事件或硬性设定时，必须在对应的描述末尾标注来源索引编号，例如：[S1], [S3]。
2. 不要为了标注而标注，只有在使用了提供的源素材（References）中的特定事实时才需要标注。
3. 文学性的描写、抒情、对话（非设定解释类）无需标注。
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
    
    val = state.get('active_scene_index')
    idx = int(val) if isinstance(val, (int, str)) else 0
    
    scene_list = state.get('scene_list')
    if not isinstance(scene_list, list) or idx >= len(scene_list):
        return {"is_audit_passed": False, "status_message": "审计信息缺失"}
        
    scene = scene_list[idx]
    if not isinstance(scene, dict):
        return {"is_audit_passed": False, "status_message": "审计信息格式错误"}
        
    worldview_rules = get_worldview_context_by_category(f"{str(scene.get('title') or '')} {str(scene.get('description') or '')}")
    
    char_status = state.get("char_status_summary", "无")
    
    prompt = f"""小说逻辑与素材锚定审计员。检查冲突与引用准确性。
禁令: {prohibited_items}

【提供的源素材】
{format_grounded_context_for_prompt(state.get('grounding_sources', []))}

官方规则: {worldview_rules}
上场快照: {char_status}

待审计正文:
{state.get('draft_content', '')}

【审计任务】
1. 检查正文是否违背了禁令或官方规则。
2. 核查正文中的 [SX] 引用是否与素材内容匹配。
3. 识别出正文中提及了特定设定但未标注引用、或标注了引用但素材中找不到对应事实的现象。

输出 JSON: {"is_consistent": true/false, "audit_log": "...", "grounding_score": 0-100}
"""
    res = get_llm(json_mode=True).invoke(prompt)
    data = parse_json_safely(res.content)
    is_ok = data.get("is_consistent", False) if data else False
    
    # B 层：在审计同时检测新实体 (增强型：带模板生成)
    entity_warning = ""
    try:
        registry = get_entity_registry()
        known_names = set()
        for names_list in registry.values():
            known_names.update(names_list)
        
        draft_content = str(state.get('draft_content') or '')
        extract_prompt = f"""从以下小说正文中提取所有专有名词实体（人物、势力、种族、科技、地点）。
只输出 JSON 数组：[{{"name": "实体名", "type": "character/faction/race/tech/location/other"}}]

正文：
{draft_content[:2000]}
"""
        ent_res = get_llm(json_mode=True).invoke(extract_prompt)
        extracted = parse_json_safely(ent_res.content)
        
        if isinstance(extracted, list):
            new_count = 0
            type_to_category = {
                "character": "general", "faction": "faction", "organization": "organization",
                "race": "race", "tech": "mechanism_tech", "mechanism": "mechanism_tech",
                "location": "geography", "planet": "planet", "weapon": "weapon",
                "creature": "creature", "religion": "religion", "history": "history", "crisis": "crisis"
            }
            
            for ent in extracted:
                if not isinstance(ent, dict): continue
                name = str(ent.get('name') or '')
                etype = str(ent.get('type') or 'other')
                if name and name not in known_names:
                    # 增强逻辑：生成模板化设定卡
                    category = type_to_category.get(etype, "general")
                    template_data = get_category_template(category)
                    template_str = json.dumps(template_data.get("template", {}), ensure_ascii=False) if template_data else "{}"
                    
                    card_prompt = f"""你是一个世界观架构师。请根据以下正文片段，为新发现的实体【{name}】（分类：{category}）生成一份结构化的设定卡。
必须遵循以下模板格式，并且必须符合 PGA 世界观规则。

【实体名】: {name}
【分类】: {category}
【参考模板】: {template_str}

【正文片段】:
{draft_content[:1500]}

TASK: 请输出该实体的完整 JSON 设定，必须匹配模板字段。
"""
                    card_res = get_llm(json_mode=True).invoke(card_prompt)
                    entity_card = parse_json_safely(card_res.content) or {"name": name, "description": "自动提取"}
                    
                    register_draft_entity(
                        entity_name=name, 
                        entity_type=category, 
                        source_context="正文中首次出现", 
                        source_agent="writing",
                        entity_card=entity_card
                    )
                    new_count += 1
            if new_count > 0:
                entity_warning = f" 同时发现 {new_count} 个未注册实体并已按模板生成设定卡，登记待审。"
    except Exception as e:
        print(f"[Entity Sentinel] 正文实体检测异常: {e}")
    
    return {
        "is_audit_passed": is_ok,
        "audit_feedback": data.get("audit_log", "解析异常") if data else "解析异常",
        "status_message": ("审计通过。" if is_ok else "审计发现逻辑冲突。") + entity_warning
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
    
    val = state.get('active_scene_index')
    idx = int(val) if isinstance(val, (int, str)) else 0
    
    scene_list = state.get('scene_list')
    if not isinstance(scene_list, list) or idx >= len(scene_list):
        return {"status_message": "存档越界"}
        
    scene = scene_list[idx]
    if not isinstance(scene, dict):
        return {"status_message": "存档场次格式错误"}

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
    
    val = state.get('active_scene_index')
    idx = (int(val) if isinstance(val, (int, str)) else 0) - 1 
    
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
    fb = str(state.get("user_feedback") or "").strip()
    if fb == "批准": return "prose_saver"
    if fb == "终止": return END
    if fb: return "write_draft" 
    return END

workflow.add_conditional_edges("human_review", route_after_human, {"prose_saver": "prose_saver", "write_draft": "write_draft", END: END})

workflow.add_edge("prose_saver", "snapshot_node")

def route_next_scene(state: WritingState):
    _idx_val = state.get('active_scene_index')
    idx = int(_idx_val) if isinstance(_idx_val, (int, str)) else 0
    scene_list = state.get('scene_list')
    total = len(scene_list) if isinstance(scene_list, list) else 0
    if idx < total: return "load_context"
    return END

workflow.add_conditional_edges("snapshot_node", route_next_scene, {"load_context": "load_context", END: END})

app = workflow.compile(checkpointer=MemorySaver())
