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
    mode: str             # 模式: 'book' 或 'chapter'
    grounding_sources: List[dict] # 绑定的源素材索引

# ==========================================
# Nodes Implementation
# ==========================================

def outline_planner(state: OutlineState):
    """大纲策划节点 (支持全局与分章细化)"""
    print(f"\n[DEBUG] outline_planner entry. State keys: {list(state.keys())}")
    query = state.get('query') or ""
    
    # 1. 意图与上下文识别
    is_chapter_detail = any(x in (query or "") for x in ["第", "章", "细化", "场景"])
    
    # 检索世界观背景 (带索引的 Grounding 模式)
    sources = get_grounded_context(query)
    grounded_context_str = format_grounded_context_for_prompt(sources)
    
    # 如果是细化章节，尝试获取全局大纲作为背景
    book_context = ""
    target_chapter_info = ""
    if is_chapter_detail:
        from lore_utils import get_latest_book_outline
        latest_book = get_latest_book_outline()
        if latest_book:
            outline = latest_book.get('outline', {})
            book_context = f"\n【全局大纲参考】\n{json.dumps(outline, ensure_ascii=False)}"
            
            # 尝试从目录中提取本章的既定目标 (基于第 X 章 或 章节标题)
            catalog = outline.get('chapter_list', []) if isinstance(outline.get('chapter_list'), list) else []
            import re
            chapter_num_match = re.search(r'第\s*(\d+|[一二三四五六七八九十]+)\s*章', (query or ""))
            if chapter_num_match:
                chapter_ref = chapter_num_match.group(1)
                for ch in catalog:
                    if not isinstance(ch, dict): continue
                    ch_title = ch.get('title')
                    if str(ch.get('chapter_num')) == chapter_ref or (isinstance(ch_title, str) and ch_title in (query or "")):
                        target_chapter_info = f"\n【既定章节目标 - 必须遵循】\n标题: {ch_title}\n既定梗概: {ch.get('summary')}\n本章核心功能: {ch.get('focus')}\n"
                        break

    feedback_section = ""
    user_feedback = state.get('user_feedback', '')
    prev_proposal = state.get('proposal', '')
    if user_feedback:
        feedback_section = f"\n【！！！当前修改需求！！！】\n{user_feedback}\n"
        if prev_proposal:
            feedback_section += f"\n【上一次生成的草案 - 请在次基础上进行增量修改】\n{prev_proposal}\n"

    # 根据模式选择 Schema 和提示词
    if is_chapter_detail:
        schema = """{
  "chapter_info": {"id": "章节号", "title": "标题", "theme": "本章主题"},
  "plot_beats": [
    {"id": 1, "action": "具体情节动作 (必须附带引用 [SX])", "logic_reason": "为什么要发生这个"},
    {"id": 2, "action": "...", "logic_reason": "..."}
  ],
  "character_snapshot": "本章结束时角色应具备的状态",
  "worldview_alignment": "本章涉及的关键设定细节 (必须附带引用 [SX])"
}"""
        mode_desc = "你正在进行【章节级细化策划】。请编写该章节详尽的剧情步进（Plot Beats），为后续的正文创作提供原子级的逻辑支撑。"
    else:
        schema = """{
  "meta_info": {"title": "...", "genre": [...], "tone": "...", "writing_style": "---"},
  "plot_beats": {"act_1": "提及关键设定时必须附带 [SX]", "midpoint": "...", "climax": "...", "resolution": "---"},
  "chapter_list": [
    {"chapter_num": 1, "title": "章节标题", "summary": "关键事实必须引用 [SX]", "focus": "..."}
  ]
}"""
        mode_desc = "你正在进行【全局小说策划】。请构建完整的故事框架、人物志、核心冲突链路，并以此为基础生成一份详尽的【章节目录】，为后续的分章创作提供指引。"

    # A 层：注入已注册实体清单
    entity_registry = get_entity_registry()
    entity_constraint = format_entity_registry_for_prompt(entity_registry)
    
    prompt = f"""你是一个专精于 PGA 世界观的小说策划专家。
{mode_desc}

【重要：素材锚定规则 (Grounding Rules)】
1. 你的所有生成内容必须严格基于下面提供的“可用源素材”。
2. 当你引用或利用源素材中的设定、历史或规则时，必须在对应的描述末尾标注来源索引编号，例如：[S1], [S3]。
3. 如果某些内容是你的文学虚构（非世界观事实），则无需标注索引，但必须确保不违背 [S] 系列中的任何禁令。

{feedback_section}
{target_chapter_info}
{book_context}

{entity_constraint}

{grounded_context_str}

【任务要求】
必须输出为 JSON 格式。
Schema: {schema}

【用户输入】
需求描述: {query}
审计建议: {state.get('review_log', '无')}

请输出符合 Schema 的 JSON。
"""
    res = get_llm(json_mode=True).invoke(prompt)
    curr_iterations = state.get('iterations', 0)
    
    return {
        "proposal": res.content,
        "grounding_sources": sources,
        "iterations": int(curr_iterations) + 1,
        "mode": "chapter" if is_chapter_detail else "book",
        "status_message": f"模式: {'章节细化' if is_chapter_detail else '全局策划'}。已引用 {len(sources)} 条素材进行锚定。"
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
    user_input = interrupt({
        "status_message": "大纲与章节目录已就绪。你可以直接批准，也可以针对全局大纲或【章节目录】中的任何项提出修改意见。",
        "proposal": proposal
    })
    print(f"[DEBUG] human_gate: Resumed. Received feedback: '{user_input}'")
    return {"user_feedback": user_input, "is_approved": user_input == "批准", "status_message": "正在处理反馈..."}

def outline_saver(state: OutlineState):
    """存档与同步节点 - 实现“批准即刷 SKILL”"""
    print(f"\n[DEBUG] outline_saver entry. State keys: {list(state.keys())}")
    proposal_str = state.get('proposal', '') # proposal is a string from LLM
    
    # Parse the proposal string into a dictionary
    try:
        proposal_dict = parse_json_safely(proposal_str)
    except Exception as e:
        return {"status_message": f"存档失败: 无法解析大纲提案 JSON: {str(e)}"}

    # 1. 存入 JSON 数据库
    db_record = {
        "id": f"outline_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(),
        "query": state.get('query', ''), # Keep query in record
        "mode": state.get('mode', 'book'),
        "outline": proposal_dict,
        "iterations": state.get('iterations', 0) # Keep iterations in record
    }
    
    with open('outlines_db.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(db_record, ensure_ascii=False) + "\n")
    
    # 2. 自动化触发 SKILL 转换 (分布式切片)
    if state.get('mode') == 'book':
        try:
            from lore_skill_converter import generate_modular_skills
            # 默认同步到第 1 章活跃窗口，或根据 state 里的进度决定
            generate_modular_skills(current_chapter_idx=1)
            print("[INFO] Modular SKILLs refreshed automatically.")
        except Exception as e:
            print(f"[ERROR] Failed to refresh SKILLs: {e}")

    return {"status_message": "大纲已确立并导出为分布式 SKILL 协议。"}

# ==========================================
# Grounding Audit (素材锚定审计)
# ==========================================
def grounding_audit_node(state: OutlineState):
    """
    基于 NotebookLM 逻辑的素材锚定审计。
    核查提案中所有的 [SX] 引用是否真实、准确。
    """
    print(f"\n[DEBUG] grounding_audit_node entry.")
    proposal = state.get('proposal', '')
    sources = state.get('grounding_sources', [])
    
    if not sources:
        return {"status_message": "无素材可审计，跳过。"}

    sources_str = format_grounded_context_for_prompt(sources)
    
    prompt = f"""你是一个严谨的文档审计员，负责核查小说大纲对【源素材】的引用真实性。
    
{sources_str}

【待审计大纲】
{proposal}

【审计任务】
1. 提取大纲中所有的 [SX] 引用。
2. 逐一核查引用内容是否在对应的 [SX] 素材中真实存在。
3. 识别出“过度发挥”或“凭空捏造”的设定（即：标了引用但在素材中找不到，或没标引用但涉及了核心设定）。

请输出 JSON 格式：
{{
    "citations_found": ["...", "..."],
    "valid_score": 0-100, 
    "hallucinations": ["发现以下凭空捏造的内容...", "..."],
    "status": "通过/不通过"
}}
"""
    try:
        res = get_llm(json_mode=True).invoke(prompt)
        audit_res = parse_json_safely(res.content)
        
        is_grounded = audit_res.get("status") == "通过"
        hallucination_log = "\n".join(audit_res.get("hallucinations", []))
        
        # 将审计结果存入 review_log
        prev_log = state.get('review_log', '')
        new_log = f"{prev_log}\n\n【素材锚定审计结果】\n状态: {audit_res.get('status')}\n有效分: {audit_res.get('valid_score')}\n问题点: {hallucination_log}"
        
        return {
            "review_log": new_log,
            "is_approved": is_grounded and state.get('is_approved', False), # 必须同时通过逻辑审计
            "status_message": f"素材锚定审计完成。置信分: {audit_res.get('valid_score')}/100"
        }
    except Exception as e:
        print(f"[Grounding Audit] Error: {e}")
        return {"status_message": "素材锚定审计解析异常"}

# ==========================================
# B 层：实体哨兵审计节点
# ==========================================
def entity_sentinel_node(state: OutlineState):
    """实体哨兵节点 (B 层) - 从大纲提案中提取实体并与注册表比对"""
    print(f"\n[DEBUG] entity_sentinel_node entry. State keys: {list(state.keys())}")
    proposal = str(state.get('proposal') or '')
    if not proposal:
        return {"status_message": "实体哨兵：无提案可审计"}
    
    # 1. 获取当前已注册实体（扁平化为名称集合）
    registry = get_entity_registry()
    known_names = set()
    for names_list in registry.values():
        known_names.update(names_list)
    
    # 2. 使用 LLM 从提案中提取所有实体
    extract_prompt = f"""你是一个实体提取专家。从以下小说大纲中提取所有提及的实体（人物、势力、种族、科技、地点等）。
只提取专有名词，不提取通用描述。
必须输出 JSON 数组格式：[{{"name": "实体名", "type": "character/faction/race/tech/location/other"}}]

大纲内容：
{proposal[:3000]}
"""
    try:
        res = get_llm(json_mode=True).invoke(extract_prompt)
        extracted = parse_json_safely(res.content)
        if not isinstance(extracted, list):
            extracted = []
    except Exception as e:
        print(f"[Entity Sentinel] 提取失败: {e}")
        return {"status_message": "实体哨兵：实体提取异常，跳过"}
    
    # 3. 差集比对并生成完整设定卡
    new_entities = []
    # 实体类型到世界观分类的映射
    type_to_category = {
        "character": "general", # 目前人物放在 general 或可由用户扩展
        "faction": "faction",
        "organization": "organization",
        "race": "race",
        "tech": "mechanism_tech",
        "mechanism": "mechanism_tech",
        "location": "geography",
        "planet": "planet",
        "weapon": "weapon",
        "creature": "creature",
        "religion": "religion",
        "history": "history",
        "crisis": "crisis"
    }

    for ent in extracted:
        if not isinstance(ent, dict):
            continue
        name = str(ent.get('name') or '')
        etype = str(ent.get('type') or 'other')
        if name and name not in known_names:
            # 找到对应分类并获取模板
            category = type_to_category.get(etype, "general")
            template_data = get_category_template(category)
            template_str = json.dumps(template_data.get("template", {}), ensure_ascii=False) if template_data else "{}"
            
            # 4. 第二次 LLM pass：根据模板生成完整的实体卡
            card_prompt = f"""你是一个世界观架构师。请根据以下大纲上下文，为新实体【{name}】（分类：{category}）生成一份结构化的设定卡。
必须遵循以下模板格式，并且必须符合 PGA 底层物理规则（能量守恒、熵增）。

【实体名】: {name}
【分类】: {category}
【参考模板】: {template_str}

【大纲上下文背景】:
{proposal[:2000]}

TASK: 请输出该实体的完整 JSON 设定，必须匹配模板字段。
"""
            try:
                card_res = get_llm(json_mode=True).invoke(card_prompt)
                entity_card = parse_json_safely(card_res.content)
            except Exception:
                entity_card = {"name": name, "description": "自动生成失败，仅保留名称"}
            
            new_entities.append({"name": name, "type": etype, "card": entity_card})
            
            # C 层：自动写入待审区（包含完整设定卡）
            register_draft_entity(
                entity_name=name,
                entity_type=category,
                source_context=f"大纲提案中首次出现",
                source_agent="outline",
                entity_card=entity_card
            )
    
    # 4. 生成报告
    if new_entities:
        names_str = ", ".join([str(e.get('name', '')) for e in new_entities])
        msg = f"⚠️ 实体哨兵发现 {len(new_entities)} 个未注册实体: [{names_str}]。已自动登记至待审区，可在仪表盘审批。"
        print(f"[Entity Sentinel] {msg}")
    else:
        msg = "✅ 实体哨兵检查通过：所有实体均已注册。"
    
    return {"status_message": msg}

# ==========================================
# Graph Definition
# ==========================================
workflow = StateGraph(OutlineState)
workflow.add_node("planner", outline_planner)
workflow.add_node("critic", outline_critic)
workflow.add_node("grounding_audit", grounding_audit_node) # 新增锚定审计
workflow.add_node("entity_sentinel", entity_sentinel_node)
workflow.add_node("human", human_gate)
workflow.add_node("saver", outline_saver)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "critic")
workflow.add_edge("critic", "grounding_audit") # 逻辑审计后接着素材映射审计

def route_after_audit(state: OutlineState):
    _ac = state.get("audit_count")
    ac = int(_ac) if isinstance(_ac, (int, str)) else 0
    # 如果逻辑审计和素材锚定审计中有任何不通过，且重试次数未满
    if not state.get("is_approved") and ac < 3:
        return "planner"
    return "entity_sentinel"

workflow.add_conditional_edges("grounding_audit", route_after_audit, {"entity_sentinel": "entity_sentinel", "planner": "planner"})

# 哨兵完成后进入人工核准
workflow.add_edge("entity_sentinel", "human")

def route_after_human(state: OutlineState):
    fb = str(state.get("user_feedback") or "").strip()
    if fb == "批准": return "saver"
    if fb == "终止": return END
    if fb: return "planner"
    return END

workflow.add_conditional_edges("human", route_after_human, {"saver": "saver", "planner": "planner", END: END})
workflow.add_edge("saver", END)

app = workflow.compile(checkpointer=MemorySaver())
