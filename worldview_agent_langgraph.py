"""
Worldview Agent (世界观 Agent) - PGA 0-4 协议小说创作引擎核心组件

本模块实现了基于 LangGraph 的世界观管理工作流。它采用 "生成-审计-人工确认-同步" 的闭环模式，
确保创意产出符合“万象星际”底层物理规则（PGA 0-4 架构）。

设计思路 (Design Philosophy):
1. 0-4 逻辑架构: 强制将设定拆分为 定义(0)、入场(1)、冲突(2)、优先级(3) 和 独立性(4) 五个维度。
2. 逻辑隔离: 严禁不同类别的设定（如物理机制与地缘政治）在同一节点中混淆。
3. 人机协作 (Human-in-the-loop): 使用 LangGraph 的 interrupt 机制，在核心设定入库前强制人工审核。
4. RAG 驱动: 每次生成都会从 MongoDB (全文) 和 ChromaDB (向量) 检索相关上下文。
"""
import os
import json
from typing import Annotated, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Import shared utilities
from lore_utils import (
    get_llm, 
    get_vector_store, 
    get_prohibited_rules, 
    get_worldview_context_by_category, 
    get_unified_context,
    get_category_template,
    upsert_category_template,
    parse_json_safely
)

# ==========================================
# 0. State Definition
# ==========================================
class AgentState(TypedDict):
    """
    Agent 运行时的状态机上下文。
    
    Attributes:
        query: 用户的原始输入请求。
        context: 检索到的相关背景知识。
        proposal: Agent 生成的当前草案/提案（通常为 JSON 格式）。
        review_log: 审计节点生成的逻辑一致性报告。
        user_feedback: 用户在中断节点输入的反馈或指令。
        iterations: 当前任务经历的生成迭代次数。
        audit_count: 当前草案经历的自审修正次数。
        is_approved: 标记当前提案是否已被审计通过或人类批准。
        category: 识别出的设定分类（如 race, faction 等）。
        doc_id: 关联的文档 ID。
        status_message: 用于 UI 显示的实时状态描述。
    """
    query: str
    context: str
    proposal: str
    review_log: str
    user_feedback: str
    iterations: int        # 总生成次数
    audit_count: int       # 当前自审重试次数
    is_approved: bool      # 用户是否批准
    category: str 
    doc_id: str
    status_message: str    # 当前执行进度描述

# ==========================================
# 0-4 Architecture & Category Specific Logic
# ==========================================
PGA_0_4_ARCHITECTURE = """
[0. Definitions] PGA protocols, star sector ecological niches, thermodynamics-based lore.
[1. Entry Logic] New elements must pass 'Entry Validation' (fit for star sector ecology).
[2. Conflict Logic] Modifications must resolve contradictions with primary lore (Thermodynamics).
[3. Priority Check] Rule Hierarchy (Thermodynamics > PGA Protocol > Regional Lore).
[4. Independence] Multi-dataset updates (Race, Faction, Mechanism) must remain modular.
"""

CATEGORY_LOGIC_TEMPLATES = {
    "race": {
        "title": "种族逻辑 (Races)",
        "logic": """
        1. 生物/机械结构：必须描述能量摄取方式、转化为熵的效率、以及抗熵变异特征。
        2. 生态位：在所属星区的物理环境下（如高引力、强辐射）的生存地位。
        3. 进化导向：对热力学第二定律的屈服或“局部逆熵”补偿机制。
        4. 隔离原则：严禁描述政治关系或国家边界，专注于生命体本身的演化与物理特性。
        """
    },
    "faction": {
        "title": "势力逻辑 (Factions)",
        "logic": """
        1. 政治与主权：组织架构、核心纲领（如何理解PGA协议）、星区势力范围（Territory）。
        2. 资源控制：对恒星能、零点能或特定星矿的配额管理与分配方式。
        3. 外交与冲突：与其他势力的条约约束力、利益纠葛、以及在星际政治中的生态位。
        4. 隔离原则：严禁描述物种演化或生理结构，专注于组织行为、权力结构与地缘政治。
        """
    },
    "geography": {
        "title": "地理逻辑 (Geographies)",
        "logic": """
        1. 物理环境：天体运行规律、重力梯度、辐射能级、空间曲率异常。
        2. 局部熵场：划分热力学有序区（低熵区）与混沌区（高熵区）的物理分布。
        3. 承载力：该区域能支持的最大能量级别 or 文明载荷量。
        4. 隔离原则：严禁描述居民的政治斗争，专注于空间本身的物理参数。
        """
    },
    "mechanism_tech": {
        "title": "机制/科技逻辑 (Mechanisms & Tech)",
        "logic": """
        1. 物理实现：技术如何通过控制局部熵增来实现功能（严禁违反能量守恒）。
        2. 协议合规：确认科技是否触及了PGA协议禁止的“时间操纵”等红线技术。
        3. 热机效率：该技术在热力学系统中的能量转化率与废热处理机制。
        4. 隔离原则：专注于技术底层逻辑，而非使用该技术的政治组织。
        """
    },
    "history": {
        "title": "历史逻辑 (History)",
        "logic": """
        1. 线性一致性：按时间轴顺序确定的因果链条，严禁出现时间旅行 or 溯因性修改.
        2. 熵增叙事：重大事件如何导致星区能量分布的彻底改变。
        3. 记录差异：不同势力对同一事件的不同热力学记录。
        4. 隔离原则：专注于事件的宏观影响，而非单一实体的设定细节。
        """
    }
}

# ==========================================
# Nodes Implementation
# ==========================================

def generator_node(state: AgentState):
    """
    生成节点 (Proposal Generator)。
    
    责任:
    1. 分类识别: 根据用户 query 自动判定所属世界观维度 (race, faction, etc.)。
    2. 上下文组装: 调用 RAG 检索文献库 (MongoDB) 和向量库 (ChromaDB) 中的相关冲突/背景。
    3. 模板注入: 获取对应分类的 JSON 模板，确保输出格式合规。
    4. 创作生成: 调用 LLM 生成符合 PGA 0-4 逻辑的设定提案。
    """
    print(f"\n[DEBUG] generator_node entry. State keys: {list(state.keys())}")
    query = state.get('query', '')
    if not query:
        print("[WARNING] 'query' is missing in state at generator_node! Using empty string.")
        
    query_lower = query.lower()
    category = state.get('category', 'general')
    if not category or category == "general":
        # 如果尚未分类，执行关键词分类
        if any(k in query_lower for k in ["势力", "阵营", "国家", "派系", "帝国", "联邦", "军团", "公约", "盟友"]):
            category = "faction"
        elif any(k in query_lower for k in ["种族", "机器人", "机械族", "生命", "进化", "族群", "演化", "物种"]):
            category = "race"
        elif any(k in query_lower for k in ["宗教", "信仰", "教会", "神说", "崇拜"]):
            category = "religion"
        elif any(k in query_lower for k in ["地理", "地形", "环境", "星域", "坐标"]):
            category = "geography"
        elif any(k in query_lower for k in ["星球", "行星", "恒星"]):
            category = "planet"
        elif any(k in query_lower for k in ["危机", "灾难", "变故", "事故"]):
            category = "crisis"
        elif any(k in query_lower for k in ["武器", "装备", "战机", "母舰"]):
            category = "weapon"
        elif any(k in query_lower for k in ["生物", "野兽", "怪物", "掠食者"]):
            category = "creature"
        elif any(k in query_lower for k in ["组织", "协会", "学术", "联盟", "公司"]):
            category = "organization"
        elif any(k in query_lower for k in ["机制", "科技", "武器", "引擎", "原理", "技术", "装置", "协议", "热力学", "物理", "发动机", "驱动"]):
            category = "mechanism_tech"
        elif any(k in query_lower for k in ["历史", "纪元", "事件", "变迁", "战争", "编年史", "记录"]):
            category = "history"
        else:
            category = "general"

    category_info = CATEGORY_LOGIC_TEMPLATES.get(category, {"title": "一般世界观", "logic": "遵循PGA底层物理与逻辑。"})
    
    # 2. 获取分类模板 (MongoDB/Local Fallback)
    template_data = get_category_template(category)
    if not template_data and category != "general":
        # 如果模板不存在，先生成一个
        meta_prompt = f"你是一个世界观架构师。请为【{category}】这个分类创建一个标准的 JSON 模板和参考例子。必须输出合法有效 JSON。"
        meta_res = get_llm(json_mode=True).invoke(meta_prompt)
        try:
            template_data = parse_json_safely(meta_res.content)
            if template_data:
                upsert_category_template(category, template_data)
        except Exception:
            template_data = {"template": "基础文本描述", "example": "无"}

    # 3. 获取上下文
    rag_context = get_unified_context(query)
    prohibited_items = get_prohibited_rules()
    worldview_rules = get_worldview_context_by_category(query)
    
    template_str = json.dumps(template_data.get("template", {}), ensure_ascii=False, indent=2) if template_data else "自由发挥"
    example_str = json.dumps(template_data.get("example", {}), ensure_ascii=False, indent=2) if template_data else "无"

    feedback_section = ""
    user_feedback = state.get('user_feedback', '')
    if user_feedback:
        feedback_section = f"""
【！！！当前核心修改需求 - 必须首先满足！！！】
用户提出以下问题或要求：
>>> {user_feedback} <<<
你必须在本次生成中优先解决上述反馈。
"""

    full_prompt = f"""你是一个专精于“万象星际协议体 (PGA)”世界观的资深创作专家。
你的任务是根据用户的查询扩展或修改世界观设定。

{feedback_section}

【当前逻辑分类：{category_info['title']}】
本类别必须严格遵守以下逻辑边界，绝不能越界：
{category_info['logic']}

【最高禁令 - 必须绝对遵循】
{prohibited_items}

【官方核心规则】
{worldview_rules}
1. 热力学第二定律：熵增不可逆，能量转换必有损耗。
2. 能量守恒：任何现象必须有物理层面的能量输入。

【PGA 0-4 架构约束】
{PGA_0_4_ARCHITECTURE}

【输出格式要求：JSON】
你必须基于以下“分类模板”进行创作，并参考其“示例”。
分类模板:
{template_str}

参考示例:
{example_str}

【生成内容】
现有背景资料: {rag_context}
用户当前需求：{query}
(之前的审计逻辑建议: {state.get('review_log', '无')})

TASK: 请完成设定提案。必须输出为 JSON 格式。
"""
    
    res = get_llm(json_mode=True).invoke(full_prompt)
    _iter_val = state.get('iterations', 0)
    curr_iterations = int(_iter_val) if isinstance(_iter_val, (int, str)) else 0
    
    return {
        "proposal": res.content, 
        "category": category,
        "iterations": curr_iterations + 1, 
        "status_message": f"[{category_info['title']}] 提议已生成并进入逻辑审查..."
    }


def reviewer_node(state: AgentState):
    """
    审计节点 (Logic Reviewer)。
    
    责任:
    1. 一致性检查: 验证提案是否违反 PGA 底层物理规则 (如能量守恒)。
    2. 隔离性审计: 检查提案是否跨越了逻辑边界 (如在种族设定中讨论地缘政治)。
    3. 逻辑闭环: 如果审计不通过，将 is_approved 设为 False，触发图回到 generator_node 进行修正。
    """
    print(f"\n[DEBUG] reviewer_node entry. State keys: {list(state.keys())}")
    query = state.get('query', '')
    proposal = str(state.get('proposal') or '')
    print(f"[DEBUG] Entering reviewer_node (proposal length: {len(proposal)})")
    category = state.get('category', 'general')
    category_info = CATEGORY_LOGIC_TEMPLATES.get(category, {"title": "一般世界观", "logic": "遵循PGA底层物理与逻辑。"})
    
    prohibited_items = get_prohibited_rules()
    worldview_rules = get_worldview_context_by_category(query)
    
    full_prompt = f"""你是一个专精于“万象星际协议体 (PGA)”的逻辑审核官。
必须输出 JSON 格式。

【审核标准：{category_info['title']}】
{category_info['logic']}

禁令: {prohibited_items}

待审核提案：
{proposal}

请根据规则审核，输出 JSON: {{"status": "合理/不合理", "audit_log": "...", "category_purity": "纯粹/混淆"}}
"""
    _count_val = state.get('audit_count', 0)
    count = int(_count_val) if isinstance(_count_val, (int, str)) else 0
    res = get_llm(json_mode=True).invoke(full_prompt)
    try:
        audit_data = parse_json_safely(res.content)
        if not audit_data:
            raise ValueError("Invalid audit JSON")
        is_purity_ok = audit_data.get("category_purity") == "纯粹"
        is_logical_ok = audit_data.get("status") == "合理"
        is_ok = is_purity_ok and is_logical_ok
        
        msg = f"完成审计：{'通过' if is_ok else '检测到逻辑混淆，正在重试'}"
        print(f"[DEBUG] Reviewer result: {msg}, is_approved: {is_ok}")
        return {
            "review_log": audit_data.get("audit_log", res.content), 
            "is_approved": is_ok,
            "audit_count": count + 1,
            "status_message": msg
        }
    except Exception as e:
        print(f"[DEBUG] Reviewer parsing error: {e}")
        return {"review_log": res.content, "is_approved": False, "audit_count": count + 1, "status_message": "审核解析异常"}

def human_node(state: AgentState):
    """
    人工节点 (Human-in-the-loop Gate)。
    
    责任:
    1. 中断执行: 在 Web 模式下发出 interrupt 信号，挂起当前线程，等待 UI 层的 Command(resume=...) 指令。
    2. 交互入口: 允许人类创作者对 Agent 的提案进行最终核准或提出修改意见。
    """
    print(f"\n[DEBUG] human_node entry. State keys: {list(state.keys())}")
    proposal = str(state.get('proposal') or '')
    category = str(state.get('category') or 'general')
    print(f"[DEBUG] Entering human_node (category: {category})")
    
    if os.getenv("AGENT_MODE") == "CLI":
        print(f"\n--- [Agent {category.upper()} 提议] ---\n{proposal}")
        choice = input("\n[a]批准 [f]反馈 [q]退出: ").strip().lower()
        if choice == 'a': return {"is_approved": True, "user_feedback": "批准"}
        if choice == 'f': return {"is_approved": False, "user_feedback": input("意见: ")}
        exit()
        
    # Web 模式：使用 interrupt 暂停图，等待用户反馈
    print("[DEBUG] human_node: Interrupting for user feedback (Web mode)...")
    user_input = interrupt({"status_message": f"{category.upper()} 设定已就绪，等待核准...", "proposal": proposal})
    print(f"[DEBUG] human_node: Resumed. Received feedback: '{user_input}'")
    return {"user_feedback": user_input, "is_approved": user_input == "批准", "status_message": "正在处理您的反馈..."}

def saver_node(state: AgentState):
    """
    存储节点 (Sync Committer)。
    
    责任:
    1. 事务写入: 将最终获批的设定同步写入本地磁盘 (worldview_db.json) 和向量库 (ChromaDB)。
    2. 状态收尾: 清理会话状态，准备进入下一个创作循环。
    """
    print(f"\n[DEBUG] saver_node entry. State keys: {list(state.keys())}")
    doc = {
        "category": state.get('category', 'general'),
        "content": state.get('proposal',''),
        "iterations": state.get('iterations', 0),
        "query": state.get('query',''),
        "timestamp": str(os.getenv("CURRENT_TIME", "2026-03-19T13:00:00"))
    }
    with open("worldview_db.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    try:
        v_store = get_vector_store()
        if v_store:
            v_store.add_texts(
                texts=[state.get('proposal','')], 
                metadatas=[{"category": state.get('category', 'general'), "source": "agent_generated"}]
            )
    except Exception: pass
    return {"is_approved": True, "status_message": "设定已通过审计并录入 PGA 数据库。"}

# ==========================================
# Graph Definition
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("generator", generator_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("human", human_node)
workflow.add_node("saver", saver_node)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "reviewer")

def route_after_review(state: AgentState):
    if state.get("is_approved") or int(state.get("audit_count", 0)) >= 3: return "human"
    return "generator"

workflow.add_conditional_edges("reviewer", route_after_review, {"human": "human", "generator": "generator"})

def route_after_human(state: AgentState):
    fb = state.get("user_feedback", "").strip()
    if fb == "批准": return "saver"
    if fb == "终止": return END
    if fb: return "generator"
    return END

workflow.add_conditional_edges("human", route_after_human, {"saver": "saver", "generator": "generator", END: END})
workflow.add_edge("saver", END)

app = workflow.compile(checkpointer=MemorySaver())
