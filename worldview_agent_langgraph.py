import os
import pymongo
import chromadb
from chromadb.config import Settings
from typing import Annotated, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import json

# ==========================================
# 0. State Definition
# ==========================================
class AgentState(TypedDict):
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
# Local DB & LLM Clients (Google AI Studio)
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"

# LLM for general generation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
# LLM for structured audit (JSON Mode)
llm_json = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY,
    model_kwargs={"response_mime_type": "application/json"}
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")

# --- MongoDB Init with Fallback ---
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    mongo_client.server_info()
    db = mongo_client["pga_worldview"]
except Exception:
    db = None

# --- ChromaDB Init with Fallback ---
try:
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    chroma_client.heartbeat()
except Exception:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

vector_store = Chroma(
    client=chroma_client,
    collection_name="pga_lore",
    embedding_function=embeddings
)

# ==========================================
# 0-4 Architecture & Prohibited Items
# ==========================================
PGA_0_4_ARCHITECTURE = """
[0. Definitions] PGA protocols, star sector ecological niches, thermodynamics-based lore.
[1. Entry Logic] New elements must pass 'Entry Validation' (fit for star sector ecology).
[2. Conflict Logic] Modifications must resolve contradictions with primary lore (Thermodynamics).
[3. Priority Check] Rule Hierarchy (Thermodynamics > PGA Protocol > Regional Lore).
[4. Independence] Multi-dataset updates (Race, Faction, Mechanism) must remain modular.
"""

def get_prohibited_rules():
    """从数据库或本地文件动态获取禁止项目"""
    try:
        if db is not None:
            rule_doc = db["prohibited_rules"].find_one({"name": "PGA核心禁令"})
            if rule_doc:
                return rule_doc["content"]
        
        # Fallback to local JSON
        if os.path.exists("worldview_db.json"):
            with open("worldview_db.json", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("category") == "prohibited_rules":
                            return data.get("content")
    except Exception:
        pass
    
    return "【最高禁令】禁止控制时间、禁止高维神明、禁止预测未来、禁止现实修改、禁止无限能量。"

def get_worldview_context_by_category(query):
    """根据查询内容识别涉及的世界观分类并获取其核心定义或上下文"""
    category_map = {
        "Races": ["种族", "智械", "机器", "生命", "熵族"],
        "Geographies": ["地理", "星域", "恒星", "行星", "戴森球", "环境"],
        "Factions": ["势力", "国家", "组织", "军团", "公约", "强国"],
        "Mechanisms": ["机制", "协议", "技术", "代偿", "热力学", "规则"],
        "History": ["历史", "记录", "演变", "纪元"]
    }
    
    detected_categories = []
    for cat, keys in category_map.items():
        if any(k in query for k in keys):
            detected_categories.append(cat)
            
    if not detected_categories:
        return ""
        
    context_blocks = []
    try:
        # Query Local JSON (Fallback)
        if os.path.exists("worldview_db.json"):
            with open("worldview_db.json", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    res_cat = data.get("category")
                    
                    # 检查此条目是否属于任一检测到的分类
                    if res_cat in detected_categories or res_cat == "race_definitions":
                        # 确保内容确切相关
                        is_relevant = False
                        for d_cat in detected_categories:
                            keywords = category_map.get(d_cat, [])
                            if any(k in data.get("name", "") or k in data.get("content", "") for k in keywords):
                                is_relevant = True
                                break
                        
                        if is_relevant:
                            context_blocks.append(f"【{res_cat} 准则】: {data.get('content')}")
    except Exception:
        pass
        
    if context_blocks:
        # Return first few blocks to avoid prompt bloat
        trimmed = []
        for i in range(min(len(context_blocks), 5)):
            trimmed.append(context_blocks[i])
        return "\n" + "\n".join(trimmed)
    return ""

# ==========================================
# Nodes Implementation
# ==========================================

def generator_node(state: AgentState):
    try:
        docs = vector_store.similarity_search(state['query'], k=2)
        rag_context = "\n".join([d.page_content for d in docs])
    except Exception:
        rag_context = ""
    
    prohibited_items = get_prohibited_rules()
    worldview_rules = get_worldview_context_by_category(state['query'])
    
    full_prompt = f"""你是一个专精于“万象星际协议体 (PGA)”世界观的资深创作专家。
你的任务是根据用户的查询扩展世界观设定。

【最高禁令 - 必须绝对遵循】
{prohibited_items}
{worldview_rules}

【物理铁律】
1. 热力学第二定律：熵增不可逆，能量转换必有损耗。
2. 能量守恒：任何现象必须有物理层面的能量输入。

【PGA 0-4 架构】
{PGA_0_4_ARCHITECTURE}

【思维链要求 (THINKING)】
请在生成内容前，先进行“深度思考”：
- 是否触犯了控制时间、高维神明、无限能量等“最高禁令”？
- 如果涉及特定分类（如种族、地理、势力），是否严格匹配了其“库内定义”？是否存在逻辑外溢？
- 该设定如何符合热力学底层逻辑？
- 如何遵循 PGA 0-4 架构中的“进入逻辑”和“冲突逻辑”？

【生成内容】
基于以下上下文信息：
现有RAG上下文: {rag_context}
用户查询：{state['query']}
用户反馈：{state['user_feedback']}
审核日志：{state['review_log']}

TASK: 产出一个“修订后的设定”（提案），遵循PGA 0-4架构，并绝对避开所有禁止项目，确保种族类型与官方定义统一。
"""
    
    res = llm.invoke(full_prompt)
    raw_val = state.get('iterations')
    if raw_val is None:
        curr_iterations = 0
    else:
        try:
            curr_iterations = int(raw_val)
        except (ValueError, TypeError):
            curr_iterations = 0
    return {
        "proposal": res.content, 
        "iterations": curr_iterations + 1, 
        "user_feedback": "",
        "status_message": "提议生成完成，正在移交逻辑审核官..."
    }

def reviewer_node(state: AgentState):
    prohibited_items = get_prohibited_rules()
    worldview_rules = get_worldview_context_by_category(state['query'])
    
    full_prompt = f"""你是一个负责维护 PGA 世界观严谨性的“逻辑审核官”。
你必须将审核结果输出为 **JSON 格式**。

【最高优先级审核项 - 禁止项目】
{prohibited_items}
{worldview_rules}

【核心标准】
1. 严禁触碰时间控制、高维神明、现实修改、无限能量等红线。
2. 严禁分类混淆：特别审核是否符合该分类（如地理、势力、种族）的“库内定义”。
3. 逻辑自洽：设定是否符合热力学与能量守恒？
4. 架构一致：是否符合 PGA 的 0-4 架构？

【输出格式】
{{
  "status": "合理" 或 "不合理",
  "audit_log": "详细的逻辑审查记录。若触犯禁令或违反官方定义，必须引用相关规则并指出冲突点。",
  "priority_check": "通过" 或 "未通过"
}}

待审核提案：
{state['proposal']}
"""
    raw_count = state.get('audit_count')
    if raw_count is None:
        count = 0
    else:
        try:
            count = int(raw_count)
        except (ValueError, TypeError):
            count = 0
    res = llm_json.invoke(full_prompt)
    try:
        import json
        audit_data = json.loads(res.content)
        is_ok = audit_data.get("status") == "合理"
        msg = f"完成第 {count + 1} 次逻辑审计：{'通过' if is_ok else '发现冲突，正在准备修正'}"
        return {
            "review_log": audit_data.get("audit_log", res.content), 
            "is_approved": is_ok,
            "audit_count": count + 1,
            "status_message": msg
        }
    except Exception:
        return {"review_log": res.content, "is_approved": False, "audit_count": count + 1, "status_message": "审核解析异常"}

def human_node(state: AgentState):
    if os.getenv("AGENT_MODE") == "CLI":
        print(f"\n--- [Agent 提议] ---\n{state['proposal']}")
        choice = input("\n[a]批准 [f]反馈 [q]退出: ").strip().lower()
        if choice == 'a': return {"is_approved": True, "user_feedback": "批准"}
        if choice == 'f': return {"is_approved": False, "user_feedback": input("意见: ")}
        exit()
    return {"status_message": "Agent 已就绪，等待您的下一步指令（反馈/批准）"}

def saver_node(state: AgentState):
    doc = {
        "category": state.get('category', 'general'),
        "content": state['proposal'],
        "iterations": state['iterations'],
        "query": state['query'],
        "timestamp": str(os.getenv("CURRENT_TIME", "2026-03-19T00:13:00"))
    }
    with open("worldview_db.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    try:
        vector_store.add_texts(texts=[state['proposal']], metadatas=[{"category": state.get('category', 'general')}])
    except Exception: pass
    return {"is_approved": True, "status_message": "设定已成功录入星际协议体 (PGA) 数据库。"}

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
    if state["is_approved"] or state.get("audit_count", 0) >= 3: return "human"
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

from langgraph.checkpoint.memory import MemorySaver

app = workflow.compile(checkpointer=MemorySaver())
