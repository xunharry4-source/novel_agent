import os
import json
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 0. State Definition
# ==========================================
class OutlineState(TypedDict):
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
# Clients Init
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
llm_json = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY,
    model_kwargs={"response_mime_type": "application/json"}
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(client=chroma_client, collection_name="pga_lore", embedding_function=embeddings)

# ==========================================
# Nodes Implementation
# ==========================================

def outline_planner(state: OutlineState):
    """大纲策划节点"""
    # 检索世界观
    docs = vector_store.similarity_search(state['query'], k=3)
    rag_context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""你是一个专精于“万象星际协议体 (PGA)”世界观的小说策划专家。
你的任务是根据用户的需求编写一份详尽的小说大纲。你必须严格按照以下 JSON Schema 输出内容：

【JSON Schema】
{{
  "meta_info": {{
    "title": "暂定书名",
    "genre": ["题材分类"],
    "tone": "基调",
    "target_audience": "目标受众",
    "writing_style": "写作风格描述：如 硬核科幻、意识流、极简主义"
  }},
  "core_hook": {{
    "logline": "一句话核心创意",
    "inciting_incident": "激励事件",
    "core_conflict": "核心矛盾"
  }},
  "world_building_ref": {{
    "base_rules": "底层逻辑",
    "key_locations": ["关键地点"],
    "power_system": "力量/科技等级体系"
  }},
  "character_roster": [
    {{
      "name": "名称",
      "role": "Protagonist/Antagonist/Supporting/Mentor",
      "motivation": "动机",
      "internal_flaw": "缺陷",
      "character_arc": "成长曲线"
    }}
  ],
  "plot_beats": {{
    "act_1": "开端与建置",
    "midpoint": "中点转换",
    "climax": "最终高潮",
    "resolution": "结局"
  }},
  "themes": ["主题"]
}}

【世界观背景】
{rag_context}

【用户需求】
{state['query']}
{state['user_feedback']}

【之前的审核意见】
{state['review_log']}

请直接输出符合 Schema 的 JSON 对象。
"""
    res = llm_json.invoke(prompt)
    curr_iterations = state.get('iterations', 0)
    if curr_iterations is None: curr_iterations = 0
    
    return {
        "proposal": res.content,
        "iterations": int(curr_iterations) + 1,
        "user_feedback": "",
        "status_message": "大纲（JSON格式）已生成，正在进行逻辑一致性审计..."
    }

def outline_critic(state: OutlineState):
    """大纲审计节点"""
    prompt = f"""你是一个负责维护 PGA 世界观与故事逻辑严谨性的“剧本审核官”。
你必须将审核结果输出为 **JSON 格式**。

【审核标准】
1. 物理一致性：是否存在非法的时间旅行、无限能量或违背热力学的情况？
2. Schema 完备性：是否完整填充了 meta_info (含 writing_style), core_hook, character_roster, plot_beats 等字段？
3. 风格一致性：大纲描述的语气与设定的 writing_style 是否契合？
4. 钩子逻辑：开端是否具备足够的张力且符合背景？
5. 节奏逻辑：事件升级是否合理，是否存在天降神兵？
6. 角色动机：角色的行为是否在其背景下具备物理与社会合理性？

【输出格式】
{{
  "status": "合理" 或 "不合理",
  "audit_log": "详细的逻辑审查记录。若涉及字段缺失或逻辑冲突，请明确指出。",
  "is_consistent": true/false
}}

待审核大纲 JSON：
{state['proposal']}
"""
    res = llm_json.invoke(prompt)
    try:
        audit_data = json.loads(res.content)
        is_ok = audit_data.get("status") == "合理"
        curr_audit_count = state.get('audit_count', 0)
        if curr_audit_count is None: curr_audit_count = 0
        
        return {
            "review_log": audit_data.get("audit_log", ""),
            "is_approved": is_ok,
            "audit_count": int(curr_audit_count) + 1,
            "status_message": "审计完成：" + ("逻辑自洽" if is_ok else "发现逻辑漏洞，正在打回重修")
        }
    except Exception:
        curr_audit_count = state.get('audit_count', 0)
        if curr_audit_count is None: curr_audit_count = 0
        return {"is_approved": False, "audit_count": int(curr_audit_count) + 1, "status_message": "审计解析异常"}

def human_gate(state: OutlineState):
    """等待用户反馈节点"""
    return {"status_message": "大纲已就绪，等待您的调整建议或批准..."}

import datetime

def outline_saver(state: OutlineState):
    """持久化节点：将通过审核的大纲存入 outlines_db.json"""
    try:
        outline_data = json.loads(state['proposal'])
        record = {
            "id": f"outline_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now().isoformat(),
            "query": state['query'],
            "outline": outline_data,
            "iterations": state['iterations']
        }
        
        with open('outlines_db.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        return {"status_message": "大纲已最终确立并成功存入 outlines_db.json。"}
    except Exception as e:
        return {"status_message": f"大纲存档失败: {str(e)}"}

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
    audit_count = state.get("audit_count", 0)
    if audit_count is None: audit_count = 0
    if state["is_approved"] or int(audit_count) >= 3:
        return "human"
    return "planner"

workflow.add_conditional_edges("critic", route_after_critic, {"human": "human", "planner": "planner"})

def route_after_human(state: OutlineState):
    fb = state.get("user_feedback")
    if fb is None: fb = ""
    fb = str(fb).strip()
    if fb == "批准": return "saver"
    if fb == "终止": return END
    if fb: return "planner" # 有反馈则重新生成
    return END

workflow.add_conditional_edges("human", route_after_human, {"saver": "saver", "planner": "planner", END: END})
workflow.add_edge("saver", END)

app = workflow.compile(checkpointer=MemorySaver())
