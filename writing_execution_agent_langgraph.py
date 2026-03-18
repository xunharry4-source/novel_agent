import os
import json
import datetime
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 0. State Definition
# ==========================================
class WritingState(TypedDict):
    # 输入信息
    outline_id: str             # 已保存的大纲 ID
    outline_content: str        # 大纲全文内容 (用于参考)
    current_act: str            # 当前编写的小说章节/幕
    
    # 过程数据
    scene_list: List[dict]      # 拆解后的场次清单 [{'title': '...', 'description': '...'}]
    active_scene_index: int     # 当前正在写的场次索引
    context_data: str           # 从 ChromaDB 检索出的背景知识
    
    # 输出数据
    draft_content: str          # 生成的初稿正文
    audit_feedback: str         # 审计意见/逻辑漏洞报告
    user_feedback: str          # 用户的人工意见
    is_audit_passed: bool       # 审计是否通过
    is_approved: bool           # 用户是否批准
    status_message: str         # 执行进度描述
    
    # 快照数据 (New)
    char_status_summary: str    # 人物逻辑快照 (位置、状态、动机)
    scene_status_summary: str   # 场景逻辑快照 (天气、物品毁损)
    visual_snapshot_path: str   # 视觉快照图片路径
    visual_description_summary: str # 视觉描述摘要 (用于保持后续场景的视觉一致性)

# ==========================================
# Clients Init
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
# JSON mode client
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

def plan_scenes_func(state: WritingState):
    """场次拆解节点 (Scene Planner)"""
    prompt = f"""你是一个专业的小说场次策划师。
你的任务是将给定的大纲拆解为具体的“原子场次（Atomic Scenes）”。
每个场次通常涵盖一个单一的行动或对话序列。

【大纲内容】
{state['outline_content']}

【当前编写部分】
{state['current_act']}

请输出一个包含场次清单的 JSON：
{{
  "scene_list": [
    {{ "id": 1, "title": "标题", "description": "场次核心内容描述，包含出场任务与目标" }},
    ...
  ]
}}
"""
    res = llm_json.invoke(prompt)
    data = json.loads(res.content)
    return {
        "scene_list": data.get("scene_list", []),
        "active_scene_index": 0,
        "status_message": f"场次拆解完成，共 {len(data.get('scene_list', []))} 场。开始加载第一场语境..."
    }

def load_context_func(state: WritingState):
    """语境加载节点 (Context Loader)"""
    idx = state['active_scene_index']
    scene = state['scene_list'][idx]
    
    # 根据场次描述检索世界观
    query = f"{scene['title']} {scene['description']}"
    docs = vector_store.similarity_search(query, k=5)
    rag_context = "\n".join([d.page_content for d in docs])
    
    return {
        "context_data": rag_context,
        "status_message": f"正在编写第 {idx+1} 场：{scene['title']}。已加载相关世界观语境。"
    }

def write_draft_func(state: WritingState):
    """正文生成节点 (Drafting Scribe)"""
    idx = state['active_scene_index']
    scene = state['scene_list'][idx]
    
    prompt = f"""你是一个拥有卓越文采的小说家（Scribe）。
你要根据世界观设定和场次计划，撰写具体的小说正文。

【大纲背景】
{state['outline_content']}

【当前场次计划】
标题：{scene['title']}
描述：{scene['description']}

【世界观语境 (严格遵守)】
{state['context_data']}

【用户反馈】
{state.get('user_feedback', '')}

【写作要求】
1. 严格遵循大纲中的写作风格 (Writing Style)。
2. 保持角色人设一致性。
3. 增加感官细节（嗅觉、听觉、视觉）。
4. 对话要自然。

直接输出小说正文内容：
"""
    res = llm.invoke(prompt)
    return {
        "draft_content": res.content,
        "user_feedback": "",
        "status_message": f"第 {idx+1} 场初稿已生成，正在提交逻辑审计..."
    }

def audit_logic_func(state: WritingState):
    """逻辑审计节点 (Logic Auditor)"""
    # 尝试加载上一步的快照以进行一致性检查
    prev_snapshot = ""
    if state.get("char_status_summary"):
        prev_snapshot = f"【上场人物状态快照】：\n{state['char_status_summary']}"

    prompt = f"""你是一个严苛的小说逻辑审计员。
你的任务是检查正文内容是否与世界观设定、角色背景以及【之前的快照状态】存在冲突。

【世界观设定】
{state['context_data']}

{prev_snapshot}

【待审计正文】
{state['draft_content']}

请输出 JSON：
{{
  "is_consistent": true/false,
  "audit_log": "如果存在逻辑漏洞、人设崩塌、物理错误（尤其是与上场快照冲突的地方），在此详细列出。若通过，写'通过'。"
}}
"""
    res = llm_json.invoke(prompt)
    data = json.loads(res.content)
    is_ok = data.get("is_consistent", False)
    
    return {
        "is_audit_passed": is_ok,
        "audit_feedback": data.get("audit_log", ""),
        "status_message": "审计通过，等待用户预览..." if is_ok else "发现逻辑冲突，正在打回重写..."
    }

def human_review_node(state: WritingState):
    """等待用户批准节点"""
    return {"status_message": "请预览正文，点击'批准'以存档，或输入修改意见。"}

def prose_saver_func(state: WritingState):
    """正文存档节点"""
    idx = state['active_scene_index']
    scene = state['scene_list'][idx]
    
    record = {
        "id": f"prose_{state['outline_id']}_{idx}_{int(datetime.datetime.now().timestamp())}",
        "outline_id": state['outline_id'],
        "scene_title": scene['title'],
        "content": state['draft_content'],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open('prose_db.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    next_idx = idx + 1
    return {
        "active_scene_index": next_idx,
        "is_approved": False, # 重置审核状态
        "status_message": f"第 {idx+1} 场已存档，正在生成状态快照..."
    }

def snapshot_node_func(state: WritingState):
    """快照生成节点 (Snapshot Agent)"""
    idx = state['active_scene_index'] - 1 # saver 已经把 index + 1 了
    
    # 尝试读取上一次的视觉描述作为参考
    prev_visual_ref = ""
    if state.get("visual_description_summary"):
        prev_visual_ref = f"【上一次的视觉风格参考】：\n{state['visual_description_summary']}"

    # 1. 提取逻辑状态
    extract_prompt = f"""请从以下正文中提取人物和场景的“状态快照”。
为了保持视觉连贯性，请参考之前的视觉描述。

【正文】
{state['draft_content']}

{prev_visual_ref}

输出 JSON：
{{
  "char_status": "人物位置、伤势、当前装备、心理动机摘要",
  "scene_status": "当前时间天气、关键物品损毁、NPC 在场情况",
  "visual_description": "一张用于生成插画的写实风格视觉描述（必须保持主角脸部、伤疤、衣服风格的一致性）"
}}
"""
    res = llm_json.invoke(extract_prompt)
    data = json.loads(res.content)
    
    char_status = data.get("char_status", "状态稳定")
    scene_status = data.get("scene_status", "环境正常")
    visual_desc = data.get("visual_description", "")
    
    # 2. 生成视觉快照 (模拟调用)
    image_path = f"snapshots/visual_{state['outline_id']}_{idx}.jpg"
    
    # 3. 持久化快照
    snapshot_record = {
        "outline_id": state['outline_id'],
        "scene_index": idx,
        "char_status": char_status,
        "scene_status": scene_status,
        "visual_path": image_path,
        "visual_description": visual_desc,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open('snapshots_db.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(snapshot_record, ensure_ascii=False) + "\n")
        
    has_more = state['active_scene_index'] < len(state['scene_list'])
    
    return {
        "char_status_summary": char_status,
        "scene_status_summary": scene_status,
        "visual_snapshot_path": image_path,
        "visual_description_summary": visual_desc,
        "status_message": f"快照及视觉参考已生成。" + (f"准备进入下一场..." if has_more else "全章节编写完成！")
    }

# ==========================================
# Graph Definition
# ==========================================
workflow = StateGraph(WritingState)

# Nodes
workflow.add_node("plan_scenes", plan_scenes_func)
workflow.add_node("load_context", load_context_func)
workflow.add_node("write_draft", write_draft_func)
workflow.add_node("audit_logic", audit_logic_func)
workflow.add_node("human_review", human_review_node)
workflow.add_node("prose_saver", prose_saver_func)
workflow.add_node("snapshot_node", snapshot_node_func)

# Edges
workflow.add_edge(START, "plan_scenes")
workflow.add_edge("plan_scenes", "load_context")
workflow.add_edge("load_context", "write_draft")
workflow.add_edge("write_draft", "audit_logic")

def route_after_audit(state: WritingState):
    if state["is_audit_passed"]:
        return "human_review"
    return "write_draft"

workflow.add_conditional_edges("audit_logic", route_after_audit, {
    "human_review": "human_review",
    "write_draft": "write_draft"
})

def route_after_human(state: WritingState):
    fb = (state.get("user_feedback") or "").strip()
    if fb == "批准": return "prose_saver"
    if fb == "终止": return END
    if fb: return "write_draft" # 有意见则重写当前场次
    return END

workflow.add_conditional_edges("human_review", route_after_human, {
    "prose_saver": "prose_saver",
    "write_draft": "write_draft",
    END: END
})

workflow.add_edge("prose_saver", "snapshot_node")

def route_next_scene(state: WritingState):
    if state['active_scene_index'] < len(state['scene_list']):
        return "load_context"
    return END

workflow.add_conditional_edges("snapshot_node", route_next_scene, {
    "load_context": "load_context",
    END: END
})

app = workflow.compile(checkpointer=MemorySaver())
