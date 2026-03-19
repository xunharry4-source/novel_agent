"""
API 入口模块 (App API) - PGA 小说创作引擎后端服务

本模块提供了基于 Flask 的 RESTful API 接口，负责连接前端 UI 与后端的 LangGraph Agents。
其核心职责包括：
1. 状态装载与持久化: 维护会话线程 (thread_id)，通过 LangGraph Checkpointer 实现状态恢复。
2. 异常处理与自愈: 捕获 API 限制错误 (429)，并自动触发 API Key 旋转机制。
3. 文献检索与管理: 提供统一的资料搜索和模板存取接口。
4. 人机交互路由: 将用户反馈正确引导至对应的 Agent 暂停点。
"""
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import os
import json
import traceback

# Import shared utilities
from lore_utils import (
    get_llm,
    get_vector_store,
    get_mongodb_db,
    rotate_api_key,
    get_category_template,
    upsert_category_template,
    get_all_templates,
    delete_category_template,
    add_new_category
)
from worldview_agent_langgraph import app as worldview_app
from novel_outline_agent_langgraph import app as outline_app
from writing_execution_agent_langgraph import app as writing_app

app = Flask(__name__)


# Agent Mapping
AGENTS = {
    "worldview": worldview_app,
    "outline": outline_app,
    "writing": writing_app
}

@app.route('/favicon.ico')
def favicon():
    return '', 204

# --- Shared Error Handler to prevent HTML responses ---
@app.errorhandler(Exception)
def handle_exception(e):
    """确保所有未捕获的异常都以 JSON 形式返回，而不是 HTML 错误页面"""
    # 如果是 404 且不是 API 路径，且不是 favicon，则静默处理
    if hasattr(e, 'code') and e.code == 404:
        return jsonify({"error": "Not Found"}), 404

    err_msg = str(e)
    print(f"[API GLOBAL ERROR]: {err_msg}")
    traceback.print_exc()
    
    # 特殊处理返回列表的接口，防止前端 .filter() 报错
    if request.path in ["/api/search", "/api/lore"]:
        return jsonify([]), 500
        
    return jsonify({
        "error": "Internal Server Error",
        "details": err_msg,
        "status_message": "系统后端发生异常，请检查控制台日志。"
    }), 500


# --- Template Management Endpoints ---
@app.route('/api/worldview/templates', methods=['GET'])
def get_templates():
    """获取所有世界观模板"""
    try:
        templates = get_all_templates()
        return jsonify(templates)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/worldview/templates', methods=['POST'])
def update_template():
    """更新或创建世界观模板"""
    data = request.json
    category = data.get('category')
    template_data = data.get('template_data')
    if not category or not template_data:
        return jsonify({"error": "缺少分类或模板数据"}), 400
    try:
        success = upsert_category_template(category, template_data)
        if success:
            return jsonify({"message": f"模板 '{category}' 更新成功"})
        else:
            return jsonify({"error": "模板更新失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/worldview/templates', methods=['DELETE'])
def delete_template():
    """删除指定分类模板"""
    data = request.json
    category = data.get('category')
    if not category:
        return jsonify({"error": "缺少分类名称"}), 400
    try:
        success = delete_category_template(category)
        if success:
            return jsonify({"message": f"分类 '{category}' 已删除"})
        else:
            return jsonify({"error": f"分类 '{category}' 不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/worldview/templates/new-category', methods=['POST'])
def create_new_category():
    """创建新的分类模板"""
    data = request.json
    category = data.get('category')
    name_zh = data.get('name_zh')
    template_fields = data.get('template')
    example_fields = data.get('example')
    
    if not category or not name_zh:
        return jsonify({"error": "缺少分类标识(category)或中文名(name_zh)"}), 400
    
    try:
        success, msg = add_new_category(category, name_zh, template_fields, example_fields)
        if success:
            return jsonify({"message": msg})
        else:
            return jsonify({"error": msg}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 用于存储大纲以便写作 Agent 使用 (模拟 MongoDB)
def get_outline_by_id(outline_id):
    if not os.path.exists('outlines_db.json'): return None
    with open('outlines_db.json', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['id'] == outline_id:
                    return data['outline']
            except: continue
    return None

@app.route('/')
def index():
    from flask import make_response
    response = make_response(send_file('dashboard.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/lore', methods=['GET'])
def get_lore():
    all_docs = []

    try:
        # 1. Worldview (JSONL)
        if os.path.exists('worldview_db.json'):
            with open('worldview_db.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        all_docs.append({
                            "name": item.get("name") or item.get("query"),
                            "content": item.get("content"),
                            "category": item.get("category", "Worldview"),
                            "timestamp": item.get("timestamp", "N/A")
                        })
                    except: pass

        # 2. Outlines (JSONL)
        if os.path.exists('outlines_db.json'):
            with open('outlines_db.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        all_docs.append({
                            "name": f"大纲: {item.get('query', '未命名')[:20]}...",
                            "content": f"大纲 ID: {item.get('id')}\n\n{item.get('proposal')}",
                            "category": "Outline",
                            "timestamp": item.get("timestamp", "刚刚")
                        })
                    except: pass

        # 3. Prose (JSONL)
        if os.path.exists('prose_db.json'):
            with open('prose_db.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        all_docs.append({
                            "name": f"正文: {item.get('scene_title')}",
                            "content": f"场次 ID: {item.get('scene_id', 'N/A')}\n\n{item.get('content')}",
                            "category": "Prose",
                            "timestamp": item.get("timestamp", "刚刚")
                        })
                    except: pass
    except: pass

    return jsonify(all_docs[::-1])

@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    data = request.json
    query = data.get('query', '')
    thread_id = data.get('thread_id', 'default_user')
    agent_type = data.get('agent_type', 'worldview')
    
    agent_app = AGENTS.get(agent_type)
    if not agent_app:
            return jsonify({"error": f"Agent '{agent_type}' 未就绪"}), 500
            
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化状态
    if agent_type == 'writing':
        outline_id = query # 初始 query 是大纲 ID
        outline_content = get_outline_by_id(outline_id)
        if not outline_content:
            return jsonify({"error": f"大纲 ID '{outline_id}' 不存在"}), 400
        
        input_state = {
            "outline_id": outline_id,
            "outline_content": json.dumps(outline_content, ensure_ascii=False),
            "current_act": "第一幕",
            "status_message": "启动中，正在拆解场次...",
            "active_scene_index": 0,
            "scene_list": [],
            "context_data": "",
            "draft_content": "",
            "audit_feedback": "",
            "user_feedback": "",
            "is_audit_passed": False,
            "is_approved": False,
            "char_status_summary": "",
            "scene_status_summary": "",
            "visual_snapshot_path": "",
            "visual_description_summary": ""
        }
    elif agent_type == 'worldview':
        input_state = {
            "query": query,
            "context": "",
            "proposal": "",
            "review_log": "",
            "user_feedback": "",
            "iterations": 0,
            "audit_count": 0,
            "is_approved": False,
            "category": "",
            "doc_id": "",
            "status_message": "正在启动万象星际探查..."
        }
    else: # outline
        input_state = {
            "query": query,
            "context": "",
            "proposal": "",
            "review_log": "",
            "user_feedback": "",
            "iterations": 0,
            "audit_count": 0,
            "is_approved": False,
            "status_message": "正在生成小说大纲提案..."
        }
    
    print(f"\n[API] Invoking Agent '{agent_type}' for thread '{thread_id}'")
    print(f"[API] Input State: {json.dumps(input_state, ensure_ascii=False)[:200]}...")
    try:
        output = agent_app.invoke(input_state, config=config)
        # 如果 graph 在 human_node 被 interrupt 暂停，output 里不包含最终结果
        # 需要从 checkpointer 获取当前 state
        state_snapshot = agent_app.get_state(config)
        current_state = dict(state_snapshot.values)
        # 检查是否有 interrupt 值（说明 graph 暂停了）
        if state_snapshot.next:
            current_state["status_message"] = current_state.get("status_message", "设定已就绪，等待审核...")
        return jsonify(current_state)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
             if rotate_api_key():
                 print("[API] Rotated key and retrying Agent invoke...")
                 output = agent_app.invoke(input_state, config=config)
                 state_snapshot = agent_app.get_state(config)
                 return jsonify(dict(state_snapshot.values))
        raise e # 继续抛给 global handler

@app.route('/api/agent/feedback', methods=['POST'])
def agent_feedback():
    data = request.json
    feedback = data.get('feedback', '')
    thread_id = data.get('thread_id', 'default_user')
    agent_type = data.get('agent_type', 'worldview')
    
    agent_app = AGENTS.get(agent_type)
    if not agent_app:
            return jsonify({"error": f"Agent '{agent_type}' 未就绪"}), 500
            
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n[API] Resuming Agent '{agent_type}' for thread '{thread_id}'")
    print(f"[API] Feedback Received: '{feedback}'")
    try:
        from langgraph.types import Command
        # 使用 Command(resume=feedback) 恢复被 interrupt 暂停的 graph
        output = agent_app.invoke(Command(resume=feedback), config=config)
        # 获取恢复后的最新 state
        state_snapshot = agent_app.get_state(config)
        current_state = dict(state_snapshot.values)
        if state_snapshot.next:
            current_state["status_message"] = current_state.get("status_message", "设定已更新，等待审核...")
        return jsonify(current_state)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
             if rotate_api_key():
                 print("[API] Rotated key and retrying Agent feedback...")
                 output = agent_app.invoke(Command(resume=feedback), config=config)
                 state_snapshot = agent_app.get_state(config)
                 return jsonify(dict(state_snapshot.values))
        raise e

@app.route('/api/search', methods=['POST'])
def search_lore(retry=True):
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify([])

    try:
        # 使用统一的 get_vector_store 获取索引
        vector_store = get_vector_store()
        if not vector_store:
            return jsonify([])
        
        docs = vector_store.similarity_search(query, k=5)
        formatted = []
        for d in docs:
            formatted.append({
                "name": d.metadata.get('name', '搜索结果'),
                "content": d.page_content,
                "category": d.metadata.get('category', 'Search'),
                "timestamp": "检索中"
            })
        print(f"[SEARCH SUCCESS] Query: '{query}', Results: {len(formatted)}")
        return jsonify(formatted)
    except Exception as e:
        if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and retry:
             if rotate_api_key():
                 print("[API] Rotated key and retrying Search...")
                 return search_lore(retry=False)
        print(f"[SEARCH ERROR]: {str(e)}")
        return jsonify([]) # 始终返回列表，防止前端 crash

@app.route('/api/snapshots/<outline_id>', methods=['GET'])
def get_snapshots(outline_id):
    snapshots = []
    if os.path.exists('snapshots_db.json'):
        with open('snapshots_db.json', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('outline_id') == outline_id:
                        snapshots.append(data)
                except: continue
    return jsonify(snapshots)

if __name__ == '__main__':
    # 启动 5005 端口
    app.run(port=5005, host='0.0.0.0', debug=True)
