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
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter

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
    add_new_category,
    get_draft_entities,
    approve_draft_entity,
    get_langfuse_callback,
    report_token_usage
)
from worldview_agent_langgraph import app as worldview_app
from novel_outline_agent_langgraph import app as outline_app
from writing_execution_agent_langgraph import app as writing_app
from router_agent_langgraph import app as router_app
from config_utils import get_config

# Load configuration for observability
CONFIG = get_config()

# --- Initialize Observability Stack ---

# 1. Sentry (Error Tracking)
if CONFIG.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=CONFIG["SENTRY_DSN"],
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
    print("[OBSERVABILITY] Sentry initialized.")

app = Flask(__name__)

# 2. Prometheus (Metrics)
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Novel Agent API Info', version='1.0.0')

# Custom metrics for Token Tracking
TOKEN_USAGE_COUNTER = Counter(
    'llm_token_usage_total', 
    'Total LLM token usage', 
    ['model', 'token_type', 'agent']
)
# token_type: prompt / completion


# Agent Mapping
AGENTS = {
    "worldview": worldview_app,
    "outline": outline_app,
    "writing": writing_app,
    "router": router_app
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
                            "id": item.get("doc_id"),
                            "type": "worldview",
                            "name": item.get("name") or item.get("query"),
                            "content": item.get("content"),
                            "category": item.get("category", "Worldview"),
                            "timestamp": item.get("timestamp", "N/A")
                        })
                    except: pass

        # 2. Outlines (JSON Array)
        if os.path.exists('outlines_db.json'):
            try:
                with open('outlines_db.json', 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        items = json.loads(content)
                        for item in items:
                            all_docs.append({
                                "id": item.get("id"),
                                "type": "outline",
                                "name": f"大纲: {item.get('query', '未命名')[:20]}...",
                                "content": item.get("proposal") or json.dumps(item.get("outline", {}), ensure_ascii=False),
                                "category": "Outline",
                                "timestamp": item.get("timestamp", "刚刚")
                            })
            except Exception as e:
                print(f"[API ERROR] Failed to load outlines: {e}")

        # 3. Prose (JSONL)
        if os.path.exists('prose_db.json'):
            with open('prose_db.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        all_docs.append({
                            "id": item.get("scene_id") or item.get("id"),
                            "type": "prose",
                            "name": f"正文: {item.get('scene_title')}",
                            "content": item.get("content"),
                            "category": "Prose",
                            "timestamp": item.get("timestamp", "刚刚")
                        })
                    except: pass
    except Exception as e:
        print(f"[API ERROR] Global get_lore error: {e}")

    return jsonify(all_docs[::-1])

@app.route('/api/archive/update', methods=['POST'])
def update_archive():
    data = request.json or {}
    item_id = data.get('id')
    item_type = data.get('type')
    new_content = data.get('content')
    
    # Diagnostic logging
    print(f"[API UPDATE] Received: id={item_id}, type={item_type}, content_len={len(new_content) if new_content is not None else 'None'}")
    
    if not item_id or not item_type or new_content is None:
        return jsonify({
            "error": "Missing id, type, or content",
            "received": {"id": item_id, "type": item_type, "has_content": new_content is not None}
        }), 400
    
    filename = {
        'worldview': 'worldview_db.json',
        'outline': 'outlines_db.json',
        'prose': 'prose_db.json'
    }.get(item_type)
    
    if not filename or not os.path.exists(filename):
        return jsonify({"error": f"Invalid type or file not found: {item_type}"}), 400
    
    updated = False
    
    # CASE 1: JSON Array (Outline)
    if item_type == 'outline':
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                items = json.loads(content) if content else []
            
            for item in items:
                if str(item.get('id')) == str(item_id):
                    # Update proposal or whole outline structure if needed
                    # If content starts with {, assume it's the full outline JSON
                    if new_content.strip().startswith('{'):
                        try:
                            item['outline'] = json.loads(new_content)
                        except:
                            item['proposal'] = new_content
                    else:
                        item['proposal'] = new_content
                    updated = True
                    break
            
            if updated:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return jsonify({"error": f"Failed to update Outline JSON: {e}"}), 500

    # CASE 2: JSONL (Worldview, Prose)
    else:
        new_lines = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        current_id = None
                        if item_type == 'worldview':
                            current_id = item.get('doc_id')
                        elif item_type == 'prose':
                            current_id = item.get('scene_id') or item.get('id')
                        
                        if str(current_id) == str(item_id):
                            item['content'] = new_content
                            line = json.dumps(item, ensure_ascii=False) + '\n'
                            updated = True
                        else:
                            if not line.endswith('\n'): line += '\n'
                    except:
                        if not line.endswith('\n'): line += '\n'
                    new_lines.append(line)
            
            if updated:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
        except Exception as e:
            return jsonify({"error": f"Failed to update JSONL: {e}"}), 500

    if updated:
        # 同步到 MongoDB, ChromaDB 和 SKILL
        try:
            from lore_utils import sync_archive_to_all_stores
            sync_archive_to_all_stores(item_id, item_type, new_content, name=data.get('name'))
        except Exception as e:
            print(f"[API] Sync error: {e}")

        return jsonify({"status": "success"})
    else:
        return jsonify({"error": f"Item {item_id} not found in {item_type}"}), 404

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
    
    # 1. 自动路由处理 (Router Logic)
    if agent_type == 'router':
        # 调用 Router 获取意图
        # 注入 LangFuse 观测回调
        langfuse_handler = get_langfuse_callback()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
            print(f"[OBSERVABILITY] LangFuse tracing enabled for router.")

        router_state = agent_app.invoke({"query": query}, config=config)
        
        # 提取 Token 消耗并上报 Prometheus
        if "metadata" in router_state and "usage_metadata" in router_state["metadata"]:
            usage = router_state["metadata"]["usage_metadata"]
            report_token_usage(
                model=CONFIG.get("DEFAULT_MODEL", "gemini-flash"),
                prompt_tokens=usage.get("prompt_token_count", 0),
                completion_tokens=usage.get("candidates_token_count", 0),
                agent_name="router"
            )
        intent = router_state.get("intent", "unknown")
        print(f"[API] Router identified intent: {intent}")
        
        if intent != 'unknown' and intent in AGENTS:
            # 自动切换到目标 Agent
            agent_type = intent
            agent_app = AGENTS[intent]
            # 继续执行后续的初始化逻辑
        else:
            # 意图不明，直接返回 Router 结果
            state_snapshot = agent_app.get_state(config)
            return jsonify(dict(state_snapshot.values))

    # 2. 状态初始化与调用子 Agent
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
        # 注入 LangFuse 观测回调
        langfuse_handler = get_langfuse_callback()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
            print(f"[OBSERVABILITY] LangFuse tracing enabled for {agent_type}.")

        output = agent_app.invoke(input_state, config=config)
        
        # 尝试上报 Token 消耗 (如果 output 中包含元数据)
        if isinstance(output, dict) and "metadata" in output and "usage_metadata" in output["metadata"]:
            usage = output["metadata"]["usage_metadata"]
            report_token_usage(
                model=CONFIG.get("DEFAULT_MODEL", "gemini-flash"),
                prompt_tokens=usage.get("prompt_token_count", 0),
                completion_tokens=usage.get("candidates_token_count", 0),
                agent_name=agent_type
            )
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
        # 再次确认回调注入
        langfuse_handler = get_langfuse_callback()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

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
                 from langgraph.types import Command
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

# ==========================================
# C 层：实体哨兵 API (Entity Sentinel Endpoints)
# ==========================================

@app.route('/api/entity-drafts', methods=['GET'])
def list_entity_drafts():
    """获取待审实体列表"""
    status = request.args.get('status', 'pending')
    drafts = get_draft_entities(status_filter=status if status != 'all' else None)
    return jsonify(drafts)

@app.route('/api/entity-drafts/approve', methods=['POST'])
def approve_entity():
    """批准待审实体 → 写入正式世界观库"""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "缺少 name 参数"}), 400
    success = approve_draft_entity(data['name'])
    if success:
        return jsonify({"status": "approved", "name": data['name']})
    return jsonify({"error": f"实体 '{data['name']}' 未找到或已处理"}), 404

@app.route('/api/entity-drafts/reject', methods=['POST'])
def reject_entity():
    """拒绝待审实体"""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "缺少 name 参数"}), 400
    
    # 更新草案库中的状态为 rejected
    if not os.path.exists('entity_drafts_db.json'):
        return jsonify({"error": "草案库不存在"}), 404
    
    all_drafts = []
    found = False
    with open('entity_drafts_db.json', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if d.get('name') == data['name'] and d.get('status') == 'pending':
                d['status'] = 'rejected'
                found = True
            all_drafts.append(d)
    
    if not found:
        return jsonify({"error": f"实体 '{data['name']}' 未找到或已处理"}), 404
    
    with open('entity_drafts_db.json', 'w', encoding='utf-8') as f:
        for d in all_drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    return jsonify({"status": "rejected", "name": data['name']})

if __name__ == '__main__':
    # 启动 5005 端口
    app.run(port=5005, host='0.0.0.0', debug=True)
