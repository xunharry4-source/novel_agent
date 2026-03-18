from flask import Flask, jsonify, request, send_file
import os
import json
from worldview_agent_langgraph import app as agent_app

app = Flask(__name__)

# 配置信息
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"

@app.route('/')
def index():
    return send_file('dashboard.html')

@app.route('/api/lore', methods=['GET'])
def get_lore():
    lore_list = []
    if os.path.exists('worldview_db.json'):
        with open('worldview_db.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    lore_list.append(json.loads(line))
    return jsonify(lore_list[::-1])

@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    try:
        data = request.json
        query = data.get('query', '')
        thread_id = data.get('thread_id', 'default_user')
        
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "query": query,
            "context": "基于当前 PGA 世界观协议进行扩展。",
            "proposal": "", "review_log": "", "user_feedback": "",
            "iterations": 0, "audit_count": 0, "is_approved": False,
            "category": "General", "doc_id": ""
        }
        
        output = agent_app.invoke(input_state, config=config)
        return jsonify(output)
    except Exception as e:
        print(f"[API ERROR] Query failed: {e}")
        return jsonify({"error": str(e), "status_message": "系统处理异常，请重试。"}), 500

@app.route('/api/agent/feedback', methods=['POST'])
def agent_feedback():
    try:
        data = request.json
        feedback = data.get('feedback', '')
        thread_id = data.get('thread_id', 'default_user')
        
        config = {"configurable": {"thread_id": thread_id}}
        # 显式传递反馈给状态机
        output = agent_app.invoke({"user_feedback": feedback}, config=config)
        return jsonify(output)
    except Exception as e:
        print(f"[API ERROR] Feedback failed: {e}")
        return jsonify({"error": str(e), "status_message": "反馈处理异常，请检查网络或会话。"}), 500

@app.route('/api/search', methods=['POST'])
def search_lore():
# ... (rest of search logic)
    query = request.json.get('query', '')
    # 延迟加载 Chroma 以避免不必要的依赖冲突
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import chromadb
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = Chroma(client=chroma_client, collection_name="pga_worldview_v1", embedding_function=embeddings)
    
    docs = vector_store.similarity_search(query, k=5)
    results = []
    for d in docs:
        results.append({
            "content": d.page_content,
            "metadata": d.metadata
        })
    return jsonify(results)

if __name__ == '__main__':
    # 启动 unified 服务
    app.run(port=5005, host='0.0.0.0')
