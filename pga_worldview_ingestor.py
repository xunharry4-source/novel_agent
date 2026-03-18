import os
import json
import uuid
import datetime
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import pymongo

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")

# 1. MongoDB 设置 (包含回退机制)
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    mongo_client.server_info()
    db = mongo_client["pga_worldview"]
    lore_coll = db["lore"]
    print("[成功] MongoDB 已连接。")
except Exception:
    print("[警告] MongoDB 不可用。回退使用 worldview_db.json。")
    lore_coll = None

# 2. ChromaDB 设置
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(
    client=chroma_client,
    collection_name="pga_worldview_v1",
    embedding_function=embeddings
)

def parse_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照 ### 分段，解析种族/机制条目
    sections = content.split('### ')
    parsed_docs = []
    category = "常规"
    
    for section in sections:
        if not section.strip(): continue
        
        lines = section.split('\n')
        title = lines[0].strip()
        body = '\n'.join(lines[1:]).strip()
        
        # 简单分类检测
        if "碳基" in section: category = "碳基生命"
        elif "硅基" in section: category = "硅基生命"
        elif "智械" in section: category = "智械生命"
        elif "能量" in section: category = "能量生命"
        
        parsed_docs.append({
            "doc_id": str(uuid.uuid4()),
            "name": title,
            "category": category,
            "content": f"### {title}\n{body}",
            "timestamp": datetime.datetime.now().isoformat()
        })
    return parsed_docs

def ingest_all(docs):
    print(f"开始同步 {len(docs)} 条设定数据...")
    for i, doc in enumerate(docs):
        # 保存到 Mongo/JSON
        if lore_coll:
            lore_coll.insert_one(doc.copy())
        else:
            with open("worldview_db.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        # 保存到 Chroma (处理频率限制)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(1) # 免费版限频缓冲
                vector_store.add_texts(
                    texts=[doc["content"]],
                    metadatas=[{"name": doc["name"], "category": doc["category"], "doc_id": doc["doc_id"]}],
                    ids=[doc["doc_id"]]
                )
                if i % 10 == 0: print(f"当前进度: {i}/{len(docs)}")
                break
            except Exception as e:
                if "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"触发限频。等待 {wait}秒后重试...")
                    time.sleep(wait)
                else:
                    print(f"错误 - 条目 {doc['name']}: {e}")
                    break

if __name__ == "__main__":
    lore_docs = parse_markdown("科幻.md")
    ingest_all(lore_docs)
