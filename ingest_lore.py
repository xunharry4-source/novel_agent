import os
import json
import uuid
import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import pymongo

# Configuration (Matching Main Agent)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDrS1FZCh0oWB4t4DCRb0f6dowtGKgEwm0"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")

# DB Connections
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    mongo_client.server_info()
    db = mongo_client["pga_worldview"]
    print("[SUCCESS] Connected to MongoDB.")
except Exception:
    print("[WARNING] MongoDB connection failed. Results will be saved to worldview_db.json.")
    db = None

try:
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    chroma_client.heartbeat()
    print("[SUCCESS] Connected to ChromaDB Server.")
except Exception:
    print("[WARNING] ChromaDB Server not found. Using local persistent storage (./chroma_db).")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

vector_store = Chroma(
    client=chroma_client,
    collection_name="pga_lore",
    embedding_function=embeddings
)

def parse_and_ingest(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_section = ""
    current_content = []
    category = "General"
    
    ingested_count = 0
    
    for line in lines:
        # Check for major categories
        if line.startswith("## "):
            category = line.strip("# ").strip()
            
        # Check for sub-sections (e.g., ### 种族)
        if line.startswith("### "):
            # Save previous section if exists
            if current_section and current_content:
                save_doc(current_section, category, "\n".join(current_content))
                ingested_count += 1
            
            current_section = line.strip("# ").strip()
            current_content = [line]
        else:
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        save_doc(current_section, category, "\n".join(current_content))
        ingested_count += 1
        
    print(f"[FINISHED] Ingested {ingested_count} sections into databases.")

def save_doc(name, category, content):
    import time
    doc_id = str(uuid.uuid4())
    payload = {
        "doc_id": doc_id,
        "name": name,
        "category": category,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # 1. Save to MongoDB or JSON
    if db:
        try:
            db.lore_collection.insert_one(payload.copy())
        except Exception as e:
            print(f"[ERROR] Mongo Insert failed: {e}")
    else:
        with open("worldview_db.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 2. Index in ChromaDB (with Retry for Rate Limits)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Add a small delay for free tier stability
            time.sleep(1) 
            vector_store.add_texts(
                texts=[content],
                metadatas=[{"name": name, "category": category, "doc_id": doc_id}],
                ids=[doc_id]
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 5 * (attempt + 1)
                print(f"[WARNING] Rate limit hit. Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Chroma Indexing failed: {e}")
                break

if __name__ == "__main__":
    parse_and_ingest("科幻.md")
