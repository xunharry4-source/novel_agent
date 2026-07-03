"""
资料与工具集 (Lore Utils) - PGA 小说创作引擎通用工具库

本模块集成了系统运行所需的底层工具，是 RAG (检索增强生成) 架构的核心支撑。
核心功能设计:
1. 双库协同检索: 
   - MongoDB: 存储权威的设定定义 (Canon)、分类模板和结构化元数据。
   - ChromaDB: 存储设定内容的向量索引，支持跨维度的语义相似度搜索。
2. 动态 Context 组装: 根据用户 Query 自动路由，从禁止项、分类规则和历史文献中提取最相关的上下文。
3. API Key 生命周期管理: 实现多 Key 自动轮询 (Rotation) 和 429 错误自愈。
4. 模型工厂: 统一初始化带 JSON 模式支持的生成模型和嵌入模型。
"""
import os
import pymongo
import chromadb
import json
import re
import datetime
import uuid
from typing import Dict, List, Optional, Any, cast
import time
from chromadb.config import Settings
from langchain_chroma import Chroma
from .config_utils import load_config, BASE_DIR
from .logger_utils import get_logger
from .dify_sync_utils import get_dify_client
from .ollama_embeddings import OllamaEmbeddings

logger = get_logger("novel_agent.lore")
try:
    from langfuse.callback import CallbackHandler
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    print("[WARN] langfuse-python not installed, LLM tracing disabled.")

from langchain_core.callbacks import BaseCallbackHandler

class AtomicLogHandler(BaseCallbackHandler):
    """
    Custom LangChain callback handler to capture 'atomic' logs from nodes.
    """
    def __init__(self, on_log_func):
        self.on_log_func = on_log_func

    def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> Any:
        if name == "atomic_log":
            self.on_log_func(data)

HAS_DOCX = False
try:
    import docx
    HAS_DOCX = True
except ImportError:
    pass

HAS_PDF = False
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    pass
    
def get_db_path(filename: str, outline_id: Optional[str] = None, worldview_id: Optional[str] = None) -> str:
    """
    获取数据文件的路径。
    层级结构:
    - 全局注册表: data/filename
    - 世界观层: data/worldviews/{worldview_id}/filename
    - 小说/大纲层: data/worldviews/{worldview_id}/outlines/{outline_id}/filename
    """
    # Use DB_PATH from config, default to "data"
    config = load_config()
    db_base = config.get("DB_PATH", "data")
    
    # If it's a relative path, make it relative to the project root (BASE_DIR)
    if not os.path.isabs(db_base):
        base_dir = os.path.join(BASE_DIR, db_base)
    else:
        base_dir = db_base
    
    # 1. 全局注册表 (不依赖任何 ID)
    global_registries = ["outlines_db.json", "worldviews_registry.json", "worldview_templates.json"]
    if filename in global_registries:
        data_dir = base_dir
    
    # 2. 层级化路径处理
    else:
        # 确定世界观 ID
        wid = worldview_id if worldview_id else "default_wv"
        
        if outline_id:
            # 大纲层 (属于某个世界观)
            data_dir = os.path.join(base_dir, "worldviews", wid, "outlines", outline_id)
        else:
            # 世界观层 (共享设定层)
            data_dir = os.path.join(base_dir, "worldviews", wid)
            
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

# ==========================================
# API Key Manager (Global Singleton for the module)
# ==========================================
class APIKeyManager:
    def __init__(self):
        config = load_config()
        self.keys = config.get("GOOGLE_API_KEYS", [])
        if not self.keys:
            # Fallback to single key if list is missing
            single_key = config.get("GOOGLE_API_KEY")
            if single_key:
                self.keys = [single_key]
        self.index = 0
        
    def get_key(self):
        if not self.keys:
            return None
        return self.keys[self.index]
    
    def rotate(self):
        if len(self.keys) <= 1:
            return False
        self.index = (self.index + 1) % len(self.keys)
        print(f"[lore_utils] API Key Rotated to Index {self.index}")
        return True

def dispatch_log(config: Dict[str, Any], message: str):
    """
    Utility to dispatch an atomic log event from within a LangGraph node.
    Requires an AtomicLogHandler to be present in the config callbacks.
    """
    from langchain_core.callbacks.manager import CallbackManager
    callbacks = config.get("callbacks")
    if callbacks:
        logger.info(f"[ATOMIC] {message}") # 同步记录到物理日志文件
        if isinstance(callbacks, list):
            for cb in callbacks:
                if hasattr(cb, "on_custom_event"):
                    try:
                        cb.on_custom_event("atomic_log", message)
                    except Exception as e:
                        logger.error(f"Error in on_custom_event: {e}")
                        raise e
        elif hasattr(callbacks, "on_custom_event"):
            try:
                callbacks.on_custom_event("atomic_log", message)
            except Exception as e:
                logger.error(f"Error in on_custom_event fallback: {e}")
                raise e
    
    # Fallback to standard logging
    logger.info(f"[ATOMIC] {message}")

_key_manager = APIKeyManager()

# --- LangFuse Callback Helper ---
def get_langfuse_callback() -> Optional[Any]:
    """Returns a LangFuse CallbackHandler if keys are configured."""
    config = load_config()
    pk = config.get("LANGFUSE_PUBLIC_KEY")
    sk = config.get("LANGFUSE_SECRET_KEY")
    host = config.get("LANGFUSE_HOST")
    
    if HAS_LANGFUSE and pk and sk:
        return CallbackHandler(
            public_key=pk, 
            secret_key=sk, 
            host=host
        )
    return None

# --- Prometheus Global Counter (Imported from app_api if needed) ---
# We use a lazy reference to avoids circular imports
_token_counter = None
_request_counter = None

def report_token_usage(model: str, prompt_tokens: int, completion_tokens: int, agent_name: str = "unknown") -> None:
    """Reports token usage to Prometheus."""
    global _token_counter
    if _token_counter is None:
        try:
            from app_api import TOKEN_USAGE_COUNTER
            _token_counter = TOKEN_USAGE_COUNTER
        except ImportError:
            return
            
    if _token_counter:
        _token_counter.labels(model=model, token_type='prompt', agent_name=agent_name).inc(prompt_tokens)
        _token_counter.labels(model=model, token_type='completion', agent_name=agent_name).inc(completion_tokens)
        
    # Increment request counter too
    global _request_counter
    if _request_counter is None:
        try:
            from app_api import LLM_REQUEST_COUNTER
            _request_counter = LLM_REQUEST_COUNTER
        except ImportError:
            pass
            
    if _request_counter:
        try:
            _request_counter.labels(model=model, agent_name=agent_name).inc(1)
        except Exception:
            pass


# --- Lore Extraction & Sync ---

def clean_text(text: str) -> str:
    """清理文本：规范化换行，移除冗余空格和噪声数据"""
    if not text:
        return ""
    
    import re
    # 1. 规范化换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. 移除连续的空行 (保留最多两个)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. 移除行尾空格
    text = re.sub(r'[ \t]+\n', '\n', text)
    
    # 4. 移除 PDF/Word 中常见的页码噪声 (简单模式)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    return text.strip()

def extract_text_from_file(file_path: str) -> str:
    """从不同格式文件中提取文本"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".md" or ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = json.dumps(data, ensure_ascii=False, indent=2)
    elif ext == ".opml":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read() # 简化处理，暂时直接读原始文本
    elif ext == ".docx":
        if not HAS_DOCX:
            raise RuntimeError("Missing docx dependency: pip install python-docx")
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        if not HAS_PDF:
            raise RuntimeError("PyPDF2 is not installed. Please install it to support .pdf files.")
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    else:
        # 尝试作为纯文本读取
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            raise ValueError(f"Unsupported file format or error reading {ext}: {e}")
            
    if not text:
        return ""
        
    return clean_text(text)

def get_lore_by_doc_id(doc_id: str, outline_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """通过 doc_id 快速获取完整实体"""
    # 优先尝试 MongoDB
    try:
        config = load_config()
        mongo_uri = config.get("MONGO_URI", "mongodb://localhost:27017/")
        mongo_db = config.get("MONGO_DB_NAME", "pga_worldview")
        mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
        coll = mongo_client[mongo_db]["lore"]
        # If MongoDB is used, it should probably be namespaced too, but for now we filter by outline_id if present in doc
        query = {"doc_id": doc_id}
        if outline_id:
            query["outline_id"] = outline_id
            
        doc = coll.find_one(query)
        if doc: 
            if "_id" in doc: del doc["_id"]
            return doc
    except Exception as e:
        logger.error(f"MongoDB lookup failed for config {doc_id}: {e}")
        raise e
        
    # 后备尝试本地 JSON (JSONL 格式)
    try:
        db_path = get_db_path("worldview_db.json", outline_id=outline_id)
            
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if data.get("doc_id") == doc_id:
                            return data
                    except Exception:
                        continue
    except Exception as e:
        logger.error(f"Failed to read data for {doc_id}: {e}")
        raise e

def sync_lore_to_db(entity: Dict[str, Any], outline_id: Optional[str] = None, worldview_id: Optional[str] = None) -> None:
    """
    同步设定实体：
    1. 写入主库 (MongoDB/JSON) 存储完整父节点。
    2. 生成精细化子片段 (Children) 同步至 ChromaDB 进行索引。
    """
    # 1. 准备元数据
    if 'doc_id' not in entity:
        import uuid
        entity['doc_id'] = str(uuid.uuid4())
    if 'timestamp' not in entity:
        import datetime
        entity['timestamp'] = datetime.datetime.now().isoformat()
    if outline_id:
        entity['outline_id'] = outline_id
    if worldview_id:
        entity['worldview_id'] = worldview_id
        
    # 2. 写入主库 (存储完整 Parent)
    try:
        db = get_mongodb_db()
        coll = db["lore"]
        coll.update_one({"doc_id": entity["doc_id"]}, {"$set": entity}, upsert=True)
    except Exception as e:
        logger.error(f"Failed to sync to MongoDB: {e}")
        raise e

    # 3. 同步至 ChromaDB (父子结构)
    try:
        config = load_config()
        
        # Use worldview_id for vector database isolation
        v_store = get_vector_store(worldview_id=(worldview_id or "default_wv"), outline_id=outline_id)
        
        # --- 实现父子切片逻辑 ---
        texts_to_index = []
        metadatas = []
        ids = []
        
        content = entity["content"]
        
        # A. 首先存储 Parent 节点本身 (用于兜底匹配)
        texts_to_index.append(content)
        metadatas.append({
            "name": entity["name"],
            "category": entity["category"],
            "doc_id": entity["doc_id"],
            "doc_type": "parent"
        })
        ids.append(entity["doc_id"])
        
        # B. 如果内容较长，生成子片段 (Children) 以提升检索精度
        if len(content) > 500:
            chunk_size = 200
            overlap = 50
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if len(chunk) < 50: continue # 太短的片段略过
                
                texts_to_index.append(chunk)
                metadatas.append({
                    "name": entity["name"],
                    "parent_id": entity["doc_id"],
                    "category": entity["category"],
                    "doc_type": "child",
                    "chunk_index": i
                })
                ids.append(f"{entity['doc_id']}_chunk_{i}")
        
        # 执行批量写入
        if v_store is not None:
            v_store.add_texts(
                texts=texts_to_index,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[SYNC] Lore '{entity['name']}' synced with {len(texts_to_index)} indices (Parent-Child).")
        else:
            logger.warning(f"[SYNC SKIP] Vector store unavailable, skipping semantic indexing for '{entity['name']}'.")
        
        # 4. 可选：同步至 Dify 知识库 (RAG API)
        try:
            dify_client = get_dify_client()
            if dify_client:
                dataset_map = config.get("DIFY_DATASET_MAP", {})
                category = entity.get("category", "").lower()
                dataset_id = dataset_map.get(category)
                
                if dataset_id:
                    # 获取现有的 Dify Document ID (如果有)
                    dify_doc_id = entity.get("metadata", {}).get("dify_document_id")
                    
                    # 生成 Dify 同步文本 (Name + Content)
                    sync_text = f"Title: {entity['name']}\n\n{entity['content']}"
                    
                    sync_res = dify_client.upsert_document(
                        dataset_id=dataset_id,
                        name=entity["name"],
                        text=sync_text,
                        document_id=dify_doc_id
                    )
                    
                    if sync_res.get("success"):
                        # 如果是第一次创建，存储 document_id 以供后续更新
                        if not dify_doc_id:
                            if "metadata" not in entity: entity["metadata"] = {}
                            entity["metadata"]["dify_document_id"] = sync_res["document_id"]
                            # 更新主库以保存此元数据
                            coll.update_one({"doc_id": entity["doc_id"]}, {"$set": {"metadata": entity["metadata"]}})
                        logger.info(f"[DIFY SYNC] Successfully synced '{entity['name']}' to dataset {dataset_id}")
                    else:
                        logger.warning(f"[DIFY ERROR] Sync failed: {sync_res.get('error')}")
                else:
                    logger.debug(f"[DIFY SKIP] No dataset ID mapped for category: {category}")
            else:
                logger.debug("[DIFY SKIP] No Dify API Key configured.")
        except Exception as de:
            logger.error(f"[DIFY ERROR] Critical sync error: {de}")
            
    except Exception as e:
        print(f"[SYNC ERROR] ChromaDB sync failed for {entity['name']}: {e}")

def get_evolution_rules() -> str:
    """提取系统在运行中自主学习沉淀的防错法则"""
    try:
        path = os.path.join(BASE_DIR, ".gemini", "skills", "evolution", "SKILL.md")
        if not os.path.exists(path):
            return "暂无进化法则。"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            start = "<!-- EVOLUTION_RECORDS_START -->"
            end = "<!-- EVOLUTION_RECORDS_END -->"
            if start in content and end in content:
                records = content.split(start)[1].split(end)[0].strip()
                return records if records else "暂无进化法则。"
            return "暂无进化法则。"
    except Exception:
        return "暂无进化法则。"

def get_llm(json_mode=False, agent_name="unknown"):
    """
    统一模型初始化入口，委托给 llm_factory 处理多提供商逻辑。
    """
    from .llm_factory import get_llm as factory_get_llm
    return factory_get_llm(json_mode=json_mode, agent_name=agent_name)

def parse_json_safely(text: str) -> Any:
    """
    更鲁棒地解析 LLM 返回的 JSON 字符串。
    1. 移除 Markdown 代码块标记 (```json ... ```)
    2. 自动修正尾部逗号 (Trailing Commas)
    3. 提取包含在文本中的 JSON 对象
    """
    if not text:
        return None
    
    # 清理 markdown 代码块
    cleaned = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    # 尝试提取第一个 { 或 [ 之后的内容
    if not (cleaned.startswith('{') or cleaned.startswith('[')):
        match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
            
    # 修正尾部逗号 (匹配 , 后面跟着 } 或 ])
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[JSON Error] Failed to parse: {e}")
        print(f"[JSON Content] {cleaned[:200]}...")
        # 最后的尝试：ast.literal_eval (能处理单引号等)
        try:
            import ast
            return ast.literal_eval(cleaned)
        except Exception as e:
            logger.error(f"AST parse failed: {e}")
            raise e

# Cache for vector stores per novel
_vector_store_cache = {}

def _embedding_model_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    catalog = config.get("EMBEDDING_MODELS") or {}
    provider_config = catalog.get(provider) or {}
    if isinstance(provider_config, str):
        provider_config = {"default": provider_config, "models": [provider_config]}
    elif isinstance(provider_config, list):
        provider_config = {"default": provider_config[0] if provider_config else None, "models": provider_config}

    default_model = provider_config.get("default")
    if not default_model:
        if provider == "ollama":
            default_model = (
                config.get("DEFAULT_EMBEDDING_MODEL")
                or config.get("OLLAMA_EMBEDDING_MODEL")
                or "embeddinggemma"
            )
        elif provider in {"gemini", "google"}:
            default_model = "models/gemini-embedding-001"

    models = provider_config.get("models") or ([default_model] if default_model else [])
    normalized = {**provider_config, "default": default_model, "models": models}
    if provider == "ollama":
        normalized["base_url"] = normalized.get("base_url") or config.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return normalized

def get_embedding_function(task_type: str = "retrieval_query"):
    config = load_config()
    provider = (config.get("EMBEDDING_PROVIDER") or "ollama").lower()
    provider_config = _embedding_model_config(config, provider)
    model_name = provider_config.get("default")

    if provider == "ollama":
        return OllamaEmbeddings(
            model=model_name,
            base_url=provider_config.get("base_url"),
        )

    if provider in {"gemini", "google"}:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = _key_manager.get_key()
        if not key:
            raise ValueError("GOOGLE_API_KEY missing for embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=key,
            task_type=task_type,
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

def get_embedding_provider_info() -> Dict[str, Any]:
    config = load_config()
    provider = (config.get("EMBEDDING_PROVIDER") or "ollama").lower()
    provider_config = _embedding_model_config(config, provider)
    return {
        "provider": provider,
        "model": provider_config.get("default"),
        "default_model": config.get("DEFAULT_EMBEDDING_MODEL") or provider_config.get("default"),
        "embedding_models": config.get("EMBEDDING_MODELS", {}),
    }

def get_vector_store(worldview_id: str = "default_wv", outline_id: Optional[str] = None):
    """
    获取向量数据库实例。支持层级化隔离。
    - 如果仅提供 worldview_id: 返回共享世界观设定库 (pga_wv_{wid})
    - 如果提供 outline_id: 返回该小说特定的语境库 (pga_prose_{oid})
    """
    global _vector_store_cache
    
    # Use a safe collection name
    if outline_id:
        safe_id = outline_id.replace("-", "_")
        collection_name = f"pga_prose_{safe_id}"
    else:
        safe_id = worldview_id.replace("-", "_")
        collection_name = f"pga_wv_{safe_id}"
    
    if collection_name in _vector_store_cache:
        return _vector_store_cache[collection_name]
        
    emb = get_embedding_function(task_type="retrieval_query")
    
    # Persistent storage for Chroma
    config = load_config()
    db_base = config.get("DB_PATH", "data")
    if not os.path.isabs(db_base):
        db_root = os.path.join(BASE_DIR, db_base)
    else:
        db_root = db_base
    
    chroma_path = os.path.join(db_root, "chroma_db")
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    
    vs = Chroma(client=client, collection_name=collection_name, embedding_function=emb)
    _vector_store_cache[collection_name] = vs
    return vs

def delete_lore_vector(doc_id: str, outline_id: Optional[str] = None, worldview_id: str = "default_wv"):
    """从向量数据库中删除指定条目"""
    try:
        vs = get_vector_store(worldview_id=worldview_id, outline_id=outline_id)
        # Chroma (LangChain) supports delete by ids
        vs.delete(ids=[doc_id])
        logger.info(f"[CHROMA DELETE] Successfully deleted vector for doc_id: {doc_id} in {outline_id}")
        return True
    except Exception as e:
        logger.error(f"[CHROMA DELETE ERROR] Failed to delete {doc_id} in {outline_id}: {e}")
        return False

def get_lore_collection_name():
    config = load_config()
    return config.get("CHROMA_COLLECTION_NAME", "pga_worldview_v1")

def rotate_api_key():
    """暴露给外部的旋转接口"""
    return _key_manager.rotate()

def get_mongodb_db():
    config = load_config()
    mongo_uri = config.get("MONGO_URI", "mongodb://localhost:27017/")
    mongo_db = config.get("MONGO_DB_NAME", "pga_worldview")
    
    try:
        # 尝试原始 URI
        mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        mongo_client.server_info()
        return mongo_client[mongo_db]
    except Exception as first_e:
        # 物理回退逻辑：如果 localhost 失败，尝试 127.0.0.1
        if "localhost" in mongo_uri:
            alt_uri = mongo_uri.replace("localhost", "127.0.0.1")
            try:
                logger.info(f"Localhost connection failed, attempting fallback to 127.0.0.1: {alt_uri}")
                mongo_client = pymongo.MongoClient(alt_uri, serverSelectionTimeoutMS=2000)
                mongo_client.server_info()
                return mongo_client[mongo_db]
            except Exception as second_e:
                logger.error(f"MongoDB physical connection failed (localhost & 127.0.0.1): {second_e}")
                raise second_e
        else:
            logger.error(f"MongoDB connection failed: {first_e}")
            raise first_e

# --- Core Context Retrieval Logic ---

def get_prohibited_rules(outline_id: Optional[str] = None):
    """从数据库或本地文件动态获取禁止项目"""
    try:
        db = get_mongodb_db()
        if db is not None:
            rule_doc = db["prohibited_rules"].find_one({"name": "PGA核心禁令"})
            if rule_doc:
                return rule_doc["content"]
        
    except Exception as e:
        logger.error(f"Failed to get prohibited rules: {e}")
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for filename in ("info.md", "technical_design_ZH.md"):
            path = os.path.join(root_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
    except Exception as e:
        logger.error(f"Failed to load local prohibited rules fallback: {e}")
    return ""

def get_worldview_context_by_category(query, outline_id: Optional[str] = None, worldview_id: Optional[str] = None):
    """
    根据查询内容识别涉及的世界观分类，并从 MongoDB/JSON 中检索“权威定义”。
    特别优先处理：种族 (Races) 和 势力 (Factions)。
    """
    category_map = {
        "faction": ["势力", "国家", "组织", "军团", "公约", "强国", "联邦", "帝国", "派系", "阵营"],
        "mechanism_tech": ["机制", "协议", "技术", "代偿", "热力学", "规则", "引擎", "武器", "装置", "科技", "原理", "戴森球", "发动机", "运作"],
        "race": ["种族", "智械", "机器", "生命", "熵族", "奥族", "秦族", "生物", "族群", "演化", "物种"],
        "history": ["历史", "记录", "演变", "纪元", "战争", "变迁", "编年史", "事件"],
        "geography": ["地理", "星域", "恒星", "行星", "环境", "坐标", "星区", "星系", "地形"]
    }

    
    detected_categories = []
    potential_names = []
    for cat, keys in category_map.items():
        for k in keys:
            if k in query:
                detected_categories.append(cat)
                if len(k) >= 2: potential_names.append(k)
            
    if not detected_categories:
        return ""
        
    context_blocks = []
    try:
        db_path = get_db_path("worldview_db.json", outline_id=outline_id, worldview_id=worldview_id)
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    res_name = data.get("name", "")
                    res_path = data.get("path", "")
                    res_content = data.get("content", "")
                    
                    is_authority = False
                    if any(pn in res_name for pn in potential_names):
                        is_authority = True
                    
                    if not is_authority:
                        for cat in detected_categories:
                            if cat == "race" and ("种族" in res_path or "生命" in res_path):
                                is_authority = True
                            if cat == "faction" and ("势力" in res_path or "国家" in res_path or "组织" in res_path):
                                is_authority = True
                            if cat == "geography" and ("地理" in res_path or "星域" in res_path):
                                is_authority = True
                            if cat == "mechanism_tech" and ("机制" in res_path or "科技" in res_path or "技术" in res_path):
                                is_authority = True
                            if cat == "history" and ("历史" in res_path or "事件" in res_path):
                                is_authority = True

                                
                    if is_authority:
                        if any(pn in res_content for pn in potential_names) or any(pn in res_name for pn in potential_names):
                            context_blocks.append(f"【权威定义 - {res_path}】:\n{res_content}")
    except Exception:
        pass
    return "\n\n".join(context_blocks[:5])
        
def get_category_template(category):
    """从 MongoDB 或本地 JSON 获取分类模板及参考例子"""
    try:
        db = get_mongodb_db()
        if db is not None:
            template_doc = db["worldview_templates"].find_one({"category": category.lower()})
            if template_doc:
                template_doc.pop("_id", None)
                return template_doc
    except Exception as e:
        logger.warning(f"MongoDB template lookup failed, using local fallback: {e}")
    
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldview", "worldview_templates.json")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        for template in templates:
            if template.get("category", "").lower() == category.lower():
                return template
    
    raise ValueError(f"Template for category '{category}' not found in MongoDB.")

def upsert_category_template(category, template_data):
    """保存或更新分类模板"""
    db = get_mongodb_db()
    data = template_data.copy()
    data["category"] = category.lower()
    
    if db is not None:
        db["worldview_templates"].update_one(
            {"category": category.lower()},
            {"$set": data},
            upsert=True
        )
        return True
    
    raise ConnectionError("MongoDB is not connected. Cannot upsert template.")

def delete_category_template(category):
    """删除指定分类模板"""
    cat_lower = category.lower()
    db = get_mongodb_db()
    
    if db is not None:
        db["worldview_templates"].delete_one({"category": cat_lower})
        return True
    
    raise ConnectionError("MongoDB is not connected. Cannot delete template.")

def add_new_category(category, name_zh, template_fields=None, example_fields=None):
    """新建一个完整的分类模板"""
    cat_lower = category.lower()
    
    # 检查是否已存在
    existing = get_category_template(cat_lower)
    if existing:
        return False, f"分类 '{category}' 已存在"
    
    data = {
        "category": cat_lower,
        "name_zh": name_zh,
        "template": template_fields or {"name": f"[{name_zh}名称]"},
        "example": example_fields or {"name": f"示例{name_zh}"}
    }
    
    success = upsert_category_template(cat_lower, data)
    return success, "创建成功" if success else "创建失败"

def get_all_templates():
    """获取所有分类模板，仅从 MongoDB 获取"""
    templates_dict = {}
    
    try:
        db = get_mongodb_db()
        cursor = db["worldview_templates"].find({})
        for doc in cursor:
            doc.pop("_id", None)
            cat = doc.get("category")
            if cat:
                templates_dict[cat] = doc
    except Exception as e:
        logger.warning(f"MongoDB template list failed, using local fallback: {e}")
    
    if templates_dict:
        return templates_dict

    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldview", "worldview_templates.json")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            for doc in json.load(f):
                cat = doc.get("category")
                if cat:
                    templates_dict[cat] = doc
    return templates_dict



def get_unified_context(query, outline_id="default", worldview_id="default_wv", retry_on_429=True):
    """
    智能路由检索：自动处理 429 错误并尝试 Key 轮换，支持按 outline_id 隔离。
    """
    context_blocks = []
    db = get_mongodb_db()
    if db is not None:
        # 简单文本匹配 - 对 query 做正则转义，防止章节内容中的特殊字符引发 MongoDB $regex 错误
        safe_query = re.escape(query.strip()[:200])  # 截断并转义，避免过长 query 导致性能问题
        try:
            cursor = db["lore"].find({"name": {"$regex": safe_query, "$options": "i"}}).limit(3)
            for doc in cursor:
                context_blocks.append(f"【权威设定: {doc['name']}】\n{doc['content']}")
        except Exception:
            pass  # MongoDB regex 仍失败则跳过，不阻塞后续检索
    else:
        raise ConnectionError("MongoDB is not available for context retrieval.")

    # 2. 尝试从 ChromaDB 检索背景资料 (隔离不同小说)
    try:
        vector_store = get_vector_store(worldview_id=worldview_id, outline_id=outline_id if outline_id != "default" else None)
        if vector_store:
            results = vector_store.similarity_search(query, k=5)
            for res in results:
                context_blocks.append(f"【背景资料: {res.metadata.get('name', '未命名')}】\n{res.page_content}")
    except Exception as e:
        if "429" in str(e) and retry_on_429:
            if rotate_api_key():
                print(f"[lore_utils] API Key Rotated. Retrying context retrieval for {outline_id}...")
                return get_unified_context(query, outline_id=outline_id, retry_on_429=False)
        print(f"[lore_utils] ChromaDB Search Error ({outline_id}): {e}")

    if context_blocks:
        unique_blocks = list(dict.fromkeys(context_blocks))
        return "\n\n".join(unique_blocks[:8])
    return ""

def get_grounded_context(query, outline_id: Optional[str] = None, worldview_id: str = "default_wv") -> List[Dict[str, str]]:
    """
    获取带索引的素材块，支持按 outline_id 隔离。
    """
    sources = []
    
    # 1. MongoDB (暂未隔离)
    db = get_mongodb_db()
    if db is not None:
        try:
            mongo_query: Dict[str, Any] = {"name": {"$regex": re.escape(query.strip()[:200]), "$options": "i"}}
            if worldview_id:
                mongo_query["worldview_id"] = worldview_id
            if outline_id:
                mongo_query["outline_id"] = outline_id
            cursor = db["lore"].find(mongo_query).limit(3)
            for doc in cursor:
                sources.append({
                    "id": f"S{len(sources)+1}",
                    "title": f"权威设定: {doc['name']}",
                    "content": doc['content']
                })
        except Exception:
            pass

    # 2. ChromaDB (已隔离)
    try:
        vector_store = get_vector_store(worldview_id=worldview_id, outline_id=outline_id)
        if vector_store:
            results = vector_store.similarity_search(query, k=5)
            for res in results:
                sources.append({
                    "id": f"S{len(sources)+1}",
                    "title": f"背景资料: {res.metadata.get('name', '未命名')}",
                    "content": res.page_content
                })
    except Exception:
        pass
        
    return sources[:10]

def format_grounded_context_for_prompt(sources: List[Dict[str, str]]) -> str:
    """将素材块格式化为 Prompt 引用段落"""
    if not sources:
        return "【无相关源素材】"
    
    formatted = "### 可用源素材 (Sources):\n"
    for s in sources:
        formatted += f"[{s['id']}] {s['title']}\n{s['content']}\n\n"
    return formatted

def get_latest_book_outline(worldview_id: Optional[str] = None):
    """获取最近一次保存的全局大纲"""
    db_path = get_db_path("outlines_db.json")
    if not os.path.exists(db_path):
        return None
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
            if not lines: return None
            for line in reversed(lines):
                item = json.loads(line.strip())
                if not worldview_id or item.get("worldview_id") == worldview_id:
                    return item
            return None
    except Exception as e:
        print(f"[lore_utils] Error reading outlines_db.json: {e}")
        raise e

def get_outline_by_id(outline_id):
    """根据 ID 获取特定大纲内容"""
    db = get_mongodb_db()
    query = {"$or": [{"id": outline_id}, {"outline_id": outline_id}]}
    try:
        data = db["outlines"].find_one(query)
        return data
    except Exception as e:
        logger.error(f"Error searching outline {outline_id}: {e}")
        raise e


# ==========================================
# Entity Sentinel (实体哨兵) Utilities
# ==========================================

def get_entity_registry(outline_id: Optional[str] = None, worldview_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    从指定小说的 worldview_db.json 中扫描所有已注册实体的名称。
    """
    registry: Dict[str, List[str]] = {}
    
    # 从本地 JSON 隔离扫描
    db_path = get_db_path("worldview_db.json", outline_id=outline_id, worldview_id=worldview_id)
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    cat = data.get("category", "general")
                    name = data.get("name", "")
                    if name:
                        if cat not in registry:
                            registry[cat] = []
                        if name not in registry[cat]:
                            registry[cat].append(name)
        except Exception as e:
            print(f"[Entity Sentinel] Error scanning worldview_db.json: {e}")
    
    # 从 MongoDB 补充
    # 从 MongoDB 扫描
    db = get_mongodb_db()
    if db is not None:
        try:
            query: Dict[str, Any] = {}
            if outline_id:
                query["outline_id"] = outline_id
            if worldview_id:
                query["worldview_id"] = worldview_id
            cursor = db["lore"].find(query, {"name": 1, "category": 1})
            for doc in cursor:
                cat = doc.get("category", "general")
                name = doc.get("name", "")
                if name:
                    if cat not in registry:
                        registry[cat] = []
                    if name not in registry[cat]:
                        registry[cat].append(name)
        except Exception:
            pass
    
    return registry


def format_entity_registry_for_prompt(registry: Dict[str, List[str]]) -> str:
    """
    将实体注册表格式化为 Prompt 可注入的文本块。
    """
    if not registry:
        return "【已注册实体清单】暂无已注册实体。你可以自由创建，但请在输出中标注 new_entities。"
    
    lines = ["【已注册实体清单 — 优先使用以下实体，如需创建新实体请在 JSON 中增加 \"new_entities\": [{\"name\": \"...\", \"type\": \"...\", \"reason\": \"...\"}] 数组】"]
    for cat, names in registry.items():
        cat_label = {
            "race": "种族", "faction": "势力", "geography": "地理",
            "mechanism_tech": "科技/机制", "history": "历史事件",
            "planet": "星球", "creature": "生物", "weapon": "武器装备",
            "organization": "组织", "religion": "宗教", "crisis": "危机事件",
        }.get(cat, cat)
        lines.append(f"  - {cat_label}: {', '.join(names)}")
    
    return "\n".join(lines)


def register_draft_entity(entity_name: str, entity_type: str, source_context: str, 
                          source_agent: str = "unknown", entity_card: Optional[Dict] = None,
                          outline_id: Optional[str] = None, worldview_id: Optional[str] = None) -> bool:
    """
    将新发现的实体写入"待审区" entity_drafts_db.json。
    C 层 - 不直接入正式库，等待用户审批。
    
    Args:
        entity_name: 实体名称 (如 "凯恩")
        entity_type: 实体分类 (如 "character", "faction", "tech")
        source_context: 实体出现的上下文描述
        source_agent: 来源 Agent (如 "outline", "writing")
        entity_card: 完整的实体设定卡 (基于分类模板生成)
    """
    import datetime as _dt
    record = {
        "name": entity_name,
        "type": entity_type,
        "source_context": source_context,
        "source_agent": source_agent,
        "entity_card": entity_card or {},
        "status": "pending",  # pending / approved / rejected
        "created_at": _dt.datetime.now().isoformat(),
        "outline_id": outline_id,
        "worldview_id": worldview_id or "default_wv"
    }
    try:
        db_path = get_db_path("entity_drafts_db.json", outline_id=outline_id, worldview_id=worldview_id)
        with open(db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[Entity Sentinel] 新实体草案已登记: {entity_name} ({entity_type})")
        return True
    except Exception as e:
        print(f"[Entity Sentinel] 登记失败: {e}")
        raise e



def get_draft_entities(status_filter: Optional[str] = "pending", outline_id: Optional[str] = None) -> List[Dict]:
    """获取待审实体列表。C 层 API 使用。"""
    drafts = []
    db_path = get_db_path("entity_drafts_db.json", outline_id=outline_id)
    if not os.path.exists(db_path):
        return drafts
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if status_filter is None or data.get("status") == status_filter:
                    drafts.append(data)
    except Exception as e:
        print(f"[Entity Sentinel] 读取草案库失败: {e}")
    return drafts


def approve_draft_entity(entity_name: str, outline_id: Optional[str] = None) -> bool:
    """批准并持久化单个实体草案"""
    all_drafts = []
    target = None
    
    db_path = get_db_path("entity_drafts_db.json", outline_id=outline_id)
    if not os.path.exists(db_path):
        return False
        
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # 仅批准第一个找到的 pending 状态实体，确保单次操作的精确性
                if not target and data.get("name") == entity_name and data.get("status") == "pending":
                    data["status"] = "approved"
                    target = data
                    print(f"[lore_utils] Approved individual draft: {entity_name}")
                all_drafts.append(data)
    except Exception as e:
        print(f"[lore_utils] Error reading drafts for approval: {e}")
        return False
    
    if not target:
        print(f"[lore_utils] No pending draft found for: {entity_name}")
        return False
        
    # 持久化到正式设定库
    try:
        add_to_worldview_db(target, outline_id=outline_id)
    except Exception as e:
        print(f"[lore_utils] Failed to add {entity_name} to worldview: {e}")
        return False
        
    # 保存更新后的草案库
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            for d in all_drafts:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[lore_utils] Error writing drafts after approval: {e}")
        return False

def add_to_worldview_db(target: Dict[str, Any], outline_id: Optional[str] = None, worldview_id: Optional[str] = None):
    """将草案实体转录并写入正式世界观库 (MongoDB)"""
    entity_card = target.get("entity_card", {})
    if entity_card:
        content = json.dumps(entity_card, ensure_ascii=False, indent=2)
    else:
        content = f"[自动注册] {target.get('source_context', '')}"
    
    canon_record = {
        "doc_id": target.get("doc_id") or f"lore_{uuid.uuid4().hex[:8]}",
        "name": target["name"],
        "category": target.get("type", "general"),
        "content": content,
        "path": f"自动注册/{target.get('type', 'general')}/{target['name']}",
        "outline_id": outline_id,
        "worldview_id": worldview_id or target.get("worldview_id") or "default_wv",
        "timestamp": datetime.datetime.now().isoformat()
    }
    sync_lore_to_db(canon_record, outline_id=outline_id, worldview_id=worldview_id)

def batch_approve_draft_entities(entity_names: List[str], outline_id: Optional[str] = None) -> Dict[str, Any]:
    """批量批准实体草案 (MongoDB)"""
    db = get_mongodb_db()
    approved_count = 0
    try:
        for name in entity_names:
            # 找到对应的待审草案
            draft = db["entity_drafts"].find_one({"name": name, "status": "pending", "outline_id": outline_id})
            if draft:
                # 1. 批准并写入世界观
                add_to_worldview_db(draft, outline_id=outline_id)
                # 2. 更新草案状态
                db["entity_drafts"].update_one({"_id": draft["_id"]}, {"$set": {"status": "approved"}})
                approved_count += 1
        return {"success": approved_count, "failed": len(entity_names) - approved_count}
    except Exception as e:
        logger.error(f"Batch approve failed: {e}")
        raise e

def batch_reject_draft_entities(entity_names: List[str], outline_id: Optional[str] = None) -> Dict[str, Any]:
    """批量拒绝实体草案 (MongoDB)"""
    db = get_mongodb_db()
    rejected_count = 0
    try:
        res = db["entity_drafts"].update_many(
            {"name": {"$in": entity_names}, "status": "pending", "outline_id": outline_id},
            {"$set": {"status": "rejected"}}
        )
        rejected_count = res.modified_count
        return {"success": rejected_count, "failed": len(entity_names) - rejected_count}
    except Exception as e:
        logger.error(f"Batch reject failed: {e}")
        raise e
    
def sync_archive_to_all_stores(item_id: str, item_type: str, content: str, name: Optional[str] = None, outline_id: Optional[str] = None, worldview_id: Optional[str] = None) -> bool:
    """
    将修改后的条目同步到 MongoDB, ChromaDB 和技能系统 (SKILL)。
    """
    print(f"[lore_utils] Syncing {item_type} ID: {item_id} to all stores (Outline: {outline_id})...")
    
    # 1. MongoDB Sync (For Worldview)
    if item_type == 'worldview':
        db = get_mongodb_db()
        db_path = get_db_path("worldview_db.json", outline_id=outline_id)
        if db is not None:
            try:
                # Add outline_id to query if present
                query = {"doc_id": item_id}
                if outline_id: query["outline_id"] = outline_id
                
                doc = db["lore"].find_one(query)
                if doc and doc.get("content") and doc.get("content") != content:
                    db["lore"].update_one(
                        query,
                        {
                            "$set": {"content": content, "name": name, "timestamp": datetime.datetime.now().isoformat()},
                            "$push": {
                                "history": {
                                    "$each": [{"timestamp": doc.get("timestamp", datetime.datetime.now().isoformat()), "content": doc.get("content")}],
                                    "$slice": -10
                                }
                            }
                        }
                    )
                else:
                    db["lore"].update_one(
                        {"doc_id": item_id},
                        {"$set": {"content": content, "name": name, "timestamp": datetime.datetime.now().isoformat()}},
                        upsert=True
                    )
                print(f"[lore_utils] MongoDB sync success for {item_id}")
            except Exception as e:
                print(f"[lore_utils] MongoDB sync error: {e}")

    # 2. ChromaDB Sync (Vector Re-indexing)
    try:
        vector_store = get_vector_store(worldview_id=worldview_id or "default_wv", outline_id=outline_id)
        if vector_store:
            # 在 ChromaDB 中，我们通常使用 doc_id 作为 metadata 的一部分
            # 这里采取：先删除旧的，再插入新的（最简单的同步方式）
            # 注意：这需要 item_id 在 ChromaDB 中是唯一的标识符
            
            # 由于 LangChain Chroma 封装的原因，直接按 metadata 删除比较慢
            # 如果我们在存储时将 doc_id 设置为 Chroma ID，则可以直接 update
            
            # 尝试删除旧记录 (如果存在)
            # 注意：item_id 必须与存储时的 ID 一致
            try:
                vector_store.delete(ids=[item_id])
            except Exception as e:
                logger.warning(f"Failed to delete old vector for {item_id}: {e}")

            vector_store.add_texts(
                texts=[content],
                ids=[item_id],
                metadatas=[{"name": name or "未命名", "type": item_type, "doc_id": item_id, "timestamp": datetime.datetime.now().isoformat()}]
            )
            print(f"[lore_utils] ChromaDB sync success for {item_id}")
    except Exception as e:
        print(f"[lore_utils] ChromaDB sync error: {e}")

    # 3. SKILL Sync (For Outlines)
    if item_type == 'outline':
        try:
            from src.common.lore_skill_converter import generate_modular_skills
            generate_modular_skills()
            print(f"[lore_utils] SKILL (ANCHORS) sync success for {item_id}")
        except Exception as e:
            print(f"[lore_utils] SKILL sync error: {e}")
            
    return True

def get_all_lore_items(outline_id=None, worldview_id=None, novel_id=None, world_id=None, page=1, page_size=100):
    """从 MongoDB 分页聚合有明确条件的资料（世界观、大纲、正文）。"""
    if not any([outline_id, worldview_id, novel_id, world_id]):
        raise ValueError("get_all_lore_items requires one of outline_id, worldview_id, novel_id, world_id")
    try:
        page = int(page)
        page_size = int(page_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("page and page_size must be integers") from exc
    if page < 1:
        raise ValueError("page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")
    skip = (page - 1) * page_size
    all_docs = []
    try:
        query = {}
        if worldview_id: query["worldview_id"] = worldview_id
        if outline_id: query["outline_id"] = outline_id
        if novel_id: query["novel_id"] = novel_id
        if world_id: query["world_id"] = world_id

        db = get_mongodb_db()
        lore_cursor = db["lore"].find(query).sort("timestamp", -1).skip(skip).limit(page_size)
        for item in lore_cursor:
            all_docs.append({
                "id": item.get("doc_id") or str(item.get("_id")),
                "type": "worldview",
                "name": item.get("name") or item.get("query") or "未命名条目",
                "content": item.get("content"),
                "category": item.get("path") or item.get("category", "Worldview"),
                "timestamp": item.get("timestamp", "N/A"),
                "outline_id": item.get("outline_id"),
                "novel_id": item.get("novel_id"),
                "worldview_id": item.get("worldview_id"),
                "world_id": item.get("world_id")
            })

        outline_query = {}
        if worldview_id:
            outline_query["worldview_id"] = worldview_id
        if novel_id:
            outline_query["novel_id"] = novel_id
        if world_id:
            outline_query["world_id"] = world_id
        if outline_id:
            outline_query["$or"] = [{"outline_id": outline_id}, {"id": outline_id}]
        outline_cursor = db["outlines"].find(outline_query).sort("timestamp", -1).skip(skip).limit(page_size)
        for item in outline_cursor:
            curr_oid = item.get("outline_id") or item.get("id")
            if not curr_oid:
                continue
            all_docs.append({
                "id": curr_oid,
                "type": "outline",
                "name": item.get("name") or item.get("title") or item.get("query") or "未命名大纲",
                "content": item.get("content") or item.get("summary") or item.get("proposal"),
                "category": f"Outlines > {item.get('book_title') or 'Novel'}",
                "timestamp": item.get("timestamp", "N/A"),
                "outline_id": curr_oid,
                "novel_id": item.get("novel_id"),
                "worldview_id": item.get("worldview_id") or "default_wv",
                "world_id": item.get("world_id") or "world_default"
            })

        prose_query = {}
        if worldview_id:
            prose_query["worldview_id"] = worldview_id
        if novel_id:
            prose_query["novel_id"] = novel_id
        if world_id:
            prose_query["world_id"] = world_id
        if outline_id:
            prose_query["outline_id"] = outline_id
        prose_cursor = db["prose"].find(prose_query).sort("timestamp", -1).skip(skip).limit(page_size)
        for item in prose_cursor:
            prose_id = item.get("prose_id") or item.get("scene_id") or item.get("id")
            if not prose_id:
                continue
            all_docs.append({
                "id": prose_id,
                "type": "prose",
                "name": item.get("title") or item.get("scene_title") or item.get("name") or item.get("query") or "未命名正文",
                "content": item.get("content"),
                "category": "Proses",
                "timestamp": item.get("timestamp", "N/A"),
                "outline_id": item.get("outline_id"),
                "novel_id": item.get("novel_id"),
                "worldview_id": item.get("worldview_id") or "default_wv",
                "world_id": item.get("world_id") or "world_default"
            })
    except Exception as e:
        logger.error(f"[LORE UTILS ERROR] get_all_lore_items: {e}")
        raise e
        
    return all_docs[:page_size]
