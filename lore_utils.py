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
from typing import Dict, List, Optional, Any
import time
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config_utils import load_config

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

_key_manager = APIKeyManager()

# --- Client Initialization Helpers ---

def get_llm(json_mode=False):
    key = _key_manager.get_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY missing. Please check config.json.")
    
    config = load_config()
    model_name = config.get("DEFAULT_MODEL", "gemini-2.0-flash")
    
    if json_mode:
        return ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=key,
            model_kwargs={"response_mime_type": "application/json"}
        )
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=key)

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
        except Exception:
            return None

def get_vector_store():
    key = _key_manager.get_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY missing for embeddings.")
    
    # 注意：搜索时使用 retrieval_query 以获得更高相关性
    emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=key, 
        task_type="retrieval_query"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    return Chroma(client=client, collection_name="pga_lore", embedding_function=emb)

def rotate_api_key():
    """暴露给外部的旋转接口"""
    return _key_manager.rotate()

def get_mongodb_db():
    try:
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        mongo_client.server_info()
        return mongo_client["pga_worldview"]
    except Exception:
        return None

# --- Core Context Retrieval Logic ---

def get_prohibited_rules():
    """从数据库或本地文件动态获取禁止项目"""
    db = get_mongodb_db()
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
    """
    根据查询内容识别涉及的世界观分类，并从 MongoDB/JSON 中检索“权威定义”。
    特别优先处理：种族 (Races) 和 势力 (Factions)。
    """
    category_map = {
        "race": ["种族", "智械", "机器", "生命", "熵族", "奥族", "秦族", "生物", "族群", "演化", "物种"],
        "geography": ["地理", "星域", "恒星", "行星", "戴森球", "环境", "坐标", "星区", "星系", "地形"],
        "faction": ["势力", "国家", "组织", "军团", "公约", "强国", "联邦", "帝国", "派系", "阵营"],
        "mechanism_tech": ["机制", "协议", "技术", "代偿", "热力学", "规则", "引擎", "武器", "装置", "科技", "原理"],
        "history": ["历史", "记录", "演变", "纪元", "战争", "变迁", "编年史", "事件"]
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
        if os.path.exists("worldview_db.json"):
            with open("worldview_db.json", "r", encoding="utf-8") as f:
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
        
def get_category_template(category):
    """从 MongoDB 或本地 JSON 获取分类模板及参考例子"""
    db = get_mongodb_db()
    try:
        if db is not None:
            template_doc = db["worldview_templates"].find_one({"category": category.lower()})
            if template_doc:
                template_doc.pop("_id", None)
                return template_doc
    except Exception:
        pass
        
    # Fallback to local JSON
    if os.path.exists("worldview_templates.json"):
        with open("worldview_templates.json", "r", encoding="utf-8") as f:
            templates = json.load(f)
            for t in templates:
                if t.get("category") == category.lower():
                    return t
    return None

def upsert_category_template(category, template_data):
    """保存或更新分类模板"""
    db = get_mongodb_db()
    data = template_data.copy()
    data["category"] = category.lower()
    
    # Try MongoDB
    try:
        if db is not None:
            db["worldview_templates"].update_one(
                {"category": category.lower()},
                {"$set": data},
                upsert=True
            )
            return True
    except Exception:
        pass
        
    # Fallback/Sync to local JSON
    all_templates = []
    if os.path.exists("worldview_templates.json"):
        with open("worldview_templates.json", "r", encoding="utf-8") as f:
            all_templates = json.load(f)
            
    # Update existing or append
    found = False
    for i, t in enumerate(all_templates):
        if t.get("category") == category.lower():
            all_templates[i] = data
            found = True
            break
    if not found:
        all_templates.append(data)
        
    with open("worldview_templates.json", "w", encoding="utf-8") as f:
        json.dump(all_templates, f, ensure_ascii=False, indent=2)
    return True

def delete_category_template(category):
    """删除指定分类模板"""
    cat_lower = category.lower()
    db = get_mongodb_db()
    
    # Delete from MongoDB
    try:
        if db is not None:
            db["worldview_templates"].delete_one({"category": cat_lower})
    except Exception:
        pass
    
    # Delete from local JSON
    if os.path.exists("worldview_templates.json"):
        with open("worldview_templates.json", "r", encoding="utf-8") as f:
            all_templates = json.load(f)
        
        original_len = len(all_templates)
        all_templates = [t for t in all_templates if t.get("category") != cat_lower]
        
        if len(all_templates) < original_len:
            with open("worldview_templates.json", "w", encoding="utf-8") as f:
                json.dump(all_templates, f, ensure_ascii=False, indent=2)
            return True
    return False

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
    """获取所有分类模板，合并 MongoDB 和本地 JSON 数据"""
    db = get_mongodb_db()
    templates_dict = {}
    
    # 1. 首先加载本地 JSON 作为基准 (Fallback)
    if os.path.exists("worldview_templates.json"):
        try:
            with open("worldview_templates.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        cat = item.get("category")
                        if cat:
                            templates_dict[cat] = item
                elif isinstance(data, dict):
                    templates_dict.update(data)
        except Exception as e:
            print(f"Error loading local templates: {e}")
            
    # 2. 然后用 MongoDB 的数据覆盖/补充 (优先使用数据库中的自定义修改)
    try:
        if db is not None:
            cursor = db["worldview_templates"].find({})
            for doc in cursor:
                doc.pop("_id", None)
                cat = doc.get("category")
                if cat:
                    templates_dict[cat] = doc
    except Exception as e:
        print(f"Error loading MongoDB templates: {e}")
        
    return templates_dict



def get_unified_context(query, retry_on_429=True):
    """
    智能路由检索：自动处理 429 错误并尝试 Key 轮换
    """
    context_blocks = []
    
    # 1. 尝试从 MongoDB 检索权威定义 (Entity Design 意图)
    db = get_mongodb_db()
    if db is not None:
        try:
            # 简单文本匹配，优先找名称一致的
            cursor = db["lore"].find({"name": {"$regex": query, "$options": "i"}}).limit(3)
            for doc in cursor:
                context_blocks.append(f"【权威设定: {doc['name']}】\n{doc['content']}")
        except Exception:
            pass

    # 2. 尝试从 ChromaDB 检索背景资料 (Supportive Lore 意图)
    try:
        vector_store = get_vector_store()
        if vector_store:
            results = vector_store.similarity_search(query, k=5)
            for res in results:
                context_blocks.append(f"【背景资料: {res.metadata.get('name', '未命名')}】\n{res.page_content}")
    except Exception as e:
        if "429" in str(e) and retry_on_429:
            if rotate_api_key():
                print(f"[lore_utils] API Key Rotated. Retrying context retrieval...")
                return get_unified_context(query, retry_on_429=False)
        print(f"[lore_utils] ChromaDB Search Error: {e}")

    if context_blocks:
        unique_blocks = list(dict.fromkeys(context_blocks))
        return "\n\n".join(unique_blocks[:8])
    return ""
