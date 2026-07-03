import os
import json
from dotenv import load_dotenv
from .logger_utils import get_logger

try:
    import yaml
except ImportError:  # pragma: no cover - only used before dependencies are installed.
    yaml = None

logger = get_logger("novel_agent.config")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LEGACY_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CONFIG_MODULES = [
    "app.yml",
    "llm.yml",
    "embeddings.yml",
    "storage.yml",
    "observability.yml",
    "integrations.yml",
    "secrets.yml",
]

CONFIG_FIELD_MODULES = {
    "app.yml": {"AUTONOMY_LEVEL"},
    "llm.yml": {
        "LLM_PROVIDER", "DEFAULT_MODEL", "DEFAULT_MODEL_MAP", "LLM_MODELS",
        "AGENT_MODELS", "OLLAMA_BASE_URL", "LOCAL_LLM_URL", "OPENAI_BASE_URL",
    },
    "embeddings.yml": {
        "EMBEDDING_PROVIDER", "DEFAULT_EMBEDDING_MODEL", "OLLAMA_EMBEDDING_MODEL",
        "EMBEDDING_MODELS",
    },
    "storage.yml": {"MONGO_URI", "MONGO_DB_NAME", "CHROMA_COLLECTION_NAME", "DB_PATH"},
    "observability.yml": {
        "LOG_PATH", "SENTRY_DSN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    },
    "integrations.yml": {"DIFY_API_KEY", "DIFY_BASE_URL", "DIFY_DATASET_MAP"},
    "secrets.yml": {"GOOGLE_API_KEY", "GOOGLE_API_KEYS", "OPENAI_API_KEY", "LOCAL_API_KEY"},
}


def _read_yaml_file(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML config files.")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config module must be a mapping: {path}")
    return data


def _load_file_config() -> dict:
    config = {}
    module_paths = [os.path.join(CONFIG_DIR, name) for name in CONFIG_MODULES]
    if any(os.path.exists(path) for path in module_paths):
        for path in module_paths:
            config.update(_read_yaml_file(path))
        return config

    if os.path.exists(LEGACY_CONFIG_PATH):
        with open(LEGACY_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return config


def _module_for_key(key: str) -> str:
    for module, keys in CONFIG_FIELD_MODULES.items():
        if key in keys:
            return module
    return "app.yml"


def _write_yaml_file(path: str, data: dict) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML config files.")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def load_config():
    """Centralized configuration loader for the Novel Agent system."""
    load_dotenv()

    try:
        config = _load_file_config()
    except Exception as e:
        logger.error(f"Failed to load config files: {e}")
        config = {}
    
    # Merge env vars for sensitive keys and LLM defaults
    env_keys = [
        "GOOGLE_API_KEY", "OPENAI_API_KEY", "LOCAL_API_KEY", "SENTRY_DSN", 
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
        "MONGO_URI", "MONGO_DB_NAME", "CHROMA_COLLECTION_NAME",
        "LLM_PROVIDER", "DEFAULT_MODEL", "OLLAMA_BASE_URL", "LOCAL_LLM_URL",
        "EMBEDDING_PROVIDER", "DEFAULT_EMBEDDING_MODEL", "OLLAMA_EMBEDDING_MODEL",
        "DEFAULT_LLM_PROVIDER", "DEFAULT_LLM_MODEL",
        "DB_PATH", "LOG_PATH"
    ]
    
    for key in env_keys:
        # Env vars take precedence over file config.
        val = os.getenv(key)
        if val:
            # Handle aliases
            if key == "DEFAULT_LLM_PROVIDER":
                config["LLM_PROVIDER"] = val
            elif key == "DEFAULT_LLM_MODEL":
                config["DEFAULT_MODEL"] = val
            else:
                config[key] = val

    # Ensure GOOGLE_API_KEYS (list) and GOOGLE_API_KEY (str) consistency
    if "GOOGLE_API_KEYS" not in config:
        config["GOOGLE_API_KEYS"] = [config.get("GOOGLE_API_KEY")] if config.get("GOOGLE_API_KEY") else []
    elif config.get("GOOGLE_API_KEYS") and not config.get("GOOGLE_API_KEY"):
        config["GOOGLE_API_KEY"] = config["GOOGLE_API_KEYS"][0]

    # Default values for essential non-sensitive fields if missing
    defaults = {
        "LLM_PROVIDER": "local",
        "DEFAULT_MODEL": "gemini-3-flash",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "AUTONOMY_LEVEL": "balanced",
        "EMBEDDING_PROVIDER": "ollama",
        "DEFAULT_EMBEDDING_MODEL": "embeddinggemma",
        "OLLAMA_EMBEDDING_MODEL": "embeddinggemma",
        "AGENT_MODELS": {},
        "LLM_MODELS": {
            "ollama": {
                "default": "gemma4:e2b",
                "models": ["gemma4:e2b"],
                "base_url": "http://localhost:11434/v1"
            },
            "gemini": {
                "default": "gemini-2.0-flash",
                "models": ["gemini-2.0-flash"]
            },
            "openai": {
                "default": "gpt-4-turbo-preview",
                "models": ["gpt-4-turbo-preview"]
            },
            "local": {
                "default": "gemini-3-flash",
                "models": ["gemini-3-flash"],
                "base_url": "http://localhost:8317/v1"
            }
        },
        "EMBEDDING_MODELS": {
            "ollama": {
                "default": "embeddinggemma",
                "models": ["embeddinggemma"],
                "base_url": "http://localhost:11434/v1"
            },
            "gemini": {
                "default": "models/gemini-embedding-001",
                "models": ["models/gemini-embedding-001"]
            }
        },
        "DEFAULT_MODEL_MAP": {
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4-turbo-preview",
            "local": "gemini-3-flash",
            "ollama": "gemma4:e2b"
        },
        "DB_PATH": "data",
        "LOG_PATH": "logs"
    }
    for key, val in defaults.items():
        if key not in config:
            config[key] = val

    # Backfill model catalogs from legacy fields so older configs stay valid.
    llm_models = config.setdefault("LLM_MODELS", {})
    model_map = config.setdefault("DEFAULT_MODEL_MAP", {})
    for provider_name, default_model in model_map.items():
        provider_models = llm_models.get(provider_name)
        if not isinstance(provider_models, dict):
            provider_models = {"default": default_model, "models": [default_model] if default_model else []}
        provider_models.setdefault("default", default_model)
        provider_models.setdefault("models", [provider_models["default"]] if provider_models.get("default") else [])
        if provider_name in {"ollama", "local"}:
            provider_models.setdefault(
                "base_url",
                config.get("OLLAMA_BASE_URL") if provider_name == "ollama" else config.get("LOCAL_LLM_URL", "http://localhost:8317/v1")
            )
        llm_models[provider_name] = provider_models

    embedding_models = config.setdefault("EMBEDDING_MODELS", {})
    default_embedding_model = (
        config.get("DEFAULT_EMBEDDING_MODEL")
        or config.get("OLLAMA_EMBEDDING_MODEL")
        or "embeddinggemma"
    )
    config["DEFAULT_EMBEDDING_MODEL"] = default_embedding_model
    config["OLLAMA_EMBEDDING_MODEL"] = config.get("OLLAMA_EMBEDDING_MODEL") or default_embedding_model
    ollama_embedding_models = embedding_models.get("ollama")
    if not isinstance(ollama_embedding_models, dict):
        ollama_embedding_models = {
            "default": default_embedding_model,
            "models": [default_embedding_model],
            "base_url": config.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        }
    ollama_embedding_models.setdefault("default", default_embedding_model)
    ollama_embedding_models.setdefault("models", [ollama_embedding_models["default"]])
    ollama_embedding_models.setdefault("base_url", config.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    embedding_models["ollama"] = ollama_embedding_models

    return config

def save_config(new_config: dict):
    """Persists configuration to module-scoped YAML files under config/."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        modules = {name: {} for name in CONFIG_MODULES}
        for key, value in new_config.items():
            if _module_for_key(key) == "secrets.yml":
                continue
            module = _module_for_key(key)
            modules[module][key] = value

        for module_name, data in modules.items():
            if module_name == "secrets.yml":
                continue
            path = os.path.join(CONFIG_DIR, module_name)
            if data:
                _write_yaml_file(path, data)
            elif os.path.exists(path):
                os.remove(path)
        return True
    except Exception as e:
        logger.error(f"Failed to save YAML config modules: {e}")
        return False

def get_config():
    return load_config()
