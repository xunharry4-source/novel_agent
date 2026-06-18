"""
LLM Factory - PGA 小说创作引擎模型工厂
支持多种模型提供商 (Gemini, OpenAI, Local) 的统一架构。
"""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from .config_utils import load_config
from .logger_utils import get_logger
from .usage_utils import update_agent_usage
from langchain_openai import ChatOpenAI

logger = get_logger("novel_agent.llm_factory")


def _parse_timeout(value, fallback: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except (TypeError, ValueError):
        return fallback

def _provider_model_config(config: dict, provider: str) -> dict:
    catalog = config.get("LLM_MODELS") or {}
    provider_config = catalog.get(provider) or {}
    if isinstance(provider_config, str):
        provider_config = {"default": provider_config, "models": [provider_config]}
    elif isinstance(provider_config, list):
        provider_config = {"default": provider_config[0] if provider_config else None, "models": provider_config}

    default_model = (
        provider_config.get("default")
        or (config.get("DEFAULT_MODEL_MAP") or {}).get(provider)
        or config.get("DEFAULT_MODEL")
    )
    models = provider_config.get("models") or ([default_model] if default_model else [])
    normalized = {**provider_config, "default": default_model, "models": models}

    if provider == "ollama":
        normalized["base_url"] = normalized.get("base_url") or config.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    elif provider == "local":
        normalized["base_url"] = normalized.get("base_url") or config.get("LOCAL_LLM_URL", "http://localhost:5000/v1")
    return normalized


def _default_timeout_for(provider: str, agent_name: str) -> int:
    if agent_name.endswith("_review_agent"):
        return 180
    if provider == "gemini":
        return 120
    return 60


def _resolve_timeout(config: dict, provider: str, agent_name: str, agent_config: dict | str | None) -> int:
    default_timeout = _default_timeout_for(provider, agent_name)
    if isinstance(agent_config, dict):
        timeout_value = agent_config.get("timeout")
        if timeout_value is not None:
            return _parse_timeout(timeout_value, default_timeout)
    provider_timeout_map = config.get("PROVIDER_TIMEOUTS") or {}
    if provider in provider_timeout_map:
        return _parse_timeout(provider_timeout_map.get(provider), default_timeout)
    if agent_name.endswith("_review_agent") and config.get("REVIEW_AGENT_TIMEOUT") is not None:
        return _parse_timeout(config.get("REVIEW_AGENT_TIMEOUT"), default_timeout)
    return default_timeout

class UsageTrackingCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks token usage per agent."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_llm_end(self, response, **kwargs) -> None:
        """Collect usage data when LLM finished."""
        try:
            for generation in response.generations:
                for chunk in generation:
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'response_metadata'):
                        meta = chunk.message.response_metadata
                        # Try OpenAI style
                        usage = meta.get('token_usage')
                        if usage:
                            input_tokens = usage.get('prompt_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0)
                            update_agent_usage(self.agent_name, input_tokens, output_tokens)
                            return

                        # Try Google Gemini style
                        usage = meta.get('usage_metadata')
                        if usage:
                            input_tokens = usage.get('prompt_token_count', 0)
                            output_tokens = usage.get('candidates_token_count', 0)
                            update_agent_usage(self.agent_name, input_tokens, output_tokens)
                            return
        except Exception as e:
            logger.error(f"Error tracking usage for {self.agent_name}: {e}")

def get_llm(json_mode: bool = False, agent_name: str = "unknown"):
    """
    根据模块化配置返回对应的 LLM 实例。
    优先级: Agent 专属配置 > Provider 默认配置 > 系统全局默认
    """
    config = load_config()
    provider = config.get("LLM_PROVIDER", "ollama").lower()

    # 1. 尝试获取 Agent 专属配置 (支持字符串或字典)
    agent_models = config.get("AGENT_MODELS", {})
    agent_config = agent_models.get(agent_name, {})

    if isinstance(agent_config, dict):
        provider = (agent_config.get("provider") or provider).lower()
        model_name = agent_config.get("model")
    else:
        model_name = agent_config

    timeout = _resolve_timeout(config, provider, agent_name, agent_config)

    # 2. 如果没有专属模型名，获取提供商默认模型
    if not model_name:
        model_name = _provider_model_config(config, provider).get("default")

    # 3. 如果依然没有，使用系统全局默认
    if not model_name:
        model_name = config.get("DEFAULT_MODEL", "gemma4:e2b")

    logger.info(f"Instantiating LLM for agent '{agent_name}' using model '{model_name}' (Provider: {provider})")

    # 4. 初始化模型
    usage_handler = UsageTrackingCallbackHandler(agent_name)

    if provider == "gemini":
        from .lore_utils import _key_manager
        key = _key_manager.get_key()
        if not key:
            raise ValueError("GOOGLE_API_KEYS missing in config/secrets.yml or environment")

        args = {
            "model": model_name,
            "google_api_key": key,
            "max_retries": 5,
            "timeout": timeout,
            "callbacks": [usage_handler]
        }
        if json_mode:
            args["response_mime_type"] = "application/json"

        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**args)

    elif provider == "openai":
        api_key = config.get("OPENAI_API_KEY")
        base_url = config.get("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in config/secrets.yml or environment")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            timeout=timeout,
            callbacks=[usage_handler],
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {}
        )

    elif provider == "local":
        base_url = _provider_model_config(config, provider).get("base_url")

        return ChatOpenAI(
            model=model_name,
            api_key="sk-not-required",
            base_url=base_url,
            temperature=0.7,
            timeout=timeout,
            callbacks=[usage_handler],
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {}
        )

    elif provider == "ollama":
        base_url = _provider_model_config(config, provider).get("base_url")

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key="ollama", # Ollama doesn't require key but some libs expect non-empty
            base_url=base_url,
            temperature=0.7,
            timeout=timeout,
            callbacks=[usage_handler],
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {}
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def get_provider_info():
    """返回当前提供商和所有模型配置信息的元数据"""
    config = load_config()
    provider = config.get("LLM_PROVIDER", "ollama")
    model_map = config.get("DEFAULT_MODEL_MAP", {})
    provider_config = _provider_model_config(config, provider)
    active_model = provider_config.get("default") or config.get("DEFAULT_MODEL", "gemma4:e2b")
    return {
        "provider": provider,
        "model": active_model,
        "default_model": config.get("DEFAULT_MODEL", "gemma4:e2b"),
        "base_url": provider_config.get("base_url"),
        "review_agent_timeout": _parse_timeout(config.get("REVIEW_AGENT_TIMEOUT"), _default_timeout_for(provider, "chapter_review_agent")),
        "agent_models": config.get("AGENT_MODELS", {}),
        "model_map": model_map,
        "llm_models": config.get("LLM_MODELS", {})
    }
