import os
from unittest.mock import patch

from src.common.config_utils import load_config
from src.common.llm_factory import get_llm


def test_load_config_reads_local_api_key_from_env_file():
    config = load_config()
    assert config.get("LOCAL_API_KEY"), "LOCAL_API_KEY should be loaded from .env or secrets"


def test_get_llm_local_uses_local_api_key():
    captured = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("src.common.llm_factory.ChatOpenAI", DummyChatOpenAI):
        llm = get_llm(agent_name="local_api_key_test_agent")

    assert llm is not None
    assert captured["api_key"] == load_config().get("LOCAL_API_KEY")
    assert captured["base_url"] == ((load_config().get("LLM_MODELS") or {}).get("local") or {}).get("base_url")
    assert captured["model"] == ((load_config().get("LLM_MODELS") or {}).get("local") or {}).get("default")


def test_get_llm_local_requires_local_api_key():
    original = os.environ.get("LOCAL_API_KEY")
    try:
        os.environ.pop("LOCAL_API_KEY", None)

        with patch("src.common.llm_factory.load_config", return_value={
            "LLM_PROVIDER": "local",
            "DEFAULT_MODEL": "gemini-3-flash",
            "LLM_MODELS": {"local": {"default": "gemini-3-flash", "models": ["gemini-3-flash"], "base_url": "http://localhost:8317/v1"}},
            "AGENT_MODELS": {},
        }):
            try:
                get_llm(agent_name="local_api_key_missing_test_agent")
            except ValueError as exc:
                assert "LOCAL_API_KEY missing" in str(exc)
            else:
                raise AssertionError("Expected get_llm(local) to require LOCAL_API_KEY")
    finally:
        if original is not None:
            os.environ["LOCAL_API_KEY"] = original
