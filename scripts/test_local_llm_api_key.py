#!/usr/bin/env python3
"""Direct local LLM connectivity test using LOCAL_API_KEY from .env/config."""

from __future__ import annotations

import json
import sys

import requests

from src.common.config_utils import load_config


def main() -> int:
    config = load_config()
    local_api_key = config.get("LOCAL_API_KEY")
    llm_models = config.get("LLM_MODELS") or {}
    local_config = llm_models.get("local") or {}
    base_url = local_config.get("base_url") or config.get("LOCAL_LLM_URL") or "http://localhost:8317/v1"
    model = local_config.get("default") or config.get("DEFAULT_MODEL") or "gemini-3-flash"

    if not local_api_key:
        print("FAILED: LOCAL_API_KEY missing in .env, config/secrets.yml, or environment")
        return 1

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {local_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a connectivity test assistant."},
            {"role": "user", "content": "Reply with exactly: LOCAL_API_KEY_OK"},
        ],
        "temperature": 0,
    }

    print(json.dumps({
        "url": url,
        "model": model,
        "auth_header_prefix": f"Bearer {local_api_key[:6]}..." if len(local_api_key) >= 6 else "Bearer ***",
    }, ensure_ascii=False, indent=2))

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as exc:
        print(f"FAILED: request error: {exc}")
        return 2

    print(f"HTTP {response.status_code}")
    print(json.dumps({
        "content_type": response.headers.get("Content-Type"),
        "server": response.headers.get("Server"),
    }, ensure_ascii=False, indent=2))
    try:
        data = response.json()
    except Exception:
        print(response.text[:2000])
        return 3

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if response.status_code >= 400:
        return 4

    choices = data.get("choices") or []
    content = ""
    if choices:
        content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    if "LOCAL_API_KEY_OK" not in content:
        print(f"FAILED: unexpected model response: {content}")
        return 5

    print("SUCCESS: local model accepted LOCAL_API_KEY and returned the expected content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
