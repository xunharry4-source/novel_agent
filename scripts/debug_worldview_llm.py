#!/usr/bin/env python3
"""Single-command worldview consistency LLM debugger."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents import review_agent
from src.common.lore_utils import get_langfuse_callback, get_llm, get_mongodb_db


OUTPUT_DIR = REPO_ROOT / "scratch" / "worldview_llm_debug"
PROMPT_FILE = OUTPUT_DIR / "worldview_consistency.prompt.txt"
RESPONSE_FILE = OUTPUT_DIR / "worldview_consistency.response.txt"

DEFAULT_PAYLOAD = {
    "world_id": "world_agent_novel_8ed501d17a",
    "worldview_id": "",
    "target_id": "",
    "name": "晶钜石",
    "summary": "超强能量石",
}


def ensure_prompt_file() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PROMPT_FILE.exists():
        return
    system_prompt, user_prompt = review_agent.build_review_messages(
        get_mongodb_db(),
        "worldview_consistency",
        DEFAULT_PAYLOAD,
    )
    PROMPT_FILE.write_text(
        f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}",
        encoding="utf-8",
    )


def parse_prompt(prompt_text: str) -> tuple[str, str]:
    system_marker = "[System]\n"
    user_marker = "\n\n[User]\n"
    if not prompt_text.startswith(system_marker) or user_marker not in prompt_text:
        raise ValueError(
            f"Prompt file format invalid: {PROMPT_FILE}\n"
            "Expected '[System]' followed by '[User]'."
        )
    body = prompt_text[len(system_marker):]
    system_prompt, user_prompt = body.split(user_marker, 1)
    return system_prompt.strip(), user_prompt.strip()


def extract_llm_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "")


def run_prompt(prompt_text: str) -> str:
    system_prompt, user_prompt = parse_prompt(prompt_text)
    config: dict[str, Any] = {}
    callback = get_langfuse_callback()
    if callback:
        config["callbacks"] = [callback]
    llm = get_llm(json_mode=True, agent_name="debug_worldview_consistency")
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        config=config if config else None,
    )
    return extract_llm_content(response)


def main() -> None:
    ensure_prompt_file()
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    raw_response = run_prompt(prompt_text)
    RESPONSE_FILE.write_text(raw_response, encoding="utf-8")
    print(f"prompt_file: {PROMPT_FILE}")
    print(f"response_file: {RESPONSE_FILE}")
    print("\n=== RESPONSE ===\n")
    print(raw_response)


if __name__ == "__main__":
    main()
