# Novel Agent Technical Design (English)

> Chinese version: [technical_design_ZH.md](./technical_design_ZH.md)

## 1. Overview

Novel Agent is a novel-production engine built on **Flask API**, **LangGraph agents**, and **MongoDB/ChromaDB**. The web UI talks to `src/app_api.py` only via REST.

## 2. Frontend (React primary)

| Item | Value |
|------|--------|
| Primary UI | `frontend/` — React 18 + Vite + TypeScript |
| Default start | `make start` → API `:5006`, UI `:5174` |
| API client | `frontend/src/api/client.ts` |

**Legacy UI (deprecated):** `ui/` NiceGUI on `:8501`. No new features. Optional install via `requirements-legacy.txt`. See [docs/frontend_strategy.md](./docs/frontend_strategy.md).

## 3. Backend & agents

- Entry: `src/app_api.py`
- Hierarchy agents: `src/agents/*_agent.py`
- Reviews: `src/agents/review_agent.py`, `src/agents/review_nodes/`
- Runs persisted in MongoDB `hierarchy_agent_runs`
- LLM: `llm_factory.py`, `config/llm.yml`, `llm_identity_registry.py`

## 4. Data

- MongoDB collections: `worlds`, `worldviews`, `novels`, `outlines`, `prose`, etc.
- ChromaDB: worldview/outline-scoped vectors
- Config: `config/storage.yml`

## 5. Further reading

- [docs/product_world_hierarchy_requirements.md](./docs/product_world_hierarchy_requirements.md)
- [docs/frontend_strategy.md](./docs/frontend_strategy.md)
- [docs/api.md](./docs/api.md)
