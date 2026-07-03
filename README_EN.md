# Novel Agent (Omni-Galaxy: Full-Stack AI Novel Writing Engine)

> This project is a professional novel creation and worldview management system driven by **LangGraph** and **Gemini**. It deconstructs complex writing processes into manageable Agent workflows using RAG (Retrieval-Augmented Generation) and Human-in-the-loop, ensuring high consistency and logical rigor in long-form narratives.

## 🌌 Core Concept: Structured Writing Process

The system splits novel creation into three core Agent stages:

1.  **Worldview Setting**: Use structured templates to define races, civilizations, technologies, and geography.
2.  **Outline Planning**: Generate plot outlines and pacing based on worldview settings.
3.  **Prose Execution**: Refine outlines into specific scenes and generate prose drafts with "Logic Snapshots".

---

## ✨ Main Features

### 1. Smart Agent Matrix
-   **Dispatcher**: Semantic recognition and multi-level routing, automatically distributing requests to the best sub-agent.
-   **Worldview Agent**: Generation and logical auditing for multiple categories like Races, Geography, Factions, etc.
-   **Outline Agent**: Structured novel planning, ensuring plot conflict aligns with worldview depth.
-   **Execution Agent**: Prose creation based on "Logic Snapshots", maintaining narrative continuity through scene deconstruction.

### 2. Full-Stack Observability
-   **Sentry**: Backend error capture and performance monitoring.
-   **LangFuse**: LangGraph execution tracing, allowing for prompt and token usage backtracking.
-   **Prometheus + Grafana**: System metrics monitoring, including custom `llm_token_usage_total` statistics.

### 3. Distributed Skill Architecture
-   **Framework**: Defines agent generation logic and "red lines" for auditing.
-   **Lore/Anchors**: Locks unchangeable plot twists and character fates.
-   **Catalog**: Implements physical slicing and "Active Window" management to keep context size manageable.

### 4. World Hierarchy Management
-   The `/worlds` page manages `World -> [Worldview, Novel] -> Outline -> Chapter`.
-   Worldviews describe a world's rules and settings; novels are stories that happen inside the same world. MongoDB stores them in `worlds`, `worldviews`, `novels`, `outlines`, and `prose`; chapters remain in the `prose` collection.
-   Create/update/delete actions run through independent hierarchy agents: `world_agent`, `worldview_agent`, `novel_agent`, `outline_agent`, and `chapter_agent`. World changes skip review; all other entity types require review plus human approval before the database write. Change requests default to partial rewrite, with full rewrite and minor edit as explicit alternatives; manually edited form fields are persisted with `manual_edit=true`.

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Configuration lives in modular `config/*.yml` files, including `llm.yml`, `embeddings.yml`, `storage.yml`, and `observability.yml`. Put secrets in `.env` or in git-ignored `config/secrets.yml`.

### 3. Start Observability Services (Optional)
Ensure Docker is installed:
```bash
cd observability
docker-compose up -d
```

### 4. Start System Services

**Recommended (React primary UI + API):**

```bash
make install   # first time: venv + Python/npm deps
make start     # API http://127.0.0.1:5006 + React http://127.0.0.1:5174
```

| Service | Port | Role |
|---------|------|------|
| Flask API | `5006` | Agents, DB, hierarchical RAG |
| React UI | `5174` | **Primary UI**: world hierarchy, novels, agent workflows |

Manual split terminals:

```bash
env PYTHONPATH=.:src .venv/bin/python src/app_api.py
cd frontend && npm run dev
```

> **Legacy NiceGUI UI (deprecated):** `ui/` is frozen. Optional: `make install-legacy` then `make start-legacy-ui` (`:8501`). See [docs/frontend_strategy.md](./docs/frontend_strategy.md).

---

## ⚙️ Core Development Principles
-   **Dual-DB Atomicity**: Approved settings sync to MongoDB (Full Text) and ChromaDB (Vector Index).
-   **Human-defined Authority**: Critical plot points are locked in SKILLs and cannot be overridden by AI.
-   **Observability First**: All agent executions must be traceable and measurable.
-   **No unbounded non-world reads**: Except for the `worlds` list, worldviews, novels, outlines, chapters, and workflow-run queries must include both a business condition and `page/page_size`. Missing filters or pagination return explicit `400` errors; APIs must not fall back to fake data or JSONL files.
-   **Idempotent Migration**: Run `PYTHONPATH=.:src:src/common .venv/bin/python scripts/migrate_world_hierarchy.py` to backfill legacy records with `world_id` and `novel_id`.
-   **Real API Verification**: Run `API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_world_hierarchy_requests.py`; the test uses `requests` against the live API and verifies every create/update/delete through follow-up queries.
-   **Real Agent Workflow Verification**: Run `API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_hierarchy_agent_workflow_requests.py`; it verifies independent agents, review nodes, human iteration, approval, real writes, and follow-up queries.
-   **Real Outline/Chapter Workflow Verification**: Run `API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_outline_chapter_workflow_requests.py`; it verifies conditional paginated state queries plus real chapter create/update/query behavior.
-   **Generated Backend API Docs**: Run `.venv/bin/python scripts/generate_api_docs.py` to generate [docs/api.md](./docs/api.md) and [docs/openapi.json](./docs/openapi.json) from Flask `@app.route` decorators in `src/app_api.py`. The generator parses source AST only; it does not import the Flask app, connect to MongoDB, or execute handlers.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
