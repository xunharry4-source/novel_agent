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

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Set up your API keys and observability DSNs in `config.json`:
```json
{
    "GOOGLE_API_KEYS": ["YOUR_API_KEY_1", "YOUR_API_KEY_2"],
    "SENTRY_DSN": "YOUR_SENTRY_DSN",
    "LANGFUSE_PUBLIC_KEY": "YOUR_LANGFUSE_PK",
    "LANGFUSE_SECRET_KEY": "YOUR_LANGFUSE_SK"
}
```

### 3. Start Observability Services (Optional)
Ensure Docker is installed:
```bash
cd observability
docker-compose up -d
```

### 4. Run Main Service
```bash
python app_api.py
```
Access `http://127.0.0.1:5005` to open the Omni-Dashboard.

---

## ⚙️ Core Development Principles
-   **Dual-DB Atomicity**: Approved settings sync to MongoDB (Full Text) and ChromaDB (Vector Index).
-   **Human-defined Authority**: Critical plot points are locked in SKILLs and cannot be overridden by AI.
-   **Observability First**: All agent executions must be traceable and measurable.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
