# Technical Design Document: Worldview & Novel Agent

[English Version](./technical_design.md) | [中文版](./technical_design_ZH.md)

## 1. Role: Workflow Architect

As a **Workflow Architect**, the design of this system prioritizes **Stability**, **Scalability**, and **Self-correction (Self-healing)**. The architecture utilizes a state-machine approach implemented via LangGraph to manage complex interactions between AI generators, logical auditors, and human overseers.

### Core Principles
* **State Management**: Using `TypedDict` for data consistency across nodes.
* **Logic Closure**: Implementing "审核-退回-重写" (Audit-Reject-Rewrite) loops to maintain high output quality.
* **Human-in-the-loop**: Explicitly defined human intervention points for critical decisions.
* **Database Synchronization**: Multi-dataset transactionality ensuring MongoDB and ChromaDB remain in sync.

---

## 2. State Management (`AgentState`)

The `AgentState` is the backbone of the workflow, tracking the query, context, current proposal, and audit history.

```python
class AgentState(TypedDict):
    query: str
    context: str
    proposal: str
    review_log: str
    user_feedback: str
    iterations: int        # Counter for total generation attempts
    audit_count: int       # Counter for self-audit retries
    is_approved: bool      # Signal from reviewer or human
    category: str 
    doc_id: str
    status_message: str    # Real-time execution status
```

---

## 3. LangGraph Workflow Definition

The system consists of 4 primary functional nodes:

1. **`generator` (The Creator)**: Uses LLM to produce or refine a "Proposal" based on the user's query and RAG context.
2. **`reviewer` (The Logical Auditor)**: Performs a zero-cheating, logic-based audit against the **0-4 Architecture** and **Highest Prohibitions**.
3. **`human` (Human Gate)**: Presents the audit results and proposal to the user for feedback or final approval.
4. **`saver` (Sync Committer)**: Executes the final "transactional" write to the databases.

### Execution Flow (DAG/State Machine)
* **Entry Point**: `generator`
* **Primary Loop**: `generator` -> `reviewer` -> (Conditional: `fail` -> `generator`, `pass` -> `human`)
* **Final Phase**: `human` -> (Conditional: `retry` -> `generator`, `approve` -> `saver`) -> `END`

---

## 4. Logical Consistency & Self-correction

The `reviewer` node is designed for **Self-correction**. If the logical auditor identifies a violation of the 0-4 architecture (e.g., an unauthorized time-control element), it provides a detailed `audit_log` which is fed back into the `generator` for a forced rewrite.

> [!NOTE]
> The loop is capped at 3 iterations to prevent infinite recursion, after which it reverts to human intervention.

---

## 5. Database Synchronization & Transactionality

The `saver` node ensures that once a worldview modification is approved:
1. **MongoDB**: The full text and metadata (including `version` and `timestamp`) are appended to `worldview_db.json` (or a remote server).
2. **ChromaDB**: The new text chunks are vectorized and indexed for future RAG retrieval.

This dual-write strategy ensures that **Retrieval (RAG)** and **Persistence (History)** are always aligned.

---

## 6. Novel Outline Agent (The Second Workflow)

The Novel Outline Agent follows a similar 0-4 architecture but focuses on narrative structure.

### JSON Schema Enforcement
The outline agent is strictly bound to a professional novel outline schema:
- **`meta_info`**: Metadata (Genre, Tone, Target Audience).
- **`core_hook`**: Logline and inciting incidents.
- **`character_roster`**: Roles and motivations.
- **`plot_beats`**: High-level pacing (Act 1, Midpoint, Climax).

### Multi-Agent Orchestration
The `app_api.py` acts as a router, selecting between `worldview_app` and `outline_app` based on the `agent_type` flag sent by the dashboard.

---

## 8. Observability & Monitoring (观测与监控)

To ensure the stability and traceability of the Multi-Agent system, a comprehensive observability stack is integrated:

1.  **Sentry (Error Tracking)**: Captures backend exceptions and performance data in real-time.
2.  **LangFuse (LLM Tracing)**: Traces every step of the LangGraph execution, providing a detailed view of prompt history, completion tokens, and transition latency.
3.  **Prometheus (Metrics)**: Exposes a `/metrics` endpoint to collect time-series data, including HTTP latencies and custom business metrics.
4.  **Grafana (Visualization)**: Provides a centralized dashboard for visualizing system health and LLM usage patterns.

### Custom Metrics
- **`llm_token_usage_total`**: A counter that tracks token consumption across different agents (worldview, outline, router) and models.

---

## 9. Project Files Mapping

| Component | Responsibility | Relevant Files |
| :--- | :--- | :--- |
| **Logic Engine (Worldview)** | Graph execution & state flow | `worldview_agent_langgraph.py` |
| **Logic Engine (Outline)** | Novel outline generation graph | `novel_outline_agent_langgraph.py` |
| **API Router & Metrics** | Multi-agent request handling & Prometheus | `app_api.py` |
| **Web Dashboard** | Multi-agent UI & JSON Rendering | `dashboard.html` |
| **Observability Config** | Sentry, LangFuse, Prometheus infra | `observability/`, `config_utils.py` |
| **Architectural Rules**| 0-4 Architecture Definitions | `info.md`, `novel_outline_info.md` |
| **Knowledge Base** | Full world setting & lore | `worldview_db.json` |
| **Vector Index** | ChromaDB persistence | `./chroma_db/` |
| **Agent Skills** | Context-level guidelines | `.gemini/skills/` |
