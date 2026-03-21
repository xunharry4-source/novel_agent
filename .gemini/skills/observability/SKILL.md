---
name: observability
description: Capabilities for monitoring, error tracking, and LLM tracing using Prometheus, Grafana, LangFuse, and Sentry.
---

# Observability Skill

This skill allows the agent to monitor system health, track errors, and trace LLM calls.

## Monitoring Stack

### 🚨 Sentry (Error Tracking)
- **Purpose**: Captures all unhandled exceptions in the Flask backend.
- **Integration**: Initialized in `app_api.py`.
- **Usage**: Check the Sentry dashboard for stack traces and performance bottlenecks.

### 🕵️ LangFuse (LLM Traces)
- **Purpose**: Provides deep visibility into LangGraph execution flows.
- **Tools**: Use `get_langfuse_callback()` from `lore_utils.py` to get a `CallbackHandler`.
- **Usage**: Pass the handler to any LangChain/LangGraph `invoke` or `stream` call via `config={"callbacks": [handler]}`.

### 📈 Prometheus + Grafana (Metrics)
- **Purpose**: Real-time system and business metrics.
- **Custom Metrics**:
    - `llm_token_usage_total`: Tracks prompt and completion tokens labeled by `model`, `agent`, and `token_type`.
- **Endpoints**: `/metrics` on the Flask app (port 5005).
- **Visualization**: Access Grafana at `http://localhost:3000`.

## Operations

### Deployment
Infrastructure is managed via Docker Compose in the `observability/` directory.
```bash
cd observability && docker-compose up -d
```

### Reporting Tokens
Use `report_token_usage(model, prompt_tokens, completion_tokens, agent_name)` from `lore_utils.py` to manually report usage if automatic tracing is unavailable.
