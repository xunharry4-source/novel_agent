# Configuration

The application loads YAML modules from this directory and merges them into the flat config dictionary returned by `src.common.config_utils.load_config()`.

- `llm.yml`: LLM providers, default LLM, model catalogs, and per-agent overrides.
- `embeddings.yml`: embedding providers, default embedding model, and embedding model catalogs.
- `storage.yml`: MongoDB, ChromaDB, and data paths.
- `app.yml`: application behavior flags.
- `observability.yml`: logs, Sentry, and LangFuse settings.
- `integrations.yml`: optional external integrations such as Dify.
- `secrets.yml`: optional local secrets. This file is ignored by git; prefer `.env` for secrets when possible.

Environment variables still override YAML values at runtime.

## Frontend

- Primary UI: `frontend/` (React). Start with `make start`.
- Legacy NiceGUI `ui/` is deprecated; see [docs/frontend_strategy.md](../docs/frontend_strategy.md).

