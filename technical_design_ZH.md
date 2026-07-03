# Novel Agent 技术设计（中文版）

> 英文摘要见 [technical_design.md](./technical_design.md)

## 1. 系统概览

Novel Agent 是以 **Flask API + LangGraph Agent + MongoDB/ChromaDB** 为核心的小说创作引擎。用户通过 Web 前端调用 `src/app_api.py` 暴露的 REST 接口，驱动世界层级实体与 Agent 工作流。

## 2. 前端架构（React 为主）

### 2.1 主前端

- **目录**：`frontend/`
- **栈**：React 18、Vite、TypeScript、Mantine、MUI、React Flow
- **入口**：`frontend/src/main.tsx` → `frontend/src/App.tsx`
- **API 客户端**：`frontend/src/api/client.ts`（`VITE_API_BASE_URL` 默认代理到 `http://127.0.0.1:5006`）
- **默认启动**：`make start`（后端 5006 + 前端 5174）

### 2.2 遗留前端（NiceGUI，逐步退役）

- **目录**：`ui/`（遗留；不再接受新功能）
- **依赖**：`requirements-legacy.txt`（可选安装）
- **启动**：`make start-legacy-ui`（已废弃，仅兼容旧环境）

详细阶段计划见 [docs/frontend_strategy.md](./docs/frontend_strategy.md)。

### 2.3 约束

- 新页面、新交互 **必须** 在 React 实现
- 禁止在 NiceGUI 与 React 之间复制业务逻辑；业务逻辑归属后端 Agent 与 API
- 前端不得绕过审查/HITL 直接写库

## 3. 后端与 Agent

- **HTTP 入口**：`src/app_api.py`
- **层级 Agent**：`src/agents/{world,worldview,novel,outline,chapter}_agent.py` 及摘要类 agent
- **审查**：`src/agents/review_agent.py`、`src/agents/review_nodes/`
- **编排**：API 层 `_run_until_human` 驱动节点序列，运行记录写入 `hierarchy_agent_runs`
- **LLM**：`src/common/llm_factory.py` + `config/llm.yml` + `src/common/llm_identity_registry.py`

## 4. 数据层

- **MongoDB**：`worlds`、`worldviews`、`novels`、`outlines`、`prose`、`hierarchy_agent_runs`、`downstream_summaries` 等
- **ChromaDB**：按 `worldview_id` / `outline_id` 隔离的向量检索
- **配置**：`config/storage.yml`

## 5. 观测与外部接入

- LangFuse、Sentry、Prometheus（见 `config/observability.yml`）
- 外部调度：`POST /api/router/dispatch`（见 `docs/router_dispatch.md`）

## 6. 文档索引

| 文档 | 用途 |
|------|------|
| [docs/product_world_hierarchy_requirements.md](./docs/product_world_hierarchy_requirements.md) | 产品需求与页面范围 |
| [docs/frontend_strategy.md](./docs/frontend_strategy.md) | 前端主路径与 NiceGUI 退役 |
| [docs/api.md](./docs/api.md) | 后端 API（自动生成） |
| [docs/*_workflow_llm_mapping.md](./docs/) | 各工作流节点与 LLM 映射 |
