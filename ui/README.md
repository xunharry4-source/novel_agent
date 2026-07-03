# Legacy NiceGUI UI (`ui/`)

**状态：已废弃（Deprecated）**

本目录为历史 NiceGUI 管理界面，**不再接受新功能**。产品主前端为 `frontend/`（React + Vite）。

- 默认启动：`make start` → React `:5174` + API `:5006`
- 遗留启动（需先 `make install-legacy`）：`make start-legacy-ui` → `:8501`
- 退役计划：[docs/frontend_strategy.md](../docs/frontend_strategy.md)

若 `main.py` 不存在，说明本地已移除或从未检出遗留 UI；请使用 React 前端。
