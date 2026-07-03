# 前端策略：React 为主，NiceGUI 逐步退役

更新日期：2026-06-26  
状态：**已确立**（React 为唯一主前端；NiceGUI 进入维护性退役阶段）

## 1. 决策摘要

| 维度 | 主路径（React） | 退役路径（NiceGUI） |
|------|----------------|---------------------|
| 目录 | `frontend/` | `ui/`（遗留，不再新增功能） |
| 技术栈 | React 18 + Vite + TypeScript + Mantine/MUI | Python NiceGUI |
| 默认端口 | `5174` | `8501` |
| 启动命令 | `make start` | `make start-legacy-ui`（已标记废弃） |
| 新功能 | **必须**在此实现 | **禁止**新增页面或业务逻辑 |
| Bug 修复 | 优先修复 | 仅阻塞性修复，且不阻塞 React 对齐 |

**原则**：所有面向用户的新 UI、工作流可视化、层级管理页，只在 React 前端交付；NiceGUI 不再作为产品演进目标。

## 2. 背景与根因

1. **产品需求已绑定 React 路由**：`docs/product_world_hierarchy_requirements.md` 中的 `/worlds`、`/novels`、`/workflow/*` 等页面均在 `frontend/src/pages/` 实现。
2. **双前端重复建设**：NiceGUI 与 React 共用 `app_api.py`，易造成文档、测试与体验分裂。
3. **NiceGUI 目录已脱离主链路**：本仓库工作区中 `ui/main.py` 已不存在（仅保留 `ui/README.md` 废弃说明）；`make start` 已是唯一有效本地 UI 启动方式。

## 3. React 主前端范围

当前 React 应用覆盖：

- 世界层级：`/worlds`、`/worlds/:worldId`
- 世界观与资料库：`/worldviews`、`/lore`、`/visualizer`
- 小说链路：`/novels` 及大纲、章节、正文管理
- Agent 工作台：`/workflow` 及五模块子路由
- 认证：`/login`、`/register`、`/me`

后端契约：**仅**通过 `src/app_api.py` REST API 交互（`frontend/src/api/client.ts`）。

## 4. NiceGUI 退役阶段

### 阶段 A — 冻结（当前）

- [x] 文档与 Makefile 标明 React 为默认前端
- [x] `make start-legacy-ui` / `make start-all` 打印废弃警告
- [x] `nicegui` 移出主 `requirements.txt`，改由 `requirements-legacy.txt` 安装
- [x] `.cursorrules` 与 `technical_design*.md` 同步前端策略

### 阶段 B — 功能对齐（进行中）

在删除 `ui/` 之前，确认 React 已覆盖 NiceGUI 仍被使用的场景：

| 能力 | React 状态 | 备注 |
|------|-----------|------|
| 世界层级 CRUD | 已覆盖 | `WorldHierarchy`、`WorldDetail` |
| 小说/大纲/章节管理 | 已覆盖 | `Novel*` 系列页面 |
| Agent 工作流 HITL | 已覆盖 | `HierarchyWorkflow` |
| 设定浏览与图谱 | 已覆盖 | `LoreDB`、`WorldviewVisualizer` |
| 旧版「创作工作室」独有功能 | 待核对 | 退役前逐项登记到本表 |

### 阶段 C — 移除（部分完成）

工作区已无 `ui/main.py`；`ui/` 仅剩废弃说明。满足以下条件后可删除 `ui/README.md` 与 `requirements-legacy.txt`：

1. 阶段 B 对照表无未覆盖的**生产必需**功能
2. README / 测试 / 部署脚本中无 NiceGUI 硬依赖
3. 团队确认无外部用户依赖 `:8501` 入口

## 5. 开发与运维约定

### 默认启动（推荐）

```bash
make install   # 不安装 NiceGUI
make start     # API :5006 + React :5174
```

### 仅调试遗留 NiceGUI（不推荐）

```bash
pip install -r requirements-legacy.txt
make start-legacy-ui   # 若 ui/main.py 不存在会明确报错
```

### 新功能 Checklist

1. 在 `frontend/src/pages/` 或 `frontend/src/components/` 实现
2. 如需新 API，先更新 `src/app_api.py` 与 `docs/api.md`（`scripts/generate_api_docs.py`）
3. 在 `docs/product_world_hierarchy_requirements.md` 或本文件阶段 B 表登记
4. **不得**在 `ui/` 添加平行实现

## 6. 影响面

- **AgentState / 工作流**：无变更；前后端边界仍为 HTTP API
- **数据库**：无变更
- **测试**：现有 `tests/test_*_requests.py` 均针对 API，与前端技术无关
- **CI**：默认流水线只需 Node + Python API，无需 NiceGUI

## 7. 回滚

若短期内必须恢复 NiceGUI：

```bash
pip install -r requirements-legacy.txt
make start-legacy-ui
```

回滚不改变 React 的主前端地位；仅作为临时兼容手段。
