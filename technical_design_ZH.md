# 技术设计文档：世界观与小说 Agent (多 Agent 架构)

## 1. 角色：工作流架构师 (Workflow Architect)

作为 **工作流架构师**，本系统的设计优先考虑 **稳定性**、**可扩展性** 和 **自愈性 (Self-healing)**。架构采用基于 LangGraph 的状态机方法，管理 AI 生成器、逻辑审计员和人类监督员之间的复杂交互。

### 核心原则
* **状态管理**：使用 `TypedDict` 确保节点间的数据一致性。
* **逻辑闭环**：实现“审计-拒绝-重写”循环，以维持高质量输出。
- **人机协作**：为关键决策定义的明确的人类干预点。
* **数据库同步**：多数据集事务性，确保 MongoDB 和 ChromaDB 保持同步。

---

## 2. 状态管理 (`AgentState`)

`AgentState` 是工作流的核心，跟踪查询、上下文、当前提案和审计历史。

```python
class AgentState(TypedDict):
    query: str
    context: str
    proposal: str
    review_log: str
    user_feedback: str
    iterations: int        # 总生成尝试计数
    audit_count: int       # 自审重试计数
    is_approved: bool      # 来自审核员或人类的信号
    category: str 
    doc_id: str
    status_message: str    # 实时执行状态
```

---

## 3. LangGraph 工作流定义

系统由 4 个主要功能节点组成：

1. **`generator` (生成器)**：利用 LLM 根据用户查询和 RAG 上下文产生或细化“提案”。
2. **`reviewer` (逻辑审计员)**：根据 **0-4 架构** 和 **最高禁令** 进行零作弊的逻辑审计。
3. **`human` (人类关卡)**：将审计结果和提案展示给用户，以获取反馈或最终批准。
4. **`saver` (同步提交器)**：执行对数据库的最终“事务性”写入。

### 执行流程 (DAG/状态机)
* **入口点**：`generator`
* **主循环**：`generator` -> `reviewer` -> (条件: `失败` -> `generator`, `通过` -> `human`)
* **最终阶段**：`human` -> (条件: `重试` -> `generator`, `批准` -> `saver`) -> `END`

---

## 4. 逻辑一致性与自愈

`reviewer` 节点设计用于 **自愈**。如果逻辑审计员识别出违反 0-4 架构（例如，未经授权的时间控制元素），它将提供详细的 `audit_log`，该日志被反馈到 `generator` 中进行强制重写。

> [!NOTE]
> 循环上限为 3 次迭代，以防止无限递归，之后会转回人类干预。

---

## 5. 数据库同步与事务性

`saver` 节点确保一旦世界观修改获得批准：
1. **MongoDB**：全文和元数据（包括版本和时间戳）被追加到 `worldview_db.json`。
2. **ChromaDB**：新的文本切片被向量化并建立索引，用于未来的 RAG 检索。

这种双写策略确保了 **检索 (RAG)** 和 **持久化 (历史)** 始终保持对齐。

---

## 6. 小说大纲 Agent (第二工作流)

小说大纲 Agent 遵循类似的 0-4 架构，但侧重于叙事结构。
- **JSON Schema 强制执行**：大纲 Agent 严格绑定到专业的小说大纲模式，包含 `meta_info`、`core_hook`、`character_roster` 和 `plot_beats`。

---

## 7. 全链路展示与观测 (Observability)

为了保证系统的可回溯性，集成了以下观测工具：
1. **Sentry**：实时捕捉后端异常。
2. **LangFuse**：追踪每一步 LangGraph 执行，查看 Prompt 历史与 Token 消耗。
3. **Prometheus**：采集 `/metrics` 指标，包括自定义的 `llm_token_usage_total`。

---

## 8. 项目文件映射

| 组件 | 职责 | 相关文件 |
| :--- | :--- | :--- |
| **逻辑引擎 (世界观)** | 图执行与状态流 | `worldview_agent_langgraph.py` |
| **逻辑引擎 (大纲)** | 小说大纲生成图 | `novel_outline_agent_langgraph.py` |
| **API 路由与指标** | 多代理请求处理与 Prometheus | `app_api.py` |
| **Web 仪表盘** | 多代理 UI 与 JSON 渲染 | `dashboard.html` |
| **观测配置** | Sentry, LangFuse, Prometheus 基础设施 | `observability/`, `config_utils.py` |
| **架构规则** | 0-4 架构定义 | `info.md`, `novel_outline_info.md` |
| **知识库** | 完整世界设定与档案 | `worldview_db.json` |
| **向量索引** | ChromaDB 持久化 | `./chroma_db/` |
| **Agent 技能** | 上下文级指南 | `.gemini/skills/` |
