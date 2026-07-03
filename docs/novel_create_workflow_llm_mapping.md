# 创建小说工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`novel`
- 业务动作：`create`
- 代码入口：[src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py)
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=novel` and `action=create`

## 结论

- 当前“创建小说”工作流不存在节点 LLM 混用问题。
- 所有会调用 LLM 的节点都使用各自独立的 `llm_agent_name`：
  - `initial_expansion` -> `novel_agent_initial_expansion`
  - `review` -> `novel_world_rules_review_agent`
  - `modify_content` -> `novel_agent_modify_content`
- 其余节点不允许调用 LLM：
  - `input`
  - `human`
  - `commit`

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py) | 只记录 `world_id`、可选 `worldview_id`、消息和 payload，禁止调用 LLM |
| `initial_expansion` | 是 | `novel_agent_initial_expansion` | `AGENT_MODELS.novel_agent_initial_expansion` | [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py) | 创建小说时的首次扩充节点 |
| `review` | 是 | `novel_world_rules_review_agent` | `AGENT_MODELS.novel_world_rules_review_agent` | [src/agents/review_nodes/novel_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/novel_review.py) | 审查是否违反父级世界规则、世界观约束和小说级规则 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `novel_agent_modify_content` | `AGENT_MODELS.novel_agent_modify_content` | [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py) | 审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py) | 只负责真实写库，禁止调用 LLM |

## 实际路由顺序

### 正常路径

1. `input`
2. `initial_expansion`
3. `review`
4. `human`
5. `commit`

### 自动审查失败或人工打回路径

1. `input`
2. `initial_expansion`
3. `review`
4. `modify_content`
5. `review`
6. `human`
7. `commit`

## 审计依据

### 代码常量

- `INITIAL_EXPANSION_AGENT_NAME = "novel_agent_initial_expansion"`
- `MODIFY_CONTENT_AGENT_NAME = "novel_agent_modify_content"`

来源：[src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py)

### review 节点装配

- `reviewer="novel_world_rules_review_agent"`

来源：[src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py)

### 全局注册表

- `WORKFLOW_NODE_LLM_IDENTITIES["novel"]["initial_expansion"] = "novel_agent_initial_expansion"`
- `WORKFLOW_NODE_LLM_IDENTITIES["novel"]["modify_content"] = "novel_agent_modify_content"`
- `REVIEW_NODE_LLM_IDENTITIES["novel_world_rules"] = "novel_world_rules_review_agent"`

来源：[src/common/llm_identity_registry.py](/Users/harry/Documents/git/novel_agent/src/common/llm_identity_registry.py)

### 配置槽位

- `AGENT_MODELS.novel_agent_initial_expansion`
- `AGENT_MODELS.novel_agent_modify_content`
- `AGENT_MODELS.novel_world_rules_review_agent`

来源：[config/llm.yml](/Users/harry/Documents/git/novel_agent/config/llm.yml)

## 风险提示

- `AGENT_NAME = "novel_agent"` 仍然保留在 [src/agents/novel_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/novel_agent.py)，但它只能作为工作流模块名、文案和兼容字段存在，不能再被任何 LLM 节点当成共享 `llm_agent_name` 使用。
- 如果未来新增更多 review、summary 或派生节点，必须先分配新的独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流。
