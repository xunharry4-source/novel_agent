# 创建分卷大纲工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`outline`
- 业务动作：`create`
- 代码入口：[src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py)
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=outline` and `action=create`

## 结论

- 当前“创建分卷大纲”工作流不存在节点 LLM 混用问题。
- 所有会调用 LLM 的节点都使用各自独立的 `llm_agent_name`：
  - `initial_expansion` -> `outline_agent_initial_expansion`
  - `world_review` -> `outline_world_rules_review_agent`
  - `worldview_review` -> `outline_worldview_rules_review_agent`
  - `novel_review` -> `outline_novel_rules_review_agent`
  - `modify_content` -> `outline_agent_modify_content`
- 其余节点不允许调用 LLM：
  - `input`
  - `human`
  - `commit`

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py) | 只接收 `novel_id`、`world_id`、`worldview_id`、消息和 payload，禁止调用 LLM |
| `initial_expansion` | 是 | `outline_agent_initial_expansion` | `AGENT_MODELS.outline_agent_initial_expansion` | [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py) | 创建分卷大纲时的首次整理与扩充节点 |
| `world_review` | 是 | `outline_world_rules_review_agent` | `AGENT_MODELS.outline_world_rules_review_agent` | [src/agents/review_nodes/world_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/world_review.py) | 审查是否违反父级世界规则与基本设定 |
| `worldview_review` | 是 | `outline_worldview_rules_review_agent` | `AGENT_MODELS.outline_worldview_rules_review_agent` | [src/agents/review_nodes/worldview_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/worldview_review.py) | 审查是否违反父级世界观设定 |
| `novel_review` | 是 | `outline_novel_rules_review_agent` | `AGENT_MODELS.outline_novel_rules_review_agent` | [src/agents/review_nodes/novel_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/novel_review.py) | 审查是否违反父级小说规则和主线约束 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `outline_agent_modify_content` | `AGENT_MODELS.outline_agent_modify_content` | [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py) | 任一审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py) | 只负责真实写入 `outlines` 集合，禁止调用 LLM |

## 实际路由顺序

### 正常路径

1. `input`
2. `initial_expansion`
3. `world_review`
4. `worldview_review`
5. `novel_review`
6. `human`
7. `commit`

### 自动审查失败或人工打回路径

1. `input`
2. `initial_expansion`
3. `world_review`
4. `worldview_review`
5. `novel_review`
6. `modify_content`
7. `world_review`
8. `worldview_review`
9. `novel_review`
10. `human`
11. `commit`

## 审计依据

### 代码常量

- `INITIAL_EXPANSION_AGENT_NAME = "outline_agent_initial_expansion"`
- `MODIFY_CONTENT_AGENT_NAME = "outline_agent_modify_content"`

来源：[src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py)

### review 节点装配

- `world_review -> reviewer="outline_world_rules_review_agent"`
- `worldview_review -> reviewer="outline_worldview_rules_review_agent"`
- `novel_review -> reviewer="outline_novel_rules_review_agent"`

来源：[src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py)

### 全局注册表

- `WORKFLOW_NODE_LLM_IDENTITIES["outline"]["initial_expansion"] = "outline_agent_initial_expansion"`
- `WORKFLOW_NODE_LLM_IDENTITIES["outline"]["modify_content"] = "outline_agent_modify_content"`
- `REVIEW_NODE_LLM_IDENTITIES["outline_world_rules"] = "outline_world_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["outline_worldview_rules"] = "outline_worldview_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["outline_novel_rules"] = "outline_novel_rules_review_agent"`

来源：[src/common/llm_identity_registry.py](/Users/harry/Documents/git/novel_agent/src/common/llm_identity_registry.py)

### 配置槽位

- `AGENT_MODELS.outline_agent_initial_expansion`
- `AGENT_MODELS.outline_agent_modify_content`
- `AGENT_MODELS.outline_world_rules_review_agent`
- `AGENT_MODELS.outline_worldview_rules_review_agent`
- `AGENT_MODELS.outline_novel_rules_review_agent`

来源：[config/llm.yml](/Users/harry/Documents/git/novel_agent/config/llm.yml)

## 风险提示

- `AGENT_NAME = "outline_agent"` 仍然保留在 [src/agents/outline_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/outline_agent.py)，但它只能作为工作流模块名、文案和兼容字段存在，不能被任何 LLM 节点当成共享 `llm_agent_name` 使用。
- 未来如果再新增 outline 审查节点或摘要派生节点，必须先分配新的独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流。
