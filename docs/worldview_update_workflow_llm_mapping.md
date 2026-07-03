# 修改世界观工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`worldview`
- 业务动作：`update`
- 代码入口：[src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py)
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=worldview` and `action=update`

## 结论

- 当前“修改世界观”工作流不存在节点 LLM 混用问题。
- 所有会调用 LLM 的节点都使用各自独立的 `llm_agent_name`：
  - `initial_expansion` -> `worldview_agent_initial_expansion`
  - `world_rule_review` -> `worldview_world_rules_review_agent`
  - `worldview_consistency_review` -> `worldview_consistency_review_agent`
  - `modify_content` -> `worldview_agent_modify_content`
- 其余节点不允许调用 LLM：
  - `input`
  - `human`
  - `commit`

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py) | 只记录 `target_id`、`world_id`、`worldview_id`、消息和 payload，禁止调用 LLM |
| `initial_expansion` | 是 | `worldview_agent_initial_expansion` | `AGENT_MODELS.worldview_agent_initial_expansion` | [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py) | 修改世界观时的首次整理与扩充节点 |
| `world_rule_review` | 是 | `worldview_world_rules_review_agent` | `AGENT_MODELS.worldview_world_rules_review_agent` | [src/agents/review_nodes/world_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/world_review.py) | 审查是否违反父级世界禁止规则与基本设定 |
| `worldview_consistency_review` | 是 | `worldview_consistency_review_agent` | `AGENT_MODELS.worldview_consistency_review_agent` | [src/agents/review_nodes/worldview_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/worldview_review.py) | 审查是否违反同一世界下已有世界观 Canon |
| `human` | 否 | `N/A` | `N/A` | [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `worldview_agent_modify_content` | `AGENT_MODELS.worldview_agent_modify_content` | [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py) | 审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py) | 只负责真实写回 lore/worldviews，禁止调用 LLM |

## 实际路由顺序

### 正常路径

1. `input`
2. `initial_expansion`
3. `world_rule_review`
4. `worldview_consistency_review`
5. `human`
6. `commit`

### 自动审查失败或人工打回路径

1. `input`
2. `initial_expansion`
3. `world_rule_review`
4. `worldview_consistency_review`
5. `modify_content`
6. `world_rule_review`
7. `worldview_consistency_review`
8. `human`
9. `commit`

## update 动作特有约束

- `action=update` 必须提供 `target_id`，否则 [src/app_api.py](/Users/harry/Documents/git/novel_agent/src/app_api.py) 会直接拒绝调度。
- `commit_node()` 在 update 场景下优先更新 `lore(type=worldview)` 条目；只有 `target_id` 指向世界观库本体时才会更新 `worldviews` 集合。

## 审计依据

### 代码常量

- `INITIAL_EXPANSION_AGENT_NAME = "worldview_agent_initial_expansion"`
- `MODIFY_CONTENT_AGENT_NAME = "worldview_agent_modify_content"`

来源：[src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py)

### review 节点装配

- `reviewer="worldview_world_rules_review_agent"`
- `reviewer="worldview_consistency_review_agent"`

来源：[src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py)

### 全局注册表

- `WORKFLOW_NODE_LLM_IDENTITIES["worldview"]["initial_expansion"] = "worldview_agent_initial_expansion"`
- `WORKFLOW_NODE_LLM_IDENTITIES["worldview"]["modify_content"] = "worldview_agent_modify_content"`
- `REVIEW_NODE_LLM_IDENTITIES["worldview_world_rules"] = "worldview_world_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["worldview_consistency"] = "worldview_consistency_review_agent"`

来源：[src/common/llm_identity_registry.py](/Users/harry/Documents/git/novel_agent/src/common/llm_identity_registry.py)

### 配置槽位

- `AGENT_MODELS.worldview_agent_initial_expansion`
- `AGENT_MODELS.worldview_agent_modify_content`
- `AGENT_MODELS.worldview_world_rules_review_agent`
- `AGENT_MODELS.worldview_consistency_review_agent`

来源：[config/llm.yml](/Users/harry/Documents/git/novel_agent/config/llm.yml)

## 风险提示

- `AGENT_NAME = "worldview_agent"` 仍然保留在 [src/agents/worldview_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/worldview_agent.py)，但它只能作为工作流模块名、文案和兼容字段存在，不能再被任何 LLM 节点当成共享 `llm_agent_name` 使用。
- 如果未来新增更多 review、summary 或派生节点，必须先分配新的独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流。
