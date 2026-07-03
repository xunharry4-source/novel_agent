# 创建世界工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`world`
- 业务动作：`create`
- 代码入口：[src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py)
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=world`

## 结论

- 当前“创建世界”工作流不存在节点 LLM 混用问题。
- 需要调用 LLM 的两个节点分别使用独立的 `llm_agent_name`：
  - `initial_expansion` -> `world_agent_initial_expansion`
  - `modify_content` -> `world_agent_modify_content`
- 其余节点不允许调用 LLM：
  - `input`
  - `human`
  - `commit`

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 代码位置 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py) `input_node()` | 只记录消息与 payload，禁止调用 LLM |
| `initial_expansion` | 是 | `world_agent_initial_expansion` | `AGENT_MODELS.world_agent_initial_expansion` | [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py) `INITIAL_EXPANSION_AGENT_NAME` / `generate_initial_expansion()` | 创建世界时的首次扩充节点 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py) `human_node()` | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `world_agent_modify_content` | `AGENT_MODELS.world_agent_modify_content` | [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py) `MODIFY_CONTENT_AGENT_NAME` / `generate_content_modification()` | 用户 `request_changes` 后的独立修改节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py) `commit_node()` | 只负责真实写库，禁止调用 LLM |

## 实际路由顺序

### 正常路径

1. `input`
2. `initial_expansion`
3. `human`
4. `commit`

### 人工打回路径

1. `input`
2. `initial_expansion`
3. `human`
4. `modify_content`
5. `human`
6. `commit`

## 审计依据

### 代码常量

- `INITIAL_EXPANSION_AGENT_NAME = "world_agent_initial_expansion"`
- `MODIFY_CONTENT_AGENT_NAME = "world_agent_modify_content"`

来源：[src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py)

### 全局注册表

- `WORKFLOW_NODE_LLM_IDENTITIES["world"]["initial_expansion"] = "world_agent_initial_expansion"`
- `WORKFLOW_NODE_LLM_IDENTITIES["world"]["modify_content"] = "world_agent_modify_content"`

来源：[src/common/llm_identity_registry.py](/Users/harry/Documents/git/novel_agent/src/common/llm_identity_registry.py)

### 配置槽位

- `AGENT_MODELS.world_agent_initial_expansion`
- `AGENT_MODELS.world_agent_modify_content`

来源：[config/llm.yml](/Users/harry/Documents/git/novel_agent/config/llm.yml)

## 风险提示

- `AGENT_NAME = "world_agent"` 仍然保留在 [src/agents/world_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/world_agent.py)，但它只能作为工作流模块名、文案和兼容字段存在，不能再被任何 LLM 节点当成共享 `llm_agent_name` 使用。
- 如果未来新增 world review 节点、summary 节点或其他派生节点，必须先新增独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流。
