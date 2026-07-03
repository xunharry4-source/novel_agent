# 创建章节大纲工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`chapter`
- 业务动作：`create`
- 页面入口：`/workflow/chapter?action=create`
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=chapter` and `action=create`
- 代码入口：[src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py)

## 结论

- 当前“创建章节大纲”工作流需要独立 LLM 的节点共有 7 个。
- 修复前存在 4 个 reviewer 名称错误，导致节点输出身份与配置槽位、注册表和真实 LLM 调用身份不一致：
  - `world_review`：`chapter_world_review_agent` -> `chapter_world_rules_review_agent`
  - `worldview_review`：`chapter_worldview_review_agent` -> `chapter_worldview_rules_review_agent`
  - `novel_review`：`chapter_novel_review_agent` -> `chapter_novel_rules_review_agent`
  - `outline_review`：`chapter_outline_review_agent` -> `chapter_outline_rules_review_agent`
- 修复后，创建章节大纲工作流不存在节点 LLM 混用问题；每个 LLM 节点都绑定独立 `llm_agent_name`。

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收章节大纲输入与父级上下文，禁止调用 LLM |
| `initial_expansion` | 是 | `chapter_agent_initial_expansion` | `AGENT_MODELS.chapter_agent_initial_expansion` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 首次整理并扩充章节大纲内容 |
| `world_review` | 是 | `chapter_world_rules_review_agent` | `AGENT_MODELS.chapter_world_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反世界禁止规则与基本设定 |
| `worldview_review` | 是 | `chapter_worldview_rules_review_agent` | `AGENT_MODELS.chapter_worldview_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反世界观设定 |
| `novel_review` | 是 | `chapter_novel_rules_review_agent` | `AGENT_MODELS.chapter_novel_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反小说规则与主线约束 |
| `outline_review` | 是 | `chapter_outline_rules_review_agent` | `AGENT_MODELS.chapter_outline_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否偏离父级分卷大纲 |
| `chapter_review` | 是 | `chapter_consistency_review_agent` | `AGENT_MODELS.chapter_consistency_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查与前置章节的一致性 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `chapter_agent_modify_content` | `AGENT_MODELS.chapter_agent_modify_content` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只负责真实写入 `prose` 集合，禁止调用 LLM |

## 不在本工作流内的章节相关 LLM

- `chapter_chapter_outline_rules_review_agent`
  - 用于 `chapter_check` 直接内容检查工作流中的“章节大纲检查”节点，不属于当前 `chapter/create` 工作流。
- `chapter_plot_errors_review_agent`
  - 用于 `chapter_check` 直接内容检查工作流中的“剧情错误检查”节点，不属于当前 `chapter/create` 工作流。

## 实际路由顺序

### 正常路径

1. `input`
2. `initial_expansion`
3. `world_review`
4. `worldview_review`
5. `novel_review`
6. `outline_review`
7. `chapter_review`
8. `human`
9. `commit`

### 自动审查失败或人工打回路径

1. `input`
2. `initial_expansion`
3. `world_review`
4. `worldview_review`
5. `novel_review`
6. `outline_review`
7. `chapter_review`
8. `modify_content`
9. `world_review`
10. `worldview_review`
11. `novel_review`
12. `outline_review`
13. `chapter_review`
14. `human`
15. `commit`

## 审计依据

### 代码常量

- `INITIAL_EXPANSION_AGENT_NAME = "chapter_agent_initial_expansion"`
- `MODIFY_CONTENT_AGENT_NAME = "chapter_agent_modify_content"`

来源：[src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py)

### review 节点装配

- `world_review -> reviewer="chapter_world_rules_review_agent"`
- `worldview_review -> reviewer="chapter_worldview_rules_review_agent"`
- `novel_review -> reviewer="chapter_novel_rules_review_agent"`
- `outline_review -> reviewer="chapter_outline_rules_review_agent"`
- `chapter_review -> reviewer="chapter_consistency_review_agent"`

来源：[src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py)

### 全局注册表

- `WORKFLOW_NODE_LLM_IDENTITIES["chapter"]["initial_expansion"] = "chapter_agent_initial_expansion"`
- `WORKFLOW_NODE_LLM_IDENTITIES["chapter"]["modify_content"] = "chapter_agent_modify_content"`
- `REVIEW_NODE_LLM_IDENTITIES["chapter_world_rules"] = "chapter_world_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["chapter_worldview_rules"] = "chapter_worldview_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["chapter_novel_rules"] = "chapter_novel_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["chapter_outline_rules"] = "chapter_outline_rules_review_agent"`
- `REVIEW_NODE_LLM_IDENTITIES["chapter_consistency"] = "chapter_consistency_review_agent"`

来源：[src/common/llm_identity_registry.py](/Users/harry/Documents/git/novel_agent/src/common/llm_identity_registry.py)

### 配置槽位

- `AGENT_MODELS.chapter_agent_initial_expansion`
- `AGENT_MODELS.chapter_agent_modify_content`
- `AGENT_MODELS.chapter_world_rules_review_agent`
- `AGENT_MODELS.chapter_worldview_rules_review_agent`
- `AGENT_MODELS.chapter_novel_rules_review_agent`
- `AGENT_MODELS.chapter_outline_rules_review_agent`
- `AGENT_MODELS.chapter_consistency_review_agent`

来源：[config/llm.yml](/Users/harry/Documents/git/novel_agent/config/llm.yml)

## 风险提示

- `config/llm.yml` 里的 `chapter_review_agent` 目前未被 `chapter/create` 工作流使用，不能把它当成任何章节创建节点的共享 LLM。
- 未来如果再给 `chapter/create` 增加新的审核节点，必须先新增独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流；禁止复用已有 reviewer 身份。
