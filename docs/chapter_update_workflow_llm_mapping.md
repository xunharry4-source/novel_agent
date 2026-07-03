# 修改章节大纲工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`chapter`
- 业务动作：`update`
- 页面入口：`/workflow/chapter?action=update`
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=chapter` and `action=update`
- 代码入口：[src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py)

## 结论

- 当前“修改章节大纲”工作流不存在节点 LLM 混用问题。
- 修改章节大纲与创建章节大纲共用同一套 7 个 LLM 节点，但每个节点都绑定自己的独立 `llm_agent_name`，不存在共用 reviewer 或共用生成节点身份。
- 本轮没有新增生产逻辑修复；修改链路直接继承了上一轮在 [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) 的 reviewer 名称修复。

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收 `target_id`、父级上下文和修改内容，禁止调用 LLM |
| `initial_expansion` | 是 | `chapter_agent_initial_expansion` | `AGENT_MODELS.chapter_agent_initial_expansion` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 修改章节大纲时的首次整理与扩充节点 |
| `world_review` | 是 | `chapter_world_rules_review_agent` | `AGENT_MODELS.chapter_world_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级世界规则与基本设定 |
| `worldview_review` | 是 | `chapter_worldview_rules_review_agent` | `AGENT_MODELS.chapter_worldview_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级世界观设定 |
| `novel_review` | 是 | `chapter_novel_rules_review_agent` | `AGENT_MODELS.chapter_novel_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级小说规则和主线约束 |
| `outline_review` | 是 | `chapter_outline_rules_review_agent` | `AGENT_MODELS.chapter_outline_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否偏离父级分卷大纲 |
| `chapter_review` | 是 | `chapter_consistency_review_agent` | `AGENT_MODELS.chapter_consistency_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查与前置章节的一致性 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `chapter_agent_modify_content` | `AGENT_MODELS.chapter_agent_modify_content` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 任一审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只负责按 `target_id` 更新 `prose` 集合，禁止调用 LLM |

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

- `config/llm.yml` 中仍保留 `chapter_review_agent` 旧槽位，但它不属于 `chapter/update` 工作流任何节点，不能被视为共享 reviewer。
- 未来如果再给 `chapter/update` 增加新的审查节点，必须先新增独立 `llm_agent_name` 和独立 `AGENT_MODELS` 槽位，再接入工作流。
