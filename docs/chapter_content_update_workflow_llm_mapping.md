# 修改章节内容工作流节点与 LLM 对应关系表

## 范围

- 工作流类型：`chapter`
- 业务动作：`update`
- 页面入口：`/workflow/chapter?action=update&chapter_outline_id=<父级章节大纲ID>&target_id=<章节内容ID>`
- 路由入口：`POST /api/hierarchy-agent/start` with `agent_type=chapter` and `action=update`
- 页面来源：[frontend/src/pages/NovelChapterContentManagement.tsx](/Users/harry/Documents/git/novel_agent/frontend/src/pages/NovelChapterContentManagement.tsx)
- 代码入口：[src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py)

## 结论

- 当前“修改章节内容”工作流不存在节点 LLM 混用问题。
- “章节内容修改”本质上仍然走 `chapter/update` 工作流，只是在 payload 中显式带 `chapter_outline_id`，保证更新后的正文仍然挂在同一个父级章节大纲下。
- 所有会调用 LLM 的节点都使用各自独立的 `llm_agent_name`：
  - `initial_expansion` -> `chapter_agent_initial_expansion`
  - `world_review` -> `chapter_world_rules_review_agent`
  - `worldview_review` -> `chapter_worldview_rules_review_agent`
  - `novel_review` -> `chapter_novel_rules_review_agent`
  - `outline_review` -> `chapter_outline_rules_review_agent`
  - `chapter_review` -> `chapter_consistency_review_agent`
  - `modify_content` -> `chapter_agent_modify_content`

## 节点与 LLM 对应表

| 工作流节点 | 是否调用 LLM | 节点使用的 `llm_agent_name` | 配置槽位 (`config/llm.yml`) | 相关文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收 `target_id`、`chapter_outline_id`、父级上下文和正文修改内容，禁止调用 LLM |
| `initial_expansion` | 是 | `chapter_agent_initial_expansion` | `AGENT_MODELS.chapter_agent_initial_expansion` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 修改章节内容时的首次整理与扩充节点 |
| `world_review` | 是 | `chapter_world_rules_review_agent` | `AGENT_MODELS.chapter_world_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级世界规则与基本设定 |
| `worldview_review` | 是 | `chapter_worldview_rules_review_agent` | `AGENT_MODELS.chapter_worldview_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级世界观设定 |
| `novel_review` | 是 | `chapter_novel_rules_review_agent` | `AGENT_MODELS.chapter_novel_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否违反父级小说规则和主线约束 |
| `outline_review` | 是 | `chapter_outline_rules_review_agent` | `AGENT_MODELS.chapter_outline_rules_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查是否偏离父级分卷大纲 |
| `chapter_review` | 是 | `chapter_consistency_review_agent` | `AGENT_MODELS.chapter_consistency_review_agent` | [src/agents/review_nodes/chapter_review.py](/Users/harry/Documents/git/novel_agent/src/agents/review_nodes/chapter_review.py) | 审查与前置章节的一致性 |
| `human` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只接收人工决定，禁止调用 LLM |
| `modify_content` | 是 | `chapter_agent_modify_content` | `AGENT_MODELS.chapter_agent_modify_content` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 任一审查失败或人工打回后的独立返工节点 |
| `commit` | 否 | `N/A` | `N/A` | [src/agents/chapter_agent.py](/Users/harry/Documents/git/novel_agent/src/agents/chapter_agent.py) | 只负责按 `target_id` 更新 `prose.chapter_outline_id` 子记录，禁止调用 LLM |

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

### 页面跳转参数

- `NovelChapterContentManagement.tsx` 的“工作流修改章节内容”会同时传 `target_id` 和 `chapter_outline_id` 到 `/workflow/chapter?...`
- `HierarchyWorkflow.tsx` 会把 `chapter_outline_id` 写回 payload

来源：
- [frontend/src/pages/NovelChapterContentManagement.tsx](/Users/harry/Documents/git/novel_agent/frontend/src/pages/NovelChapterContentManagement.tsx)
- [frontend/src/pages/HierarchyWorkflow.tsx](/Users/harry/Documents/git/novel_agent/frontend/src/pages/HierarchyWorkflow.tsx)

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

- `chapter_outline_id` 只决定这条正文更新后仍挂在哪个父级章节大纲下，不会引入新的共享 LLM reviewer。
- `config/llm.yml` 里的 `chapter_review_agent` 旧槽位仍不属于本工作流任何节点，不能被视为共享 reviewer。
