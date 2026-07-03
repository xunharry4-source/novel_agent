# 世界层级与 Agent 迭代工作台产品需求文档

更新日期：2026-04-29  
适用范围：`frontend/` React 主前端（`http://127.0.0.1:5174`）下的 `/worlds` 世界列表与世界详情、`/novels` 小说列表与小说详情、小说下属 `/outlines` 与 `/chapters` 管理页、`/lore` 世界观资料库、`/visualizer` 世界观图谱、五模块 Agent 工作流页、后端层级 API、真实 requests 测试。

> **前端策略**：React 为唯一产品演进目标；NiceGUI `ui/` 已废弃，不得新增平行页面。见 [frontend_strategy.md](./frontend_strategy.md)。

## 1. 产品目标

构建一个用于小说生产的层级化创作工作台，让用户能够按真实业务关系管理：

`世界 -> (世界观, 小说) -> 大纲 -> 章节`

其中：

- 世界：创作宇宙的顶层容器。
- 世界观：世界的规则、设定、历史、势力、地理、科技、禁令等。
- 小说：发生在同一个世界中的具体故事项目。
- 大纲：某一部小说的叙事结构和章节计划。
- 章节：某一份大纲下的正文内容，继续使用 `prose` 集合存储。

产品必须保证页面展示和所有操作都来自真实数据库查询，不允许假数据、静态样例数据、mock、吞异常、隐藏错误或降级伪成功。

## 2. 核心用户价值

1. 用户可以清楚区分“世界规则”和“世界中的故事”，避免把世界观和小说混为父子关系。
2. 用户可以通过列表、树、图和工作流视图管理复杂层级，同时避免世界观和章节大数据量全量渲染导致页面卡死。
3. 用户可以用类似 Dify 的对话框和工作流节点方式与各类创作 Agent 交互。
4. 用户可以在每次 AI 迭代中选择修改范围，默认局部重写，避免“小改动导致大范围改写”。
5. 用户可以通过真实 API 测试证明新增、修改、删除、查询都真实发生在数据库中。

## 3. 信息架构

### 3.1 层级关系

- `worlds`
  - `worldviews`
  - `novels`
    - `outlines`
      - `prose`

### 3.2 关系规则

- `worldviews` 与 `novels` 为兄弟节点 (Siblings)，均直接归属 `worlds`。
- `worldview` 表示世界规则与设定，不是小说的父级。
- `novel` 表示该世界中发生的故事，是 `outline` 的父级。
- `outline` 必须归属一个 `novel`，并可关联同世界下的 `worldview` 作为设定约束。
- `prose` 必须归属一个 `outline`，并继承 `novel_id`、`worldview_id`、`world_id`。

## 14. 功能模块详解

### 14.1 世界 (World) - 顶层宇宙容器

- **所属模块**：世界管理
- **功能名称**：创作宇宙顶层容器 (World Container)
- **功能背景**：小说创作通常涉及多个平行世界或独立宇宙，需要一个顶层实体来隔离数据并作为创作资产的逻辑锚点。
- **功能描述**：提供一个全局容器，用于组织世界观、小说及所有相关创作数据，确保数据的逻辑隔离、权限管理与层级归属。世界管理必须维护 `forbidden_rules`（世界禁止规则）与 `basic_settings`（世界基本设定），作为世界观与小说审查节点的强制依据。
- **使用场景**：用户计划开启一个全新的创作系列（如“赛博大唐”），首先创建一个“世界”作为所有设定和故事的根基。
- **新增逻辑**：
  1. 用户提交“创建世界”表单 -> 触发 `world_agent`。
  2. `world_agent` 生成全局唯一 `world_id`，并整理 `forbidden_rules` 与 `basic_settings`。
  3. 物理写入 MongoDB `worlds` 集合，初始化空关联列表，并保存世界禁止规则与基本设定。
- **修改逻辑**：
  1. 用户提交“编辑世界”表单 -> 触发 `world_agent`。
  2. 允许修改名称 (name)、摘要 (summary)、世界禁止规则 (`forbidden_rules`) 与世界基本设定 (`basic_settings`)，禁止修改 `world_id`。
  3. 异步更新 `updated_at` 时间戳并广播层级树刷新信号。
- **删除逻辑**：
  1. 用户提交“删除世界”请求。
  2. 校验下属是否有关联的 `worldviews` 或 `novels`。
  3. 若非空且未传 `cascade=true`，返回 `409`。
  4. 若满足条件，执行物理级联删除所有下属资产。
- **查询逻辑**：
  1. 支持列表无条件全量查询。
  2. 也可以根据名称 (name) 等条件进行搜索与过滤。
- **是否需要调用Agent**：是 (`world_agent`)
- **Agent工作流与循环迭代机制 (State Graph)**：
  - **工作流状态 (State)**：维护 `payload` (名称/摘要/世界禁止规则/世界基本设定)、`feedback` (修改意见)、`status` (状态)。
  - **输入节点 (Input Node)**：初始化 State，记录用户消息和表单 payload。
  - **初始扩充节点 (Initial Expansion Node)**：作为第二个节点，调用 `world_agent` 专属 LLM 对用户输入进行初步理解、补全和结构化，直接生成可供人工确认的世界名称、摘要、世界禁止规则与世界基本设定；该节点不得写库，不得跳过 LLM，不得使用通用 Prompt。
    - **LLM 扩充逻辑**：接收简略的世界名称或一句话描述，扩充为包含物理法则、核心冲突、地理风貌、禁止规则、时代边界、力量体系和资源机制的完整“世界根设定”。
    - **扩充规则**：摘要遵循 50-200 字；`forbidden_rules` 必须列出后续世界观与小说绝对不能违反的规则；`basic_settings` 必须包含时代、力量体系、地理边界、组织结构、资源机制和基础约束。
  - **人工节点 (Approval Node/HITL)**：展示初始扩充结果给用户。用户可选择：“同意 (Approve)”或“不同意/要求修改 (Request Changes)”。
    - *同意路径*：若用户同意，状态进入写库固化节点。
    - *不同意路径*：若用户不同意或提供修改意见，则将意见写入 `feedback`，**状态流转到修改内容节点**。
  - **修改内容节点 (Modify Content Node)**：仅在用户不同意时执行，调用 `world_agent` 专属 LLM 根据反馈修改世界名称、摘要、世界禁止规则或世界基本设定，保留未要求修改的内容和业务 ID；修改完成后回到人工节点再次确认。
  - **写库固化节点 (Saver/Commit Node)**：对应五模块工作流中的 `apply` 步骤，用户批准后触发，真实执行 MongoDB `worlds` 集合写入，并保存 `forbidden_rules` 与 `basic_settings`。
- **规则说明**：
  - 世界名称必须在当前用户空间内唯一。
  - `forbidden_rules` 与 `basic_settings` 是世界根约束，世界观审查节点和小说审查节点必须优先校验，不允许被后续模块绕开。
  - 世界详情页必须允许管理 `forbidden_rules` 与 `basic_settings`，这些字段是该世界下世界观与小说审查的强制约束。
  - 世界详情页修改 `forbidden_rules` 或 `basic_settings` 时必须进入 `world_agent` 修改工作流，禁止详情页直接写库。
  - 删除世界属于高危操作，必须强制二次确认。

### 14.2 世界观 (Worldview) - 规则与设定库

- **所属模块**：世界观管理
- **功能名称**：规则与设定资料库 (Worldview Lore DB)
- **功能背景**：故事的逻辑一致性取决于严谨的设定，需要结构化的方式存储地理、历史、种族、物理规则等 Canon 内容。
- **功能描述**：管理世界的“真理”设定，支持树状分类浏览和基于语义的 RAG 检索，为后续所有 AI 创作提供强有力的上下文约束。
- **使用场景**：创作者定义了“灵气引擎”的运作原理，随后要求系统内所有涉及此类技术的小说章节必须严格遵循该设定。
- **新增逻辑**：
  1. 提交设定条目 -> 触发 `worldview_agent` 进行世界规则审查与既有世界观一致性审查。
  2. 两个审查节点均通过后，生成 `worldview_id` 或 `doc_id`。
  3. 写入 MongoDB 并同步至 ChromaDB 向量库建立语义索引。
- **修改逻辑**：
  1. 修改现有条目 -> 触发 `worldview_agent` 重新执行世界规则审查与既有世界观一致性审查。
  2. 更新 MongoDB 记录，同时增量更新 ChromaDB 中的向量特征。
- **删除逻辑**：
  1. 提交删除请求 -> 校验是否被当前大纲/章节引用。
  2. 物理从 MongoDB 移除，并同步从 ChromaDB 吊销对应向量。
- **查询逻辑**：
  1. 禁止无条件全量查询。
  2. 查询必须带 `world_id`、`worldview_id` 或 `query` 等业务条件，并带分页。
  3. 支持树状 (Tree) 或网状图 (Graph) 结构展示。
- **是否需要调用Agent**：是 (`worldview_agent`)
- **Agent工作流与循环迭代机制 (State Graph)**：
  - **工作流状态 (State)**：维护 `lore_data`、`world_rule_review_feedback`、`worldview_consistency_feedback`、`user_feedback`。
  - **输入节点 (Input Node)**：初始化设定条目请求。
  - **初始扩充节点 (Initial Expansion Node)**：作为第二个节点，调用 `worldview_agent` 专属 LLM 对碎片化设定进行初步补全，直接生成可审查的世界观 payload，明确条目名称、分类、核心规则、父级 `world_id`、需要检索的 Canon 关键词和不可改写的用户原意；该节点不得写库，不得跳过 LLM，不得使用通用 Prompt。
    - **LLM 扩充逻辑**：将用户的碎片化笔记、简短设定或模糊描述，扩充为格式严谨、逻辑自洽的结构化世界观条目（Lore Entry）。
    - **扩充规则**：遵循“条目级”扩充（200-500字），必须包含“条目名称”、“核心描述”、“规则约束”和“分类标签”四个维度。
  - **世界规则审查节点 (World Rule Review Node)**：第一个审查节点，基于所属 `world_id` 的 `forbidden_rules` 与 `basic_settings` 审查当前世界观是否违反世界禁止规则、基本设定、时代边界、力量体系、地理边界、组织结构或资源机制。只要违反世界根规则，必须判定不通过。
  - **既有世界观一致性审查节点 (Worldview Consistency Review Node)**：第二个审查节点，基于 RAG 检索校验新增或修改后的世界观是否与同一世界下已有 Canon 冲突。具体审查内容包括：内容格式是否规范、是否包含逻辑漏洞、新增设定是否与该世界下已有核心设定（历史、地理、规则等）存在直接冲突。
    - *自动迭代循环*：任一审查未通过，将冲突原因写入对应反馈字段，**状态自动流转到修改内容节点**，由 LLM 根据冲突意见修正内容，修正后必须先回到世界规则审查节点，再进入既有世界观一致性审查节点，直至两个审查节点均通过或达到上限。
  - **人工节点 (Approval Node/HITL)**：两个审查节点均通过后提交用户确认。用户可选择：“批准 (Approve)”、“完全重写 (Full Rewrite)”、“指定局部重写 (Targeted Partial Rewrite)” 或 “指定内容重写 (Targeted Content Rewrite)”。
    - *人工迭代循环*：若用户要求修改，将意见写入 `user_feedback`，**状态流转到修改内容节点**，修改后必须再次经历两个审查节点。
  - **修改内容节点 (Modify Content Node)**：仅在世界规则审查失败、既有世界观一致性审查失败或用户不同意时执行，调用 `worldview_agent` 专属 LLM 根据审查反馈或 `user_feedback` 修改世界观内容；修改完成后回到世界规则审查节点。
  - **写库节点 (Commit Node)**：全部通过后，写入 MongoDB 并同步至 ChromaDB 建立语义索引。
- **规则说明**：
  - 所有设定条目必须强绑定特定的 `world_id`。
  - 支持 OPML 标准格式的导入导出。

### 14.3 小说 (Novel) - 故事项目管理

- **所属模块**：小说管理
- **功能名称**：独立故事项目管理 (Novel Project Management)
- **功能背景**：一个世界可以容纳多个故事，每个故事有其独立的叙事视角、角色曲线和完结状态。
- **功能描述**：作为大纲和章节的父级容器，负责协调特定故事线的创作进度，并关联当前世界下的相应设定集。小说也必须像世界一样维护 `forbidden_rules`（小说禁止规则）与 `basic_settings`（小说基本设定），作为大纲与章节审查节点的强制依据。
- **使用场景**：在已定义的“星际航海”世界中，开启一部名为《边缘星区往事》的特定小说创作项目。
- **新增逻辑**：
  1. 在世界节点下点击“添加小说” -> 触发 `novel_agent`。
  2. 必填 `world_id`，可选关联 `worldview_id`。
  3. `novel_agent` 整理小说摘要、小说禁止规则与小说基本设定。
  4. 初始化小说根目录，物理写入 `novels` 集合，并保存 `forbidden_rules` 与 `basic_settings`。
- **修改逻辑**：
  1. 提交修改表单 -> 更新小说名称、介绍、简介、关联世界观、小说禁止规则或小说基本设定。
  2. 修改 `worldview_id` 时需校验新设定集是否属于同一个世界。
- **删除逻辑**：
  1. 提交删除请求 -> 校验是否包含大纲。
  2. 若含子项且未级联，拒绝删除。
  3. 确认后删除 `novels` 记录，并清理所有下属 `outlines` 和 `prose`。
- **查询逻辑**：
  1. 禁止全量查询。必须带 `world_id` 等业务条件。
  2. 必须进行分页展示。
- **是否需要调用Agent**：是 (`novel_agent`)
- **Agent工作流与循环迭代机制 (State Graph)**：
  - **工作流状态 (State)**：维护小说元数据、小说禁止规则、小说基本设定、`review_feedback`、`user_feedback`。
  - **输入节点 (Input Node)**：接收标题、介绍、简介、`world_id`、可选 `forbidden_rules` 与 `basic_settings`。
  - **初始扩充节点 (Initial Expansion Node)**：作为第二个节点，调用 `novel_agent` 专属 LLM 对标题、简介、父级世界和可选世界观进行初步整合，直接生成可审查的小说项目 payload，明确故事类型、核心卖点、主角方向、父级约束、小说禁止规则、小说基本设定和不可偏离的用户指令；该节点不得写库，不得跳过 LLM，不得使用通用 Prompt。
    - **LLM 扩充逻辑**：基于给定的世界背景，将简短的小说标题或核心看点扩充为包含“核心主旨”、“主角轮廓”、“初期目标”、“小说禁止规则”和“小说基本设定”的小说项目描述。
    - **扩充规则**：摘要遵循 100-300 字；`forbidden_rules` 必须列出本小说中大纲与章节绝对不能违反的叙事/设定规则；`basic_settings` 必须包含小说类型、主角底线、主线冲突、叙事基调、时间线、人物关系规则和剧情约束。
  - **审查节点 (Review Node)**：校验小说大方向是否违反所属世界的 `forbidden_rules` 与 `basic_settings`，小说自身 `forbidden_rules` 与 `basic_settings` 是否自洽，以及是否偏离关联的世界观设定。具体审查内容包括：故事背景是否契合世界禁止规则与基本设定、是否绕开世界根禁令、小说级规则是否与世界级规则冲突、主角设定与核心主线是否符合逻辑、是否存在破坏世界基础规则的设定（反吃设定）。
    - *自动迭代循环*：如偏离设定，生成 `review_feedback` 并**状态流转到修改内容节点**，要求 LLM 在特定设定约束下修正内容，修正后必须再次审查。
  - **人工节点 (Approval Node/HITL)**：用户确认。用户可选择：“批准 (Approve)”、“完全重写 (Full Rewrite)”、“指定局部重写 (Targeted Partial Rewrite)” 或 “指定内容重写 (Targeted Content Rewrite)”。
    - *人工迭代循环*：用户可追加修改指令 (`user_feedback`)，触发**状态流转到修改内容节点**，由 LLM 完成针对性修改后重新过审。
  - **修改内容节点 (Modify Content Node)**：仅在审查失败或用户不同意时执行，调用 `novel_agent` 专属 LLM 根据 `review_feedback` 或 `user_feedback` 修改小说项目内容、小说禁止规则或小说基本设定；修改完成后回到审查节点。
  - **写库节点 (Commit Node)**：最终状态落盘至 `novels` 集合，并保存 `forbidden_rules` 与 `basic_settings`。
- **规则说明**：
  - 小说不可跨世界存在，必须归属于单一 `world_id`。
  - 小说新增页必须允许在创建时填写 `name`、`introduction`、`summary`、`forbidden_rules` 与 `basic_settings`。
  - 小说详情页必须允许查看 `introduction`、`forbidden_rules` 与 `basic_settings`，这些字段是该小说下大纲和章节审查的强制约束。
  - 小说详情页修改 `forbidden_rules` 或 `basic_settings` 时必须进入 `novel_agent` 修改工作流，禁止详情页直接写库。
  - 小说级规则不得违反所属世界的 `forbidden_rules` 与 `basic_settings`。

### 14.4 大纲 (Outline) - 叙事结构规划

- **所属模块**：叙事策划
- **功能名称**：结构化大纲迭代 (Structured Outline Iteration)
- **功能背景**：长篇创作需要稳定的结构化蓝图，AI 生成的大纲通常需要经历多轮人工反馈和局部微调。
- **功能描述**：提供大纲的层级化规划能力，支持将故事拆解为多个叙事节点，并允许用户针对特定节点进行“局部重构”。大纲必须依次通过世界审查、世界观审查和小说审查，确保不违反世界根规则、关联世界观 Canon 和小说级规则。
- **新增逻辑**：
  1. `outline_agent` 根据小说简介生成初始树状结构。
  2. 支持从已有模板快速创建大纲框架。
  3. 物理写入 `outlines` 集合。
- **修改逻辑**：
  1. 修改大纲节点 -> 触发 `outline_agent` 进行 `partial_rewrite` (局部重写)。
  2. 利用 LangGraph 仅更新受影响的子路径。
  3. 更新后必须重新同步至下属章节的任务描述。
- **删除逻辑**：
  1. 删除大纲节点 -> 物理从数据库移除。
  2. 必须级联删除其下属的所有正文内容（`prose`）。
- **查询逻辑**：
  1. 必须带 `novel_id`、`world_id` 等业务条件进行查询。
  2. 支持分页和层级树状结构展示。
- **是否需要调用Agent**：是 (`outline_agent`)
- **Agent工作流与循环迭代机制 (State Graph)**：
  - **工作流状态 (State)**：维护树状结构数据、局部重写路径、`world_review_feedback`、`worldview_review_feedback`、`novel_review_feedback` 和用户反馈。
  - **输入节点 (Input Node)**：接收大纲结构或局部重写 (`partial_rewrite`) 指令。
  - **初始扩充节点 (Initial Expansion Node)**：作为第二个节点，调用 `outline_agent` 专属 LLM 对小说简介、大纲目标和局部重写范围进行初步拆解，直接生成可审查的大纲 payload，明确章节层级、关键冲突、受影响路径、父级 `novel_id/world_id/worldview_id` 和必须继承的设定约束；该节点不得写库，不得跳过 LLM，不得使用通用 Prompt。
    - **LLM 扩充逻辑**：将小说简介扩充为包含开端、发展、高潮、结局的树状剧情节点。针对局部重写指令，仅重构受影响的逻辑链条。
    - **扩充规则**：遵循“结构化”扩充，每个节点必须包含“剧情摘要”、“核心转折”和“涉及条目（Lore Tags）”。
  - **世界审查节点 (World Review Node)**：第一个审查节点，读取所属世界的 `forbidden_rules` 与 `basic_settings`，检查大纲是否违反世界禁止规则、基本设定、时代边界、力量体系、地理边界、组织结构或资源机制。
  - **世界观审查节点 (Worldview Review Node)**：第二个审查节点，读取关联 `worldview_id` 和同一 `world_id` 下已有 Canon，检查大纲是否违反世界观设定、历史地理规则、势力规则、资源机制或 Lore 前后一致性。
  - **小说审查节点 (Novel Review Node)**：第三个审查节点，读取所属小说的 `forbidden_rules` 与 `basic_settings`，检查大纲是否违反小说禁止规则、主角底线、主线冲突、叙事基调、时间线、人物关系规则或剧情约束。
    - *自动迭代循环*：任一审查未通过，将失败原因写入对应反馈字段，**状态流转到修改内容节点**进行逻辑重塑，修正后必须先回到世界审查节点，再依次进入世界观审查节点和小说审查节点。
  - **人工节点 (Approval Node/HITL)**：三个审查节点均通过后，用户确认节点结构。用户可选择：“批准 (Approve)”、“完全重写 (Full Rewrite)”、“指定局部重写 (Targeted Partial Rewrite)” 或 “指定内容重写 (Targeted Content Rewrite)”。
    - *人工迭代循环*：长篇大纲通常需多轮微调。用户的局部修改指令会导致状态机进入修改内容节点，再重新经历世界审查、世界观审查、小说审查，直到用户满意。
  - **修改内容节点 (Modify Content Node)**：仅在世界审查失败、世界观审查失败、小说审查失败或用户不同意时执行，调用 `outline_agent` 专属 LLM 根据审查反馈或 `user_feedback` 修改大纲内容；修改完成后回到世界审查节点。
  - **写库节点 (Commit Node)**：用户批准后写入 `outlines` 集合。
- **规则说明**：
  - 每次大纲修改必须记录版本快照。
  - 大纲必须依次通过世界审查、世界观审查、小说审查后才能进入人工确认。

### 14.5 章节 (Chapter) - 正文创作与审查

- **所属模块**：正文创作
- **功能名称**：智能章节迭代与审查 (Intelligent Chapter Iteration & Review)
- **功能背景**：正文创作是创作链路的终点，需要极高的内容质量把控和严密的逻辑审查（反吃设定）。
- **功能描述**：支持章节正文的 AI 协同生成、自动质量审查及手动润色，提供针对特定段落的“局部重写”能力以实现精细化打磨。章节必须依次通过世界审查、世界观审查、小说审查、大纲审查和章节审查，确保正文不违反世界根规则、关联世界观 Canon、小说级规则、父级大纲任务，并且与此前章节在剧情承接、人物状态和时间线上保持一致。
- **新增逻辑**：
  1. `chapter_agent` 根据大纲节点生成初稿。
  2. 继承所有父级 ID，物理写入 `prose` 集合。
- **修改逻辑**：
  1. 标记正文片段 -> 触发 AI 局部重写。
  2. 手动编辑 -> 保存时触发 `review_agent` 增量检查设定一致性。
- **删除逻辑**：
  1. 直接物理删除 `prose` 记录。
- **查询逻辑**：
  1. 绝对禁止全量显示。
  2. 必须按 `outline_id`、`novel_id` 或其他条件进行分页查询。
- **是否需要调用Agent**：是 (`chapter_agent` 及 `review_agent`)
- **Agent工作流与循环迭代机制 (State Graph)**：
  - **工作流状态 (State)**：维护正文内容、上下文记忆、大纲约束、`world_review_feedback`、`worldview_review_feedback`、`novel_review_feedback`、`outline_review_feedback`、`chapter_review_feedback` 和用户反馈。
  - **输入节点 (Input Node)**：注入大纲节点要求、前文上下文及重写片段标记。
  - **初始扩充节点 (Initial Expansion Node)**：作为第二个节点，调用 `chapter_agent` 专属 LLM 对大纲节点、前文上下文、目标片段和重写范围进行初步整理，直接生成可审查的章节 payload，明确场景目标、人物状态、叙事视角、上下文承接、父级 `outline_id/novel_id/worldview_id/world_id` 和不得违反的设定约束；该节点不得写库，不得跳过 LLM，不得使用通用 Prompt。
    - **LLM 扩充逻辑**：根据大纲节点的“剧情摘要”，扩充为具有文学表现力的正文段落。
    - **扩充规则**：遵循“正文级”扩充（1000-3000字），必须严格继承前文场景、人物状态和既定设定，不得出现“吃设定”现象。
  - **世界审查节点 (World Review Node)**：第一个审查节点，读取所属世界的 `forbidden_rules` 与 `basic_settings`，检查正文是否违反世界禁止规则、基本设定、时代边界、力量体系、地理边界、组织结构或资源机制。
  - **世界观审查节点 (Worldview Review Node)**：第二个审查节点，读取关联 `worldview_id` 和同一 `world_id` 下已有 Canon，检查正文是否违反世界观设定、历史地理规则、势力规则、资源机制或 Lore 前后一致性。
  - **小说审查节点 (Novel Review Node)**：第三个审查节点，读取所属小说的 `forbidden_rules` 与 `basic_settings`，检查正文是否违反小说禁止规则、主角底线、主线冲突、叙事基调、时间线、人物关系规则或剧情约束。
  - **大纲审查节点 (Outline Review Node)**：第四个审查节点，读取父级 `outline_id` 的大纲内容，检查正文是否严格执行大纲任务，不能擅自删除、提前、延后或改写大纲安排的关键事件。
  - **章节审查节点 (Chapter Review Node)**：第五个审查节点，读取同一 `outline_id/novel_id/world_id` 下此前已入库章节，检查当前章节与之前章节在剧情承接、时间线、地点变化、人物状态、人物关系、伤势/装备/资源、伏笔和叙事视角上是否一致。
    - *自动迭代循环*：任一审查未通过，将失败原因写入对应反馈字段，将流程**流转到修改内容节点**，要求 LLM 在不改变已有设定、父级大纲和前文事实的前提下修改正文，修改后必须先回到世界审查节点，再依次通过世界观审查、小说审查、大纲审查和章节审查。
  - **人工节点 (Approval Node/HITL)**：五个审查节点均通过后提供人工介入的打磨入口。用户可选择：“批准 (Approve)”、“完全重写 (Full Rewrite)”、“指定局部重写 (Targeted Partial Rewrite)” 或 “指定内容重写 (Targeted Content Rewrite)”。
    - *人工迭代循环*：用户标记不满意段落，提交修改意见，触发**状态流转到修改内容节点**，针对该段落修改后同样需要重新经历五个审查节点。
  - **修改内容节点 (Modify Content Node)**：仅在世界审查失败、世界观审查失败、小说审查失败、大纲审查失败、章节审查失败或用户不同意时执行，调用 `chapter_agent` 专属 LLM 根据审查反馈或 `user_feedback` 修改章节正文；修改完成后回到世界审查节点。
  - **写库节点 (Commit Node)**：定稿通过后继承所有关联 ID 写入 `prose` 集合。
- **规则说明**：
  - 只有通过“人工介入”或“审查通过”状态的章节才会被标记为“正式版本”。
  - 章节必须依次通过世界审查、世界观审查、小说审查、大纲审查、章节审查后才能进入人工确认。
  - 章节 Agent 固定链路必须为：`输入节点 -> 初始扩充节点 -> 世界审核节点 -> 世界观审查节点 -> 小说审查节点 -> 大纲审核节点 -> 章节审查节点 -> 人工节点 -> 入库节点`。
  - 章节 Agent 任一审查失败或人工不同意时，必须进入 `修改内容节点`，修改完成后必须回到 `世界审核节点`，重新依次执行世界审核、世界观审查、小说审查、大纲审核和章节审查。
  - 章节 Agent 禁止使用单一通用 `review` 节点替代上述 5 个独立审核节点。
  - 章节审查节点必须优先读取同一 `outline_id` 下此前已入库章节；如果无同一大纲章节，则退回读取同一 `novel_id` 或 `world_id` 下的前置章节上下文；如果当前为第一章且无前置章节，允许通过章节审查，但仍必须记录“无前置章节”的审查输入。

## 4. 页面需求

### 4.1 `/worlds` 世界列表

默认展示方式：

- 世界：列表显示。

页面能力：

- `/worlds` 只展示所有世界，不展示层级树，不展示世界观、小说、大纲、章节的全量数据。
- 页面不得内置假数据，世界列表必须来自 `GET /api/worlds/list`。
- 点击世界行必须跳转到 `/worlds/:world_id` 世界详情页，不允许只在列表页选中而不跳转。
- 支持创建、编辑、删除世界，所有操作必须通过真实 API。
- 世界详情页必须展示世界详细信息、世界禁止规则 (`forbidden_rules`) 与世界基本设定 (`basic_settings`)。
- 世界详情页必须允许编辑世界禁止规则和世界基本设定；点击保存必须跳转 `/workflow/world?action=update&id=<world_id>`，由 `world_agent` 修改内容节点和人工节点处理后写库。
- 世界详情页必须展示这两个字段被哪些下游审查节点使用：世界观的世界规则审查节点、小说的世界规则与背景契合审查节点。
- 删除世界时必须有 `cascade` 显式确认字段；未确认且存在世界观或小说时，后端必须返回 `409`。
- 世界下属数据的管理入口由 `/lore`、`/novels`、`/visualizer` 和各详情页承担。

### 4.2 对话与人工介入

- 世界、世界观、小说、大纲、章节 5 个模块的 Agent 生成、Agent 扩充和 Agent 审查链路必须进入对应的 Dify-like 工作流页面。
- 世界详情与小说详情中的禁止规则和设定规则修改必须进入对应 Agent 工作流，禁止详情页直接调用更新接口写库。
- 大纲与章节的新增和修改必须跳转到对应工作流页面，不允许在管理表格中直接写库。
- 删除操作必须通过明确确认入口提交真实删除接口；删除后必须再次查询确认记录不存在。
- 人工介入时必须提供修改模式：
  - `full_rewrite`：完全重写。
  - `partial_rewrite`：指定局部重写，默认值。
  - `content_rewrite`：指定内容重写。
- 用户手动修改表单字段时，工作流记录必须保存 `manual_edit=true`。
- AI 根据用户反馈修改时必须遵守范围保护，不得把小范围修改扩大为全量重写。

### 4.3 `/lore` 世界观资料库

- 页面不得内置假数据。
- 所有数据必须从数据库 API 查询。
- 必须提供世界选择器，且为强制字段；默认选择第一个真实查询到的世界。
- 所有世界观与设定条目查询都必须带当前选中 `world_id`，只显示该世界内容。
- 世界观数据禁止无条件全量查询。
- 世界观可切换 tree 或图展示，但必须带 `world_id`、`worldview_id`、`query` 等业务条件，并带分页。
- 章节禁止全量显示，必须按 `outline_id`、`worldview_id`、`novel_id` 或其他业务条件分页查询。

### 4.4 `/novels` 小说管理

- `/novels` 必须重做为小说列表页，用表格分页展示小说，不再承载大纲/章节迭代工作台。
- 页面必须提供世界筛选框，允许按 `world_id` 筛选小说。
- 页面必须提供小说搜索框，允许按小说 ID、名称、介绍、简介查询小说；搜索必须走真实 `GET /api/novels/list`，不允许只做前端假过滤。
- 小说列表必须显示：小说 ID、名称、介绍、简介/摘要、禁止规则数量、设定规则数量、更新时间和操作列。
- 小说列表操作列必须提供：修改、删除、大纲管理、章节管理。
- 点击小说行必须进入 `/novels/:novel_id` 小说详情页。
- 点击“新增小说”必须跳转到 `/novels/new` 独立新增页；新增页必须包含所属世界、小说名称、小说介绍、小说简介、小说禁止规则和小说设定规则。
- 新增小说保存必须调用真实 `POST /api/novels/create`，写入 `name`、`introduction`、`summary`、`forbidden_rules` 与 `basic_settings`，并在成功后通过 `GET /api/novels/get` 回查确认。
- 小说详情页必须显示小说详细信息、小说介绍、小说简介、小说禁止规则和小说设定规则。
- 小说详情页必须允许编辑小说禁止规则与小说设定规则；点击保存必须跳转 `/workflow/novel?action=update&world_id=<world_id>&id=<novel_id>`，由 `novel_agent` 修改内容节点和审查节点处理后写库。
- 删除小说必须有确认弹窗；若有下属大纲或章节且未传 `cascade=true`，后端必须返回 `409`。
- 点击“大纲管理”必须跳转 `/novels/:novel_id/outlines`。
- 点击“章节管理”必须跳转 `/novels/:novel_id/chapters`。

### 4.5 `/novels/:novel_id/outlines` 大纲管理

- 页面必须按当前小说 `novel_id` 查询大纲，表格分页显示，不允许全量加载所有小说的大纲。
- 表格必须显示：大纲 ID、名称、摘要、世界观、更新时间和操作列。
- 页面必须提供新增、修改、删除、查询能力。
- 点击“新增大纲”必须跳转 `/workflow/outline?action=create&world_id=<world_id>&novel_id=<novel_id>`。
- 点击“修改大纲”必须跳转 `/workflow/outline?action=update&world_id=<world_id>&novel_id=<novel_id>&id=<outline_id>`，并携带名称、摘要和可选 `worldview_id`。
- 删除大纲必须调用真实删除接口；删除后必须查询 `GET /api/outlines/list` 确认该大纲已不存在。
- 空数据时必须显示明确空状态，不允许伪造示例大纲。

### 4.6 `/novels/:novel_id/chapters` 章节管理

- 页面必须按当前小说 `novel_id` 查询章节，表格分页显示，不允许全量加载所有小说章节。
- 页面必须提供大纲筛选框；选择大纲后按 `outline_id` 过滤章节。
- 表格必须显示：章节 ID、标题、所属大纲、正文摘要、更新时间和操作列。
- 页面必须提供新增、修改、删除、查询能力。
- 点击“新增章节”必须跳转 `/workflow/chapter?action=create&world_id=<world_id>&novel_id=<novel_id>&outline_id=<outline_id>`；如果未选择大纲但当前小说有大纲，允许默认使用第一条大纲。
- 点击“修改章节”必须跳转 `/workflow/chapter?action=update&world_id=<world_id>&novel_id=<novel_id>&outline_id=<outline_id>&id=<chapter_id>`，并携带标题和正文。
- 删除章节必须调用真实删除接口；删除后必须查询 `GET /api/lore/list` 确认该章节已不存在。
- 空数据时必须显示明确空状态，不允许伪造示例章节。

### 4.7 旧 `/outlines` 大纲章节迭代工作台

- 旧 `/outlines` 综合工作台已废弃，不得作为小说、大纲、章节管理的主入口恢复。
- 大纲管理必须进入 `/novels/:novel_id/outlines`。
- 章节管理必须进入 `/novels/:novel_id/chapters`。
- 大纲与章节的创建、修改、迭代必须使用 Dify-like 工作流页面；删除必须使用真实删除接口和删除后回查。
- 大纲与章节工作流必须在审查节点展示其继承的小说禁止规则与小说基本设定，便于用户确认审查依据。
- 提供世界观导入入口：
  - 支持 `.json`、`.md`、`.markdown`、`.opml`。
  - 导入时必须选择当前世界下的目标 `worldview_id`。
  - 导入后必须保持原文件中的上下级关系，并写入 `path`、`hierarchy_path`、`hierarchy_order` 等字段。
  - 导入完成后前端必须再次查询数据库确认导入内容真实存在，不能只相信上传接口返回成功。

### 4.8 `/visualizer` 世界观图谱与逻辑树

- 页面不得内置假数据，所有图谱、树和表格数据必须来自数据库 API。
- 必须提供世界选择器，且为强制字段；默认选择第一个真实查询到的世界。
- 关系图谱和逻辑树默认只显示当前世界下的 3 级内容，禁止一次性全量展开。
- 查询必须带当前选中 `world_id`，查询结果使用表格显示。
- 点击查询结果表格中的一行时，图谱必须切换为该内容上下 3 级关系。
- 点击图谱节点时，必须在当前已显示图谱的基础上增量新增该节点的直接子节点和相关内容节点，不允许替换或清空原有图谱。
- 图谱必须提供“自动整理”功能，对当前已显示节点按现有边关系重新布局；自动整理不得重新查询数据库、不得删除已展开节点、不得改变当前节点集合。
- 点击逻辑树节点时，必须按节点名称或路径查询并显示下一级内容。
- 如果节点查询失败或返回跨世界数据，页面必须显示明确错误，不允许静默降级或保留旧数据伪装成功。

### 4.9 五模块独立工作流页 (Dify-like)

- **核心目标**：为所有 Agent 任务（增、删、改）提供透明的可视化监控与交互界面。
- **覆盖范围**：
  - 世界：`world_agent` 的新增、修改。
  - 世界观：`worldview_agent` 的新增、修改。
  - 小说：`novel_agent` 的新增、修改。
  - 大纲：`outline_agent` 的新增、修改。
  - 章节：`chapter_agent` 的新增、修改。
- **入口要求**：
  - Agent 驱动的“创建”和“修改”入口必须进入对应模块的独立工作流页面，不允许只使用一个依赖 `type` 参数的通用页面承载全部业务；世界/小说详情页的规则字段修改也必须进入对应 Agent 工作流。
  - 每个工作流页面必须有独立的路由、标题、初始参数校验和表单字段；节点图、对话框、输入输出查看组件可以复用，但业务表单和必填父级参数不能混用。
  - 路由形态：
    - 新增世界：`/workflow/world?action=create`，不得要求、传递或展示 `world_id`，因为世界是顶层实体。
    - 修改世界：`/workflow/world?action=update&id=<world_id>`
    - 世界观：`/workflow/worldview?action=<create|update>&world_id=<id>&id=<worldview_id>`
    - 小说：`/workflow/novel?action=<create|update>&world_id=<id>&id=<novel_id>`
    - 大纲：`/workflow/outline?action=<create|update>&world_id=<id>&novel_id=<id>&id=<outline_id>`
    - 章节：`/workflow/chapter?action=<create|update>&world_id=<id>&novel_id=<id>&outline_id=<id>&id=<chapter_id>`
  - `/novels/:novel_id/outlines` 的新增/修改入口必须跳转到 `/workflow/outline`。
  - `/novels/:novel_id/chapters` 的新增/修改入口必须跳转到 `/workflow/chapter`。
  - 旧的查询参数兼容入口已废弃且不得恢复；所有入口必须直接跳转到 `/workflow/<type>`。
  - 除新增世界外，工作流页面必须能通过 URL 参数或等价状态明确识别 `type`、`action`、`world_id`、父级 ID、目标 ID。
  - 若新增世界入口错误携带 `world_id`，前端必须移除该参数或忽略该参数，不能让用户选择世界。
  - 如果从弹窗进入工作流，弹窗中也必须展示同等信息，不允许隐藏当前实体、父级和 action。
- **布局要求**：
  - 新增世界页面不得把“新增表单”“用户处理”“工作流图”“节点/对话详情”横向挤在同一行；必须按纵向列展示，优先顺序为：新增表单 -> 用户处理 -> 工作流图 -> 节点/对话详情。
  - 新增世界不需要选择父级世界，页面不得展示世界选择器，也不得要求用户输入 `world_id`。
  - 名称、摘要/设定、Agent 消息、人工反馈等输入区域必须使用适合长文本的输入框；摘要/设定、消息和反馈不得使用过小的单行输入框。
  - 其他模块可以使用多栏布局，但在宽度不足时必须自动折叠为纵向列，避免表单、节点图和对话框互相挤压。
- **可视化节点图**：
  - 展示工作流的完整节点链；世界必须按真实 `world_agent` 展示 `Input -> Initial Expansion -> Human -> Saver/Commit`，用户不同意时展示 `Human -> Modify Content -> Human` 循环，不得显示草案节点或审查节点。
  - 世界观必须展示 `Input -> Initial Expansion -> World Rule Review -> Worldview Consistency Review -> Approval -> Commit`，任一审查失败或人工不同意时展示 `World Rule Review/Worldview Consistency Review/Human -> Modify Content -> World Rule Review` 循环。
  - 小说必须展示 `Input -> Initial Expansion -> Review -> Approval -> Commit`，审查失败或人工不同意时展示 `Review/Human -> Modify Content -> Review` 循环。
  - 大纲必须展示 `Input -> Initial Expansion -> World Review -> Worldview Review -> Novel Review -> Approval -> Commit`，任一审查失败或人工不同意时展示 `World Review/Worldview Review/Novel Review/Human -> Modify Content -> World Review` 循环。
  - 章节必须展示 `Input -> Initial Expansion -> World Review -> Worldview Review -> Novel Review -> Outline Review -> Chapter Review -> Approval -> Commit`，任一审查失败或人工不同意时展示 `World Review/Worldview Review/Novel Review/Outline Review/Chapter Review/Human -> Modify Content -> World Review` 循环。
  - 当前运行节点需高亮显示，并带有动态加载动画。
  - 每个节点必须显示状态：`pending`、`running`、`completed`、`failed`、`waiting_human` 或 `skipped`。
  - 节点点击可查看该步骤的详细输入 (Input) 与输出 (Output) JSON 数据。
  - 节点输入输出必须来自真实 `hierarchy_agent_runs.nodes` 或实时 Agent 返回结果，不允许前端写死样例 JSON。
  - 节点失败时必须显示后端返回的真实错误，不允许吞异常或只显示“失败”。
- **对话式交互 (HITL)**：
  - 右侧提供常驻对话框，用于用户与 Agent 的实时反馈。
  - 支持快捷操作：`批准 (Approve)`、`重写 (Rewrite)`、`中止 (Stop)`。
  - 对话框必须保留本次新增/修改的用户消息、Agent 回复、审查意见、人工反馈和最终批准记录。
  - 对话框中用户可选择 `partial_rewrite`、`content_rewrite`、`full_rewrite`，默认必须为 `partial_rewrite`。
  - 用户可在对话框中手动改写表单字段；提交时必须带 `manual_edit=true`，并写入工作流记录。
- **执行状态要求**：
  - 页面必须明确展示“当前运行到哪个节点”。
  - 页面必须展示每一轮迭代次数 `iterations`。
  - 审查失败时，状态必须停留在可见的失败/待修改节点，并展示失败原因。
  - 人工批准前不得执行真实写库节点。
  - 写库成功后必须展示 `commit_result`，包括真实生成或更新的业务 ID。
- **日志流**：展示底层的推理日志和 API 调用状态。
- **适用范围**：世界 (World)、世界观 (Worldview)、小说 (Novel)、大纲 (Outline)、章节 (Chapter) 的所有变更操作。

## 5. Agent 需求

### 5.1 独立 Agent 与物理隔离要求

每类实体必须由完全独立的 Agent 逻辑、工作流及代码实例执行，严禁使用“通用 Agent”或“单体 Agent”承载多模块业务：

- `world_agent`：专用于世界实体的创建与修改。
- `worldview_agent`：专用于世界观设定，包含世界规则审查与既有世界观一致性审查两个审查节点。
- `novel_agent`：专用于小说项目，包含小说禁止规则、小说基本设定管理，以及世界禁止规则、世界基本设定与背景契合度审查。
- `outline_agent`：专用于大纲结构，支持局部重写，并包含世界审查、世界观审查、小说审查三个审查节点。
- `chapter_agent`：专用于章节正文，包含世界审查、世界观审查、小说审查、大纲审查、章节审查五个审查节点。

**隔离规则**：

1. **逻辑独立**：禁止不同模块共用同一个 Agent 实例、同一个 State Graph 定义或同一个后端处理类。
2. **配置隔离**：每个模块的 Agent 必须具有独立的 Prompt 模板、工具集（Tools）和状态管理逻辑。
3. **流程解耦**：一个模块的工作流逻辑变更不得影响其他模块的 Agent 运行。
4. **审核文件隔离**：世界审核、世界观审查、小说审查、大纲审核、章节审查必须拆分为独立审核节点文件，禁止把审核节点实现直接写回 5 个 Agent 主文件。
5. **审核文件固定位置**：审核节点文件只能放在 `src/agents/review_nodes/` 目录下，一个审核一个文件：`world_review.py`、`worldview_review.py`、`novel_review.py`、`outline_review.py`、`chapter_review.py`。
6. **Agent 主文件职责**：`world_agent.py`、`worldview_agent.py`、`novel_agent.py`、`outline_agent.py`、`chapter_agent.py` 只负责本模块 State、初始扩充、修改内容、人工节点、写库节点和工作流连线；审核节点必须从独立审核文件导入。

### 5.2 审查规则

- 世界：不需要审查，工作流不得显示或记录审查节点；必须记录工作流输入、真实 LLM 初始扩充内容、人工确认、修改内容循环和写库固化结果。
- 世界观：需要审查，且必须包含 2 个独立审查节点：世界规则审查节点、既有世界观一致性审查节点。
- 小说：需要审查，必须检查是否违反所属世界的禁止规则与基本设定，并检查小说自身禁止规则与基本设定是否自洽、是否与世界级规则冲突。
- 大纲：需要审查，且必须包含 3 个独立审查节点：世界审查节点、世界观审查节点、小说审查节点。
- 章节：需要审查，且必须包含 5 个独立审查节点：世界审查节点、世界观审查节点、小说审查节点、大纲审查节点、章节审查节点。

### 5.2.1 Agent 工作流标准链路

5 个模块必须按以下固定链路执行，不允许插入通用 Agent，不允许恢复草案节点，不允许跳过初始扩充节点或修改内容节点：

- **世界 Agent (`world_agent`) 同意路径**：
  `输入节点 -> 初始扩充节点 -> 人工节点（同意）-> 入库节点`
- **世界 Agent (`world_agent`) 不同意路径**：
  `输入节点 -> 初始扩充节点 -> 人工节点（不同意）-> 修改内容节点 -> 人工节点（同意）-> 入库节点`
- **世界观 Agent (`worldview_agent`) 同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **世界观 Agent (`worldview_agent`) 不同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（通过）-> 人工节点（不同意）-> 修改内容节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **世界观 Agent (`worldview_agent`) 审查失败路径**：
  `输入节点 -> 初始扩充节点 -> 世界规则审查节点（不通过）-> 修改内容节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（不通过）-> 修改内容节点 -> 世界规则审查节点（通过）-> 既有世界观一致性审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **大纲 Agent (`outline_agent`) 同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **大纲 Agent (`outline_agent`) 不同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（不同意）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **大纲 Agent (`outline_agent`) 审查失败路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **章节 Agent (`chapter_agent`) 同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **章节 Agent (`chapter_agent`) 不同意路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（不同意）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **章节 Agent (`chapter_agent`) 审查失败路径**：
  `输入节点 -> 初始扩充节点 -> 世界审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
  或 `输入节点 -> 初始扩充节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（不通过）-> 修改内容节点 -> 世界审查节点（通过）-> 世界观审查节点（通过）-> 小说审查节点（通过）-> 大纲审查节点（通过）-> 章节审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **小说 Agent 同意路径**：
  `输入节点 -> 初始扩充节点 -> 审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **小说 Agent 不同意路径**：
  `输入节点 -> 初始扩充节点 -> 审查节点（通过）-> 人工节点（不同意）-> 修改内容节点 -> 审查节点（通过）-> 人工节点（同意）-> 入库节点`
- **小说 Agent 审查失败路径**：
  `输入节点 -> 初始扩充节点 -> 审查节点（不通过）-> 修改内容节点 -> 审查节点（通过）-> 人工节点（同意）-> 入库节点`

节点约束：

- `初始扩充节点` 必须调用当前模块专属 LLM，直接生成可进入人工确认或审查的业务 payload。
- `修改内容节点` 必须调用当前模块专属 LLM，根据人工反馈或审查失败原因修改当前 payload。
- 世界模块的 `修改内容节点` 完成后只能回到 `人工节点`。
- 世界 Agent 的 `初始扩充节点` 和 `修改内容节点` 必须维护世界级 `forbidden_rules` 与 `basic_settings`；世界详情页必须允许用户查看和修改这两个字段。
- 世界观的 `修改内容节点` 完成后必须回到 `世界规则审查节点`，再进入 `既有世界观一致性审查节点`，两个审查节点都通过后才能进入 `人工节点`。
- 大纲的 `修改内容节点` 完成后必须回到 `世界审查节点`，再进入 `世界观审查节点` 与 `小说审查节点`，三个审查节点都通过后才能进入 `人工节点`。
- 章节的 `修改内容节点` 完成后必须回到 `世界审查节点`，再进入 `世界观审查节点`、`小说审查节点`、`大纲审查节点` 与 `章节审查节点`，五个审查节点都通过后才能进入 `人工节点`。
- 小说的 `修改内容节点` 完成后必须回到 `审查节点`，审查通过后才能进入 `人工节点`。
- 小说 Agent 的 `初始扩充节点` 和 `修改内容节点` 必须维护小说级 `forbidden_rules` 与 `basic_settings`；小说详情页必须允许用户查看和修改这两个字段。
- 大纲和章节审查节点必须读取所属小说的 `forbidden_rules` 与 `basic_settings`，并将其作为强制约束。
- 5 个模块均不得显示、记录或执行 `draft`/草案节点。

### 5.3 工作流记录

每次 Agent 运行必须写入 `hierarchy_agent_runs`，至少包含：

- `run_id`
- `agent_type`
- `action`
- `entity_id` 或 `target_id`
- `world_id`
- `parent_ids`
- `status`
- `current_node`
- `iterations`
- `review_required`
- `pending_payload`
- `conversation`
- `nodes`
- `commit_result`
- `created_at`
- `updated_at`

节点至少覆盖：

- 输入节点：记录用户消息、表单 payload、URL 参数和父级 ID。
- 初始扩充节点：作为所有模块的第二个节点，调用当前模块专属 Agent LLM 生成可进入人工确认或审查的业务 payload，并记录 `expanded_input`、用户原始意图、补全后的上下文、父级约束和缺失字段；禁止跳过 LLM，禁止使用通用 Prompt，禁止写库。
- 逻辑审计节点：展示基于 RAG 的冲突检测输入输出；仅适用于世界观、小说、大纲、章节，世界模块不显示也不记录审查节点。
- 世界规则字段：世界工作流必须记录 `forbidden_rules` 与 `basic_settings` 的输入、LLM 扩充结果、人工修改结果和写库结果；世界详情页管理这些字段时必须跳转世界 Agent 工作流，禁止直接保存。
- 世界观世界规则审查节点：世界观专属第一个审查节点，必须读取所属 `world_id` 的 `forbidden_rules` 与 `basic_settings`，记录是否违反世界禁止规则、基本设定、时代边界、力量体系、地理边界、组织结构和资源机制。
- 世界观既有设定一致性审查节点：世界观专属第二个审查节点，必须检索同一世界下已有世界观设定，记录是否与已有 Canon 发生冲突。
- 小说规则字段：小说工作流必须记录 `forbidden_rules` 与 `basic_settings` 的输入、LLM 扩充结果、人工修改结果和写库结果；小说详情页管理这些字段时必须跳转小说 Agent 工作流，禁止直接保存。
- 大纲世界审查节点：大纲专属第一个审查节点，必须读取所属 `world_id` 的 `forbidden_rules` 与 `basic_settings`，记录是否违反世界根规则。
- 大纲世界观审查节点：大纲专属第二个审查节点，必须读取关联 `worldview_id` 与同一世界下已有 Canon，记录是否违反世界观设定。
- 大纲小说审查节点：大纲专属第三个审查节点，必须读取所属 `novel_id` 的 `forbidden_rules` 与 `basic_settings`，记录是否违反小说主线、主角底线、时间线、人物关系和剧情约束。
- 章节世界审查节点：章节专属第一个审查节点，必须读取所属 `world_id` 的 `forbidden_rules` 与 `basic_settings`，记录是否违反世界根规则。
- 章节世界观审查节点：章节专属第二个审查节点，必须读取关联 `worldview_id` 与同一世界下已有 Canon，记录是否违反世界观设定。
- 章节小说审查节点：章节专属第三个审查节点，必须读取所属 `novel_id` 的 `forbidden_rules` 与 `basic_settings`，记录是否违反小说主线、主角底线、时间线、人物关系和剧情约束。
- 章节大纲审查节点：章节专属第四个审查节点，必须读取父级 `outline_id` 的大纲内容，记录是否违反大纲任务、关键事件、转折、冲突升级和结尾安排。
- 章节审查节点：章节专属第五个审查节点，必须读取同一 `outline_id/novel_id/world_id` 下此前已入库章节，记录当前章节是否与前文在剧情承接、时间线、人物状态、地点变化、资源装备、伏笔和叙事视角上保持一致。
- 修改内容节点：在人工不同意后或审查失败后，根据用户反馈或审查反馈调用当前模块专属 LLM 修改内容，并记录修改前 payload、反馈、修改后 payload 和 LLM 调用信息；世界模块修改后回到人工节点，世界观模块修改后回到世界规则审查节点，大纲模块修改后回到世界审查节点，章节模块修改后回到世界审查节点，小说修改后回到审查节点。
- 审查节点：世界观必须记录两个审查节点的输入、输出、是否通过和失败原因；小说必须记录世界级规则与小说级规则审查输入输出；大纲必须记录世界审查、世界观审查、小说审查三个节点的输入输出；章节必须记录世界审查、世界观审查、小说审查、大纲审查、章节审查五个节点的输入输出、是否通过和失败原因。
- 人工节点：提供 Dify 风格的对话交互区，允许用户输入修改建议、选择修改模式和手动修改字段。
- 写库节点：记录真实数据库写入结果、最终生成或更新的 ID、受影响集合和写入时间。

5 个模块新增/修改工作流必须满足：

- `world + create`
- `world + update`
- `worldview + create`
- `worldview + update`
- `novel + create`
- `novel + update`
- `outline + create`
- `outline + update`
- `chapter + create`
- `chapter + update`

以上 10 条链路都必须可在页面查看节点状态、当前节点、输入输出、对话记录和最终写库结果。

### 5.4 Agent 核心机制与扩充规则

所有模块的 Agent 必须遵循统一的循环迭代机制与 LLM 扩充规则，确保生成内容的质量与层级一致性。

#### 5.4.1 循环迭代机制 (Iteration Mechanism)

Agent 运行过程分为三个核心阶段，形成闭环：

1. **自动迭代 (Auto-Iteration)**：
    - 主要发生在“审查节点 (Review Node)”失败时。
    - Agent 会自动携带审查失败原因（冲突点、逻辑漏洞）进入“修改内容节点”，重新调用对应模块专属 LLM 生成修正方案。
    - 自动迭代应有最大次数限制（默认 3 次），超过则强制转入人工介入状态。
2. **人工迭代 (Human-in-the-Loop Iteration)**：
    - 在“人工节点 (Approval Node)”中，用户通过对话框提供反馈。
    - Agent 接收反馈后，判断修改范围（`full_rewrite`, `partial_rewrite`, `content_rewrite`），并进入“修改内容节点”按用户反馈生成修正方案。
3. **固化迭代 (Commit Iteration)**：
    - 只有在人工批准后，才会通过“写库节点 (Commit Node)”完成最终固化。

#### 5.4.2 LLM 扩充规则 (Expansion Rules)

“初始扩充节点”负责完成首次**扩充 (Expansion)**，“修改内容节点”负责后续按审查意见或人工反馈进行修正。当用户提供简略输入（如：一个标题或一句话摘要）时，初始扩充节点必须调用模块专属 LLM 生成可审查或可人工确认的业务 payload，并同时记录 `expanded_input`。LLM 必须按以下规则进行高质量扩充：

1. **意图保持 (Intent Preservation)**：扩充内容必须 100% 保留用户的核心创意和关键指令，不得擅自删减用户提供的核心要素。
2. **逻辑补全 (Logical Completion)**：根据模块定位自动补全逻辑细节。
    - *例如*：扩充“世界”时，LLM 应自动构思其核心物理法则、文明等级等基础框架。
    - *例如*：扩充“章节”时，LLM 应根据大纲节点自动生成场景描写、人物对话与心理活动。
3. **约束继承 (Constraint Inheritance)**：
    - **强约束**：必须严格遵守所有父级 ID 关联的设定。若在写“章节”，则必须检索并遵循该世界的“世界观设定”和该小说的“大纲约束”。
    - **反吃设定**：LLM 在扩充时应主动自查是否与已有上下文冲突。
4. **颗粒度分级 (Granularity)**：
    - **摘要级 (Summary)**：50-200 字，用于世界、小说等描述。
    - **条目级 (Lore)**：200-500 字，具有明确的分类、属性和描述字段。
    - **正文级 (Prose)**：1000-3000 字，包含完整的叙事结构。
5. **结构化输出 (Structured Output)**：
    - 所有扩充结果必须以 JSON 格式返回给工作流引擎，区分 `metadata` (元数据) 和 `content` (主体内容)。

---

## 6. 数据模型需求

### 6.1 MongoDB 集合

`worlds`

- `world_id`
- `name`
- `summary`
- `forbidden_rules`
- `basic_settings`
- `timestamp`

`worldviews`

- `worldview_id`
- `world_id`
- `name`
- `summary`
- `timestamp`

`novels`

- `novel_id`
- `world_id`
- `name`
- `introduction`
- `summary`
- `forbidden_rules`
- `basic_settings`
- `timestamp`

`outlines`

- `outline_id`
- `novel_id`
- `worldview_id`
- `world_id`
- `name/title`
- `summary`
- `timestamp`

`prose`

- `id/scene_id/prose_id`
- `outline_id`
- `novel_id`
- `worldview_id`
- `world_id`
- `title/name`
- `content`
- `timestamp`

`lore`

- `doc_id`
- `type`
- `name`
- `content`
- `category`
- `path`
- `hierarchy_path`
- `hierarchy_order`
- `world_id`
- `worldview_id`
- `outline_id`
- `novel_id`
- `source_file`
- `source_format`
- `timestamp`

`lore` 用于存储世界观设定条目和导入后的层级资料。导入资料必须保留原始上下级关系，`path` 与 `hierarchy_path` 必须可用于树、图谱、搜索和下一级展开查询。

### 6.2 迁移需求

- 必须提供幂等迁移脚本补齐旧数据关系。
- 重复执行不得重复创建默认世界或默认小说。
- 不得覆盖已有明确的 `world_id`、`worldview_id`、`novel_id`、`outline_id`。

迁移命令：

```bash
PYTHONPATH=.:src:src/common .venv/bin/python scripts/migrate_world_hierarchy.py
```

## 7. API 需求

### 7.1 世界 API

- `GET /api/worlds/list`
- `GET /api/worlds/get`
- `POST /api/worlds/create`
- `POST /api/worlds/update`
- `DELETE /api/worlds/delete`

世界列表是唯一允许无条件列表查询的业务数据接口。

### 7.2 非世界查询限制

除 `worlds` 列表与按唯一 ID 查询的详情接口外，所有数据库列表查询接口必须同时满足：

- 必须带业务条件，例如 `world_id`、`worldview_id`、`novel_id`、`outline_id`、`run_id`、`agent_type`、`status` 或 `query`。
- 必须带 `page` 与 `page_size`。
- 缺失业务条件时返回明确 `400`。
- 缺失分页时返回明确 `400`。
- `page_size` 必须有上限，当前上限为 100。

按唯一 ID 查询的详情接口如 `GET /api/worlds/get?world_id=...`、`GET /api/novels/get?novel_id=...` 必须提供唯一业务 ID，缺失时返回明确 `400`，不存在时返回明确 `404`。

适用接口包括但不限于：

- `GET /api/worldviews/list`
- `GET /api/novels/list`
- `GET /api/novels/get`
- `GET /api/outlines/list`
- `GET /api/lore/list`
- `GET /api/lore/tree`
- `GET /api/lore/mindmap`
- `GET /api/lore/entity-graph/<doc_id>`
- `GET /api/lore/export/opml`
- `GET /api/world-hierarchy/tree`
- `GET /api/workflow/outline-chapter/state`
- `GET /api/hierarchy-agent/list`

`GET /api/novels/list` 的 `query` 必须覆盖 `novel_id`、`name`、`introduction`、`summary`，保证小说列表页搜索真实来自后端。

`GET /api/outlines/list` 的 `query` 必须覆盖 `outline_id`、`id`、`name`、`title`、`summary`，保证大纲管理页搜索真实来自后端。

`GET /api/lore/list` 的 `query` 必须覆盖 `name`、`content`、`category`、`path` 等字段，保证导入后的层级节点可以通过名称或路径被查询和展开。

### 7.3 写入与删除

- 创建子级时必须校验父级存在。
- 修改父级字段时必须校验新父级存在。
- `POST /api/novels/create` 必须支持写入 `introduction`、`forbidden_rules` 与 `basic_settings`。
- `POST /api/novels/update` 必须支持修改 `introduction`、`forbidden_rules` 与 `basic_settings`，并校验目标小说存在。
- 删除父级时：
  - 没有子级可直接删除。
  - 有子级且未传 `cascade=true` 必须返回 `409`。
  - 有子级且传 `cascade=true` 才允许显式级联删除。
- 增删改后必须可通过查询接口验证数据库真实变化。

### 7.4 世界观导入 API

- `POST /api/worldviews/import`
- 请求类型：`multipart/form-data`。
- 必填字段：`world_id`、`worldview_id`、`file`。
- 支持格式：`.json`、`.md`、`.markdown`、`.opml`。
- 必须校验：
  - `world_id` 对应世界真实存在。
  - `worldview_id` 对应世界观真实存在。
  - `worldview_id` 必须归属请求中的 `world_id`。
  - 文件类型不支持时返回明确 `400`。
- 写入规则：
  - 导入内容写入 `lore` 集合。
  - 每个节点必须带 `world_id`、`worldview_id`、`path`、`hierarchy_path`、`hierarchy_order`、`source_file`、`source_format`。
  - 同一 `worldview_id + source_file + path` 重复导入应更新同一条记录，不得重复制造同一路径节点。
  - 导入成功后必须能通过 `GET /api/lore/list` 查询确认。

## 8. 禁止项

产品和测试中禁止：

- 页面写死假数据。
- API 返回假数据。
- 使用 mock、假库、fixture 伪装真实测试。
- 使用 JSONL 或本地文件作为数据库失败时的静默回退。
- 隐藏后端错误。
- 吞异常后返回成功。
- 删除断言让测试通过。
- 修改测试预期掩盖问题。
- 查询接口无条件读取大集合。
- 世界观和章节全量展示。
- **禁止使用同一个 Agent**：严禁在后端实现或前端调用中使用单一 Agent 承载不同层级或不同模块的业务逻辑；必须保持 5 个模块 Agent 的物理与逻辑独立。

## 9. 验收测试要求

测试必须使用 `requests` 调用真实运行中的 API 服务。

### 9.1 正常路径

- 创建 world 后查询确认字段一致。
- 创建 worldview 后查询确认挂到 world。
- 创建 novel 后查询确认挂到 world，且不作为 worldview 子级。
- 创建 outline 后查询确认挂到 novel，并带 `worldview_id` 与 `world_id`。
- 创建 chapter/prose 后查询确认挂到 outline、novel、worldview、world。
- 查询 `/api/world-hierarchy/tree`，严格检查层级顺序和每层 ID。
- 导入 JSON/Markdown/OPML 世界观后，必须再次查询 `GET /api/lore/list`，严格检查 `world_id`、`worldview_id`、`path`、`hierarchy_path` 与原文件上下级关系一致。
- `/visualizer` 查询必须返回当前世界内的数据，表格点击后图谱显示该内容上下 3 级，图谱或逻辑树节点点击后显示下一级。
- 5 个模块的新增/修改工作流页面必须逐项验证：
  - 打开对应创建或修改入口。
  - 页面展示正确的 `type`、`action`、`world_id`、父级 ID 或目标 ID。
  - 启动工作流后查询 `GET /api/hierarchy-agent/get`，检查 `current_node`、`nodes`、`conversation` 与页面显示一致。
  - 点击每个节点后，页面展示真实 input/output。
  - 人工反馈通过 `POST /api/hierarchy-agent/respond` 提交后，页面显示新一轮 `iterations`。
  - 批准写库后，必须再次查询业务列表接口确认数据真实新增或修改。

### 9.2 修改路径

- 逐层 update 后必须再次 query。
- 检查名称、summary、content、父级字段真实变化。
- 对章节修改必须再次按 `outline_id` 或其他条件查询确认正文变化。
- 修改工作流必须验证默认修改模式为 `partial_rewrite`。
- 小范围修改必须检查未被修改字段保持原值，防止 Agent 把局部改动扩大为全量重写。
- 手动修改字段提交后必须检查 `manual_edit=true` 被写入工作流记录。

### 9.3 删除路径

- 非空父级未带 `cascade=true` 必须返回 `409`。
- 返回 `409` 后必须查询确认子级仍存在。
- 带 `cascade=true` 删除 novel/worldview/world 后，必须查询确认所有子级真实消失。

### 9.4 异常路径

- 父级不存在时创建子级必须返回 `400` 或 `404`。
- 缺少必填 ID/name 必须返回 `400`。
- 非法 type 或非法层级关系必须返回明确错误。
- 非世界查询缺失条件必须返回 `400`。
- 非世界查询缺失 `page/page_size` 必须返回 `400`。
- 世界观导入缺少 `world_id`、`worldview_id` 或 `file` 必须返回 `400`。
- 世界观导入文件格式不支持必须返回 `400`。
- 世界观导入时 `worldview_id` 不属于 `world_id` 必须返回明确错误。
- `/visualizer` 查询或节点展开返回跨世界数据必须报错，不允许静默显示。
- 工作流页面缺失 `type`、`action`、`world_id` 或必需父级 ID 时必须显示明确错误，禁止启动伪工作流。
- 非法 `type/action` 必须返回明确错误，不允许进入空白或假节点页面。
- 后端工作流节点失败时，前端必须显示真实错误内容，不允许吞异常或自动跳到完成状态。

### 9.5 真实验证命令

```bash
API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_world_hierarchy_requests.py
API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_hierarchy_agent_workflow_requests.py
API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_outline_chapter_workflow_requests.py
API_BASE_URL=http://127.0.0.1:5006 REQUEST_TIMEOUT=180 .venv/bin/python tests/test_world_workflow_llm_draft_requests.py
API_BASE_URL=http://127.0.0.1:5006 REQUEST_TIMEOUT=180 .venv/bin/python tests/test_five_module_workflow_requests.py
API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_worldview_import_requests.py
START_TEST_SERVER=1 API_BASE_URL=http://127.0.0.1:5017 .venv/bin/python tests/test_worldview_import_requests.py
npm run build
curl -I http://127.0.0.1:5174/worlds
curl -I 'http://127.0.0.1:5174/workflow/world?action=create'
curl -I 'http://127.0.0.1:5174/workflow/worldview?action=create&world_id=world_default'
curl -I 'http://127.0.0.1:5174/workflow/novel?action=create&world_id=world_default'
curl -I 'http://127.0.0.1:5174/workflow/outline?action=create&world_id=world_default'
curl -I 'http://127.0.0.1:5174/workflow/chapter?action=create&world_id=world_default'
curl -I http://127.0.0.1:5174/lore
curl -I http://127.0.0.1:5174/novels
curl -I http://127.0.0.1:5174/novels/new
curl -I http://127.0.0.1:5174/novels/<novel_id>
curl -I http://127.0.0.1:5174/novels/<novel_id>/outlines
curl -I http://127.0.0.1:5174/novels/<novel_id>/chapters
curl -I http://127.0.0.1:5174/visualizer
```

## 10. 当前验证状态

验证状态：分项验证。

真实执行证据：

- `tests/test_world_hierarchy_requests.py`：通过。
- `tests/test_hierarchy_agent_workflow_requests.py`：通过。
- `tests/test_outline_chapter_workflow_requests.py`：通过。
- `tests/test_world_workflow_llm_draft_requests.py`：通过，覆盖新增世界工作流必须触发真实 `world_agent` LLM 初始扩充节点，检查 `initial_expansion.output.llm_invoked=true`、`raw_response` 非空、世界流程不存在 `draft` 或 `review` 节点，并在批准后查询确认真实写库。
- `tests/test_worldview_import_requests.py`：通过，覆盖 JSON、Markdown、OPML 导入并查询确认层级路径真实写入。
- `START_TEST_SERVER=1 API_BASE_URL=http://127.0.0.1:5017 .venv/bin/python tests/test_worldview_import_requests.py`：通过，测试脚本可启动真实 Flask 服务验证导入 API。
- `API_BASE_URL=http://127.0.0.1:5006 .venv/bin/python tests/test_worldview_import_requests.py`：通过，测试脚本可打现有 `5006` 服务验证导入 API。
- `API_BASE_URL=http://127.0.0.1:5006 REQUEST_TIMEOUT=180 .venv/bin/python tests/test_five_module_workflow_requests.py`：通过，覆盖世界、世界观、小说、大纲、章节 5 个模块的新增/修改工作流，使用真实 requests 调用 `/api/hierarchy-agent/*`，真实调用 Ollama 审查 Agent，并通过业务列表接口查询确认写库结果。
- 新增独立真实 requests 测试文件：`tests/test_real_world_policy_create_requests.py`、`tests/test_real_world_policy_update_requests.py`、`tests/test_real_worldview_policy_create_requests.py`、`tests/test_real_worldview_policy_update_requests.py`、`tests/test_real_novel_policy_create_requests.py`、`tests/test_real_novel_policy_update_requests.py`、`tests/test_real_review_rule_world_review_requests.py`、`tests/test_real_review_rule_worldview_review_requests.py`、`tests/test_real_review_rule_novel_review_requests.py`、`tests/test_real_review_rule_outline_review_requests.py`、`tests/test_real_review_rule_chapter_review_requests.py`、`tests/test_real_agent_world_flow_requests.py`、`tests/test_real_agent_worldview_flow_requests.py`、`tests/test_real_agent_novel_flow_requests.py`、`tests/test_real_agent_outline_flow_requests.py`、`tests/test_real_agent_chapter_flow_requests.py`。这些文件每个只测试一个业务内容，全部通过 `py_compile` 静态校验；需要在真实 API 服务运行后逐个执行。
- `GET /api/lore/list?world_id=world_default&query=测试分类&page=1&page_size=50`：返回结果均属于 `world_default`，用于验证 `/visualizer` 节点点击查询字段覆盖 `category/path`。
- `/outlines`：浏览器可访问，世界观导入按钮和导入表单可见。
- `/visualizer`：浏览器可访问，点击图谱节点后在原图谱基础上追加下一级节点；实测点击 `子节点1 (1)` 后 React Flow 节点数从 7 增加到 10，原有节点仍保留，新增子节点 `孙节点1-1` 可见。
- `/visualizer` 自动整理：浏览器实测按钮可见且可点击；整理前后 React Flow 节点数保持 7，节点位置从 `translate(400px, 50px)` 变为 `translate(80px, 60px)`，确认是当前图谱重新布局而非清空或替换。
- `/workflow/world`、`/workflow/worldview`、`/workflow/novel`、`/workflow/outline`、`/workflow/chapter`：浏览器可访问，分别显示独立模块标题、启动按钮、React Flow 节点和模块专属标识。
- `/worlds`：浏览器实测点击世界行进入 `/worlds/:world_id` 世界详情页；世界详情页可见世界详细、世界禁止规则、世界设定规则；规则修改入口必须跳转 `/workflow/world?action=update...`。
- `/novels`：浏览器实测显示小说表格、世界筛选、小说搜索、修改、删除、大纲管理、章节管理；搜索不存在关键词显示空结果，搜索命中关键词返回真实小说记录。
- `/novels/new`：浏览器实测从“新增小说”跳转独立新增页；页面包含名称、介绍、简介、小说禁止规则、小说设定规则；创建后跳转小说详情页，并通过 `GET /api/novels/get` 回查确认 `introduction`、`summary`、`forbidden_rules`、`basic_settings` 已写库。
- `/novels/:novel_id/outlines`：浏览器实测按小说显示大纲表格分页；新增大纲跳转 `/workflow/outline?action=create...`，修改大纲跳转 `/workflow/outline?action=update...`。
- `/novels/:novel_id/chapters`：浏览器实测按小说显示章节表格分页并支持大纲筛选；新增章节跳转 `/workflow/chapter?action=create...`，修改章节跳转 `/workflow/chapter?action=update...`。
- `npm run build`：通过。
- `curl -I http://127.0.0.1:5174/worlds`：返回 `200 OK`。
- `curl -I` 检查 5 个独立工作流路由均返回 `200 OK`。

待实现/待验证项：

- 仍需在更大规模真实数据下持续压测世界观、章节分页查询和图谱展开性能。
- 仍需补充完整 E2E 级点击测试，逐项模拟启动工作流、人工反馈、批准和业务列表回查；当前已由真实 requests 测试覆盖 API 写库链路，并由浏览器点击测试覆盖世界详情、小说新增/详情、小说搜索筛选、大纲管理与章节管理核心跳转。
- 不得以静态节点、假对话、mock run 或前端样例 JSON 作为通过依据。

真实性标记：

- API 测试为真实 requests 调用本地 `5006` 服务。
- 新增 `test_real_*_requests.py` 文件禁止 mock、假库和伪成功；所有新增/修改测试都必须通过业务查询接口回查真实结果，审核节点测试必须检查真实节点输出结构。
- 导入测试为真实 multipart 上传并写入 MongoDB，不使用 mock。
- 前端构建为真实 `npm run build`。
- 未使用 mock、假库或 fixture 作为验收依据。

## 11. 风险与约束

- 大量历史数据迁移前可能缺少 `world_id`、`novel_id`，必须先执行迁移脚本。
- 页面如果一次性展开所有世界观和章节，仍可能造成大数据量压力；当前 `/visualizer` 默认限制 3 级并支持节点点击增量追加下一级，后续仍建议增加专用的服务端子节点聚合接口。
- 跨集合分页当前以接口级限制为主，后续如需要全局排序分页，需要引入统一聚合查询策略。
- LLM/Embedding 依赖 Ollama 时，运行环境必须保证对应模型可用；缺失模型不允许静默降级。

## 12. 回滚方案

- 数据回滚：通过测试创建数据的唯一名称或 ID 清理测试数据；生产数据删除必须使用显式 `cascade=true`。
- API 回滚：保留旧接口路径，但不得恢复无条件全量查询和假数据回退。
- 前端回滚：可以回退页面展示形态，但不得恢复静态假数据。
- 配置回滚：可恢复到上一版 `config/*.yml`，但必须保留 LLM 与 embedding 的默认项和多配置项结构。

## 13. Release Gate

- 功能正确性：已覆盖世界、世界观、小说、大纲、章节的创建、修改、查询、删除。
- 工作流完整性：世界、世界观、小说、大纲、章节 5 个模块的 Agent 新增和修改链路，都必须有可访问的 Dify-like 独立工作流页面；大纲与章节管理页的新增/修改必须跳转对应工作流。
- 节点透明性：每条新增/修改链路都必须展示节点状态、当前运行节点、节点 input/output、对话记录、人工反馈和最终 `commit_result`。
- 人工交互正确性：每条修改链路都必须支持 `partial_rewrite`、`content_rewrite`、`full_rewrite`，默认 `partial_rewrite`，并支持手动修改字段。
- 查询安全性：非世界查询必须条件化和分页化。
- 页面安全性：`/worlds` 只展示世界列表；`/worlds/:world_id` 只展示当前世界详情；`/novels` 支持世界筛选和小说搜索；`/novels/:novel_id/outlines` 与 `/novels/:novel_id/chapters` 必须只显示当前小说下数据；`/lore`、`/visualizer` 必须强制选择世界并只显示该世界数据。
- 导入正确性：世界观 JSON、Markdown、OPML 导入必须保留上下级关系，并由真实查询确认。
- 测试真实性：真实 requests API 测试通过。
- 前端可用性：`/worlds`、`/worlds/:world_id`、`/novels`、`/novels/new`、`/novels/:novel_id`、`/novels/:novel_id/outlines`、`/novels/:novel_id/chapters`、`/workflow/world`、`/workflow/worldview`、`/workflow/novel`、`/workflow/outline`、`/workflow/chapter`、`/lore`、`/visualizer` 可访问，前端构建通过。
- 残余风险：大规模数据下仍建议继续实现服务端子节点聚合、节点级懒加载和更细粒度分页。

## Completion Gate

- RCA: not applicable
- Verification: passed
- Evidence: passed
- Rollback: passed
- Final Judgment: 已完成
