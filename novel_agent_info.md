# 万象小说创作引擎 (Novel Agent) 核心协议

本文档汇总了系统的全链路架构规则（PGA 0-4 架构），是所有 Agent 开发与逻辑审计的最高准则。

## 1. 多级 SKILL 映射体系 (Tiered Skill Mapping)

系统通过物理文件隔离实现注意力聚焦，各 Agent 依职责调用对应的 SKILL：

### 1层：框架法典 (Framework Layer)

- **文件**: [PGA_LOGIC.md](file:///Users/harry/Documents/git/novel_agent/.gemini/skills/framework/PGA_LOGIC.md)
- **核心内容**: 0-4 逻辑闭环、节点执行 Schema、自审审计标准、Dispatcher 路由规则。
- **作用**: 定义"如何做 Agent"，属于底层操作系统级指令。

### 2层：剧情锚点 (Lore/Anchor Layer)

- **文件**: [ANCHORS.md](file:///Users/harry/Documents/git/novel_agent/.gemini/skills/lore/ANCHORS.md)
- **核心内容**: 核心冲突节点、人物生死、世界观不可逆转折。
- **作用**: 定义"故事的底线"，属于叙事宪法级指令，阻止"吃书"。

### 3层：执行目录 (Catalog Layer)

- **文件**: [ACTIVE_WINDOW.md](file:///Users/harry/Documents/git/novel_agent/.gemini/skills/catalog/ACTIVE_WINDOW.md)
- **核心内容**: 当前写作任务（Chapter N）及其前后 5 章的精细大纲。
- **作用**: 定义"施工蓝图"，提供高精细节，防止逻辑漂移。
- **补充**: 历史章节数据通过物理切片存放在 `catalog/ARCHIVE/` 中，由 [MASTER_INDEX.md](file:///Users/harry/Documents/git/novel_agent/.gemini/skills/catalog/MASTER_INDEX.md) 统一索引。

---

## 2. 动态维护流程：Lore-to-Skill
1. **大纲确立**: 策划 Agent 生成章节目录 -> 用户批准。
2. **物理切片**: 触发 `lore_skill_converter.py`。
3. **SKILL 刷新**: 自动更新 `ACTIVE_WINDOW` 与 `ANCHORS`，确保正文创作 Agent 持有最新"法令"。

---

## 3. 核心架构：PGA 0-4 Logic
本系统严格遵循 0-4 逻辑图进行分步构建，实现"框架管人、锚点管书、目录管章"：

任何 Agent 的节点逻辑必须映射至以下五阶段：


### 0 - 定义与状态 (Definitions & State)

- **输入定义**: 明确 `query` (原始需求) 与 `context` (RAG 检索内容)。
- **状态维护**: 必须使用 `TypedDict` 定义持久化状态，严禁在节点间传递未定义的裸数据。

### 1 - 策划与分析 (Planning/Analysis)

- **逻辑栅栏 (Gate)**: 在生成前，必须检查是否违反 [最高禁令](#4-最高禁令)。
- **意图对齐**: 识别用户请求的子类（如世界观中的 Faction vs. Geography）。

### 2 - 生成与提案 (Generation/Drafting)

- **模板驱动**: 必须使用预定义的 JSON 模板进行输出。
- **关联引用**: 正文生成必须引用大纲 ID，大纲生成必须引用世界观 Context。

### 3 - 审计与反馈 (Audit/Feedback)

- **逻辑闭环**: `Reviewer` 节点必须检查逻辑一致性（如：能量是否守恒、人物是否瞬移）。
- **人机协同**: 关键节点必须通过 `interrupt` 暂停，等待人类确认。

### 4 - 确立与持久化 (Canon/Persistence)

- **双库并行**: 最终结果必须同时写入 `worldview_db.json` (结构化数据) 与 ChromaDB (向量索引)。

---

## 2. 路由分发逻辑 (Routing Logic)

### 意图判定

- **Worldview**: 关键词涉及“设定、背景、势力、地理、技术、历史”。
- **Outline**: 关键词涉及“大纲、策划、剧情、冲突、节拍、章节概要”。
- **Writing**: 关键词涉及“正文、第一章、编写、描写、润色”。

### 异常处理

- 对于无法识别或超出上述范围的请求，系统必须返回：“**该功能不支持，超出系统使用范围。**”

---

## 3. 长内容处理策略 (Handling Long Content)

系统通过以下两种核心机制应对 LLM 的上下文限制与生成遗忘问题：

### A. 叙事分块 (Narrative Chunking)

- **场次拆解**: 正文 Agent 严禁一次性生成数千字的完整章节。必须先通过 `outline_to_scenes` 将其拆解为“原子场次（Atomic Scenes）”。
- **迭代生成**: 每次仅针对一个场次进行描写（1000 字左右），通过 `active_scene_index` 在 Graph 中循环迭代。
- **状态快照**: 每个场次生成后，会更新当前的人物理智、环境状态快照，作为下一场的输入。

### B. RAG 语境压测 (Context Truncation)

- **8 块限制**: `get_unified_context` 在合并 MongoDB 与 ChromaDB 检索结果时，仅保留相关度最高的 **Top 8** 唯一内容块。
- **精化检索**: 检索时使用当前场次的 Title 与 Description，而非全量大纲，确保语境聚焦。

---

## 4. 最高禁令 (Highest Prohibitions)

> [!CAUTION]
> 1. 严禁时间旅行: 物理层面上禁止任何形式的因果倒置。
> 2. 严禁全知全能: 所有智慧体必须受限于光速与熵增逻辑。
> 3. 严禁无限能源: 任何技术必须有明确的负熵来源。
> 4. 严禁现实微调: 禁止跨越底层物理常数的突发修改。
