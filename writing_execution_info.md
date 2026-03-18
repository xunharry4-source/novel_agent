# Writing Execution Agent: 0-4 Architecture (PGA PROTOCOL)

## 0. Definitions & Terms
- **Atomic Scene (原子场次)**: The smallest unit of narrative execution. A scene must have a clear "Entry -> Conflict -> Outcome" loop.
- **Context Window (动态语境)**: The specific set of character/world data retrieved for a single scene, ensuring high retrieval accuracy.
- **Style Invariant (风格守恒)**: The requirement that prose MUST strictly adhere to the `writing_style` defined in the outline.

## 1. Scene Decomposition Logic (Atomicity)
- The **Scene Planner** takes the "Plot Beats" from the outline and splits them into discrete scenes.
- Each scene is a "Hook -> Interaction -> Result" cycle.
- **Prohibition**: No scene can span multiple plot beats without explicit architect approval. 

## 2. Dynamic Context Loading (RAG)
- Before writing a scene, the **Context Loader** identifies key entities (Characters/Locations) in the scene plan.
- It performs a focused search in ChromaDB to "anchor" the AI's memory.
- **Rule**: If a character is "broke" in the worldview, they cannot possess luxury items in the prose.

## 3. Style Enforcement & Drafting
- The **Scribe** uses the `writing_style` as a system instruction.
- Output is generated in a "Drafting Mode" first, optimized for flow and sensory details.
- **Invariant**: The tone MUST match the `tone` field in `meta_info`.

## 4. Logic Audit & Consistency (The Great Wall)
- The **Auditor** compares the draft against the retrieved context data.
- **Priority**: Worldview Rules > Character Motivation > Narrative Flow.
- If a reversal occurs without a documented cause in the worldview, the draft is REJECTED and sent back to the Scribe with a detailed error log.

---
*Independence*: The Writing Execution Agent does not modify the Worldview. It is a consumer of the Worldview and an executor of the Outline.
