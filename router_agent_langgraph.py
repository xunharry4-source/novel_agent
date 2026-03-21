"""
Dispatcher Agent (中转 Agent) - PGA 小说创作引擎路由层

本模块负责对用户输入的原始请求进行语义分类，并将其分发（Dispatch）至最合适的子策略 Agent。
设计思路:
1. 意图分类 (Intent Classification): 利用 LLM 对 Query 进行多分类识别（世界观/大纲/正文）。
2. 模糊处理: 当意图不明确时，标记为 unknown 并请求用户澄清，避免误触。
3. 架构一致性: 遵循 PGA 0-4 框架，确保与子 Agent 的状态流转逻辑对齐。
"""
import os
import json
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import shared utilities
from lore_utils import get_llm, parse_json_safely

# ==========================================
# 0. State Definition
# ==========================================
class RouterState(TypedDict):
    """
    路由 Agent 的状态机上下文。
    """
    query: str                                     # 用户原始输入
    intent: Literal["worldview", "outline", "writing", "unknown"] # 识别出的意图
    confidence: float                              # 分类置信度
    reasoning: str                                 # 分类逻辑简述
    status_message: str                            # 执行进度描述

# ==========================================
# Nodes Implementation
# ==========================================

def intent_classifier(state: RouterState):
    """
    意图识别节点 (Classification Node)
    """
    query = state.get('query', '')
    llm = get_llm(json_mode=True)
    
    prompt = f"""你是一个小说创作引擎的中转路由（Dispatcher）。
你的任务是分析用户的 Query，并将其分类到以下三个 Agent 之一：
1. worldview: 涉及世界观设定、势力、地理、武装、历史、种族、基础规则等背景资料的创建或修改。
2. outline: 涉及小说整体大纲、章节概要、剧情脉络、节拍设计等策划性工作。
3. writing: 涉及具体章节正文内容的描写、场次拆解、正文逻辑审计或修辞润色。

如果意图不明确，请返回 unknown。

用户 Query: "{query}"

请直接返回 JSON 格式:
{{
    "intent": "worldview" | "outline" | "writing" | "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "说明为什么这么分类"
}}
"""
    response = llm.invoke(prompt)
    result = parse_json_safely(response.content)
    
    intent = result.get("intent", "unknown")
    status_msg = f"已识别意图: {intent}"
    if intent == 'unknown':
        status_msg = "该功能不支持，超出系统使用范围。"
        
    return {
        "intent": intent,
        "confidence": result.get("confidence", 0.0),
        "reasoning": result.get("reasoning", ""),
        "status_message": status_msg
    }

# ==========================================
# 1. Routing Logic (PGA 0-4 Schema)
# ==========================================

# Define Graph
workflow = StateGraph(RouterState)

# Add Nodes
workflow.add_node("classifier", intent_classifier)

# Build Graph
workflow.add_edge(START, "classifier")
workflow.add_edge("classifier", END)

# Compile
app = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    # Test
    inputs = {"query": "帮我写一下这个故事的第一章正文"}
    config = {"configurable": {"thread_id": "test_router"}}
    for output in app.stream(inputs, config):
        print(output)
