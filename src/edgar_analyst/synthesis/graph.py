"""No-op LangGraph state machine for the synthesis spine.

Future iterations replace the single ``echo`` node with the real
retrieve / rerank / synthesize / cite flow. The shape of ``GraphState``
is intentionally narrow so that downstream extensions can add fields
(retrieved chunks, citations, token usage) without churn here.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class GraphState(TypedDict, total=False):
    query: str
    response: str


def _echo(state: GraphState) -> GraphState:
    return {"query": state.get("query", ""), "response": state.get("query", "")}


def build_graph() -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    builder = StateGraph(GraphState)
    builder.add_node("echo", _echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile()


graph = build_graph()
