"""Three-node LangGraph state machine: embed_query -> retrieve -> synthesize.

The graph is the only place that wires retrieval to the chat model.
Each node is async and returns a state slice; the conditional edge
after ``retrieve`` short-circuits to END when no chunks come back so
synthesis is never called with an empty context.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from edgar_analyst.ingestion.store import connect
from edgar_analyst.retrieval.embedder import embed_query as voyage_embed_query
from edgar_analyst.retrieval.vector_search import search
from edgar_analyst.settings import get_settings
from edgar_analyst.synthesis.citations import extract_citations
from edgar_analyst.synthesis.prompts import format_chunks, synthesis_prompt
from edgar_analyst.synthesis.state import QueryState

logger = logging.getLogger(__name__)

CHAT_MODEL = "claude-sonnet-4-5-20250929"


def _build_chat_model() -> ChatAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    return ChatAnthropic(
        model=CHAT_MODEL,
        api_key=settings.anthropic_api_key,
        timeout=60,
        stop=None,
    )


async def embed_query_node(state: QueryState) -> QueryState:
    settings = get_settings()
    if not settings.voyage_api_key:
        return {"error": "VOYAGE_API_KEY is not configured"}
    question = state.get("question", "")
    vector = await voyage_embed_query(question, api_key=settings.voyage_api_key)
    return {"query_embedding": vector}


async def retrieve_node(state: QueryState) -> QueryState:
    settings = get_settings()
    embedding = state.get("query_embedding")
    if embedding is None:
        return {"error": "missing query embedding"}
    ticker = state.get("ticker", "").upper()
    conn = connect(settings.postgres_dsn_psycopg)
    try:
        chunks = search(conn, embedding, ticker, k=settings.retrieval_top_k)
    finally:
        conn.close()
    if not chunks:
        return {
            "retrieved_chunks": [],
            "error": f"no chunks found for ticker {ticker}",
        }
    return {"retrieved_chunks": chunks}


async def synthesize_node(state: QueryState) -> QueryState:
    chunks = state.get("retrieved_chunks") or []
    question = state.get("question", "")
    ticker = state.get("ticker", "")

    chat = _build_chat_model()
    prompt_value = synthesis_prompt.invoke(
        {
            "ticker": ticker,
            "question": question,
            "context": format_chunks(chunks),
        }
    )
    response = await chat.ainvoke(prompt_value)
    text = _coerce_text(response.content)
    citations = extract_citations(text, chunks)
    return {"synthesis": text, "citations": citations}


def _coerce_text(content: Any) -> str:
    """Flatten ChatAnthropic message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict):
                text_value = piece.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return str(content)


def _route_after_retrieve(state: QueryState) -> str:
    if state.get("error"):
        return END
    if not state.get("retrieved_chunks"):
        return END
    return "synthesize"


def build_graph() -> CompiledStateGraph[QueryState, None, QueryState, QueryState]:
    builder = StateGraph(QueryState)
    builder.add_node("embed_query", embed_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "embed_query")
    builder.add_edge("embed_query", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"synthesize": "synthesize", END: END},
    )
    builder.add_edge("synthesize", END)
    return builder.compile()


graph = build_graph()


async def run_query(ticker: str, question: str) -> QueryState:
    """Run the full graph end-to-end and return the final state."""
    result = await graph.ainvoke(
        {
            "ticker": ticker.upper(),
            "question": question,
        }
    )
    return cast(QueryState, result)
