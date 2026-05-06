"""Tests for the LangGraph synthesis pipeline.

All external dependencies (Voyage embedder, Postgres connection,
Anthropic chat model) are patched. The graph itself is exercised
end-to-end so node wiring, conditional routing, and citation
extraction are all covered.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from edgar_analyst.retrieval.vector_search import RetrievedChunk
from edgar_analyst.synthesis import run_query
from edgar_analyst.synthesis.state import Citation


def _chunk(idx: int, content: str = "body") -> RetrievedChunk:
    return RetrievedChunk(
        item_id="1A",
        section_title="Risk Factors",
        content=content,
        similarity_score=0.9,
        accession_no="0000000001-24-000001",
        chunk_index=idx,
    )


@pytest.fixture
def patched_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage")


async def test_graph_runs_end_to_end_with_citations(
    patched_settings: None,
) -> None:
    chunks = [
        _chunk(0, content="Supply concentration in Asia."),
        _chunk(1, content="Foreign currency exposure."),
    ]
    chat_response = AIMessage(
        content=(
            "Apple faces supply concentration [chunk_id=0] and currency exposure [chunk_id=1]."
        )
    )

    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(return_value=chat_response)

    with (
        patch(
            "edgar_analyst.synthesis.graph.voyage_embed_query",
            AsyncMock(return_value=[0.1] * 1024),
        ),
        patch("edgar_analyst.synthesis.graph.connect", return_value=MagicMock()),
        patch("edgar_analyst.synthesis.graph.search", return_value=chunks),
        patch("edgar_analyst.synthesis.graph._build_chat_model", return_value=fake_chat),
    ):
        result = await run_query("AAPL", "What are Apple's risks?")

    assert result["synthesis"].startswith("Apple faces supply concentration")
    assert len(result["retrieved_chunks"]) == 2
    citations = result["citations"]
    assert [c.chunk_index for c in citations] == [0, 1]
    assert all(isinstance(c, Citation) for c in citations)
    assert not result.get("error")


async def test_graph_short_circuits_on_empty_retrieval(
    patched_settings: None,
) -> None:
    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(side_effect=AssertionError("must not be called"))

    with (
        patch(
            "edgar_analyst.synthesis.graph.voyage_embed_query",
            AsyncMock(return_value=[0.1] * 1024),
        ),
        patch("edgar_analyst.synthesis.graph.connect", return_value=MagicMock()),
        patch("edgar_analyst.synthesis.graph.search", return_value=[]),
        patch(
            "edgar_analyst.synthesis.graph._build_chat_model",
            return_value=fake_chat,
        ),
    ):
        result = await run_query("AAPL", "What are Apple's risks?")

    assert result.get("error", "").startswith("no chunks found")
    assert result.get("retrieved_chunks") == []
    assert "synthesis" not in result or not result.get("synthesis")
    fake_chat.ainvoke.assert_not_called()


async def test_graph_handles_list_message_content(
    patched_settings: None,
) -> None:
    """ChatAnthropic sometimes returns content as a list of dicts."""
    chunks = [_chunk(5)]
    list_content: list[Any] = [
        {"type": "text", "text": "Answer text [chunk_id=5]."},
    ]
    chat_response = AIMessage(content=list_content)

    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(return_value=chat_response)

    with (
        patch(
            "edgar_analyst.synthesis.graph.voyage_embed_query",
            AsyncMock(return_value=[0.1] * 1024),
        ),
        patch("edgar_analyst.synthesis.graph.connect", return_value=MagicMock()),
        patch("edgar_analyst.synthesis.graph.search", return_value=chunks),
        patch("edgar_analyst.synthesis.graph._build_chat_model", return_value=fake_chat),
    ):
        result = await run_query("AAPL", "anything")

    assert "Answer text" in result["synthesis"]
    assert [c.chunk_index for c in result["citations"]] == [5]
