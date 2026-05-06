"""Tests for the FastAPI app: health endpoint and SSE query endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from edgar_analyst import __version__
from edgar_analyst.api.main import create_app
from edgar_analyst.retrieval.vector_search import RetrievedChunk


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.content = text


def _retrieved() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            item_id="1A",
            section_title="Risk Factors",
            content="Apple faces concentration risk in Asia.",
            similarity_score=0.91,
            accession_no="0000320193-24-000123",
            chunk_index=0,
        )
    ]


async def _fake_astream_events(
    _state: dict[str, Any], **_kwargs: Any
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "event": "on_chain_end",
        "name": "retrieve",
        "data": {"output": {"retrieved_chunks": _retrieved()}},
    }
    yield {
        "event": "on_chat_model_stream",
        "name": "ChatAnthropic",
        "data": {"chunk": _FakeChunk("Apple faces concentration risk")},
    }
    yield {
        "event": "on_chat_model_stream",
        "name": "ChatAnthropic",
        "data": {"chunk": _FakeChunk(" [chunk_id=0].")},
    }


def test_query_endpoint_streams_token_citations_done_events() -> None:
    client = TestClient(create_app())
    with patch("edgar_analyst.api.routes.graph") as fake_graph:
        fake_graph.astream_events = _fake_astream_events
        with client.stream(
            "POST",
            "/v1/query",
            json={"ticker": "AAPL", "question": "What are Apple's risks?"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = b"".join(response.iter_bytes()).decode()

    assert "event: token" in body
    assert '"token": "Apple faces concentration risk"' in body
    assert "event: citations" in body
    assert '"chunk_index": 0' in body
    assert "event: done" in body


async def _fake_astream_events_empty(
    _state: dict[str, Any], **_kwargs: Any
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "event": "on_chain_end",
        "name": "retrieve",
        "data": {"output": {"retrieved_chunks": [], "error": "no chunks found for ticker AAPL"}},
    }


def test_query_endpoint_emits_error_event_on_empty_retrieval() -> None:
    client = TestClient(create_app())
    with patch("edgar_analyst.api.routes.graph") as fake_graph:
        fake_graph.astream_events = _fake_astream_events_empty
        with client.stream(
            "POST",
            "/v1/query",
            json={"ticker": "AAPL", "question": "anything"},
        ) as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes()).decode()

    assert "event: error" in body
    assert "no chunks found" in body
    assert "event: done" in body
    assert "event: citations" not in body
