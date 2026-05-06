"""HTTP route definitions.

The query route uses LangGraph's ``astream_events`` to surface
per-token chat output as Server-Sent Events. Once the synthesis
node completes we emit the citation list and a terminal ``done``
event so clients know the stream is fully drained.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from edgar_analyst.synthesis.citations import extract_citations
from edgar_analyst.synthesis.graph import graph
from edgar_analyst.synthesis.state import Citation

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. AAPL.")
    question: str = Field(..., min_length=1, description="Question about the filing.")


class CitationModel(BaseModel):
    item_id: str
    section_title: str
    accession_no: str
    chunk_index: int
    snippet: str


class QueryResponse(BaseModel):
    synthesis: str
    citations: list[CitationModel]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _coerce_chunk_text(content: Any) -> str:
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
    return ""


async def _stream_query(req: QueryRequest) -> AsyncIterator[str]:
    state_input = {"ticker": req.ticker.upper(), "question": req.question}
    accumulated: list[str] = []
    retrieved: list[Any] = []
    error: str | None = None

    async for event in graph.astream_events(state_input, version="v2"):
        kind = event.get("event")
        name = event.get("name")
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            text = _coerce_chunk_text(getattr(chunk, "content", "")) if chunk else ""
            if text:
                accumulated.append(text)
                yield _sse("token", {"token": text})
        elif kind == "on_chain_end" and name == "retrieve":
            output = event.get("data", {}).get("output") or {}
            retrieved = list(output.get("retrieved_chunks") or [])
            if output.get("error"):
                error = str(output["error"])
        elif kind == "on_chain_end" and name == "LangGraph":
            output = event.get("data", {}).get("output") or {}
            if not retrieved:
                retrieved = list(output.get("retrieved_chunks") or [])
            if not error and output.get("error"):
                error = str(output["error"])

    if error:
        yield _sse("error", {"error": error})
        yield _sse("done", {})
        return

    full_text = "".join(accumulated)
    citations: list[Citation] = extract_citations(full_text, retrieved)
    yield _sse(
        "citations",
        {"citations": [asdict(c) for c in citations]},
    )
    yield _sse("done", {})


@router.post("/v1/query")
async def post_query(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(_stream_query(req), media_type="text/event-stream")
