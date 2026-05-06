"""Typed state for the LangGraph synthesis flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from edgar_analyst.retrieval.vector_search import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    item_id: str
    section_title: str
    accession_no: str
    chunk_index: int
    snippet: str


class QueryState(TypedDict, total=False):
    ticker: str
    question: str
    query_embedding: list[float] | None
    retrieved_chunks: list[RetrievedChunk]
    synthesis: str
    citations: list[Citation]
    error: str | None
