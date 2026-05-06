"""Voyage embedder for the query side of the pipeline.

Wraps ``langchain_voyageai.VoyageAIEmbeddings`` with the same model
(``voyage-3``, 1024-dim) used by the ingestion path so that the
query and document vectors live in the same space.
"""

from __future__ import annotations

import logging

from langchain_voyageai import VoyageAIEmbeddings

from edgar_analyst.ingestion.embedder import DEFAULT_DIMENSIONS, DEFAULT_MODEL

logger = logging.getLogger(__name__)


def _build_client(api_key: str, model: str = DEFAULT_MODEL) -> VoyageAIEmbeddings:
    return VoyageAIEmbeddings(voyage_api_key=api_key, model=model)


async def embed_query(text: str, *, api_key: str, model: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single query string and return a 1024-dim float vector."""
    if not text.strip():
        raise ValueError("query text must be non-empty")
    client = _build_client(api_key, model)
    vector = await client.aembed_query(text)
    if len(vector) != DEFAULT_DIMENSIONS:
        raise RuntimeError(f"Voyage returned {len(vector)} dims; expected {DEFAULT_DIMENSIONS}")
    return list(vector)
