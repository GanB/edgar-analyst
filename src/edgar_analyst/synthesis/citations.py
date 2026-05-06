"""Inline ``[chunk_id=N]`` citation extraction.

Parses the synthesis text for chunk-id markers, looks each one up
against the retrieved set, and returns deduped ``Citation`` records
in the order they appear. Unknown ids are logged and skipped so a
mild model hallucination does not crash the response path.
"""

from __future__ import annotations

import logging
import re

from edgar_analyst.retrieval.vector_search import RetrievedChunk
from edgar_analyst.synthesis.state import Citation

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[chunk_id=(\d+)\]")
SNIPPET_LIMIT = 200


def _make_snippet(content: str, *, limit: int = SNIPPET_LIMIT) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "..."


def extract_citations(synthesis_text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Return ordered, deduplicated citations referenced inline in the text."""
    by_index: dict[int, RetrievedChunk] = {c.chunk_index: c for c in chunks}
    seen: set[int] = set()
    out: list[Citation] = []

    for match in CITATION_PATTERN.finditer(synthesis_text):
        chunk_index = int(match.group(1))
        if chunk_index in seen:
            continue
        chunk = by_index.get(chunk_index)
        if chunk is None:
            logger.warning("synthesis cited chunk_id=%d not present in retrieved set", chunk_index)
            continue
        seen.add(chunk_index)
        out.append(
            Citation(
                item_id=chunk.item_id,
                section_title=chunk.section_title,
                accession_no=chunk.accession_no,
                chunk_index=chunk.chunk_index,
                snippet=_make_snippet(chunk.content),
            )
        )
    return out
