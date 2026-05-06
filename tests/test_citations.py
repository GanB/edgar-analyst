"""Tests for the inline chunk_id citation extractor."""

from __future__ import annotations

from edgar_analyst.retrieval.vector_search import RetrievedChunk
from edgar_analyst.synthesis.citations import extract_citations


def _chunk(idx: int, content: str = "body content") -> RetrievedChunk:
    return RetrievedChunk(
        item_id="1A",
        section_title="Risk Factors",
        content=content,
        similarity_score=0.9,
        accession_no="0000000001-24-000001",
        chunk_index=idx,
    )


def test_extract_citations_returns_one_per_unique_marker() -> None:
    chunks = [_chunk(3), _chunk(7), _chunk(12)]
    text = "Apple faces supply concentration [chunk_id=3] and FX risk [chunk_id=7]."

    citations = extract_citations(text, chunks)

    assert [c.chunk_index for c in citations] == [3, 7]
    assert citations[0].item_id == "1A"
    assert citations[0].section_title == "Risk Factors"


def test_extract_citations_dedupes_repeats_in_order() -> None:
    chunks = [_chunk(3), _chunk(7)]
    text = "[chunk_id=7] foo [chunk_id=3] bar [chunk_id=7] baz [chunk_id=3]"

    citations = extract_citations(text, chunks)

    assert [c.chunk_index for c in citations] == [7, 3]


def test_extract_citations_skips_unknown_chunk_ids() -> None:
    chunks = [_chunk(3)]
    text = "Known [chunk_id=3] and bogus [chunk_id=99]."

    citations = extract_citations(text, chunks)

    assert [c.chunk_index for c in citations] == [3]


def test_extract_citations_returns_empty_when_no_markers() -> None:
    chunks = [_chunk(3), _chunk(7)]
    text = "An answer with no citations whatsoever."

    citations = extract_citations(text, chunks)

    assert citations == []


def test_extract_citations_truncates_long_snippet_with_ellipsis() -> None:
    long_body = "x" * 500
    chunks = [_chunk(3, content=long_body)]
    text = "Reference [chunk_id=3]"

    citations = extract_citations(text, chunks)

    assert len(citations) == 1
    assert citations[0].snippet.endswith("...")
    assert len(citations[0].snippet) == 203  # 200 chars + "..."


def test_extract_citations_keeps_short_snippet_intact() -> None:
    chunks = [_chunk(3, content="short body")]
    text = "[chunk_id=3]"

    citations = extract_citations(text, chunks)

    assert citations[0].snippet == "short body"
