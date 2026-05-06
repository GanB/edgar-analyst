"""Tests for the token-aware chunker."""

from __future__ import annotations

from edgar_analyst.ingestion.chunker import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    chunk_sections,
    token_count,
)
from edgar_analyst.ingestion.parser import Section

# Roughly tokens-per-word: 0.75 in English; pad with extra words.
WORD = "compliance "


def _section(item_id: str, title: str, words: int) -> Section:
    text = (WORD * words).strip()
    return Section(item_id=item_id, title=title, text=text)


def test_chunker_emits_chunks_below_target_token_size() -> None:
    section = _section("1A", "Risk Factors", words=2000)
    chunks = chunk_sections(
        [section],
        ticker="AAPL",
        cik="0000320193",
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
    )

    assert len(chunks) > 1
    # Allow a small overshoot because RecursiveCharacterTextSplitter rounds at
    # separator boundaries, but no chunk should be wildly larger.
    for chunk in chunks:
        assert token_count(chunk.text) <= DEFAULT_CHUNK_TOKENS * 1.1


def test_chunker_metadata_propagates_to_every_chunk() -> None:
    sections = [
        _section("1", "Business", words=1500),
        _section("1A", "Risk Factors", words=1500),
    ]
    chunks = chunk_sections(
        sections,
        ticker="AAPL",
        cik="0000320193",
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
    )

    for chunk in chunks:
        assert chunk.ticker == "AAPL"
        assert chunk.cik == "0000320193"
        assert chunk.accession_no == "0000320193-24-000123"
        assert chunk.form_type == "10-K"
        assert chunk.filing_date == "2024-11-01"
        assert chunk.item_id in {"1", "1A"}


def test_chunker_chunk_index_is_globally_unique_within_filing() -> None:
    sections = [
        _section("1", "Business", words=1500),
        _section("1A", "Risk Factors", words=1500),
        _section("7", "MD&A", words=1500),
    ]
    chunks = chunk_sections(
        sections,
        ticker="AAPL",
        cik="0000320193",
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
    )

    indexes = [c.chunk_index for c in chunks]
    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)
    assert indexes[0] == 0


def test_chunker_section_chunk_count_matches_per_section_count() -> None:
    sections = [
        _section("1", "Business", words=1500),
        _section("1A", "Risk Factors", words=3000),
    ]
    chunks = chunk_sections(
        sections,
        ticker="AAPL",
        cik="0000320193",
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
    )

    counts_per_section: dict[str, int] = {}
    declared_per_section: dict[str, int] = {}
    for c in chunks:
        counts_per_section[c.item_id] = counts_per_section.get(c.item_id, 0) + 1
        declared_per_section[c.item_id] = c.section_chunk_count

    assert counts_per_section == declared_per_section


def test_chunker_overlap_is_configured() -> None:
    # The default overlap is non-zero; verify by checking the splitter config
    # observable through token_count behavior on a long section: more chunks
    # than the strict no-overlap minimum of ceil(total / chunk).
    section = _section("1A", "Risk Factors", words=3000)
    chunks_with_overlap = chunk_sections(
        [section],
        ticker="AAPL",
        cik="000",
        accession_no="acc",
        form_type="10-K",
        filing_date="2024-01-01",
    )
    chunks_no_overlap = chunk_sections(
        [section],
        ticker="AAPL",
        cik="000",
        accession_no="acc",
        form_type="10-K",
        filing_date="2024-01-01",
        overlap_tokens=0,
    )
    # Overlap means the same content is sliced into more pieces.
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)
    # And the default overlap is the documented value.
    assert DEFAULT_OVERLAP_TOKENS == 50
