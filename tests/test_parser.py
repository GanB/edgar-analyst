"""Tests for the filing parser."""

from __future__ import annotations

from pathlib import Path

from edgar_analyst.ingestion.parser import Section, parse_filing

FIXTURES = Path(__file__).parent / "fixtures" / "filings"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_parse_filing_detects_named_sections() -> None:
    sections = parse_filing(_read("sample_10k.html"))

    item_ids = [s.item_id for s in sections]
    assert "1" in item_ids
    assert "1A" in item_ids
    assert "7" in item_ids
    assert "8" in item_ids


def test_parse_filing_skips_toc_entries() -> None:
    sections = parse_filing(_read("sample_10k.html"))

    # The TOC lists Items 1, 1A, 7, 8 as one-liners. The parser must
    # not return zero-body or near-zero-body sections for those entries.
    for section in sections:
        assert len(section.text) >= 500, (
            f"Section {section.item_id!r} has body {len(section.text)} chars; "
            "TOC entries should have been folded out."
        )


def test_parse_filing_section_titles_are_normalized() -> None:
    sections = {s.item_id: s for s in parse_filing(_read("sample_10k.html"))}

    assert sections["1A"].title.lower().startswith("risk factors")
    assert "management" in sections["7"].title.lower()


def test_parse_filing_falls_back_to_whole_when_no_headers() -> None:
    sections = parse_filing(_read("no_sections.html"))

    assert len(sections) == 1
    assert sections[0].item_id == "whole"
    assert "no recognizable section headers" in sections[0].text


def test_parse_filing_strips_script_and_style() -> None:
    html = """
    <html><body>
    <style>p { color: red; }</style>
    <script>var noise = "should not appear";</script>
    <p>Item 1. Business</p>
    <p>Visible body content for the test fixture.</p>
    </body></html>
    """
    sections = parse_filing(html)
    joined = " ".join(s.text for s in sections)

    assert "noise" not in joined
    assert "should not appear" not in joined


def test_section_dataclass_is_frozen() -> None:
    import dataclasses

    import pytest

    s = Section(item_id="1", title="Business", text="body")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "mutated"  # type: ignore[misc]
