"""Tests for the click CLI."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from edgar_analyst import __version__
from edgar_analyst.cli import main
from edgar_analyst.retrieval.vector_search import RetrievedChunk
from edgar_analyst.synthesis.state import Citation, QueryState


def test_hello_prints_banner() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["hello"])

    assert result.exit_code == 0
    assert f"edgar-analyst v{__version__}" in result.output
    assert "RAG over SEC EDGAR filings" in result.output


def _stub_state(*, error: str | None = None) -> QueryState:
    if error:
        return {
            "ticker": "AAPL",
            "question": "Q",
            "retrieved_chunks": [],
            "error": error,
        }
    chunks = [
        RetrievedChunk(
            item_id="1A",
            section_title="Risk Factors",
            content="Apple operates globally and faces foreign currency risk.",
            similarity_score=0.9,
            accession_no="0000320193-24-000123",
            chunk_index=2,
        )
    ]
    return {
        "ticker": "AAPL",
        "question": "Q",
        "retrieved_chunks": chunks,
        "synthesis": "Apple faces FX risk [chunk_id=2].",
        "citations": [
            Citation(
                item_id="1A",
                section_title="Risk Factors",
                accession_no="0000320193-24-000123",
                chunk_index=2,
                snippet="Apple operates globally and faces foreign currency risk.",
            )
        ],
        "error": None,
    }


def test_ask_prints_synthesis_and_sources() -> None:
    runner = CliRunner()
    with patch("edgar_analyst.cli.run_query", return_value=_stub_state()):
        result = runner.invoke(main, ["ask", "--ticker", "AAPL", "What are Apple's risk factors?"])

    assert result.exit_code == 0
    assert "Apple faces FX risk [chunk_id=2]." in result.output
    assert "Sources:" in result.output
    assert "Item 1A" in result.output
    assert "Risk Factors" in result.output
    assert "accession=0000320193-24-000123" in result.output
    assert "snippet:" in result.output


def test_ask_exits_nonzero_on_error_state() -> None:
    runner = CliRunner()
    with patch(
        "edgar_analyst.cli.run_query",
        return_value=_stub_state(error="no chunks found for ticker AAPL"),
    ):
        result = runner.invoke(main, ["ask", "--ticker", "AAPL", "What are Apple's risk factors?"])

    assert result.exit_code == 1
    assert "no chunks found" in result.output
