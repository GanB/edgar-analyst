"""Tests for the click CLI."""

from __future__ import annotations

from click.testing import CliRunner

from edgar_analyst import __version__
from edgar_analyst.cli import main


def test_hello_prints_banner() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["hello"])

    assert result.exit_code == 0
    assert f"edgar-analyst v{__version__}" in result.output
    assert "RAG over SEC EDGAR filings" in result.output
