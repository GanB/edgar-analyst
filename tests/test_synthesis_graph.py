"""Tests for the no-op synthesis graph."""

from __future__ import annotations

from edgar_analyst.synthesis import build_graph


def test_graph_echoes_query() -> None:
    compiled = build_graph()
    result = compiled.invoke({"query": "What did Apple report in 10-K Item 1A?"})

    assert result["response"] == "What did Apple report in 10-K Item 1A?"
    assert result["query"] == "What did Apple report in 10-K Item 1A?"
