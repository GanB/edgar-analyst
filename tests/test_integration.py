"""End-to-end integration test against live SEC and Voyage.

Skipped by default. Run locally with:

    INTEGRATION=1 uv run pytest -m integration

Requires:
    - Local Postgres running (``docker compose up -d postgres``)
    - ``VOYAGE_API_KEY`` and ``SEC_EDGAR_USER_AGENT`` set in ``.env``
"""

from __future__ import annotations

import os

import pytest

from edgar_analyst.ingestion.pipeline import ingest_filing
from edgar_analyst.ingestion.store import (
    connect,
    count_for_ticker,
    init_schema,
)
from edgar_analyst.settings import get_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("INTEGRATION"),
        reason="set INTEGRATION=1 to run live ingestion test",
    ),
]


async def test_ingest_nke_10k_end_to_end() -> None:
    settings = get_settings()
    assert settings.voyage_api_key, "VOYAGE_API_KEY must be set"

    conn = connect(settings.postgres_dsn_psycopg)
    try:
        init_schema(conn)
    finally:
        conn.close()

    result = await ingest_filing("NKE", "10-K", settings)
    assert result.rows_upserted >= 30, f"Expected at least 30 rows; got {result.rows_upserted}"

    conn = connect(settings.postgres_dsn_psycopg)
    try:
        count = count_for_ticker(conn, "NKE")
    finally:
        conn.close()
    assert count >= 30


async def test_query_aapl_risks_end_to_end() -> None:
    """Live query path: Voyage embed + pgvector + Anthropic synthesis."""
    settings = get_settings()
    assert settings.voyage_api_key, "VOYAGE_API_KEY must be set"
    assert settings.anthropic_api_key, "ANTHROPIC_API_KEY must be set"

    from edgar_analyst.synthesis import run_query

    state = await run_query("AAPL", "What are Apple's biggest risk factors?")

    assert not state.get("error"), state.get("error")
    assert state.get("synthesis"), "synthesis should be non-empty"
    assert len(state.get("citations") or []) >= 1, "expected at least one citation"
