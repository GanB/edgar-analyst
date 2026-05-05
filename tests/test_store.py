"""Tests for the Postgres + pgvector store.

Requires the local Docker Compose Postgres to be running. If it's
not reachable on the configured DSN, these tests are skipped so the
rest of the unit suite remains runnable in CI without Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import Connection
from psycopg.sql import SQL, Identifier

from edgar_analyst.ingestion.chunker import Chunk
from edgar_analyst.ingestion.store import (
    EMBEDDING_DIMS,
    connect,
    count_for_ticker,
    init_schema,
    sample_sections_for_ticker,
    upsert_chunks,
)
from edgar_analyst.settings import get_settings

TEST_TABLE = "documents_test"


def _settings_dsn() -> str:
    return get_settings().postgres_dsn_psycopg


def _postgres_reachable() -> bool:
    try:
        with (
            psycopg.connect(_settings_dsn(), connect_timeout=2) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="local Postgres not reachable; start with `docker compose up -d postgres`",
)


@pytest.fixture
def conn() -> Iterator[Connection]:
    connection = connect(_settings_dsn())
    try:
        # Drop any leftover test table from a previous failing run.
        with connection.cursor() as cur:
            cur.execute(
                SQL("DROP TABLE IF EXISTS {table} CASCADE").format(table=Identifier(TEST_TABLE))
            )
        connection.commit()
        init_schema(connection, table=TEST_TABLE)
        yield connection
    finally:
        with connection.cursor() as cur:
            cur.execute(
                SQL("DROP TABLE IF EXISTS {table} CASCADE").format(table=Identifier(TEST_TABLE))
            )
        connection.commit()
        connection.close()


def _make_chunk(idx: int, *, ticker: str = "TEST", item_id: str = "1") -> Chunk:
    return Chunk(
        ticker=ticker,
        cik="0000000001",
        accession_no="0000000001-24-000001",
        form_type="10-K",
        filing_date="2024-01-01",
        item_id=item_id,
        section_title=f"Title for {item_id}",
        chunk_index=idx,
        section_chunk_count=10,
        text=f"chunk body {idx}",
    )


def _make_embedding(seed: float) -> list[float]:
    return [seed] * EMBEDDING_DIMS


def test_init_schema_is_idempotent(conn: Connection) -> None:
    # Fixture already initialized once; calling again should not raise.
    init_schema(conn, table=TEST_TABLE)
    init_schema(conn, table=TEST_TABLE)


def test_upsert_then_count_returns_inserted_rows(conn: Connection) -> None:
    chunks = [_make_chunk(i) for i in range(5)]
    embeddings = [_make_embedding(0.1 * i) for i in range(5)]

    upsert_chunks(conn, chunks, embeddings, table=TEST_TABLE)

    assert count_for_ticker(conn, "TEST", table=TEST_TABLE) == 5


def test_upsert_is_idempotent_on_unique_key(conn: Connection) -> None:
    chunks = [_make_chunk(i) for i in range(5)]
    embeddings = [_make_embedding(0.1) for _ in range(5)]

    upsert_chunks(conn, chunks, embeddings, table=TEST_TABLE)
    first_count = count_for_ticker(conn, "TEST", table=TEST_TABLE)

    # Second run: same chunks, different embeddings. Row count unchanged;
    # rows updated in place.
    new_embeddings = [_make_embedding(0.9) for _ in range(5)]
    upsert_chunks(conn, chunks, new_embeddings, table=TEST_TABLE)
    second_count = count_for_ticker(conn, "TEST", table=TEST_TABLE)

    assert first_count == second_count == 5


def test_upsert_updates_content_on_conflict(conn: Connection) -> None:
    chunk = _make_chunk(0)
    upsert_chunks(conn, [chunk], [_make_embedding(0.1)], table=TEST_TABLE)

    updated = Chunk(
        ticker=chunk.ticker,
        cik=chunk.cik,
        accession_no=chunk.accession_no,
        form_type=chunk.form_type,
        filing_date=chunk.filing_date,
        item_id="1A",
        section_title="Risk Factors",
        chunk_index=chunk.chunk_index,
        section_chunk_count=chunk.section_chunk_count,
        text="updated body",
    )
    upsert_chunks(conn, [updated], [_make_embedding(0.2)], table=TEST_TABLE)

    with conn.cursor() as cur:
        cur.execute(
            SQL(
                "SELECT item_id, section_title, content FROM {table} WHERE chunk_index = %s"
            ).format(table=Identifier(TEST_TABLE)),
            (0,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[0] == "1A"
    assert row[1] == "Risk Factors"
    assert row[2] == "updated body"


def test_sample_sections_returns_distinct_pairs(conn: Connection) -> None:
    chunks = [
        _make_chunk(0, item_id="1"),
        _make_chunk(1, item_id="1"),
        _make_chunk(2, item_id="1A"),
        _make_chunk(3, item_id="7"),
        _make_chunk(4, item_id="7"),
    ]
    embeddings = [_make_embedding(0.1) for _ in range(5)]
    upsert_chunks(conn, chunks, embeddings, table=TEST_TABLE)

    sample = sample_sections_for_ticker(conn, "TEST", n=5, table=TEST_TABLE)
    item_ids = [s[0] for s in sample]

    assert sorted(item_ids) == ["1", "1A", "7"]


def test_count_for_ticker_returns_zero_for_unknown_ticker(conn: Connection) -> None:
    assert count_for_ticker(conn, "UNKNOWN", table=TEST_TABLE) == 0
