"""Tests for the retrieval embedder and vector search.

The embedder test mocks ``VoyageAIEmbeddings`` so no live API call
is made. The vector search test inserts a small known dataset into
a temporary table on the local pgvector and asserts ordering, so it
is gated on Postgres reachability the same way ``test_store.py`` is.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from psycopg import Connection
from psycopg.sql import SQL, Identifier

from edgar_analyst.ingestion.chunker import Chunk
from edgar_analyst.ingestion.store import EMBEDDING_DIMS, connect, init_schema, upsert_chunks
from edgar_analyst.retrieval.embedder import embed_query
from edgar_analyst.retrieval.vector_search import RetrievedChunk, search
from edgar_analyst.settings import get_settings

TEST_TABLE = "documents_retrieval_test"


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


pg_required = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="local Postgres not reachable; start with `docker compose up -d postgres`",
)


async def test_embed_query_returns_1024_dim_vector() -> None:
    fake_vector = [0.01] * 1024
    with patch("edgar_analyst.retrieval.embedder.VoyageAIEmbeddings") as factory:
        instance = factory.return_value
        instance.aembed_query = AsyncMock(return_value=fake_vector)

        result = await embed_query("What are Apple's risk factors?", api_key="fake-key")

    assert len(result) == 1024
    assert result == fake_vector
    factory.assert_called_once()
    instance.aembed_query.assert_awaited_once_with("What are Apple's risk factors?")


async def test_embed_query_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await embed_query("   ", api_key="fake-key")


async def test_embed_query_rejects_wrong_dim() -> None:
    with patch("edgar_analyst.retrieval.embedder.VoyageAIEmbeddings") as factory:
        factory.return_value.aembed_query = AsyncMock(return_value=[0.0] * 512)
        with pytest.raises(RuntimeError, match="512 dims"):
            await embed_query("hello world", api_key="fake-key")


@pg_required
class TestVectorSearch:
    @pytest.fixture
    def conn(self) -> Iterator[Connection]:
        connection = connect(_settings_dsn())
        try:
            with connection.cursor() as cur:
                cur.execute(
                    SQL("DROP TABLE IF EXISTS {t} CASCADE").format(t=Identifier(TEST_TABLE))
                )
            connection.commit()
            init_schema(connection, table=TEST_TABLE)
            yield connection
        finally:
            with connection.cursor() as cur:
                cur.execute(
                    SQL("DROP TABLE IF EXISTS {t} CASCADE").format(t=Identifier(TEST_TABLE))
                )
            connection.commit()
            connection.close()

    @staticmethod
    def _make_chunk(idx: int, *, ticker: str = "TEST", item_id: str = "1A") -> Chunk:
        return Chunk(
            ticker=ticker,
            cik="0000000001",
            accession_no="0000000001-24-000001",
            form_type="10-K",
            filing_date="2024-01-01",
            item_id=item_id,
            section_title=f"Section {item_id}",
            chunk_index=idx,
            section_chunk_count=10,
            text=f"chunk body number {idx}",
        )

    @staticmethod
    def _seeded_vector(seed: float) -> list[float]:
        return [seed] * EMBEDDING_DIMS

    def test_search_orders_by_similarity_with_higher_score_first(self, conn: Connection) -> None:
        chunks = [self._make_chunk(i) for i in range(3)]
        # Embeddings: 0 is closest to the query (same direction); 2 is opposite.
        embeddings = [
            self._seeded_vector(0.5),
            self._seeded_vector(0.2),
            self._seeded_vector(-0.5),
        ]
        upsert_chunks(conn, chunks, embeddings, table=TEST_TABLE)

        query_vec = self._seeded_vector(0.5)
        results = search(conn, query_vec, "TEST", k=3, table=TEST_TABLE)

        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        # First result is the exact-match embedding.
        assert results[0].chunk_index == 0
        # Scores are monotonically non-increasing.
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)
        # Score for an exact-direction match should be near 1.0.
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-6)

    def test_search_filters_by_ticker(self, conn: Connection) -> None:
        upsert_chunks(
            conn,
            [self._make_chunk(0, ticker="ALPHA")],
            [self._seeded_vector(0.3)],
            table=TEST_TABLE,
        )
        upsert_chunks(
            conn,
            [self._make_chunk(1, ticker="BETA")],
            [self._seeded_vector(0.3)],
            table=TEST_TABLE,
        )

        results = search(conn, self._seeded_vector(0.3), "ALPHA", k=5, table=TEST_TABLE)

        assert len(results) == 1
        assert results[0].chunk_index == 0

    def test_search_respects_k_limit(self, conn: Connection) -> None:
        chunks = [self._make_chunk(i) for i in range(5)]
        embeddings = [self._seeded_vector(0.1 * (i + 1)) for i in range(5)]
        upsert_chunks(conn, chunks, embeddings, table=TEST_TABLE)

        results = search(conn, self._seeded_vector(0.4), "TEST", k=2, table=TEST_TABLE)

        assert len(results) == 2

    def test_search_returns_empty_for_unknown_ticker(self, conn: Connection) -> None:
        upsert_chunks(
            conn,
            [self._make_chunk(0)],
            [self._seeded_vector(0.1)],
            table=TEST_TABLE,
        )

        results = search(conn, self._seeded_vector(0.1), "UNKNOWN", k=4, table=TEST_TABLE)

        assert results == []

    def test_search_with_zero_k_returns_empty(self, conn: Connection) -> None:
        results = search(conn, self._seeded_vector(0.1), "TEST", k=0, table=TEST_TABLE)
        assert results == []
