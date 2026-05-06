"""Postgres + pgvector store for ingested filing chunks.

Direct psycopg, no ORM, no langchain-postgres for the schema; the
table is owned by ingestion and queried later by retrieval. The
schema layout is explicit so columns map 1:1 to ``Chunk`` metadata.

Idempotency is enforced by ``UNIQUE (accession_no, chunk_index)`` so
that re-running ingestion for the same filing rewrites rows in place
rather than duplicating them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.sql import SQL, Identifier

from edgar_analyst.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "documents"
EMBEDDING_DIMS = 1024


def connect(dsn: str) -> Connection:
    """Open a psycopg connection and register the pgvector type adapter."""
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
    except Exception:
        conn.close()
        raise
    return conn


def init_schema(conn: Connection, *, table: str = DEFAULT_TABLE) -> None:
    """Create the documents table, indexes, and pgvector extension.

    Idempotent: safe to call repeatedly.
    """
    table_id = Identifier(table)
    ddl = SQL(
        """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {table} (
            id BIGSERIAL PRIMARY KEY,
            accession_no TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date DATE NOT NULL,
            item_id TEXT NOT NULL,
            section_title TEXT NOT NULL,
            section_chunk_count INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1024) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (accession_no, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS {idx_ticker} ON {table} (ticker);
        CREATE INDEX IF NOT EXISTS {idx_accession} ON {table} (accession_no);
        CREATE INDEX IF NOT EXISTS {idx_embedding} ON {table}
            USING hnsw (embedding vector_cosine_ops);
        """
    ).format(
        table=table_id,
        idx_ticker=Identifier(f"idx_{table}_ticker"),
        idx_accession=Identifier(f"idx_{table}_accession"),
        idx_embedding=Identifier(f"idx_{table}_embedding"),
    )
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def upsert_chunks(
    conn: Connection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    table: str = DEFAULT_TABLE,
) -> int:
    """Insert or update chunks keyed on ``(accession_no, chunk_index)``.

    Returns the number of rows affected.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )
    if not chunks:
        return 0

    sql = SQL(
        """
        INSERT INTO {table} (
            accession_no, chunk_index, ticker, cik, form_type, filing_date,
            item_id, section_title, section_chunk_count, content, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (accession_no, chunk_index) DO UPDATE SET
            ticker = EXCLUDED.ticker,
            cik = EXCLUDED.cik,
            form_type = EXCLUDED.form_type,
            filing_date = EXCLUDED.filing_date,
            item_id = EXCLUDED.item_id,
            section_title = EXCLUDED.section_title,
            section_chunk_count = EXCLUDED.section_chunk_count,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding
        """
    ).format(table=Identifier(table))

    rows_affected = 0
    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cur.execute(
                sql,
                (
                    chunk.accession_no,
                    chunk.chunk_index,
                    chunk.ticker,
                    chunk.cik,
                    chunk.form_type,
                    chunk.filing_date,
                    chunk.item_id,
                    chunk.section_title,
                    chunk.section_chunk_count,
                    chunk.text,
                    list(embedding),
                ),
            )
            rows_affected += cur.rowcount
    conn.commit()
    return rows_affected


def count_for_ticker(conn: Connection, ticker: str, *, table: str = DEFAULT_TABLE) -> int:
    sql = SQL("SELECT COUNT(*) FROM {table} WHERE ticker = %s").format(table=Identifier(table))
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        row = cur.fetchone()
    if row is None:
        return 0
    return int(row[0])


def sample_sections_for_ticker(
    conn: Connection,
    ticker: str,
    *,
    n: int = 5,
    table: str = DEFAULT_TABLE,
) -> list[tuple[str, str]]:
    """Return up to ``n`` distinct (item_id, section_title) pairs for a ticker."""
    sql = SQL(
        """
        SELECT DISTINCT item_id, section_title
        FROM {table}
        WHERE ticker = %s
        ORDER BY item_id
        LIMIT %s
        """
    ).format(table=Identifier(table))
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, n))
        rows = cur.fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]
