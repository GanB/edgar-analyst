"""pgvector cosine search returning chunks with provenance metadata.

Direct psycopg, parameterized SQL, no ORM. Mirrors the ingestion
store's table layout so a retrieved row carries enough metadata to
build inline citations downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from psycopg import Connection
from psycopg.sql import SQL, Identifier

from edgar_analyst.ingestion.store import DEFAULT_TABLE


@dataclass(frozen=True)
class RetrievedChunk:
    item_id: str
    section_title: str
    content: str
    similarity_score: float
    accession_no: str
    chunk_index: int


def search(
    conn: Connection,
    query_embedding: list[float],
    ticker: str,
    *,
    k: int = 8,
    table: str = DEFAULT_TABLE,
) -> list[RetrievedChunk]:
    """Return top-``k`` chunks for ``ticker`` ordered by cosine similarity.

    Uses pgvector's ``<=>`` operator (cosine distance, smaller is better).
    The returned ``similarity_score`` is ``1 - distance`` so callers can
    treat higher as better.
    """
    if k <= 0:
        return []
    sql = SQL(
        """
        SELECT
            item_id,
            section_title,
            content,
            accession_no,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM {table}
        WHERE ticker = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
    ).format(table=Identifier(table))

    with conn.cursor() as cur:
        cur.execute(sql, (query_embedding, ticker, query_embedding, k))
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            item_id=str(row[0]),
            section_title=str(row[1]),
            content=str(row[2]),
            accession_no=str(row[3]),
            chunk_index=int(row[4]),
            similarity_score=float(row[5]),
        )
        for row in rows
    ]
