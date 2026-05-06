"""End-to-end ingestion orchestrator.

Wires together the EDGAR client, parser, chunker, embedder, and store
behind one async entry point. Logs progress at each stage at INFO so a
single ingestion run produces a readable trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from edgar_analyst.ingestion.chunker import chunk_sections
from edgar_analyst.ingestion.edgar import EdgarClient
from edgar_analyst.ingestion.embedder import VoyageEmbedder
from edgar_analyst.ingestion.parser import parse_filing
from edgar_analyst.ingestion.store import connect, upsert_chunks
from edgar_analyst.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    ticker: str
    cik: str
    accession_no: str
    form_type: str
    filing_date: str
    section_count: int
    chunk_count: int
    rows_upserted: int


async def ingest_filing(ticker: str, form_type: str, settings: Settings) -> IngestionResult:
    if settings.voyage_api_key is None:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set; cannot embed chunks. Add it to .env and re-run."
        )

    ticker = ticker.upper()
    async with EdgarClient(user_agent=settings.sec_edgar_user_agent) as client:
        cik = await client.resolve_cik(ticker)
        logger.info("Resolved CIK %s for %s", cik, ticker)

        filings = await client.get_recent_filings(cik, form_type, limit=1)
        if not filings:
            raise LookupError(f"No {form_type} filings found for {ticker} (CIK {cik})")
        filing = filings[0]
        logger.info(
            "Found filing %s dated %s (form %s, primary doc %s)",
            filing.accession_no,
            filing.filing_date,
            filing.form_type,
            filing.primary_document,
        )

        html = await client.fetch_filing(cik, filing.accession_no, filing.primary_document)

    sections = parse_filing(html)
    chunks = chunk_sections(
        sections,
        ticker=ticker,
        cik=cik,
        accession_no=filing.accession_no,
        form_type=filing.form_type,
        filing_date=filing.filing_date,
    )
    logger.info("Parsed %d sections, %d chunks", len(sections), len(chunks))

    if not chunks:
        return IngestionResult(
            ticker=ticker,
            cik=cik,
            accession_no=filing.accession_no,
            form_type=filing.form_type,
            filing_date=filing.filing_date,
            section_count=len(sections),
            chunk_count=0,
            rows_upserted=0,
        )

    embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
    vectors = embedder.embed([c.text for c in chunks])
    logger.info("Embedded %d chunks via Voyage", len(vectors))

    conn = connect(settings.postgres_dsn_psycopg)
    try:
        rows = upsert_chunks(conn, chunks, vectors)
    finally:
        conn.close()
    logger.info("Upserted %d rows", rows)

    return IngestionResult(
        ticker=ticker,
        cik=cik,
        accession_no=filing.accession_no,
        form_type=filing.form_type,
        filing_date=filing.filing_date,
        section_count=len(sections),
        chunk_count=len(chunks),
        rows_upserted=rows,
    )
