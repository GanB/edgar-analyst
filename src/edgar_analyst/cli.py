"""Command-line entry point for edgar-analyst."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict

import click

from edgar_analyst import __version__
from edgar_analyst.ingestion.pipeline import ingest_filing
from edgar_analyst.ingestion.store import (
    connect,
    count_for_ticker,
    init_schema,
    sample_sections_for_ticker,
)
from edgar_analyst.settings import get_settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__, prog_name="edgar-analyst")
def main() -> None:
    """edgar-analyst — RAG over public SEC EDGAR filings."""


@main.command()
def hello() -> None:
    """Print a project banner and exit."""
    click.echo(
        f"edgar-analyst v{__version__} — RAG over SEC EDGAR filings. "
        "Run `uv run uvicorn edgar_analyst.api.main:app --reload` to start the API."
    )


@main.command("init-db")
def init_db() -> None:
    """Create the documents table and pgvector extension."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    conn = connect(settings.postgres_dsn_psycopg)
    try:
        init_schema(conn)
    finally:
        conn.close()
    click.echo(
        "Schema initialized at "
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


@main.command()
@click.option("--ticker", required=True, help="Stock ticker, e.g. AAPL.")
@click.option(
    "--form",
    "form_type",
    required=True,
    type=click.Choice(["10-K", "10-Q"], case_sensitive=False),
    help="Filing form type.",
)
def ingest(ticker: str, form_type: str) -> None:
    """Ingest the most recent filing of a given form type for a ticker."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    result = asyncio.run(ingest_filing(ticker, form_type.upper(), settings))
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@main.command()
@click.option("--ticker", required=True, help="Stock ticker, e.g. AAPL.")
def verify(ticker: str) -> None:
    """Report row count and a sample of section headers for a ticker."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    ticker_upper = ticker.upper()
    conn = connect(settings.postgres_dsn_psycopg)
    try:
        count = count_for_ticker(conn, ticker_upper)
        sample = sample_sections_for_ticker(conn, ticker_upper, n=5)
    finally:
        conn.close()
    click.echo(f"Ticker {ticker_upper}: {count} chunks")
    if sample:
        click.echo("Sample sections:")
        for item_id, title in sample:
            click.echo(f"  Item {item_id}: {title}")
    else:
        click.echo("No sections found. Run `edgar-analyst ingest` first.")


if __name__ == "__main__":
    main()
