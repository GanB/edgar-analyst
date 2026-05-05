"""Command-line entry point for edgar-analyst."""

from __future__ import annotations

import click

from edgar_analyst import __version__


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


if __name__ == "__main__":
    main()
