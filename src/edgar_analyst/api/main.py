"""FastAPI application factory and route registration."""

from __future__ import annotations

from fastapi import FastAPI

from edgar_analyst import __version__


def create_app() -> FastAPI:
    app = FastAPI(
        title="edgar-analyst",
        version=__version__,
        description="RAG over public SEC EDGAR filings.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
