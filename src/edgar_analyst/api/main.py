"""FastAPI application factory and route registration."""

from __future__ import annotations

from fastapi import FastAPI

from edgar_analyst import __version__
from edgar_analyst.api.routes import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="edgar-analyst",
        version=__version__,
        description="RAG over public SEC EDGAR filings.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(query_router)
    return app


app = create_app()
