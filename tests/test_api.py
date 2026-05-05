"""Tests for the FastAPI health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from edgar_analyst import __version__
from edgar_analyst.api.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
