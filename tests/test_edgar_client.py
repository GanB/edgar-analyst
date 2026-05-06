"""Tests for the EDGAR HTTP client.

All HTTP traffic is mocked via pytest-httpx. No live SEC requests.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from edgar_analyst.ingestion.edgar import (
    DATA_BASE,
    TICKERS_URL,
    WWW_BASE,
    EdgarClient,
    EdgarRateLimitError,
    FilingMeta,
)

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def disable_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edgar_analyst.ingestion.edgar.MIN_REQUEST_SPACING_SECONDS", 0.0)


async def test_resolve_cik_success(httpx_mock: HTTPXMock, disable_throttle: None) -> None:
    httpx_mock.add_response(url=TICKERS_URL, text=_read("company_tickers.json"))

    async with EdgarClient(user_agent="test ua") as client:
        cik = await client.resolve_cik("aapl")

    assert cik == "0000320193"


async def test_resolve_cik_unknown_ticker(httpx_mock: HTTPXMock, disable_throttle: None) -> None:
    httpx_mock.add_response(url=TICKERS_URL, text=_read("company_tickers.json"))

    async with EdgarClient(user_agent="test ua") as client:
        with pytest.raises(LookupError, match="ZZZZ"):
            await client.resolve_cik("ZZZZ")


async def test_resolve_cik_caches_tickers(httpx_mock: HTTPXMock, disable_throttle: None) -> None:
    httpx_mock.add_response(url=TICKERS_URL, text=_read("company_tickers.json"))

    async with EdgarClient(user_agent="test ua") as client:
        first = await client.resolve_cik("AAPL")
        second = await client.resolve_cik("MSFT")

    assert first == "0000320193"
    assert second == "0000789019"
    # Only one upstream call despite two resolve_cik invocations.
    assert len(httpx_mock.get_requests(url=TICKERS_URL)) == 1


async def test_get_recent_filings_filters_form_type(
    httpx_mock: HTTPXMock, disable_throttle: None
) -> None:
    httpx_mock.add_response(
        url=f"{DATA_BASE}/submissions/CIK0000320193.json",
        text=_read("submissions_aapl.json"),
    )

    async with EdgarClient(user_agent="test ua") as client:
        filings = await client.get_recent_filings("0000320193", "10-K", limit=10)

    assert len(filings) == 2
    assert filings[0] == FilingMeta(
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
        primary_document="aapl-20240928.htm",
    )
    assert all(f.form_type == "10-K" for f in filings)


async def test_get_recent_filings_respects_limit(
    httpx_mock: HTTPXMock, disable_throttle: None
) -> None:
    httpx_mock.add_response(
        url=f"{DATA_BASE}/submissions/CIK0000320193.json",
        text=_read("submissions_aapl.json"),
    )

    async with EdgarClient(user_agent="test ua") as client:
        filings = await client.get_recent_filings("0000320193", "10-K", limit=1)

    assert len(filings) == 1
    assert filings[0].accession_no == "0000320193-24-000123"


async def test_fetch_filing_returns_document_text(
    httpx_mock: HTTPXMock, disable_throttle: None
) -> None:
    expected_url = f"{WWW_BASE}/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    httpx_mock.add_response(url=expected_url, text=_read("aapl-20240928.htm"))

    async with EdgarClient(user_agent="test ua") as client:
        body = await client.fetch_filing("0000320193", "0000320193-24-000123", "aapl-20240928.htm")

    assert "Apple 10-K excerpt body for testing." in body


async def test_user_agent_header_is_sent(httpx_mock: HTTPXMock, disable_throttle: None) -> None:
    httpx_mock.add_response(url=TICKERS_URL, text=_read("company_tickers.json"))

    async with EdgarClient(user_agent="Ganesh Babu test@example.com") as client:
        await client.resolve_cik("AAPL")

    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["User-Agent"] == "Ganesh Babu test@example.com"
    assert "gzip" in request.headers["Accept-Encoding"]


async def test_request_retries_on_429_then_succeeds(
    httpx_mock: HTTPXMock, disable_throttle: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Speed up tenacity backoff during the test.
    from tenacity import stop_after_attempt, wait_fixed

    import edgar_analyst.ingestion.edgar as edgar_mod

    real_request = edgar_mod.EdgarClient._request

    async def fast_request(self: edgar_mod.EdgarClient, url: str) -> httpx.Response:
        from tenacity import AsyncRetrying, retry_if_exception_type

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            retry=retry_if_exception_type(EdgarRateLimitError),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(url)
                if response.status_code in {429, 503}:
                    raise EdgarRateLimitError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
        raise RuntimeError("unreachable")

    monkeypatch.setattr(edgar_mod.EdgarClient, "_request", fast_request)

    httpx_mock.add_response(url=TICKERS_URL, status_code=429)
    httpx_mock.add_response(url=TICKERS_URL, text=_read("company_tickers.json"))

    async with EdgarClient(user_agent="test ua") as client:
        cik = await client.resolve_cik("AAPL")

    assert cik == "0000320193"
    assert len(httpx_mock.get_requests(url=TICKERS_URL)) == 2
    # Re-bind to the original to keep the test module clean.
    monkeypatch.setattr(edgar_mod.EdgarClient, "_request", real_request)
