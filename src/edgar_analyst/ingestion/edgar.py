"""HTTP client for SEC EDGAR.

Implements the small slice of EDGAR needed for ingestion:
ticker → CIK resolution, recent filings lookup, and primary-document
fetch. Rate-limited to 5 req/sec (well under SEC's 10 req/sec ceiling)
and retried on 429/503 with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self, cast

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DATA_BASE = "https://data.sec.gov"
WWW_BASE = "https://www.sec.gov"
TICKERS_URL = f"{WWW_BASE}/files/company_tickers.json"

# Self-imposed limit; SEC's published ceiling is 10 req/sec. Stay well below.
MIN_REQUEST_SPACING_SECONDS = 0.2

# Treat these as transient and worth retrying.
RETRY_STATUS_CODES = frozenset({429, 503})


class EdgarRateLimitError(httpx.HTTPStatusError):
    """Raised on 429/503 so tenacity can retry."""


@dataclass(frozen=True)
class FilingMeta:
    accession_no: str
    form_type: str
    filing_date: str  # ISO YYYY-MM-DD
    primary_document: str


class EdgarClient:
    """Async EDGAR client with polite rate limiting and retry."""

    def __init__(self, user_agent: str, *, timeout: float = 30.0) -> None:
        self._user_agent = user_agent
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=timeout,
            follow_redirects=True,
        )
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0
        self._tickers_cache: dict[str, str] | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_allowed_at:
                await asyncio.sleep(self._next_allowed_at - now)
                now = loop.time()
            self._next_allowed_at = now + MIN_REQUEST_SPACING_SECONDS

    async def _request(self, url: str) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(EdgarRateLimitError),
            reraise=True,
        ):
            with attempt:
                await self._throttle()
                response = await self._client.get(url)
                if response.status_code in RETRY_STATUS_CODES:
                    raise EdgarRateLimitError(
                        f"EDGAR returned {response.status_code} for {url}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
        raise RuntimeError("unreachable: AsyncRetrying exited without returning")

    async def _load_tickers(self) -> dict[str, str]:
        if self._tickers_cache is not None:
            return self._tickers_cache
        response = await self._request(TICKERS_URL)
        payload = json.loads(response.text)
        # Format: { "0": {"cik_str": 320193, "ticker": "AAPL", ...}, ... }
        mapping: dict[str, str] = {}
        for entry in payload.values():
            ticker = str(entry["ticker"]).upper()
            cik_int = int(entry["cik_str"])
            mapping[ticker] = f"{cik_int:010d}"
        self._tickers_cache = mapping
        return mapping

    async def resolve_cik(self, ticker: str) -> str:
        mapping = await self._load_tickers()
        key = ticker.upper()
        if key not in mapping:
            raise LookupError(f"No CIK found for ticker {ticker!r}")
        return mapping[key]

    async def get_recent_filings(
        self, cik: str, form_type: str, limit: int = 10
    ) -> list[FilingMeta]:
        url = f"{DATA_BASE}/submissions/CIK{cik}.json"
        response = await self._request(url)
        payload: dict[str, Any] = json.loads(response.text)
        recent = payload.get("filings", {}).get("recent", {})
        accession_numbers = cast(list[str], recent.get("accessionNumber", []))
        forms = cast(list[str], recent.get("form", []))
        filing_dates = cast(list[str], recent.get("filingDate", []))
        primary_docs = cast(list[str], recent.get("primaryDocument", []))

        out: list[FilingMeta] = []
        for accession, form, filing_date, primary in zip(
            accession_numbers, forms, filing_dates, primary_docs, strict=True
        ):
            if form != form_type:
                continue
            out.append(
                FilingMeta(
                    accession_no=accession,
                    form_type=form,
                    filing_date=filing_date,
                    primary_document=primary,
                )
            )
            if len(out) >= limit:
                break
        return out

    async def fetch_filing(self, cik: str, accession_no: str, primary_document: str) -> str:
        cik_int = int(cik)
        accession_compact = accession_no.replace("-", "")
        url = f"{WWW_BASE}/Archives/edgar/data/{cik_int}/{accession_compact}/{primary_document}"
        response = await self._request(url)
        return response.text
