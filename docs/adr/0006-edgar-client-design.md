# ADR 0006 — EDGAR client design

## Status

Accepted, 2026-05-05.

## Context

SEC EDGAR is a public, free, high-traffic system. Their fair-access policy
asks clients to:

- Send a descriptive `User-Agent` header that identifies the requester
  (typically a name and contact email).
- Stay under 10 requests per second across all endpoints.
- Accept compressed responses.

EDGAR responds to abuse with 429 and 503 status codes; sustained abuse can
result in IP-level blocks. The ingestion path is small (one filing per
invocation in v0.2.0) so we are nowhere near the ceiling, but the client must
behave well as a default so future batch runs do not trip rate limits.

The rest of the ingestion stack is async (LangGraph, FastAPI, langchain
async APIs), so the client should be async too.

## Decision

- `httpx.AsyncClient` configured at construction with the `User-Agent` from
  `Settings.sec_edgar_user_agent` and `Accept-Encoding: gzip, deflate`.
- A self-imposed rate limit of 5 requests per second (200ms minimum spacing
  between requests), enforced by an `asyncio.Lock` plus a monotonic
  next-allowed timestamp. This sits well under SEC's 10 req/sec ceiling and
  leaves headroom for any other process the user is running locally.
- Tenacity-driven retry on 429 and 503 only, with exponential backoff
  (initial 1s, max 8s) and a maximum of 3 attempts. Other 4xx/5xx codes are
  not retried; they are surfaced as `httpx.HTTPStatusError` so the caller
  can decide.
- Tickers JSON is fetched once per process and cached in-memory; subsequent
  `resolve_cik` calls are zero-network.
- All HTTP traffic in tests is mocked via `pytest-httpx`. Fixtures live in
  `tests/fixtures/edgar/`. CI never hits SEC. The live path is exercised only
  by the integration test, which is gated on `INTEGRATION=1`.

## Consequences

- The client is polite by construction: even a developer running the CLI in a
  loop is well below SEC's published limits.
- Retries handle the most common transient failures (rate-limit blip, brief
  upstream unavailability) without manual reruns.
- The async surface composes cleanly with the rest of the pipeline; the
  pipeline orchestrator awaits the client end-to-end and never blocks the
  event loop.
- We own retry, timeout, and rate-limit logic, which is a small surface to
  maintain at this scope.

## Alternatives considered

- **Synchronous `requests` client** — simpler, but the pipeline is already
  async for downstream LangGraph and Voyage calls. Mixing sync HTTP with an
  otherwise async pipeline forces thread-pool bridging that adds complexity
  for no benefit.
- **`sec-edgar-downloader` package** — convenient bulk downloader, but it
  abstracts away the rate-limiting details and the parsing path, which we
  want explicit control over (different chunking, different retention,
  different test surface).
- **No retry / no rate limit; rely on httpx defaults** — works on a happy
  day, fails on the first 429 burst with no recovery. Not acceptable.
- **Cache `company_tickers.json` on disk between runs** — defers a known
  upstream change (new listings, ticker changes) for marginal speedup. The
  in-memory cache is enough for one CLI invocation; revisit if a long-running
  worker needs it.
