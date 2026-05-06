# ADR 0002 — Anthropic API direct + Voyage embeddings

## Status

Accepted, 2026-05-05.

## Context

The system needs a long-context chat model for synthesis with citations and a strong embeddings model for retrieval over financial prose (10-K, 10-Q, 8-K). We want one provider per concern, both reachable directly via well-supported SDKs.

## Decision

Use Anthropic's API directly for chat with Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) and Voyage AI for embeddings with `voyage-3` (1024-dim).

## Consequences

- Claude Sonnet 4.5's 200K context comfortably fits 5–10 retrieved chunks plus system instructions, leaving headroom for long filings and per-section answers.
- Voyage `voyage-3` performs well on financial text in published benchmarks, and at 1024 dimensions it sits in a sweet spot for pgvector index size and retrieval latency.
- Two provider relationships (Anthropic, Voyage) instead of one, which is acceptable given the concern split is clean.
- Direct SDKs avoid an additional gateway layer; we accept that we own retry, timeout, and rate-limit handling, which is a small surface at this scope.

## Alternatives considered

- **AWS Bedrock for chat** — viable but adds an AWS dependency before there is a deploy story; deferred to ADR 0004's deployment posture.
- **OpenAI for chat** — capable, but no concrete benefit over Sonnet 4.5 for this use case and we are already standardizing on Anthropic.
- **OpenAI `text-embedding-3-large` for embeddings** — viable, but adds a second provider for chat-adjacent tooling without a measurable retrieval win on financial text.
