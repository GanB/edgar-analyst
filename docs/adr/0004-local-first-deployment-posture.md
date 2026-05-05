# ADR 0004 — Local-first deployment posture

## Status

Accepted, 2026-05-05.

## Context

The portfolio value of edgar-analyst is in the application code: the LangGraph design, the retrieval and citation discipline, and the eval harness. Continuously running cloud infrastructure adds cost and operational tax without adding to that value, until there is a specific reason (a live URL for hiring conversations, an interview demo).

## Decision

v1 runs entirely locally: Postgres + pgvector via Docker Compose, FastAPI via `uv run uvicorn`, and (in a later release) the Vite dev server for the React UI. The eventual production target — ECS Fargate for the API, RDS for Postgres + pgvector, ALB in front, Secrets Manager for credentials, CloudWatch for logs and metrics — is documented as a target topology but is not provisioned.

## Consequences

- A clean `git clone && uv sync && docker compose up` path keeps onboarding and contribution friction low.
- No standing cloud costs; the project does not become a maintenance burden when attention is elsewhere.
- The deploy story is deferred, not absent: the `/health` endpoint, env-var-driven configuration, and stateless API design are chosen to make the eventual ECS rollout incremental.
- Activation triggers are explicit (live URL needed, demo scheduled), so the cloud path is opened deliberately rather than drifted into.

## Alternatives considered

- **Continuous AWS deploy from day one** — meaningful operational tax for a project where the visible artifact is the code, the README, and the eval results.
- **Fully serverless (Lambda + API Gateway + a managed vector store)** — harder to demonstrate the full stack and constrains the architecture for no current benefit.
- **Cloud-only with no local path** — would block contributors and slow iteration, and would tie any demo to an always-on environment.
