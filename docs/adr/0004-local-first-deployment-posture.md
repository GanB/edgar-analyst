# ADR 0004 — Local-first deployment posture

## Status

Accepted, 2026-05-05.

## Context

The substantive work in edgar-analyst lives in the application code: the LangGraph design, the retrieval and citation discipline, and the eval harness. Continuously running cloud infrastructure adds cost and operational tax without contributing to that work, and there is no current need for an always-on deployment.

## Decision

v1 runs entirely locally: Postgres + pgvector via Docker Compose, FastAPI via `uv run uvicorn`, and (in a later release) the Vite dev server for the React UI. The eventual production target — ECS Fargate for the API, RDS for Postgres + pgvector, ALB in front, Secrets Manager for credentials, CloudWatch for logs and metrics — is documented as a target topology but is not provisioned.

## Consequences

- A clean `git clone && uv sync && docker compose up` path keeps onboarding and contribution friction low.
- No standing cloud costs; the project does not become a maintenance burden when attention is elsewhere.
- The deploy story is deferred, not absent: the `/health` endpoint, env-var-driven configuration, and stateless API design are chosen to make the eventual ECS rollout incremental.
- Activation triggers are explicit (a hosted endpoint becomes necessary, sustained multi-user load, integration with another service), so the cloud path is opened deliberately rather than drifted into.

## Alternatives considered

- **Continuous AWS deploy from day one** — meaningful operational tax for a project at this scope, where iteration on the application code and eval results is the primary work.
- **Fully serverless (Lambda + API Gateway + a managed vector store)** — constrains the architecture (cold starts on the synthesis path, no long-running connections, awkward fit for the local Postgres dev loop) for no current benefit.
- **Cloud-only with no local path** — would block contributors and slow iteration, and would tie any demo to an always-on environment.
