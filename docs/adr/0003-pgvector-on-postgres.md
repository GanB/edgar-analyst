# ADR 0003 — pgvector on Postgres

## Status

Accepted, 2026-05-05.

## Context

The system needs a vector store for retrieval over filing chunks and a relational store for filing metadata, ingestion bookkeeping, and eventual user data. Operating two separate datastores in a portfolio-scope project is needless surface.

## Decision

Use the `pgvector` extension on Postgres 16, run locally via Docker Compose, with HNSW indexes for retrieval.

## Consequences

- One datastore covers both vector search and relational queries, so backups, migrations, and connection pooling are configured in one place.
- `pgvector` is mature, widely deployed, and integrates cleanly with `langchain-postgres`, so the application code stays small.
- HNSW gives sub-100ms retrieval for the corpus sizes we expect (single-digit thousands of filings, low millions of chunks); we accept slightly slower index builds in exchange for fast queries.
- Postgres operational knowledge (psql, EXPLAIN, pg_stat_statements) carries straight over; no separate observability for the vector store.

## Alternatives considered

- **Pinecone** — managed and fast, but adds a third-party dependency and a billing relationship for a self-contained portfolio project.
- **Weaviate or Qdrant** — capable, but introduce a second datastore alongside Postgres for no concrete benefit at this scope.
- **FAISS in-process** — fast and dependency-light, but has no persistence or transactional story and would need to be rebuilt every restart.
