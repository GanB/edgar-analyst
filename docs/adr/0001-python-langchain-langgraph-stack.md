# ADR 0001 — Python + LangChain + LangGraph stack

## Status

Accepted, 2026-05-05.

## Context

edgar-analyst is a retrieval-augmented question-answering system over public SEC filings. The query path is multi-step (retrieve → optional rerank → synthesize → cite) and benefits from explicit, typed state. The ingestion path needs filing loaders, HTML/XBRL parsers, chunkers, and an embedding step. We want a single ecosystem that covers both with first-party integrations for the chosen model and vector store.

## Decision

Build the system in Python 3.12, using LangChain for retrieval and document primitives and LangGraph for the orchestration of the query flow.

## Consequences

- LangGraph's `StateGraph` gives the query path explicit state and edges, which makes adding rerank, citation, and self-check nodes a localized change rather than an orchestration rewrite.
- LangChain provides off-the-shelf document loaders, splitters, retrievers, and output parsers, so the ingestion and synthesis code stays focused on EDGAR-specific concerns.
- The Python ecosystem (LangSmith for tracing/eval, vector store integrations, the Anthropic and Voyage SDKs) is concentrated and well-maintained, so most integration work is wiring rather than building.
- We accept that LangChain churns its public surface; we mitigate by pinning versions and keeping our use of it to stable primitives (loaders, splitters, prompt templates, output parsers).

## Alternatives considered

- **Spring AI on the JVM** — capable but the Python LangChain/LangGraph ecosystem is meaningfully richer for RAG today; tooling for evaluation and observability lives there.
- **LlamaIndex** — strong on ingestion ergonomics but weaker on stateful multi-step agentic flows than LangGraph.
- **Direct Anthropic SDK without an orchestration layer** — lowest dependency cost but pushes state, retries, and graph branching into hand-rolled code, which is not the right trade for this project's scope.
