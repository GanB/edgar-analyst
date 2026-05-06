# ADR 0007 — LangGraph synthesis design

## Status

Accepted, 2026-05-05.

## Context

The query path has four discrete steps: embed the user question via Voyage,
retrieve top-k chunks from pgvector, synthesize an answer with the chat
model, and extract inline citations. The phases are independently testable
and have different failure modes (network embed errors vs. empty retrieval
vs. chat-model timeouts), so we want a structure that exposes each step as
a unit. The FastAPI surface needs to stream tokens to the client as they
arrive from the chat model, so the orchestrator must support per-token
event surfacing without coupling the graph to the HTTP layer. Future
phases will add a reranker (v0.4.0) and conversation history; the design
should accommodate both without a large refactor.

## Decision

A LangGraph `StateGraph` with three nodes and one typed state object.

- Nodes: `embed_query`, `retrieve`, `synthesize`. Each is async and returns
  a partial `QueryState` slice; LangGraph merges the slice into the state
  between nodes.
- Edges: `START -> embed_query -> retrieve -> {synthesize | END}`. A
  conditional edge after `retrieve` short-circuits to `END` when the chunk
  set is empty so the chat model is never called with no context.
- State shape (`QueryState` TypedDict): `ticker`, `question`,
  `query_embedding`, `retrieved_chunks`, `synthesis`, `citations`, `error`.
- Streaming: HTTP layer drives the graph via `astream_events(version="v2")`
  and forwards `on_chat_model_stream` chunks as SSE `event: token` frames.
  Citations and the terminal `done` frame are emitted from the HTTP layer
  after the stream drains.

## Consequences

- Each node is mockable in isolation: unit tests patch the embedder, the
  Postgres connection, and the chat model independently. The graph itself
  is exercised end-to-end with all three patched.
- Adding a reranker in v0.4.0 is a new node between `retrieve` and
  `synthesize` plus one edge change; no node currently in the graph needs
  to be modified.
- Conversation history is a state field addition (`history: list[Message]`)
  consumed inside `synthesize`. No node-graph topology change required.
- `astream_events` is the integration seam for streaming. The graph stays
  unaware of HTTP, and the HTTP layer stays unaware of LangGraph internals
  beyond the public event protocol.
- One operational caveat: every node opens its own dependencies (Postgres
  connection, chat client). At this scale that overhead is dwarfed by
  network latency to Voyage and the chat model; revisit if a connection
  pool becomes worth its cost.

## Alternatives considered

- **Plain async function pipeline.** Simpler at first, but per-token
  streaming forces a callback or a generator threading through every layer,
  which gets ugly. LangGraph's event protocol is the cleanest seam.
- **LangChain Chain (RunnableSequence).** Composable, but stateful flows
  fit `StateGraph` better. Chains do not express conditional edges or
  partial state updates as naturally, and the LangGraph surface is the
  current ecosystem direction.
- **External DAG framework (Prefect, Dagster).** Overkill for an
  in-process query flow that finishes in a few seconds. Operational
  overhead and dependency surface are not justified at this scope.
