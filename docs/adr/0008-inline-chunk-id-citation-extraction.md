# ADR 0008 — Inline chunk_id citation extraction

## Status

Accepted, 2026-05-05.

## Context

The synthesis step must produce a grounded answer with attributable
citations back to specific filing sections. Two viable shapes were on the
table:

1. Prompt the chat model to emit structured JSON whose fields include
   the answer prose and a list of cited chunk ids.
2. Prompt the chat model to interleave inline markers like
   `[chunk_id=N]` in plain prose, then parse the markers post-hoc and
   look each one up against the retrieved set.

Constraints: the response must stream token-by-token (FastAPI SSE), the
output renders both in a CLI and a future React frontend, and the parser
must be deterministic enough to unit-test.

## Decision

Inline `[chunk_id=N]` markers, parsed by a small regex (`r"\[chunk_id=(\d+)\]"`)
post-synthesis. Each unique id is mapped to its `RetrievedChunk` to build
a `Citation` (item id, section title, accession number, chunk index, and
a 200-char snippet). Unknown ids that the model invents are logged at
WARNING and skipped.

## Consequences

- The model produces natural prose with inline references that read well
  in both CLI output and frontend display. No format-coupling between the
  prompt design and the renderer.
- Streaming is trivial: token frames flow as the chat model emits them,
  and citation extraction runs on the accumulated text after the stream
  drains. The client receives `event: token` repeatedly, then a single
  `event: citations` with the deduped list, then `event: done`.
- Parsing is deterministic, easy to unit-test (multiple, dedupe, missing
  marker, unknown id), and tolerates mild model misbehavior (an unknown
  id is dropped, not crashed on).
- Failure mode: the model writes well-formed prose but emits no markers
  at all. The CLI surface and the API both still return the synthesis;
  the citations list is empty. We treat that as a quality signal in eval
  rather than a hard error.

## Alternatives considered

- **Structured JSON output (or tool calls).** Breaks naturally streaming
  prose. Anthropic supports streaming structured output, but the rendered
  surface (CLI, frontend) wants prose with inline references, not a JSON
  blob. Tool-use mode also adds latency for marginal gain at this scope.
- **Vector match between synthesis sentences and chunks.** Heuristic and
  fragile. Embedding alignment between a synthesized sentence and the
  source chunk is not guaranteed, and tuning a threshold per filing type
  is more work than it is worth.
- **Footnote-style citations (`[^1]`, `[^2]`).** Indistinguishable from
  inline markers in robustness, but harder to parse safely without
  conflicting with markdown footnote rendering downstream.
