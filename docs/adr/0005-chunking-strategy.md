# ADR 0005 — Chunking strategy for filing ingestion

## Status

Accepted, 2026-05-05.

## Context

Retrieval over SEC filings has to balance two pressures. Chunks need to be small
enough that a handful of them fit in the synthesis model's effective context
window alongside the question and instructions, and small enough that a single
relevant passage does not get drowned out by a noisy neighbor. They also need
to be large enough to carry sufficient context for the synthesis model to cite
the source meaningfully — a one-sentence chunk is usually too thin to support a
grounded answer.

10-K and 10-Q filings have a strong native structure (Item 1A Risk Factors,
Item 7 MD&A, etc.) and topical coherence is highest within an Item. Crossing
section boundaries inside a chunk mixes unrelated subject matter and degrades
both retrieval precision and answer faithfulness.

## Decision

- Section-aware chunking. The parser emits one logical section per Item; the
  chunker chunks each section independently and never spans section boundaries.
- Target chunk size: 500 tokens, with 50 tokens of overlap between consecutive
  chunks within the same section.
- LangChain's `RecursiveCharacterTextSplitter`, configured with a `tiktoken`
  `cl100k_base` length function. Voyage does not publish its tokenizer; cl100k
  is a close-enough proxy for chunk sizing at this scale.
- Each chunk carries: ticker, CIK, accession number, form type, filing date,
  Item id, section title, chunk index (0-indexed within the filing, the unique
  key alongside accession number in the store), and the section's total chunk
  count.

## Consequences

- Retrieval can filter or rerank by section before ranking by similarity, which
  is useful when a question is scoped to a known Item ("according to Item 1A,
  ...").
- Synthesis can cite both filing and Item, so answers carry a more specific
  provenance than "this is from Apple's 10-K."
- Chunk indexes are global per filing, so re-ingestion replaces a filing's
  chunks deterministically via `ON CONFLICT (accession_no, chunk_index)`.
- The 500/50 sizing is a starting default; the eval harness in v0.5.0 will
  tune it against measured retrieval and faithfulness scores.

## Alternatives considered

- **Fixed-size chunking with no section awareness** — simplest possible, but
  routinely splits topically coherent passages and merges unrelated ones across
  Item boundaries. Rejected.
- **Semantic chunking (embedding-based boundary detection)** — adds an extra
  embedding pass per filing and a hyperparameter (similarity threshold) that
  itself needs tuning. The marginal retrieval win at this scale does not
  justify the complexity for v0.2.0. Reconsider after the eval harness exists.
- **Smaller chunks (200–300 tokens)** — improves retrieval precision but
  produces thin context that hurts synthesis fidelity. Rejected for now;
  revisit with eval data.
- **Larger chunks (1000+ tokens)** — fewer chunks, simpler index, but a single
  chunk often spans multiple subtopics within an Item, blurring retrieval
  signal. Rejected for now.
