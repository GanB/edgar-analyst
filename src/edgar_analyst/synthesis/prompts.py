"""Prompt templates for the synthesis chain.

The system prompt anchors the model in the financial-analyst role
and constrains it to grounded answers. The human prompt embeds the
formatted retrieval context plus the user's question. Inline
citations use the ``[chunk_id=N]`` marker scheme parsed by
``synthesis.citations``.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from edgar_analyst.retrieval.vector_search import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a financial analyst answering questions about a specific SEC filing. "
    "Use only the provided chunks to answer. Cite every factual claim inline "
    "using markers of the form [chunk_id=N], where N is the chunk_id shown above "
    "the chunk you used. Do not invent facts that are not present in the chunks. "
    "If the chunks do not answer the question, say so directly."
)

HUMAN_PROMPT = (
    "Question about ticker {ticker}:\n"
    "{question}\n\n"
    "Retrieved filing chunks:\n"
    "{context}\n\n"
    "Write a focused answer with inline [chunk_id=N] citations for each fact. "
    "Cite chunk_ids you actually used. Do not invent facts not present in the chunks."
)


synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into the prompt-friendly block."""
    if not chunks:
        return "(no chunks retrieved)"
    blocks: list[str] = []
    for c in chunks:
        header = f"[chunk_id={c.chunk_index}, item={c.item_id}, section={c.section_title}]"
        blocks.append(f"{header}\n{c.content}\n---")
    return "\n".join(blocks)
