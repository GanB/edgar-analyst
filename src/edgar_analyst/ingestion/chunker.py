"""Token-aware chunker that preserves section metadata.

Wraps LangChain's ``RecursiveCharacterTextSplitter`` configured with a
``tiktoken`` ``cl100k_base`` length function. Voyage does not expose
its own tokenizer; cl100k is a close-enough proxy for chunk sizing
at this scale (target 500, overlap 50).

Each output chunk carries the metadata needed to round-trip a
retrieval result back to the source: ticker, CIK, accession number,
form type, filing date, item id, section title, and the chunk's
position within its section.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from edgar_analyst.ingestion.parser import Section

DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
TIKTOKEN_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    ticker: str
    cik: str
    accession_no: str
    form_type: str
    filing_date: str
    item_id: str
    section_title: str
    chunk_index: int
    section_chunk_count: int
    text: str


_ENCODING = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _build_splitter(chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_sections(
    sections: list[Section],
    *,
    ticker: str,
    cik: str,
    accession_no: str,
    form_type: str,
    filing_date: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk each section independently and return a flat list of chunks.

    ``chunk_index`` is 0-indexed and globally unique within a filing, so
    that ``UNIQUE (accession_no, chunk_index)`` in the store identifies
    a single chunk. ``section_chunk_count`` reports how many chunks
    came out of the chunk's own section, which preserves per-section
    provenance for retrieval.
    """
    splitter = _build_splitter(chunk_tokens, overlap_tokens)
    out: list[Chunk] = []
    global_index = 0
    for section in sections:
        pieces = splitter.split_text(section.text)
        section_chunk_count = len(pieces)
        for piece in pieces:
            out.append(
                Chunk(
                    ticker=ticker,
                    cik=cik,
                    accession_no=accession_no,
                    form_type=form_type,
                    filing_date=filing_date,
                    item_id=section.item_id,
                    section_title=section.title,
                    chunk_index=global_index,
                    section_chunk_count=section_chunk_count,
                    text=piece,
                )
            )
            global_index += 1
    return out


def token_count(text: str) -> int:
    """Public helper for token counting; useful in tests and logging."""
    return _token_len(text)
