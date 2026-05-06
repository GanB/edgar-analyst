"""Voyage embeddings wrapper.

Thin layer over ``langchain_voyageai.VoyageAIEmbeddings`` that:

- caps each upstream call at ``BATCH_SIZE`` documents (Voyage's
  ``voyage-3`` accepts up to 128 inputs per request);
- retries transient failures with exponential backoff so a brief
  network blip does not abort an entire ingestion run.
"""

from __future__ import annotations

import logging

from langchain_voyageai import VoyageAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "voyage-3"
DEFAULT_DIMENSIONS = 1024
BATCH_SIZE = 128


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._client = VoyageAIEmbeddings(
            voyage_api_key=api_key,
            model=model,
            batch_size=batch_size,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            logger.info(
                "Embedding batch %d..%d via Voyage %s",
                start,
                start + len(batch),
                self._model,
            )
            vectors = self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Voyage returned {len(vectors)} vectors for batch of {len(batch)}"
                )
            out.extend(vectors)
        return out
