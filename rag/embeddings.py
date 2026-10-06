"""Sentence-Transformers embeddings (all-MiniLM-L6-v2, runs on CPU)."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or used."""


@lru_cache(maxsize=1)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(
            f"Could not load the embedding model '{MODEL_NAME}'. The first run needs an internet "
            f"connection to download it once (about 90 MB). Details: {exc}"
        ) from exc


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (n, 384)."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise EmbeddingError("There is no text to embed.")
    model = get_model()
    try:
        vectors = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
    return np.ascontiguousarray(vectors, dtype="float32")
