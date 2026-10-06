"""FAISS vector store (cosine similarity via inner product on normalised vectors)."""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from langchain_core.documents import Document

from rag.embeddings import EmbeddingError, embed_texts


class VectorStoreError(Exception):
    """Raised when the FAISS index cannot be built or searched."""


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float
    chunk_id: int


class FaissVectorStore:
    def __init__(self, name: str = "store") -> None:
        self.name = name
        self.index: faiss.Index | None = None
        self.documents: list[Document] = []

    def __len__(self) -> int:
        return len(self.documents)

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise VectorStoreError(f"No documents to index for '{self.name}'.")
        try:
            vectors = embed_texts([d.page_content for d in documents])
            if vectors.shape[0] != len(documents):
                raise VectorStoreError("Embedding count does not match chunk count.")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        except (VectorStoreError, EmbeddingError):
            raise
        except Exception as exc:  # noqa: BLE001 - FAISS / numpy problems
            raise VectorStoreError(f"Could not build the FAISS index for '{self.name}': {exc}") from exc
        self.index = index
        self.documents = list(documents)

    def search_vector(self, vector: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        if self.index is None or not self.documents:
            raise VectorStoreError(f"The '{self.name}' index has not been built yet.")
        k = max(1, min(k, len(self.documents)))
        try:
            query = np.ascontiguousarray(vector.reshape(1, -1), dtype="float32")
            scores, ids = self.index.search(query, k)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            results.append(
                RetrievedChunk(
                    source=doc.metadata.get("source", self.name),
                    text=doc.page_content,
                    score=float(score),
                    chunk_id=int(doc.metadata.get("chunk_id", idx)),
                )
            )
        return results

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        return self.search_vector(embed_texts([query])[0], k)
