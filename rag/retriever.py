"""High-level retriever that searches the resume and job-description indexes."""
from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from rag.vector_store import FaissVectorStore, RetrievedChunk, VectorStoreError
from utils.helpers import SOURCE_JD, SOURCE_RESUME

RESUME = "resume"
JOB = "job"
BOTH = "both"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Turn retrieved chunks into the text block that is placed in the LLM prompt."""
    if not chunks:
        return "(no relevant text was retrieved)"
    return "\n\n".join(f"[{c.source} | chunk {c.chunk_id}]\n{c.text}" for c in chunks)


class CareerRetriever:
    def __init__(self) -> None:
        self.resume_store = FaissVectorStore(SOURCE_RESUME)
        self.jd_store = FaissVectorStore(SOURCE_JD)

    def build(self, resume_docs: list[Document], jd_docs: list[Document]) -> None:
        self.resume_store.build(resume_docs)
        self.jd_store.build(jd_docs)

    @property
    def is_ready(self) -> bool:
        return self.resume_store.index is not None and self.jd_store.index is not None

    @property
    def resume_chunk_count(self) -> int:
        return len(self.resume_store)

    @property
    def jd_chunk_count(self) -> int:
        return len(self.jd_store)

    def chunks(self, source: str) -> list[Document]:
        return self._store(source).documents

    def _store(self, source: str) -> FaissVectorStore:
        if source == RESUME:
            return self.resume_store
        if source == JOB:
            return self.jd_store
        raise VectorStoreError(f"Unknown source '{source}'.")

    def retrieve_multi(self, queries: Sequence[str], source: str, k: int = 4) -> list[RetrievedChunk]:
        """Top-k chunks per query from one source, de-duplicated, best score first."""
        best: dict[tuple[str, int], RetrievedChunk] = {}
        sources = [RESUME, JOB] if source == BOTH else [source]
        for src in sources:
            store = self._store(src)
            for query in queries:
                for chunk in store.search(query, k):
                    key = (chunk.source, chunk.chunk_id)
                    if key not in best or chunk.score > best[key].score:
                        best[key] = chunk
        return sorted(best.values(), key=lambda c: (-c.score))

    def retrieve(self, query: str, source: str = BOTH, k: int = 4) -> list[RetrievedChunk]:
        return self.retrieve_multi([query], source, k)
