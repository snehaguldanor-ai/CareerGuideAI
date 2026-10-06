"""Split documents into overlapping chunks (LangChain text splitter)."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def chunk_text(text: str, source: str, filename: str = "") -> list[Document]:
    """Return LangChain Documents tagged with their source (Resume / Job Description)."""
    if not text or not text.strip():
        raise ValueError(f"Cannot chunk an empty {source}.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
    if not pieces:
        raise ValueError(f"No usable text chunks were produced from the {source}.")
    return [
        Document(page_content=piece, metadata={"source": source, "chunk_id": i, "filename": filename})
        for i, piece in enumerate(pieces)
    ]
