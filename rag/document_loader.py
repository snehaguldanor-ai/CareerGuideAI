"""Extract text from uploaded PDF / TXT files with clear, user-friendly errors."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from utils.helpers import clean_text


class DocumentLoadError(Exception):
    """Raised when an uploaded document cannot be read."""


@dataclass
class LoadedDocument:
    name: str
    text: str
    pages: int
    kind: str  # "pdf" or "txt"


def load_pdf(data: bytes, name: str = "document.pdf") -> LoadedDocument:
    if not data:
        raise DocumentLoadError(f"'{name}' is empty (0 bytes).")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - any decrypt problem means unreadable
                unlocked = 0
            if not unlocked:
                raise DocumentLoadError(f"'{name}' is password-protected. Please upload an unlocked PDF.")
        page_texts: list[str] = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - skip an unreadable page, keep the rest
                page_texts.append("")
        page_count = len(reader.pages)
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"'{name}' is not a valid PDF or is corrupted ({exc}).") from exc

    if page_count == 0:
        raise DocumentLoadError(f"'{name}' has no pages.")
    text = clean_text("\n\n".join(page_texts))
    if not text:
        raise DocumentLoadError(
            f"No text could be extracted from '{name}'. It may be a scanned image PDF - "
            "please upload a text-based PDF or a .txt file."
        )
    return LoadedDocument(name=name, text=text, pages=page_count, kind="pdf")


def load_txt(data: bytes, name: str = "document.txt") -> LoadedDocument:
    if not data:
        raise DocumentLoadError(f"'{name}' is empty (0 bytes).")
    text = ""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    text = clean_text(text)
    if not text:
        raise DocumentLoadError(f"'{name}' contains no readable text.")
    return LoadedDocument(name=name, text=text, pages=1, kind="txt")


def load_document(name: str, data: bytes) -> LoadedDocument:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(data, name)
    if suffix == ".txt":
        return load_txt(data, name)
    raise DocumentLoadError(f"Unsupported file type '{suffix}'. Please upload a PDF or TXT file.")
