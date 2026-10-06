"""Small shared helpers: text cleaning, JSON recovery, skill-text matching."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"

SOURCE_RESUME = "Resume"
SOURCE_JD = "Job Description"

_STOPWORDS = {"and", "of", "the", "for", "in", "to", "a", "an", "with", "on"}


def clean_text(text: str) -> str:
    """Normalise whitespace, bullets and line-break hyphenation in extracted text."""
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[•●▪■◦►▶◆]", "-", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def truncate(text: str, limit: int = 1200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def safe_md(text: str) -> str:
    """Escape '$' so Streamlit does not treat prices/amounts as LaTeX."""
    return (text or "").replace("$", "\\$")


def extract_json(raw: str) -> Any | None:
    """Best-effort recovery of a JSON object/array from an LLM reply."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (ValueError, TypeError):
                continue
    return None


def _tokens(text: str) -> list[str]:
    text = (text or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return [t.strip(".") for t in text.split() if t.strip(".")]


def _stem(token: str) -> str:
    if len(token) > 4:
        return token[:4]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def normalize_skill(skill: str) -> str:
    return " ".join(_tokens(skill))


def skill_in_text(skill: str, text: str) -> bool:
    """True if the skill is literally present in the text (plural/stem tolerant
    for multi-word skills). Single words need a whole-word match, so 'Java'
    does not match 'JavaScript'."""
    skill_n = normalize_skill(skill)
    if not skill_n or not text:
        return False
    text_tokens = _tokens(text)
    text_n = " ".join(text_tokens)
    pattern = r"(?<![a-z0-9+#])" + re.escape(skill_n) + r"(?:s|es)?(?![a-z0-9+#])"
    if re.search(pattern, text_n):
        return True
    words = [w for w in skill_n.split() if w not in _STOPWORDS]
    if len(words) >= 2:
        stems = {_stem(t) for t in text_tokens}
        return all(_stem(w) in stems for w in words)
    return False


def unique_skills(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = normalize_skill(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name.strip())
    return out
