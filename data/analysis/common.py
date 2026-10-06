"""Shared types and prompt helpers for the analysis modules."""
from __future__ import annotations

from dataclasses import dataclass, field

from CareerGuideAI.rag.vector_store import RetrievedChunk


class AnalysisError(Exception):
    """Raised when an analysis step cannot produce a usable result."""


@dataclass
class AnalysisOutput:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)


def skills_block(result) -> str:
    """Human-readable skill summary (built from the RAG skill-gap result) for prompts."""

    def fmt(items) -> str:
        return ", ".join(i.skill for i in items) if items else "none"

    return (
        f"EXISTING SKILLS (found in resume and required by the job): {fmt(result.strong)}\n"
        f"PARTIALLY MATCHING SKILLS (related evidence only): {fmt(result.partial)}\n"
        f"MISSING SKILLS (required by the job, not found in resume): {fmt(result.missing)}"
    )
