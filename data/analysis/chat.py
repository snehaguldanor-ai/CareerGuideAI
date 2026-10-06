"""RAG-grounded career chat: retrieve first, then answer."""
from __future__ import annotations

from CareerGuideAI.data.analysis.common import skills_block
from llm.prompts import NOT_FOUND_SENTENCE, SYSTEM_PROMPT
from CareerGuideAI.rag.retriever import BOTH, CareerRetriever, format_context
from CareerGuideAI.rag.vector_store import RetrievedChunk

SUGGESTED_QUESTIONS = [
    "What skills am I missing for this job?",
    "Am I suitable for this role?",
    "What should I learn first?",
    "Which projects should I build?",
    "What questions can be asked about my project?",
    "How can I improve my resume?",
    "What should I prepare for the interview?",
]


def prepare_chat(
    retriever: CareerRetriever,
    question: str,
    history: list[dict],
    skills=None,
) -> tuple[str, str, list[RetrievedChunk]]:
    """Retrieve context for the question and build (system, prompt, chunks)."""
    chunks = retriever.retrieve(question, BOTH, k=4)
    recent = history[-6:]
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent) or "(no previous messages)"
    analysis = f"\nSKILL ANALYSIS FROM THE DOCUMENTS:\n{skills_block(skills)}\n" if skills else ""
    prompt = f"""Answer the user's QUESTION using only the CONTEXT below (resume and job description excerpts).
If the context does not contain the answer, reply: "{NOT_FOUND_SENTENCE}"
Label anything not supported by the documents as "General advice:". Keep the answer focused and under 250 words.
{analysis}
CONTEXT:
{format_context(chunks)}

PREVIOUS CONVERSATION:
{convo}

QUESTION: {question}"""
    return SYSTEM_PROMPT, prompt, chunks
