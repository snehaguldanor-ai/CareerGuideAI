"""Personalised interview questions (4 categories, 6 questions each) with suggested answers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from CareerGuideAI.data.analysis.common import skills_block
from CareerGuideAI.data.analysis.skill_gap import SkillGapResult
from llm.ollama_client import OllamaClient
from llm.prompts import SYSTEM_PROMPT
from CareerGuideAI.rag.retriever import BOTH, JOB, RESUME, CareerRetriever, format_context
from CareerGuideAI.rag.vector_store import RetrievedChunk

QUESTIONS_PER_CATEGORY = 6

CATEGORIES: dict[str, dict] = {
    "Technical Questions": {
        "icon": "🧠",
        "source": BOTH,
        "queries": ["required technical skills and tools", "programming languages and technologies"],
        "instruction": (
            "Ask technical questions about the skills the job requires. Include a few questions about the "
            "MISSING skills so the candidate can prepare, and a few about skills the candidate already lists."
        ),
    },
    "Project Questions": {
        "icon": "🛠️",
        "source": RESUME,
        "queries": ["projects built and technologies used", "project description results"],
        "instruction": (
            "Ask questions about the specific projects written in the resume (design choices, challenges, "
            "tools, results). If the resume lists no projects, say so instead of inventing any."
        ),
    },
    "HR Questions": {
        "icon": "🤝",
        "source": BOTH,
        "queries": ["education and background", "role responsibilities and teamwork"],
        "instruction": (
            "Ask behavioural / HR questions (motivation, teamwork, strengths, weaknesses, learning) "
            "personalised to the candidate's background and to this role."
        ),
    },
    "Job-Specific Questions": {
        "icon": "🎯",
        "source": JOB,
        "queries": ["responsibilities of the role", "required qualifications", "company and team"],
        "instruction": "Ask questions that test fit for the responsibilities and requirements written in the job description.",
    },
}


@dataclass
class InterviewResult:
    sections: dict[str, str] = field(default_factory=dict)
    sources: dict[str, list[RetrievedChunk]] = field(default_factory=dict)


def generate_interview(
    retriever: CareerRetriever,
    llm: OllamaClient,
    skills: SkillGapResult,
    progress: Callable[[float, str], None] | None = None,
) -> InterviewResult:
    result = InterviewResult()
    total = len(CATEGORIES)
    for n, (name, cfg) in enumerate(CATEGORIES.items()):
        if progress:
            progress(n / total, f"Generating {name.lower()}...")
        chunks = retriever.retrieve_multi(cfg["queries"], cfg["source"], k=4)
        prompt = f"""Write exactly {QUESTIONS_PER_CATEGORY} interview questions for the category "{name}".
{cfg['instruction']}

{skills_block(skills)}

Format every item exactly like this:
**Q1. <question>**
- *Why it may be asked:* <one short sentence tied to the context>
- *Suggested answer:* <3-4 sentences. Use only facts from the RESUME context. Where the candidate has no matching experience, say they should honestly explain how they would learn it instead of claiming experience.>

CONTEXT:
{format_context(chunks)}"""
        result.sections[name] = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.4, max_tokens=1500)
        result.sources[name] = chunks
    if progress:
        progress(1.0, "Interview preparation ready.")
    return result
