"""Resume improvement suggestions grounded in resume + job description chunks."""
from __future__ import annotations

from analysis.common import AnalysisOutput, skills_block
from analysis.skill_gap import SkillGapResult
from llm.ollama_client import OllamaClient
from llm.prompts import NOT_FOUND_SENTENCE, SYSTEM_PROMPT
from rag.retriever import JOB, RESUME, CareerRetriever, format_context


def analyze_resume(retriever: CareerRetriever, llm: OllamaClient, skills: SkillGapResult) -> AnalysisOutput:
    resume_chunks = retriever.retrieve_multi(
        ["projects and experience", "skills and technologies", "education and achievements"], RESUME, k=5
    )
    jd_chunks = retriever.retrieve_multi(["required skills and qualifications", "responsibilities"], JOB, k=4)
    prompt = f"""Review the RESUME against the JOB DESCRIPTION and suggest improvements.

{skills_block(skills)}

Write the report with EXACTLY these markdown sections:
## 🔑 Missing Keywords
Keywords from the job description that do not appear in the resume.
## ⭐ Skills To Highlight
Existing resume skills/projects that match the job and should be made more visible.
## ⚠️ Weak Project / Experience Descriptions
Quote a SHORT phrase from the resume and explain why it is weak (vague, no outcome, no tools).
## 🧩 Missing Technical Skills
Skills from the MISSING list. Label each "Missing". Tell the candidate to learn them first and NOT to add them to the resume until learned.
## 🛠️ Formatting & Content Suggestions
Practical structure and wording advice.
## ✍️ Better Bullet-Point Suggestions
Rewrite 3 existing resume bullets using only facts present in the resume. Use placeholders like [metric] where a number is unknown. Never invent experience.
## 🏷️ Skill Classification
A markdown table with columns: Skill | Category (Existing / Suggested / Missing).

If a section has no support in the context, write: {NOT_FOUND_SENTENCE}

RESUME CONTEXT:
{format_context(resume_chunks)}

JOB DESCRIPTION CONTEXT:
{format_context(jd_chunks)}"""
    text = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=1700)
    return AnalysisOutput(text=text, sources=resume_chunks + jd_chunks)
