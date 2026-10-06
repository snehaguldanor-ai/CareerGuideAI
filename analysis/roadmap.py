"""Personalised learning roadmap built from the detected skill gaps."""
from __future__ import annotations

from analysis.common import AnalysisOutput, skills_block
from analysis.skill_gap import SkillGapResult
from llm.ollama_client import OllamaClient
from llm.prompts import SYSTEM_PROMPT
from rag.retriever import JOB, RESUME, CareerRetriever, format_context


def generate_roadmap(
    retriever: CareerRetriever, llm: OllamaClient, skills: SkillGapResult, weeks: int = 4
) -> AnalysisOutput:
    jd_chunks = retriever.retrieve_multi(["required skills", "responsibilities and tools used"], JOB, k=4)
    resume_chunks = retriever.retrieve_multi(["projects", "skills and technologies"], RESUME, k=3)

    if skills.recommendations:
        gap_lines = "\n".join(
            f"- {r['skill']} ({r['status']}, priority {r['priority']}): {r['reason']}" for r in skills.recommendations
        )
    else:
        gap_lines = "- No gaps were detected."

    prompt = f"""Create a realistic {weeks}-week learning roadmap for this candidate so they can become a strong fit for the job.

{skills_block(skills)}

SKILL GAPS ORDERED BY PRIORITY:
{gap_lines}

RULES:
- Plan ONLY around the skill gaps listed above, highest priority first. Do not add skills the job does not ask for.
- If there are no gaps, make a plan to deepen the strongest required skills and prepare for interviews.
- The last week must be a mini-project that combines the new skills and fits the job (build on the candidate's existing projects if the resume shows any).
- Do not name specific URLs or invent course names; describe resource types instead (official docs, tutorials, practice problems).

Use exactly this format for every week:
### Week N: <focus skills>
- **Goal:** one sentence
- **Tasks:** 3-4 concrete bullet points
- **Practice / deliverable:** one item
- **Time:** estimated hours

End with a short "General advice:" line.

JOB DESCRIPTION CONTEXT:
{format_context(jd_chunks)}

RESUME CONTEXT:
{format_context(resume_chunks)}"""
    text = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.3, max_tokens=1700)
    return AnalysisOutput(text=text, sources=jd_chunks + resume_chunks)
