"""Skill match + skill gap analysis: RAG retrieval -> Ollama extraction -> embedding/text matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from analysis.common import AnalysisError
from llm.ollama_client import LLMError, ModelNotFoundError, OllamaClient, OllamaUnavailableError
from llm.prompts import SYSTEM_PROMPT
from rag.embeddings import embed_texts
from rag.retriever import BOTH, JOB, RESUME, CareerRetriever, format_context
from rag.vector_store import RetrievedChunk
from utils.helpers import (
    SOURCE_JD,
    SOURCE_RESUME,
    extract_json,
    normalize_skill,
    skill_in_text,
    truncate,
    unique_skills,
)

# Tunable similarity thresholds for "partial match" (cosine similarity, MiniLM).
PARTIAL_SKILL_SIM = 0.55   # JD skill vs. a skill listed on the resume
PARTIAL_CHUNK_SIM = 0.45   # JD skill vs. best resume chunk
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
ProgressFn = Callable[[float, str], None]


@dataclass
class SkillItem:
    skill: str
    importance: str          # "Required" or "Preferred"
    status: str              # "strong" | "partial" | "missing"
    evidence: str = ""
    similarity: float = 0.0


@dataclass
class SkillGapResult:
    items: list[SkillItem] = field(default_factory=list)
    resume_skills: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)

    @property
    def strong(self) -> list[SkillItem]:
        return [i for i in self.items if i.status == "strong"]

    @property
    def partial(self) -> list[SkillItem]:
        return [i for i in self.items if i.status == "partial"]

    @property
    def missing(self) -> list[SkillItem]:
        return [i for i in self.items if i.status == "missing"]

    @property
    def match_percent(self) -> float:
        if not self.items:
            return 0.0
        return round((len(self.strong) + 0.5 * len(self.partial)) / len(self.items) * 100, 1)


# --------------------------------------------------------------------------- extraction
def _parse_skill_list(raw: str) -> list[tuple[str, str]] | None:
    data = extract_json(raw)
    if isinstance(data, dict):
        items = data.get("skills")
        if items is None:
            items = next((v for v in data.values() if isinstance(v, list)), None)
    elif isinstance(data, list):
        items = data
    else:
        return None
    if not isinstance(items, list):
        return None
    out: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str):
            name, importance = item, "required"
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("skill") or "").strip()
            importance = str(item.get("importance") or "required")
        else:
            continue
        name = name.strip()
        if name:
            out.append((name, importance.strip().lower()))
    return out


def _extract(llm: OllamaClient, context: str, full_text: str, kind: str, limit: int) -> list[tuple[str, str]]:
    if kind == "job":
        task = (
            "From the JOB DESCRIPTION excerpts below, list the skills, programming languages, tools, "
            "technologies, concepts and qualifications the job asks for. Only include items that are "
            'explicitly written in the text. Mark each one "required" or "preferred".\n'
            'Return ONLY JSON like: {"skills":[{"name":"Python","importance":"required"}]}'
        )
    else:
        task = (
            "From the RESUME excerpts below, list every skill, programming language, tool, technology "
            "and technical concept the candidate has actually written down (including those used in "
            "projects). Only include items explicitly written in the text.\n"
            'Return ONLY JSON like: {"skills":["Python","SQL"]}'
        )
    prompt = f"{task}\n\nCONTEXT:\n{context}"
    parsed = None
    for _ in range(2):
        raw = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.0, json_mode=True, max_tokens=700)
        parsed = _parse_skill_list(raw)
        if parsed is not None:
            break
    if parsed is None:
        raise AnalysisError("The model did not return a readable skill list. Please click Analyze again.")
    # Hallucination guard: keep only skills that literally occur in the uploaded document.
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, importance in parsed:
        key = normalize_skill(name)
        if not key or key in seen or len(name) > 40:
            continue
        if skill_in_text(name, full_text):
            seen.add(key)
            kept.append((name, importance))
    return kept[:limit]


# --------------------------------------------------------------------------- matching
def _classify(
    retriever: CareerRetriever,
    jd_skills: list[tuple[str, str]],
    resume_skills: list[str],
    resume_text: str,
) -> list[SkillItem]:
    jd_vecs = embed_texts([s for s, _ in jd_skills])
    resume_vecs = embed_texts(resume_skills) if resume_skills else None
    resume_docs = retriever.chunks(RESUME)
    items: list[SkillItem] = []
    for (skill, importance), vec in zip(jd_skills, jd_vecs):
        level = "Required" if importance in ("required", "must", "mandatory", "must-have") else "Preferred"
        if skill_in_text(skill, resume_text):
            evidence = next(
                (d.page_content for d in resume_docs if skill_in_text(skill, d.page_content)), ""
            )
            items.append(SkillItem(skill, level, "strong", truncate(evidence, 220), 1.0))
            continue
        best_sim, related = 0.0, ""
        if resume_vecs is not None:
            sims = resume_vecs @ vec
            idx = int(np.argmax(sims))
            best_sim, related = float(sims[idx]), resume_skills[idx]
        top_chunk = retriever.resume_store.search_vector(vec, 1)[0]
        if best_sim >= PARTIAL_SKILL_SIM:
            items.append(SkillItem(skill, level, "partial", f"Related to resume skill '{related}'", best_sim))
        elif top_chunk.score >= PARTIAL_CHUNK_SIM:
            items.append(SkillItem(skill, level, "partial", truncate(top_chunk.text, 220), top_chunk.score))
        else:
            items.append(SkillItem(skill, level, "missing", "No evidence found in the resume.", max(best_sim, top_chunk.score)))
    return items


# --------------------------------------------------------------------------- recommendations
def _fallback_rec(item: SkillItem) -> dict:
    if item.status == "missing":
        priority = "High" if item.importance == "Required" else "Medium"
        reason = f"{item.importance} by the job description but not found in your resume."
    else:
        priority = "Medium" if item.importance == "Required" else "Low"
        reason = f"Only related evidence found in your resume; strengthen '{item.skill}' with a concrete project."
    return {
        "skill": item.skill,
        "status": item.status,
        "priority": priority,
        "reason": reason,
        "how_to_learn": "Learn the fundamentals, then apply it in a small project and add it to your resume.",
    }


def _recommendations(llm: OllamaClient, retriever: CareerRetriever, items: list[SkillItem]) -> list[dict]:
    gaps = [i for i in items if i.status != "strong"]
    if not gaps:
        return []
    fallback = {normalize_skill(i.skill): _fallback_rec(i) for i in gaps}
    jd_ctx = format_context(retriever.retrieve_multi([i.skill for i in gaps[:6]], JOB, k=2)[:6])
    listing = "\n".join(f"- {i.skill} | status: {i.status} | job importance: {i.importance}" for i in gaps)
    prompt = (
        "The candidate lacks or only partially shows the skills below. For EACH skill give a learning priority "
        '("High", "Medium" or "Low") based on how central it is in the JOB DESCRIPTION, a one-sentence reason '
        "and one short practical way to learn it. Use only the listed skills.\n"
        'Return ONLY JSON: {"recommendations":[{"skill":"...","priority":"High","reason":"...","how_to_learn":"..."}]}\n\n'
        f"SKILLS:\n{listing}\n\nJOB DESCRIPTION CONTEXT:\n{jd_ctx}"
    )
    llm_recs: dict[str, dict] = {}
    try:
        data = extract_json(llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.1, json_mode=True, max_tokens=900))
        recs = data.get("recommendations") if isinstance(data, dict) else data
        for rec in recs or []:
            if isinstance(rec, dict) and rec.get("skill"):
                llm_recs[normalize_skill(str(rec["skill"]))] = rec
    except (OllamaUnavailableError, ModelNotFoundError):
        raise
    except LLMError:
        pass
    merged: list[dict] = []
    for key, base in fallback.items():
        rec = dict(base)
        got = llm_recs.get(key)
        if got:
            priority = str(got.get("priority", "")).strip().capitalize()
            if priority in PRIORITY_ORDER:
                rec["priority"] = priority
            for field_name in ("reason", "how_to_learn"):
                value = str(got.get(field_name, "")).strip()
                if value:
                    rec[field_name] = value
        merged.append(rec)
    merged.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 3))
    return merged


# --------------------------------------------------------------------------- entry point
def run_skill_gap(
    retriever: CareerRetriever,
    llm: OllamaClient,
    resume_text: str,
    jd_text: str,
    progress: ProgressFn | None = None,
) -> SkillGapResult:
    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    if not retriever.is_ready:
        raise AnalysisError("The document index is not ready. Upload both documents first.")

    tick(0.1, "Retrieving job requirements from the job description...")
    jd_chunks = retriever.retrieve_multi(
        ["required skills technologies tools", "qualifications requirements responsibilities", "preferred nice to have"],
        JOB,
        k=5,
    )
    tick(0.2, "Retrieving your skills and projects from the resume...")
    resume_chunks = retriever.retrieve_multi(
        ["skills technologies programming languages tools", "projects experience education"], RESUME, k=6
    )

    tick(0.3, "Ollama is extracting the skills the job asks for...")
    jd_skills = _extract(llm, format_context(jd_chunks), jd_text, "job", 25)
    if not jd_skills:
        raise AnalysisError("No skills could be identified in the job description. Check the uploaded text.")

    tick(0.5, "Ollama is extracting the skills in your resume...")
    resume_skills = [s for s, _ in _extract(llm, format_context(resume_chunks), resume_text, "resume", 40)]
    resume_skills = unique_skills(resume_skills)

    tick(0.7, "Comparing skills with embeddings + FAISS evidence search...")
    items = _classify(retriever, jd_skills, resume_skills, resume_text)

    tick(0.85, "Prioritising what to learn...")
    recommendations = _recommendations(llm, retriever, items)

    tick(1.0, "Skill analysis complete.")
    return SkillGapResult(
        items=items,
        resume_skills=resume_skills,
        recommendations=recommendations,
        sources=jd_chunks + resume_chunks,
    )
