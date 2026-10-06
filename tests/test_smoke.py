"""Offline smoke tests: no Ollama and no model download needed.

Run from the project folder:   python tests/test_smoke.py
They use a deterministic fake embedder and a fake LLM to verify the full pipeline logic
(loading -> chunking -> FAISS -> retrieval -> skill matching -> every analysis module).
"""
import hashlib
import io
import json
import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import rag.embeddings as emb
import rag.vector_store as vs


def fake_embed(texts):
    """Bag-of-words hashing embedding: similar words -> similar vectors, L2-normalised."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.strip(".,:;()").encode()).hexdigest(), 16) % 128] += 1.0
        n = np.linalg.norm(out[i])
        out[i] /= n if n else 1.0
    return out


emb.embed_texts = fake_embed
vs.embed_texts = fake_embed

import analysis.skill_gap as sg  # noqa: E402

sg.embed_texts = fake_embed

from analysis.chat import prepare_chat  # noqa: E402
from analysis.interview import generate_interview  # noqa: E402
from analysis.resume_analyzer import analyze_resume  # noqa: E402
from analysis.roadmap import generate_roadmap  # noqa: E402
from rag.chunker import chunk_text  # noqa: E402
from rag.document_loader import DocumentLoadError, load_document  # noqa: E402
from rag.retriever import CareerRetriever  # noqa: E402
from utils.helpers import JD_DIR, RESUME_DIR, extract_json, skill_in_text  # noqa: E402


class FakeLLM:
    model = "fake"

    def generate(self, prompt, system="", temperature=0.2, json_mode=False, max_tokens=None):
        if json_mode and "JOB DESCRIPTION excerpts" in prompt:
            names = ["Python", "SQL", "Git", "REST API", "DSA", "Problem Solving", "Docker", "Kubernetes (invented)"]
            return json.dumps({"skills": [{"name": n, "importance": "required"} for n in names]})
        if json_mode and "RESUME excerpts" in prompt:
            return json.dumps({"skills": ["Python", "Java", "SQL", "Streamlit", "Pandas", "Quantum Computing (invented)"]})
        if json_mode:
            return "not json at all"
        return "FAKE ANSWER"

    def stream(self, *a, **k):
        yield "ok"


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = load_document("r.txt", (RESUME_DIR / "sample_resume.txt").read_bytes())
        j = load_document("j.txt", (JD_DIR / "sample_job_description.txt").read_bytes())
        cls.resume, cls.jd = r, j
        cls.retriever = CareerRetriever()
        cls.retriever.build(chunk_text(r.text, "Resume"), chunk_text(j.text, "Job Description"))
        cls.result = sg.run_skill_gap(cls.retriever, FakeLLM(), r.text, j.text)

    def test_loader_errors(self):
        with self.assertRaises(DocumentLoadError):
            load_document("x.pdf", b"")
        with self.assertRaises(DocumentLoadError):
            load_document("x.pdf", b"this is not a pdf")
        with self.assertRaises(DocumentLoadError):
            load_document("x.txt", b"   \n  ")
        with self.assertRaises(DocumentLoadError):
            load_document("x.docx", b"abc")

    def test_pdf_roundtrip(self):
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest("pypdf missing")
        w = PdfWriter()
        w.add_blank_page(200, 200)
        buf = io.BytesIO()
        w.write(buf)
        with self.assertRaises(DocumentLoadError):  # blank page -> no extractable text
            load_document("blank.pdf", buf.getvalue())

    def test_index_counts(self):
        self.assertGreater(self.retriever.resume_chunk_count, 0)
        self.assertGreater(self.retriever.jd_chunk_count, 0)
        hits = self.retriever.retrieve("SQL database", "both", 3)
        self.assertTrue(hits and {h.source for h in hits} <= {"Resume", "Job Description"})

    def test_skill_text_matching(self):
        self.assertTrue(skill_in_text("Java", "I know Java and SQL"))
        self.assertFalse(skill_in_text("Java", "I know JavaScript"))
        self.assertTrue(skill_in_text("REST API", "built RESTful APIs"))
        self.assertFalse(skill_in_text("Docker", self.resume.text))

    def test_skill_gap_is_dynamic_and_grounded(self):
        by_name = {i.skill: i.status for i in self.result.items}
        self.assertEqual(by_name["Python"], "strong")
        self.assertEqual(by_name["SQL"], "strong")
        self.assertEqual(by_name["Docker"], "missing")
        self.assertNotIn("Kubernetes (invented)", by_name)  # not in the JD text -> filtered
        self.assertNotIn("Quantum Computing (invented)", self.result.resume_skills)
        self.assertTrue(0 < self.result.match_percent < 100)
        self.assertTrue(self.result.recommendations)  # fallback recs work even when LLM JSON fails
        for r in self.result.recommendations:
            self.assertIn(r["priority"], ("High", "Medium", "Low"))

    def test_other_modules(self):
        llm = FakeLLM()
        self.assertEqual(analyze_resume(self.retriever, llm, self.result).text, "FAKE ANSWER")
        self.assertEqual(generate_roadmap(self.retriever, llm, self.result, 4).text, "FAKE ANSWER")
        iv = generate_interview(self.retriever, llm, self.result)
        self.assertEqual(len(iv.sections), 4)
        system, prompt, chunks = prepare_chat(self.retriever, "What skills am I missing?", [], self.result)
        self.assertIn("QUESTION: What skills am I missing?", prompt)
        self.assertTrue(chunks)

    def test_extract_json(self):
        self.assertEqual(extract_json('```json\n{"a":1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! {"a": [1,2]} done'), {"a": [1, 2]})
        self.assertIsNone(extract_json("nope"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
