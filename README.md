# 🚀 CareerGuide AI – Personalized Career Assistant

A local, privacy-friendly career assistant that reads **your resume** and **a job description**, retrieves the most relevant passages with **RAG**, and uses a local **Ollama (llama3.2:3b)** model to give grounded career guidance. No paid APIs, no cloud.

## Problem statement
Students apply to jobs without knowing how their resume compares with a job description. Generic chatbots give generic advice and often invent skills or experience.

## Proposed solution
CareerGuide AI compares the uploaded resume with the uploaded job description. Every answer is built from retrieved chunks of those two documents, shown on screen as evidence, and the model is instructed (and the code is designed) not to invent anything.

## Features
- Resume upload (PDF/TXT) and job description upload (PDF/TXT) with page count and text preview
- 🚀 Analyze My Career: ✅ strong / 🟡 partial / ❌ missing skills, computed dynamically with an overall match %
- 📊 Skill Gap Analysis with recommended skills and High/Medium/Low priority
- 🗺️ Personalized learning roadmap (2–8 weeks)
- 📝 Resume improvement (keywords, weak descriptions, better bullets; existing vs suggested vs missing skills)
- 🎤 Interview prep: 6 questions + suggested answers in each of 4 categories
- 💬 Career chat with retrieval before every answer
- 📚 "Retrieved Information" evidence panel, then "🤖 AI Analysis", on every page
- Friendly error handling (bad/empty PDF, Ollama down, model missing, FAISS/embedding errors)

## Architecture
```
Upload (PDF/TXT)
  -> pypdf text extraction -> cleaning
  -> LangChain RecursiveCharacterTextSplitter (500 chars, 80 overlap)
  -> Sentence-Transformers all-MiniLM-L6-v2 embeddings (384-d, normalised)
  -> FAISS IndexFlatIP (one index for resume, one for job description)
  -> similarity retrieval (top-k chunks)
  -> prompt = rules + retrieved context + task
  -> Ollama llama3.2:3b -> grounded answer (shown with the evidence)
```

```
CareerGuideAI/
├── app.py                  Streamlit UI (7 pages)
├── requirements.txt
├── run.bat                 Windows launcher
├── data/resumes, data/job_descriptions   sample files
├── rag/        document_loader, chunker, embeddings, vector_store, retriever
├── llm/        ollama_client, prompts (hallucination guard)
├── analysis/   skill_gap, resume_analyzer, roadmap, interview, chat, common
├── utils/      helpers
└── tests/      test_smoke.py (offline tests with fake LLM/embedder)
```

## Technology stack
Python 3.11+, Streamlit, Ollama + llama3.2:3b, LangChain (core + text splitters), FAISS (faiss-cpu), Sentence-Transformers, pypdf.

## How skill matching works (not hard-coded)
1. Retrieve job-requirement chunks and resume chunks from FAISS.
2. Ollama extracts skill lists as JSON from those chunks.
3. **Guard:** a skill is kept only if it literally appears in the uploaded document.
4. A job skill is **Strong** if it appears in the resume text, **Partial** if embeddings find a related resume skill/chunk (thresholds in `analysis/skill_gap.py`), otherwise **Missing**.
5. Ollama assigns priorities/reasons for the gaps (a rule-based fallback is used if its JSON is unreadable).
6. Match % = (strong + 0.5 × partial) ÷ total job skills.

## Installation (Windows, PowerShell or CMD)
```
mkdir CareerGuideAI
cd CareerGuideAI
python --version
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In PowerShell, if activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.

## Ollama setup
```
ollama --version
ollama list
ollama pull llama3.2:3b
ollama run llama3.2:3b "Say hello"
```
Ollama usually runs in the background after installing. If not: `ollama serve` in a separate terminal.
The first launch of the app downloads `all-MiniLM-L6-v2` (~90 MB) once, so be online for that run; afterwards it works offline.

## Run
```
.venv\Scripts\activate
streamlit run app.py
```
Open http://localhost:8501.

## Example workflow
1. Click **🧪 Load sample data** in the sidebar (or upload your own files on 📄 Documents).
2. Go to 🏠 Dashboard and click **🚀 Analyze My Career**.
3. Open 📊 Skill Gap: show the *Retrieved Information* panel, then the *AI Analysis*.
4. Generate resume tips, roadmap and interview questions.
5. Ask the chatbot "What should I learn first?" and show the retrieved chunks.

## Tests
`python tests/test_smoke.py` runs offline tests (fake embedder + fake LLM) of the whole pipeline logic.

## Future enhancements
Support for multiple job descriptions, DOCX resumes, OCR for scanned PDFs, saved sessions, ATS score, exporting a PDF report, a cross-encoder re-ranker, and a larger local model.
