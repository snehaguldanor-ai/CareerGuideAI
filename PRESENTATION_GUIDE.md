# Presentation Guide & Viva Q&A – CareerGuide AI

## 1. Problem Statement
Students and fresh graduates cannot easily tell how well their resume fits a specific job. Generic chatbots give generic advice and may invent skills, degrees or experience.

## 2. Proposed Solution
A local AI assistant that reads the user's own resume and a job description, retrieves the relevant passages (RAG), and uses a local LLM to produce grounded skill-gap analysis, resume tips, a roadmap, interview questions and chat answers.

## 3. Objectives
- Compare a resume with a job description automatically
- Ground every answer in the uploaded documents
- Show the evidence (retrieved chunks) to build trust
- Run fully locally and free

## 4. Technology Stack
Python, Streamlit (UI), pypdf (PDF text), LangChain text splitter (chunking), Sentence-Transformers all-MiniLM-L6-v2 (embeddings), FAISS (vector search), Ollama + llama3.2:3b (local LLM).

## 5. System Architecture
UI (Streamlit) → document loader → cleaner → chunker → embedder → FAISS (2 indexes: resume, job) → retriever → prompt builder (rules + context) → Ollama → answer + evidence panel.

## 6. RAG Workflow
Document → text extraction → cleaning → chunking → embeddings → FAISS → similarity retrieval → relevant context → Ollama → final answer. The retrieval happens **before** generation, so the model answers from the documents instead of from memory.

## 7. How Ollama is used
Ollama runs llama3.2:3b on the laptop. The app sends it a system prompt (strict rules) plus the retrieved context and a task. JSON mode is used for skill extraction; streaming is used in chat.

## 8. How FAISS is used
Each chunk's embedding is stored in a FAISS `IndexFlatIP` index. Vectors are normalised, so inner product equals cosine similarity. A query is embedded and FAISS returns the nearest chunks.

## 9. How embeddings work
An embedding model converts text to a 384-number vector so that texts with similar meaning are close together. "REST API" and "building web services" end up near each other even without sharing words.

## 10. Key Features
Skill match (strong/partial/missing) with dynamic %, skill gap with priorities, roadmap, resume improvement, interview prep (4 categories × 6 questions), RAG chat, evidence panels, hallucination guard, error handling.

## 11. Advantages
Private (data stays on the laptop), free, grounded and explainable, works offline after the first model download, personalised to the actual documents.

## 12. Limitations
A 3B model can make mistakes or give shorter answers; scanned-image PDFs are not supported (no OCR); skill matching depends on how skills are written; thresholds for partial match are heuristic; responses take tens of seconds on a CPU laptop; English documents only.

## 13. Future Enhancements
OCR, DOCX support, multiple jobs ranking, ATS score, report export, saved history, re-ranking, larger models.

## 14. 2–3 minute demo script
1. (20s) "CareerGuide AI compares my resume with a job description using RAG and a local LLM." Show sidebar: Ollama green.
2. (20s) Click **Load sample data** (B.Tech CSE student vs Software Developer Intern). Show Documents: page count, text preview, FAISS index with chunk counts.
3. (30s) Dashboard → **🚀 Analyze My Career**. While it runs, say: "It retrieves chunks, Ollama extracts skills, embeddings and FAISS compare them."
4. (40s) Skill Gap page: point to **📚 Retrieved Information** (Resume + Job Description chunks with similarity), then **🤖 AI Analysis**: ✅ Python, SQL; ❌ Git, REST API, DSA. "Nothing is hard-coded; change the files and the result changes."
5. (30s) Learning Roadmap and Interview Prep: show weeks driven by missing skills; show project questions from the resume.
6. (30s) Career Chat: ask "What should I learn first?" and then "Do I have a PhD?" to show the "couldn't find enough information" behaviour.

> Tip: run the analysis once before presenting, so the first-time embedding model download and model warm-up are done.

---

# Viva Questions and Answers

1. **What is RAG?** Retrieval-Augmented Generation: first retrieve relevant text from our documents, then give it to the LLM so the answer is based on that text.
2. **Why did you use RAG?** Resumes and job descriptions are private and specific; the LLM has not seen them. RAG gives it the right facts and reduces made-up answers.
3. **Why Ollama?** It runs LLMs locally with a simple API: free, private and offline.
4. **Why llama3.2:3b?** It is small enough for a student laptop (about 2 GB) but good enough to follow instructions and produce JSON/markdown.
5. **What are embeddings?** Lists of numbers (384 here) that represent the meaning of text; similar meaning = nearby vectors.
6. **Why Sentence Transformers?** Easy, free, local library; all-MiniLM-L6-v2 is small and fast with good semantic quality.
7. **What is FAISS?** A library from Meta for fast similarity search over vectors.
8. **Why FAISS?** Fast, runs locally, no database server needed, simple Python API.
9. **What is chunking?** Splitting long text into small overlapping pieces (500 characters, 80 overlap) so retrieval returns focused passages.
10. **Why overlap in chunks?** So a sentence cut at a boundary still appears whole in one chunk.
11. **What is semantic search?** Searching by meaning using embeddings instead of exact keyword match.
12. **How does retrieval work?** The question is embedded, FAISS finds the chunks with highest cosine similarity, and the top-k are put in the prompt.
13. **What is cosine similarity here?** Vectors are normalised, so inner product (`IndexFlatIP`) equals cosine similarity.
14. **What is hallucination?** When an LLM states things that are not true or not in the data, such as inventing a certification.
15. **How does your project reduce hallucination?** (a) answers use retrieved context only, (b) the system prompt forbids inventing facts and requires the "couldn't find enough information" sentence, (c) extracted skills are kept only if they literally appear in the document, (d) general advice is labelled, (e) low temperature, (f) evidence is displayed.
16. **Why Streamlit?** Pure-Python UI with widgets, tabs, metrics and chat in a few lines; ideal for demos.
17. **What happens when a user uploads a PDF?** pypdf extracts text per page → cleaned → chunked → embedded → added to FAISS. Errors (corrupt, empty, scanned, encrypted) show a friendly message.
18. **How is skill matching performed?** Retrieve chunks → Ollama extracts skills → literal-presence guard → job skill is Strong if in the resume text, Partial if embeddings find related evidence, else Missing.
19. **How is the match percentage calculated?** (strong + 0.5 × partial) ÷ total job skills × 100, computed at runtime.
20. **Why use a local LLM?** Privacy of resumes, no cost, no internet needed, no API keys.
21. **What are the limitations?** Small model errors, no OCR, heuristic thresholds, slower on CPU, English only.
22. **Why two separate FAISS indexes?** So we can retrieve specifically from the resume or specifically from the job description and label the source.
23. **What is the role of the system prompt?** It sets strict rules: use only context, don't invent, label general advice, separate existing/suggested/missing skills.
24. **Why use JSON mode?** It makes the skill lists machine-readable so the app can classify and count them; a fallback handles bad JSON.
25. **What does temperature mean and what did you use?** Randomness of the output; low values (0–0.2) for extraction and analysis for consistency, a bit higher (0.3–0.4) for roadmap/interview creativity.
26. **How do you handle Ollama not running?** The client catches connection errors and shows "Ollama is not running – start it", and a missing model shows the `ollama pull` command; the app doesn't crash.
27. **Why cache the embedding model?** Loading it takes seconds; `st.cache_resource` loads it once per session.
28. **How would you improve it?** OCR, DOCX support, re-ranking, ATS scoring, bigger model, multi-job comparison.
29. **Difference between RAG and fine-tuning?** RAG adds knowledge at question time from documents; fine-tuning changes the model's weights. RAG is cheaper and works with private, changing data.
30. **Is any data sent to the cloud?** No, except downloading the models once; everything else runs locally.
