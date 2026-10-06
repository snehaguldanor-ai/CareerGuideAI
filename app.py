"""CareerGuide AI - Your Personalized AI Career Assistant (Streamlit entry point).

Run with:  streamlit run app.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")      # avoid OpenMP clash (faiss + torch) on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CareerGuide AI", page_icon="🚀", layout="wide")

from analysis.chat import SUGGESTED_QUESTIONS, prepare_chat
from analysis.common import AnalysisError
from analysis.interview import CATEGORIES, generate_interview
from analysis.resume_analyzer import analyze_resume
from analysis.roadmap import generate_roadmap
from analysis.skill_gap import run_skill_gap
from llm.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    LLMError,
    ModelNotFoundError,
    OllamaClient,
    OllamaUnavailableError,
)
from rag.chunker import chunk_text
from rag.document_loader import DocumentLoadError, load_document
from rag.embeddings import EmbeddingError, get_model
from rag.retriever import CareerRetriever
from rag.vector_store import VectorStoreError
from utils.helpers import JD_DIR, RESUME_DIR, SOURCE_JD, SOURCE_RESUME, file_hash, safe_md

PAGES = [
    "🏠 Dashboard",
    "📄 Documents",
    "📊 Skill Gap",
    "📝 Resume Improvement",
    "🗺️ Learning Roadmap",
    "🎤 Interview Prep",
    "💬 Career Chat",
]

st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
       padding: 1.4rem 1.8rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.05rem;}
.pill {display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.82rem; margin:.15rem .2rem .15rem 0;}
.pill-green {background:#dcfce7; color:#166534;}
.pill-yellow {background:#fef9c3; color:#854d0e;}
.pill-red {background:#fee2e2; color:#991b1b;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- state
def init_state() -> None:
    defaults = {
        "resume": None,
        "jd": None,
        "resume_hash": None,
        "jd_hash": None,
        "retriever": None,
        "skill_result": None,
        "resume_report": None,
        "roadmap": None,
        "interview": None,
        "chat": [],
        "queued_q": None,
        "nonce": 0,
        "model": DEFAULT_MODEL,
        "host": DEFAULT_HOST,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results() -> None:
    for key in ("skill_result", "resume_report", "roadmap", "interview"):
        st.session_state[key] = None
    st.session_state.chat = []


def get_llm() -> OllamaClient:
    return OllamaClient(model=st.session_state.model, host=st.session_state.host)


@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def load_embedder():
    return get_model()


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama(host: str, model: str) -> tuple[bool, str]:
    return OllamaClient(model=model, host=host).check()


def show_error(exc: Exception) -> None:
    """Friendly, non-crashing error messages."""
    if isinstance(exc, OllamaUnavailableError):
        st.error(f"🔌 **Ollama is not running.** {exc}")
    elif isinstance(exc, ModelNotFoundError):
        st.error(f"📦 **Model missing.** {exc}")
    elif isinstance(exc, LLMError):
        st.error(f"🤖 **LLM error.** {exc}")
    elif isinstance(exc, EmbeddingError):
        st.error(f"🧬 **Embedding error.** {exc}")
    elif isinstance(exc, VectorStoreError):
        st.error(f"🗂️ **Vector store (FAISS) error.** {exc}")
    elif isinstance(exc, AnalysisError):
        st.warning(f"⚠️ {exc}")
    elif isinstance(exc, DocumentLoadError):
        st.error(f"📄 {exc}")
    else:
        st.error(f"Something unexpected happened: {exc}")


# --------------------------------------------------------------------------- documents / index
def rebuild_index() -> None:
    st.session_state.retriever = None
    reset_results()
    resume, jd = st.session_state.resume, st.session_state.jd
    if not (resume and jd):
        return
    try:
        with st.spinner("Chunking, embedding and indexing both documents (FAISS)..."):
            load_embedder()
            retriever = CareerRetriever()
            retriever.build(
                chunk_text(resume.text, SOURCE_RESUME, resume.name),
                chunk_text(jd.text, SOURCE_JD, jd.name),
            )
        st.session_state.retriever = retriever
    except Exception as exc:  # noqa: BLE001 - never crash the app on index errors
        show_error(exc)


def handle_upload(kind: str, uploaded) -> bool:
    """kind is 'resume' or 'jd'. Returns True when a new file was loaded."""
    if uploaded is None:
        return False
    data = uploaded.getvalue()
    digest = file_hash(data)
    if st.session_state[f"{kind}_hash"] == digest:
        return False
    try:
        doc = load_document(uploaded.name, data)
    except DocumentLoadError as exc:
        show_error(exc)
        return False
    except Exception as exc:  # noqa: BLE001
        show_error(DocumentLoadError(f"Could not read '{uploaded.name}': {exc}"))
        return False
    st.session_state[kind] = doc
    st.session_state[f"{kind}_hash"] = digest
    return True


def load_sample_data() -> None:
    try:
        resume = load_document("sample_resume.txt", (RESUME_DIR / "sample_resume.txt").read_bytes())
        jd = load_document("sample_job_description.txt", (JD_DIR / "sample_job_description.txt").read_bytes())
    except (OSError, DocumentLoadError) as exc:
        st.sidebar.error(f"Could not load sample data: {exc}")
        return
    st.session_state.resume, st.session_state.jd = resume, jd
    st.session_state.resume_hash, st.session_state.jd_hash = "sample", "sample"
    rebuild_index()


def clear_all() -> None:
    for key in ("resume", "jd", "resume_hash", "jd_hash", "retriever"):
        st.session_state[key] = None
    reset_results()
    st.session_state.nonce += 1


def require_ready() -> bool:
    if not st.session_state.resume and not st.session_state.jd:
        st.warning("📄 Please upload a **resume** and a **job description** on the Documents page first.")
        return False
    if not st.session_state.resume:
        st.warning("📄 No resume uploaded yet. Add your resume on the Documents page.")
        return False
    if not st.session_state.jd:
        st.warning("💼 No job description uploaded yet. Add one on the Documents page.")
        return False
    if st.session_state.retriever is None or not st.session_state.retriever.is_ready:
        st.warning("⚙️ The document index is not ready. Re-upload the files or use the sample data.")
        return False
    return True


# --------------------------------------------------------------------------- shared UI
def hero() -> None:
    st.markdown(
        '<div class="hero"><h1>🚀 CareerGuide AI</h1><p>Your Personalized AI Career Assistant</p></div>',
        unsafe_allow_html=True,
    )


def render_sources(chunks, title: str = "📚 Retrieved Information", expanded: bool = True) -> None:
    with st.expander(f"{title} - {len(chunks)} chunks from FAISS", expanded=expanded):
        if not chunks:
            st.write("No relevant text was retrieved.")
            return
        for c in chunks:
            icon = "📄" if c.source == SOURCE_RESUME else "💼"
            st.markdown(f"**{icon} Source: {c.source}** &nbsp;·&nbsp; chunk #{c.chunk_id} &nbsp;·&nbsp; similarity `{c.score:.2f}`")
            st.text(c.text)


def ai_header() -> None:
    st.markdown("### 🤖 AI Analysis")


def ollama_ok() -> bool:
    ok, msg = check_ollama(st.session_state.host, st.session_state.model)
    if not ok:
        st.error(f"🔌 {msg}")
    return ok


def run_skill_analysis(extras: bool = False) -> None:
    if not require_ready() or not ollama_ok():
        return
    retriever, llm = st.session_state.retriever, get_llm()
    resume, jd = st.session_state.resume, st.session_state.jd
    try:
        with st.status("Analyzing your career profile...", expanded=True) as status:
            bar = st.progress(0.0)

            def progress(frac: float, msg: str) -> None:
                bar.progress(min(frac, 1.0), text=msg)

            st.session_state.skill_result = run_skill_gap(retriever, llm, resume.text, jd.text, progress)
            if extras:
                st.write("📝 Writing resume improvement report...")
                st.session_state.resume_report = analyze_resume(retriever, llm, st.session_state.skill_result)
                st.write("🗺️ Building learning roadmap...")
                st.session_state.roadmap = generate_roadmap(retriever, llm, st.session_state.skill_result, 4)
                st.write("🎤 Preparing interview questions...")
                st.session_state.interview = generate_interview(retriever, llm, st.session_state.skill_result)
            status.update(label="Analysis complete ✅", state="complete", expanded=False)
    except Exception as exc:  # noqa: BLE001
        show_error(exc)


def need_skills() -> bool:
    if st.session_state.skill_result is None:
        st.info("Run **🚀 Analyze My Career** on the Dashboard (or below) first so the skill gaps are known.")
        if st.button("🚀 Analyze My Career", key=f"analyze_{st.session_state.get('_page', '')}", type="primary"):
            run_skill_analysis()
            st.rerun()
        return False
    return True


# --------------------------------------------------------------------------- pages
def page_dashboard() -> None:
    hero()
    res, jd, ret, sk = (st.session_state[k] for k in ("resume", "jd", "retriever", "skill_result"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resume", "✅ Loaded" if res else "⏳ Missing", f"{res.pages} page(s)" if res else None)
    c2.metric("Job Description", "✅ Loaded" if jd else "⏳ Missing")
    c3.metric("Resume chunks", ret.resume_chunk_count if ret else 0)
    c4.metric("Job-description chunks", ret.jd_chunk_count if ret else 0)

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("🎯 Career Fit Overview")
        if sk:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Skill match", f"{sk.match_percent:.0f}%")
            m2.metric("✅ Matched", len(sk.strong))
            m3.metric("🟡 Partial", len(sk.partial))
            m4.metric("❌ Missing", len(sk.missing))
            st.progress(min(sk.match_percent / 100, 1.0), text=f"Overall skill match: {sk.match_percent:.1f}%")
            st.caption("Match % = (strong matches + 0.5 × partial matches) ÷ skills the job asks for.")
        else:
            st.info("No analysis yet. Upload both documents, then press the button below.")
        extras = st.checkbox("Also generate resume tips, roadmap and interview prep (slower)", value=False)
        if st.button("🚀 Analyze My Career", type="primary", use_container_width=True):
            run_skill_analysis(extras)
            st.rerun()
    with right:
        with st.container(border=True):
            st.markdown("**🔄 RAG pipeline**")
            st.markdown(
                "1. 📄 Upload & extract text\n2. 🧹 Clean text\n3. ✂️ Chunk\n4. 🧬 Embed (MiniLM)\n"
                "5. 🗂️ Index in FAISS\n6. 🔍 Retrieve top chunks\n7. 🦙 Ollama `llama3.2:3b`\n8. ✅ Grounded answer"
            )
    if sk:
        st.subheader("Skill snapshot")
        pills = "".join(f'<span class="pill pill-green">✅ {i.skill}</span>' for i in sk.strong)
        pills += "".join(f'<span class="pill pill-yellow">🟡 {i.skill}</span>' for i in sk.partial)
        pills += "".join(f'<span class="pill pill-red">❌ {i.skill}</span>' for i in sk.missing)
        st.markdown(pills, unsafe_allow_html=True)


def page_documents() -> None:
    hero()
    st.subheader("📄 Upload your documents")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 👤 Resume")
            up = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt"], key=f"resume_up_{st.session_state.nonce}")
    with col2:
        with st.container(border=True):
            st.markdown("#### 💼 Job Description")
            up2 = st.file_uploader("Job description (PDF or TXT)", type=["pdf", "txt"], key=f"jd_up_{st.session_state.nonce}")
    changed = handle_upload("resume", up)
    changed = handle_upload("jd", up2) or changed
    if changed:
        rebuild_index()

    c1, c2 = st.columns(2)
    for col, key, label in ((c1, "resume", "Resume"), (c2, "jd", "Job description")):
        doc = st.session_state[key]
        with col:
            if doc:
                st.success(f"✅ {label} uploaded successfully: **{doc.name}**")
                st.caption(f"Pages: {doc.pages} · Characters extracted: {len(doc.text):,}")
                with st.expander("Extracted text preview", expanded=True):
                    st.text(doc.text[:1500] + (" ..." if len(doc.text) > 1500 else ""))
            else:
                st.info(f"No {label.lower()} uploaded yet.")
    ret = st.session_state.retriever
    if ret and ret.is_ready:
        st.success(f"🗂️ FAISS index ready · {ret.resume_chunk_count} resume chunks · {ret.jd_chunk_count} job chunks")
    elif st.session_state.resume and st.session_state.jd:
        st.warning("Documents are loaded but the index failed to build. See the error above.")


def page_skill_gap() -> None:
    hero()
    st.subheader("📊 Skill Gap Analysis")
    if not require_ready() or not need_skills():
        return
    sk = st.session_state.skill_result
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skill match", f"{sk.match_percent:.0f}%")
    c2.metric("✅ Strong", len(sk.strong))
    c3.metric("🟡 Partial", len(sk.partial))
    c4.metric("❌ Missing", len(sk.missing))
    st.progress(min(sk.match_percent / 100, 1.0))

    render_sources(sk.sources)
    ai_header()
    col_a, col_b, col_c = st.columns(3)
    for col, title, items in (
        (col_a, "✅ Strong Match (skills you have)", sk.strong),
        (col_b, "🟡 Partial Match", sk.partial),
        (col_c, "❌ Missing Skills", sk.missing),
    ):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            if items:
                for i in items:
                    st.markdown(f"- {i.skill} _({i.importance})_")
            else:
                st.caption("None")

    st.markdown("#### 🎓 Recommended skills to learn (with priority)")
    if sk.recommendations:
        icons = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
        df = pd.DataFrame(
            [
                {
                    "Priority": icons.get(r["priority"], r["priority"]),
                    "Skill": r["skill"],
                    "Status": r["status"].capitalize(),
                    "Why": r["reason"],
                    "How to learn": r["how_to_learn"],
                }
                for r in sk.recommendations
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No gaps found - your resume covers every skill the job description asks for.")

    with st.expander("🔍 Evidence for every job skill"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Skill": i.skill, "Importance": i.importance, "Result": i.status.capitalize(), "Evidence from resume": i.evidence}
                    for i in sk.items
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Skills found on your resume"):
        st.write(", ".join(sk.resume_skills) if sk.resume_skills else "No skills could be detected on the resume.")


def page_resume_improvement() -> None:
    hero()
    st.subheader("📝 Resume Improvement")
    if not require_ready() or not need_skills():
        return
    if st.button("✨ Generate resume suggestions", type="primary") and ollama_ok():
        try:
            with st.spinner("Ollama is reviewing your resume against the job description..."):
                st.session_state.resume_report = analyze_resume(
                    st.session_state.retriever, get_llm(), st.session_state.skill_result
                )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
    out = st.session_state.resume_report
    if out:
        render_sources(out.sources)
        ai_header()
        st.markdown(safe_md(out.text))
        st.download_button("⬇️ Download report", out.text, file_name="resume_improvement.md")


def page_roadmap() -> None:
    hero()
    st.subheader("🗺️ Personalized Learning Roadmap")
    if not require_ready() or not need_skills():
        return
    weeks = st.slider("Roadmap length (weeks)", 2, 8, 4)
    if st.button("🗺️ Generate roadmap", type="primary") and ollama_ok():
        try:
            with st.spinner("Ollama is planning your roadmap..."):
                st.session_state.roadmap = generate_roadmap(
                    st.session_state.retriever, get_llm(), st.session_state.skill_result, weeks
                )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
    out = st.session_state.roadmap
    if out:
        render_sources(out.sources)
        ai_header()
        st.markdown(safe_md(out.text))
        st.download_button("⬇️ Download roadmap", out.text, file_name="learning_roadmap.md")


def page_interview() -> None:
    hero()
    st.subheader("🎤 Interview Preparation")
    if not require_ready() or not need_skills():
        return
    if st.button("🎤 Generate interview questions", type="primary") and ollama_ok():
        try:
            with st.status("Generating personalised questions (4 categories)...", expanded=True) as status:
                bar = st.progress(0.0)
                st.session_state.interview = generate_interview(
                    st.session_state.retriever,
                    get_llm(),
                    st.session_state.skill_result,
                    lambda f, m: bar.progress(min(f, 1.0), text=m),
                )
                status.update(label="Interview prep ready ✅", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
    res = st.session_state.interview
    if res:
        tabs = st.tabs([f"{CATEGORIES[n]['icon']} {n}" for n in res.sections])
        for tab, name in zip(tabs, res.sections):
            with tab:
                render_sources(res.sources.get(name, []), expanded=False)
                ai_header()
                st.markdown(safe_md(res.sections[name]))


def page_chat() -> None:
    hero()
    st.subheader("💬 Ask CareerGuide AI")
    if not require_ready():
        return
    st.caption("Every answer is generated after retrieving the most relevant resume and job-description chunks.")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True, on_click=lambda q=q: st.session_state.update(queued_q=q))

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(safe_md(msg["content"]))
            if msg.get("sources"):
                render_sources(msg["sources"], expanded=False)

    typed = st.chat_input("Ask about your skills, resume, projects or interview...")
    question = typed or st.session_state.queued_q
    st.session_state.queued_q = None
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    if not ollama_ok():
        return
    try:
        system, prompt, chunks = prepare_chat(
            st.session_state.retriever, question, st.session_state.chat, st.session_state.skill_result
        )
        with st.chat_message("assistant"):
            render_sources(chunks)
            ai_header()
            answer = st.write_stream(get_llm().stream(prompt, system))
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": answer, "sources": chunks})
    except Exception as exc:  # noqa: BLE001
        show_error(exc)


# --------------------------------------------------------------------------- main
def sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🚀 CareerGuide AI")
        st.caption("Your Personalized AI Career Assistant")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        ok, msg = check_ollama(st.session_state.host, st.session_state.model)
        (st.success if ok else st.error)(("🟢 " if ok else "🔴 ") + msg)
        if st.button("🔄 Re-check Ollama", use_container_width=True):
            check_ollama.clear()
            st.rerun()
        st.divider()
        st.markdown("**Progress**")
        st.markdown(
            f"{'✅' if st.session_state.resume else '⬜'} Resume  \n"
            f"{'✅' if st.session_state.jd else '⬜'} Job description  \n"
            f"{'✅' if st.session_state.retriever else '⬜'} FAISS index  \n"
            f"{'✅' if st.session_state.skill_result else '⬜'} Career analysis"
        )
        st.divider()
        if st.button("🧪 Load sample data", use_container_width=True):
            load_sample_data()
            st.rerun()
        if st.button("🗑️ Clear everything", use_container_width=True):
            clear_all()
            st.rerun()
        with st.expander("⚙️ Settings"):
            st.session_state.model = st.text_input("Ollama model", st.session_state.model)
            st.session_state.host = st.text_input("Ollama host", st.session_state.host)
    return page


def main() -> None:
    init_state()
    page = sidebar()
    st.session_state["_page"] = page
    {
        PAGES[0]: page_dashboard,
        PAGES[1]: page_documents,
        PAGES[2]: page_skill_gap,
        PAGES[3]: page_resume_improvement,
        PAGES[4]: page_roadmap,
        PAGES[5]: page_interview,
        PAGES[6]: page_chat,
    }[page]()


main()
