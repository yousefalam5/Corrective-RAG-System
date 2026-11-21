import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crag_engine import CRAGConfig, CRAGEngine

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "sample_docs"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"

st.set_page_config(
    page_title="Corrective RAG (CRAG) System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .badge-correct { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-weak { background-color: #FEF08A; color: #854D0E; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-incorrect { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .rewrite-box { background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 12px; border-radius: 4px; margin: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing CRAG Engine & Sentence Transformer...")
def get_crag_engine() -> CRAGEngine:
    config = CRAGConfig(
        data_dir=DATA_DIR,
        chroma_dir=CHROMA_DIR,
        relevance_threshold=0.35,
    )
    return CRAGEngine(config)


engine = get_crag_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("CRAG Control Center")
    st.caption("Corrective Retrieval-Augmented Generation")

    st.markdown("---")
    st.subheader("1. LLM Provider")
    llm_choice = st.radio(
        "Inference Engine",
        ["Fast Semantic Mode (Local)", "Google Gemini API"],
        help="Gemini API provides LLM-based query rewriting and full answer generation.",
    )

    if llm_choice == "Google Gemini API":
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Free key from Google AI Studio (aistudio.google.com)",
        )
        if api_key:
            success = engine.update_gemini_key(api_key)
            if success:
                st.success("Gemini API connected! ✅")
            else:
                st.error("Failed to initialize Gemini. Check your key.")
    else:
        engine.update_gemini_key("")

    st.markdown("---")
    st.subheader("2. Pipeline Parameters")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=2, max_value=8, value=4)
    rel_thresh = st.slider(
        "Relevance Threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.35,
        step=0.05,
        help="Chunks scoring below this threshold are marked IRRELEVANT and trigger Query Rewriting.",
    )
    engine.config.relevance_threshold = rel_thresh

    st.markdown("---")
    st.subheader("3. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, PPTX, TXT, CSV)",
        type=["pdf", "docx", "pptx", "txt", "csv"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Uploads", type="primary", use_container_width=True):
            if uploaded_files:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for file in uploaded_files:
                    target_path = DATA_DIR / file.name
                    target_path.write_bytes(file.read())
                    saved_paths.append(target_path)
                with st.spinner("Extracting, chunking & embedding..."):
                    report = engine.ingest_files(saved_paths)
                st.success(f"Indexed {report['chunks']} chunks from {report['files']} files.")
                st.rerun()
            else:
                st.warning("Please upload files first.")

    with col2:
        if st.button("Load Samples", use_container_width=True):
            with st.spinner("Ingesting sample academic documents..."):
                report = engine.ingest_files()
            st.success(f"Indexed {report['chunks']} sample chunks.")
            st.rerun()

    # Database Metric
    chunk_count = engine.collection.count()
    st.metric("Indexed Knowledge Chunks", chunk_count)
    if chunk_count == 0:
        st.info("Click 'Load Samples' or upload files to populate the database.")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">🛡️ Corrective RAG (CRAG) System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluates retrieval quality, detects weak context, '
    'rewrites queries dynamically, and generates hallucination-free answers.</div>',
    unsafe_allow_html=True,
)

# Quick Questions
st.markdown("**💡 Try these test queries:**")
q_cols = st.columns(4)
sample_q = ""
with q_cols[0]:
    if st.button("CAP Theorem (Direct)", use_container_width=True):
        sample_q = "Explain the CAP theorem and the trade-off between consistency and availability in distributed systems."
with q_cols[1]:
    if st.button("Neural Nets (Vague)", use_container_width=True):
        sample_q = "Can you tell me about those deep neural network layer things and activation functions?"
with q_cols[2]:
    if st.button("Solar Energy (Direct)", use_container_width=True):
        sample_q = "What is the difference between monocrystalline and polycrystalline solar cells?"
with q_cols[3]:
    if st.button("Out-of-Domain (Refusal)", use_container_width=True):
        sample_q = "What were the major naval tactics used in the Battle of Trafalgar in 1805?"

# Query Input
with st.form("crag_form"):
    user_query = st.text_area(
        "Enter your question:",
        value=sample_q,
        height=90,
        placeholder="Ask anything about the ingested documents in English or Arabic...",
    )
    submit_btn = st.form_submit_button("Run CRAG Pipeline 🚀", type="primary")

if submit_btn and user_query.strip():
    if engine.collection.count() == 0:
        st.error("The knowledge base is empty! Please click 'Load Samples' in the sidebar first.")
    else:
        with st.spinner("Executing Corrective RAG pipeline (Retrieval ➡️ Evaluation ➡️ Rewriting ➡️ Generation)..."):
            trace = engine.run_crag_pipeline(user_query, top_k=top_k)

        st.markdown("---")

        # --- STEP 1 & 2: RETRIEVAL EVALUATION VERDICT ---
        st.subheader("1. Retrieval Quality Evaluation")
        v_col1, v_col2, v_col3 = st.columns([1.5, 2, 2])

        with v_col1:
            verdict = trace["retrieval_grade"]
            if verdict == "CORRECT":
                st.markdown('<span class="badge-correct">🟢 VERDICT: CORRECT</span>', unsafe_allow_html=True)
                st.caption("Retrieved passages have high confidence and relevance.")
            elif verdict == "WEAK":
                st.markdown('<span class="badge-weak">🟡 VERDICT: WEAK CONTEXT</span>', unsafe_allow_html=True)
                st.caption("Context is marginal. Query Rewriter was triggered.")
            else:
                st.markdown('<span class="badge-incorrect">🔴 VERDICT: INCORRECT / IRRELEVANT</span>', unsafe_allow_html=True)
                st.caption("Initial retrieval had zero relevant passages.")

        with v_col2:
            st.metric("Initial Confidence Score", f"{trace['initial_avg_score']:.3f}", delta=f"Threshold: {rel_thresh}")

        with v_col3:
            st.metric("Verified Passages Used", len(trace["verified_chunks"]))

        # --- STEP 3: QUERY REWRITER TRACE (IF TRIGGERED) ---
        if trace["was_rewritten"]:
            st.subheader("2. 🔄 Query Correction & Re-Retrieval")
            st.markdown(
                f"""
                <div class="rewrite-box">
                    <b>Original Query:</b> <i>"{trace['original_query']}"</i><br>
                    <b>Rewritten Query:</b> <code style="color: #16A34A; font-weight: bold;">"{trace['rewritten_query']}"</code><br>
                    <small>The query was reformulated with domain keywords to improve semantic search precision.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Inspect Chunks Expander
        with st.expander("🔍 Inspect Graded Chunks & Relevance Breakdown"):
            st.markdown("**Initial Retrieval Chunks:**")
            for i, chunk in enumerate(trace["initial_chunks"], start=1):
                st.markdown(
                    f"**Chunk #{i}** | Source: `{chunk['metadata'].get('source')}` | "
                    f"Similarity: `{chunk['similarity']:.3f}` | Grade: `{chunk['grade_score']:.3f}` | "
                    f"Status: **{chunk['verdict']}**"
                )
                st.caption(f"_{chunk['text']}_")
                st.divider()

        # --- STEP 4: GROUNDED ANSWER ---
        st.subheader("3. 📝 Grounded Answer")
        if trace["status"] == "UNANSWERED":
            st.warning(trace["answer"])
        else:
            st.markdown(trace["answer"])

        # --- STEP 5: SOURCES & ATTRIBUTIONS ---
        if trace["sources"]:
            st.subheader("4. 📚 Verified Source References")
            src_df = pd.DataFrame(trace["sources"])
            st.dataframe(
                src_df,
                column_config={
                    "source": "Document Source",
                    "page": "Page / Slide",
                    "similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.2f"),
                    "grade_score": st.column_config.ProgressColumn("Evaluation Score", min_value=0.0, max_value=1.0, format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
            )
