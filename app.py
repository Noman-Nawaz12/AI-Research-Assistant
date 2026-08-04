"""
app.py
-------
Main Streamlit interface for the AI Research Assistant
(Multi-Source RAG with Dual Groq Model Architecture)
"""

import streamlit as st
import time

from document_processor import process_multiple_documents
from chunking import chunk_documents
from vector_store import VectorStore
from llm_pipeline import run_dual_llm_pipeline, LLMPipelineError
from memory import ConversationMemory
from utils import show_error, show_warning, show_success, show_info, validate_api_key


# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — makes the interface attractive instead of plain default Streamlit
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Overall app background ---------- */
    .stApp {
        background: linear-gradient(180deg, #f8f9fd 0%, #f0f2fc 100%);
    }

    /* ---------- Main title banner (subtle block, black text) ---------- */
    .main-title {
        text-align: center;
        padding: 24px 28px;
        margin-bottom: 20px;
        background: #F0F5FF;
        border: 1.5px solid #A9CDF7;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(91, 79, 224, 0.08);
    }
    .main-title h1 {
        font-size: 2.3rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: 0.3px;
        color: #1f1f2e;
    }
    .main-title p {
        margin: 8px 0 0 0;
        font-size: 0.98rem;
        color: #6b6680;
        font-weight: 500;
    }

    /* ---------- Section cards ---------- */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(31, 25, 90, 0.05);
        margin-bottom: 16px;
        border: 1px solid #e9e8f7;
    }

    /* ---------- Chat bubbles ---------- */
    .chat-user {
        background: linear-gradient(135deg, #5B4FE0, #7C6CF0);
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(91,79,224,0.22);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-assistant {
        background: #ffffff;
        color: #262730;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #e9e8f7;
        box-shadow: 0 3px 10px rgba(31,25,90,0.05);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ---------- Sidebar background only — do NOT force color on every child ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #241F4E 0%, #372F80 100%);
        border-right: none;
    }

    /* Sidebar plain text (markdown/headings/captions) -> light color */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stCaption {
        color: #EDEBFF !important;
    }

    /* Sidebar INPUT widgets keep normal dark-on-white/native styling (fixes invisible text) */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        color: #1f1f2e !important;
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* File uploader box — force readable text/icons regardless of parent rule */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1.5px dashed #B7ADF5 !important;
        border-radius: 14px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
        color: #3a3160 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: #6C5CE7 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background-color: #6C5CE7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {
        color: #ffffff !important;
    }

    /* Sliders / number inputs value labels */
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] [data-baseweb="slider"] div {
        color: #EDEBFF !important;
    }

    /* Sidebar horizontal rule */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* ---------- Buttons (main area + sidebar action buttons) ---------- */
    .stButton>button {
        background: linear-gradient(90deg, #5B4FE0, #7C6CF0);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 9px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 3px 10px rgba(91,79,224,0.25);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(91,79,224,0.4);
    }
    .stButton>button p {
        color: #ffffff !important;
    }

    /* ---------- Badge / pill for source references ---------- */
    .source-pill {
        display: inline-block;
        background: #EDEBFF;
        color: #4B3FBF;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 3px 5px 3px 0;
        font-weight: 600;
        border: 1px solid #DAD5FA;
    }

    /* ---------- Divider ---------- */
    .pretty-divider {
        height: 3px;
        background: linear-gradient(90deg, #6C5CE7, #A78BFA, transparent);
        border-radius: 3px;
        margin: 18px 0;
    }

    /* ---------- Expander headers in main area ---------- */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [{"role", "content", "sources": [...]}, ...]
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "last_condensed_context" not in st.session_state:
    st.session_state.last_condensed_context = ""


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="main-title">
    <h1>🔬 AI Research Assistant</h1>
    <p>✨ Multi-Source Retrieval-Augmented Generation • Powered by Dual Groq LLMs ⚡</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.markdown("## 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("## ⚙️ Retrieval Settings")
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=6)
    chunk_size = st.number_input("Chunk Size", min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=150, step=50)

    st.markdown("---")
    process_btn = st.button("🚀 Process Documents", use_container_width=True)

    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.memory = ConversationMemory()
        st.session_state.chat_display = []
        st.session_state.documents_processed = False
        st.session_state.last_retrieved_chunks = []
        st.session_state.last_condensed_context = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.caption("Streamlit • Groq (Mixtral 8x7B + Llama 3.1 8B) • LangChain • FAISS • Sentence Transformers")


# ----------------------------------------------------------------------------
# DOCUMENT PROCESSING
# ----------------------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        show_warning("Please upload at least one document (PDF, DOCX, or TXT).")
    else:
        with st.spinner("📄 Extracting text from documents..."):
            extracted_data, errors = process_multiple_documents(uploaded_files)

        for err in errors:
            show_error(err)

        if extracted_data:
            with st.spinner("✂️ Chunking documents..."):
                chunks = chunk_documents(extracted_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            with st.spinner("🧬 Generating embeddings & building FAISS index..."):
                vs = VectorStore()
                vs.build_index(chunks)
                st.session_state.vector_store = vs
                st.session_state.documents_processed = True

            show_success(f"Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks. Ready to chat! 🎉")
        elif not errors:
            show_warning("No text could be extracted from the uploaded documents.")


# ----------------------------------------------------------------------------
# MAIN LAYOUT — Chat + Expandable Info Panels
# ----------------------------------------------------------------------------
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.markdown("### 💬 Chat")

    if not st.session_state.documents_processed:
        show_info("Upload documents and click **Process Documents** in the sidebar to get started.")

    # Render chat history
    for turn in st.session_state.chat_display:
        if turn["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {turn["content"]}</div>', unsafe_allow_html=True)
            if turn.get("sources"):
                pills = "".join(
                    f'<span class="source-pill">📄 {s}</span>' for s in turn["sources"]
                )
                st.markdown(pills, unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        if not st.session_state.documents_processed:
            show_warning("Please process your documents first.")
        elif not validate_api_key(groq_api_key):
            show_error("Please enter a valid Groq API key in the sidebar.")
        else:
            st.session_state.chat_display.append({"role": "user", "content": question})
            st.session_state.memory.add_turn("user", question)

            with st.spinner("🔍 Retrieving relevant chunks..."):
                retrieved = st.session_state.vector_store.search(question, top_k=top_k)
                st.session_state.last_retrieved_chunks = retrieved

            try:
                with st.spinner("🧩 Mixtral summarizing context..."):
                    time.sleep(0.2)  # tiny UX pause so spinner is visible
                    result = run_dual_llm_pipeline(
                        api_key=groq_api_key,
                        retrieved_chunks=retrieved,
                        conversation_history_summary=st.session_state.memory.get_summary(),
                        current_question=question
                    )

                st.session_state.last_condensed_context = result["condensed_context"]
                st.session_state.memory.update_summary(result["condensed_context"])
                st.session_state.memory.add_turn("assistant", result["final_answer"])

                sources = sorted(set(f'{c["source"]} (p.{c["page"]})' for c in retrieved))
                st.session_state.chat_display.append({
                    "role": "assistant",
                    "content": result["final_answer"],
                    "sources": sources
                })

            except LLMPipelineError as e:
                show_error(str(e))
            except Exception as e:
                show_error(f"Unexpected error: {str(e)}")

            st.rerun()

with col_info:
    st.markdown("### 📊 Response Dashboard")

    with st.expander("📚 Retrieved Chunks", expanded=False):
        if st.session_state.last_retrieved_chunks:
            for c in st.session_state.last_retrieved_chunks:
                st.markdown(f"**📄 {c['source']}** — page {c['page']} (score: {c['score']:.3f})")
                st.caption(c["chunk_text"][:300] + ("..." if len(c["chunk_text"]) > 300 else ""))
                st.markdown('<div class="pretty-divider"></div>', unsafe_allow_html=True)
        else:
            st.caption("No chunks retrieved yet.")

    with st.expander("🧩 Summarized Context (Model 1 Output)", expanded=False):
        if st.session_state.last_condensed_context:
            st.write(st.session_state.last_condensed_context)
        else:
            st.caption("No summary generated yet.")

    with st.expander("🕓 Full Chat History", expanded=False):
        history_text = st.session_state.memory.get_full_history_text()
        if history_text:
            st.text(history_text)
        else:
            st.caption("No conversation yet.")

    st.markdown("---")
    if st.session_state.vector_store and st.session_state.vector_store.is_ready():
        st.success(f"✅ Vector DB ready — {len(st.session_state.vector_store.chunks)} chunks indexed")
    else:
        st.caption("Vector DB not built yet.")