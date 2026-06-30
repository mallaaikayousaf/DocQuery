"""
RAG Application — PDF Question Answering with Streamlit
Upload PDFs, ask questions, and get AI-generated answers grounded in your documents.
"""

import time
import streamlit as st

from src.config import API_KEY, EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, VECTOR_DB_PATH
from src.pdf_processor import extract_text_from_bytes, validate_pdf
from src.chunker import create_chunks, get_chunk_statistics
from src.embeddings import load_embedding_model, generate_embeddings
from src.vector_store import create_chroma_client, create_collection, store_embeddings, get_collection_stats
from src.generator import configure_groq, generate_answer
from src.retriever import Retriever
from utils.helpers import format_file_size, truncate_text, format_elapsed_time


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3B3486 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        line-height: 1.3;
        overflow: visible;
        display: block;
        width: 100%;
    }
    .sub-header {
        color: #8B8FA3;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #1A1D29;
        border: 1px solid #2D3250;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8B8FA3;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2D3250;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2D3250;
        border-radius: 12px;
        padding: 8px;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #6C63FF;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: #8B8FA3;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ready {
        background: rgba(108, 99, 255, 0.15);
        color: #6C63FF;
    }
    .badge-warning {
        background: rgba(255, 165, 0, 0.15);
        color: #FFA500;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "groq_client" not in st.session_state:
    st.session_state.groq_client = None

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None

if "collection" not in st.session_state:
    st.session_state.collection = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ── Model & Client Loading ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load and cache the embedding model."""
    return load_embedding_model(EMBEDDING_MODEL)


def init_services():
    """Initialize all services (embedding model, Groq client, ChromaDB)."""
    # Embedding model
    if st.session_state.embedding_model is None:
        with st.spinner("🔄 Loading embedding model..."):
            st.session_state.embedding_model = get_embedding_model()

    # Groq client
    if st.session_state.groq_client is None and API_KEY:
        st.session_state.groq_client = configure_groq(API_KEY)

    # ChromaDB
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client = create_chroma_client(VECTOR_DB_PATH)
        st.session_state.collection = create_collection(
            st.session_state.chroma_client, "pdf_chunks"
        )

    # Retriever
    if (st.session_state.retriever is None
            and st.session_state.embedding_model is not None
            and st.session_state.collection is not None):
        st.session_state.retriever = Retriever(
            st.session_state.embedding_model,
            st.session_state.collection
        )


# ── PDF Processing Pipeline ─────────────────────────────────────────────────

def process_pdf(uploaded_file) -> dict:
    """
    Full pipeline: validate → extract text → chunk → embed → store.
    Returns a dict with processing stats.
    """
    stats = {}
    file_bytes = uploaded_file.read()
    stats['file_name'] = uploaded_file.name
    stats['file_size'] = len(file_bytes)

    # Step 1: Validate
    if not validate_pdf(file_bytes):
        raise ValueError(f"'{uploaded_file.name}' is not a valid PDF file.")

    # Step 2: Extract text
    t0 = time.time()
    text = extract_text_from_bytes(file_bytes)
    stats['extract_time'] = time.time() - t0
    stats['char_count'] = len(text)

    if not text.strip():
        raise ValueError(
            f"No readable text found in '{uploaded_file.name}'. "
            "It may be a scanned document."
        )

    # Step 3: Chunk
    t0 = time.time()
    chunks = create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    stats['chunk_time'] = time.time() - t0
    stats['chunk_stats'] = get_chunk_statistics(chunks)

    # Step 4: Embed
    t0 = time.time()
    embeddings = generate_embeddings(
        chunks, st.session_state.embedding_model, show_progress=False
    )
    stats['embed_time'] = time.time() - t0

    # Step 5: Store
    metadatas = [{"source": uploaded_file.name, "chunk_index": i} for i in range(len(chunks))]
    t0 = time.time()
    store_embeddings(
        st.session_state.collection,
        chunks,
        embeddings.tolist(),
        metadatas=metadatas
    )
    stats['store_time'] = time.time() - t0

    stats['total_time'] = (
        stats['extract_time'] + stats['chunk_time']
        + stats['embed_time'] + stats['store_time']
    )

    return stats


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Document Manager")
    st.markdown("---")

    # PDF Upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base."
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            for uploaded_file in uploaded_files:
                try:
                    with st.status(f"Processing **{uploaded_file.name}**...", expanded=True) as status:
                        st.write("🔍 Validating PDF...")
                        time.sleep(0.2)
                        st.write("📝 Extracting text...")
                        stats = process_pdf(uploaded_file)

                        st.write(f"✂️ Created **{stats['chunk_stats']['count']}** chunks")
                        st.write(f"🧠 Generated embeddings")
                        st.write(f"💾 Stored in vector database")
                        status.update(
                            label=f"✅ **{uploaded_file.name}** processed!",
                            state="complete"
                        )

                    st.session_state.documents_processed += 1
                    st.session_state.total_chunks += stats['chunk_stats']['count']

                    # Show stats
                    with st.expander(f"📊 Stats for {uploaded_file.name}"):
                        col1, col2 = st.columns(2)
                        col1.metric("Chunks", stats['chunk_stats']['count'])
                        col2.metric("Characters", f"{stats['char_count']:,}")
                        col1.metric("File Size", format_file_size(stats['file_size']))
                        col2.metric("Total Time", format_elapsed_time(stats['total_time']))

                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    st.markdown("---")

    # Knowledge Base Stats
    st.markdown("### 📊 Knowledge Base")
    if st.session_state.collection is not None:
        try:
            col_stats = get_collection_stats(st.session_state.collection)
            col1, col2 = st.columns(2)
            col1.metric("Total Chunks", col_stats['count'])
            col2.metric("Documents", st.session_state.documents_processed)
        except Exception:
            st.info("No documents in knowledge base yet.")
    else:
        st.info("Knowledge base not initialized.")

    st.markdown("---")

    # API Status
    st.markdown("### ⚙️ System Status")
    if API_KEY:
        st.markdown(
            '<span class="status-badge badge-ready">🟢 Groq API Connected</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-badge badge-warning">🟡 API Key Missing</span>',
            unsafe_allow_html=True
        )
        st.caption("Add `API_KEY` to your `.env` file to enable Q&A.")

    model_status = "Loaded" if st.session_state.embedding_model else "Not loaded"
    st.caption(f"Embedding Model: `{EMBEDDING_MODEL}` — {model_status}")
    st.caption(f"LLM: `{LLM_MODEL}`")

    st.markdown("---")

    # Clear Database
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        try:
            if st.session_state.chroma_client:
                st.session_state.chroma_client.delete_collection("pdf_chunks")
                st.session_state.collection = create_collection(
                    st.session_state.chroma_client, "pdf_chunks"
                )
                st.session_state.retriever = Retriever(
                    st.session_state.embedding_model,
                    st.session_state.collection
                )
                st.session_state.documents_processed = 0
                st.session_state.total_chunks = 0
                st.session_state.chat_history = []
                st.success("Knowledge base cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {e}")


# ── Main Area ────────────────────────────────────────────────────────────────

# Initialize services
init_services()

# Header
st.markdown('<p class="main-header">RAG — PDF Question Answering</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Upload your PDF documents and ask questions — answers are grounded in your data.'
    '</p>',
    unsafe_allow_html=True
)

# Check readiness
if not API_KEY:
    st.warning(
        "⚠️ **Groq API key not configured.** "
        "Add `API_KEY=your_groq_key` to the `.env` file and restart the app."
    )

if st.session_state.collection is not None:
    try:
        col_stats = get_collection_stats(st.session_state.collection)
        if col_stats['count'] == 0:
            st.info(
                "👋 **Getting started:** Upload a PDF in the sidebar, "
                "then ask questions about it here."
            )
    except Exception:
        pass

# Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("📚 Retrieved Context", expanded=False):
                for i, chunk in enumerate(message["context"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(truncate_text(chunk, 500))
                    if i < len(message["context"]):
                        st.markdown("---")
        if "time" in message:
            st.caption(f"⏱️ {format_elapsed_time(message['time'])}")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not API_KEY:
            response_text = (
                "❌ I can't generate answers without a Groq API key. "
                "Please add `API_KEY` to your `.env` file."
            )
            st.markdown(response_text)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text
            })

        elif st.session_state.retriever is None:
            response_text = (
                "⚠️ The system is still initializing. Please wait a moment and try again."
            )
            st.markdown(response_text)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text
            })

        else:
            try:
                t0 = time.time()

                # Retrieve
                with st.spinner("🔍 Searching documents..."):
                    retrieved_chunks = st.session_state.retriever.retrieve(
                        prompt, top_k=TOP_K
                    )

                if not retrieved_chunks:
                    response_text = (
                        "🤔 I couldn't find any relevant information in the uploaded documents. "
                        "Try uploading more documents or rephrasing your question."
                    )
                    st.markdown(response_text)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                else:
                    # Generate
                    with st.spinner("💡 Generating answer..."):
                        response_text = generate_answer(
                            st.session_state.groq_client,
                            prompt,
                            retrieved_chunks,
                            model_name=LLM_MODEL
                        )

                    elapsed = time.time() - t0

                    st.markdown(response_text)

                    # Show context
                    with st.expander("📚 Retrieved Context", expanded=False):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.caption(truncate_text(chunk, 500))
                            if i < len(retrieved_chunks):
                                st.markdown("---")

                    st.caption(f"⏱️ {format_elapsed_time(elapsed)}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "context": retrieved_chunks,
                        "time": elapsed
                    })

            except Exception as e:
                error_msg = f"❌ An error occurred: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
