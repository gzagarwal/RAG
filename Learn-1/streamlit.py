# app.py
import os
import tempfile
import uuid
from typing import List

import streamlit as st

# --- LlamaIndex imports (OpenAI LLM + embeddings) ---
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

st.set_page_config(
    page_title="PDF Chat (LlamaIndex + OpenAI)", page_icon="📄", layout="wide"
)

# ----------------------------
# Sidebar: configuration
# ----------------------------
st.sidebar.title("⚙️ Settings")

# 1) OpenAI API key
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Starts with sk-... (stored only in memory for this session)",
)

# 2) Model choices
default_llm = "gpt-4o-mini"  # you can change to "gpt-4o" or "gpt-3.5-turbo" if needed
default_embed = "text-embedding-3-small"  # fast & cheap; or "text-embedding-3-large"
llm_model = st.sidebar.text_input("LLM model", value=default_llm)
embed_model = st.sidebar.text_input("Embedding model", value=default_embed)

chunk_size = st.sidebar.slider("Chunk size (characters)", 512, 4096, 2048, step=256)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 512, 128, step=16)

st.sidebar.caption(
    "Tip: Larger chunks capture more context but can be slower and cost more tokens; "
    "overlap helps preserve context between chunks."
)

# ----------------------------
# App header
# ----------------------------
st.title("📄 Chat with your PDFs (LlamaIndex + OpenAI)")
st.markdown(
    "Upload one or more PDFs, build an index with OpenAI embeddings, then chat with them.\n"
    "No data is persisted unless you choose to export it."
)

# ----------------------------
# File uploader
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDFs. Max size depends on your Streamlit config.",
)

# ----------------------------
# Session state helpers
# ----------------------------
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": str}

if "index_id" not in st.session_state:
    st.session_state.index_id = None  # UUID for this index session (used for temp dir)


def require_api_key():
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        return False
    os.environ["OPENAI_API_KEY"] = api_key
    return True


def build_index_from_pdfs(
    pdf_files: List[st.runtime.uploaded_file_manager.UploadedFile],
):
    """Save uploaded PDFs to a temp dir, read them with LlamaIndex, then build a VectorStoreIndex."""
    # Configure LlamaIndex global settings
    Settings.llm = OpenAI(model=llm_model, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=embed_model)
    # Configure text splitting (node parser) globally
    from llama_index.core.node_parser import SentenceSplitter

    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Create a unique temp directory for this session's PDFs
    session_dir = tempfile.mkdtemp(prefix=f"pdf_chat_{uuid.uuid4().hex}_")
    saved_paths = []
    for f in pdf_files:
        dest = os.path.join(session_dir, f.name)
        with open(dest, "wb") as out:
            out.write(f.getvalue())
        saved_paths.append(dest)

    # Read PDFs
    docs = SimpleDirectoryReader(input_files=saved_paths).load_data()

    # Build index (in-memory)
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    return index


def ensure_chat_engine(index: VectorStoreIndex):
    """Create a chat engine from the index."""
    # "condense_question" helps when the user follows up with pronouns or shorthand
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=False,
        streaming=True,  # we'll stream tokens to the UI
    )
    return chat_engine


# ----------------------------
# Build index button
# ----------------------------
col_left, col_right = st.columns([2, 1])
with col_left:
    build_clicked = st.button(
        "🔧 Build / Rebuild Index", type="primary", use_container_width=True
    )
with col_right:
    clear_clicked = st.button("🧹 Clear Chat", use_container_width=True)

if clear_clicked:
    st.session_state.messages = []

if build_clicked:
    if not require_api_key():
        st.stop()
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
        st.stop()
    with st.spinner("Indexing PDFs… this may take a moment for big files."):
        index = build_index_from_pdfs(uploaded_files)
        st.session_state.chat_engine = ensure_chat_engine(index)
        st.session_state.index_id = uuid.uuid4().hex
    st.success("Index ready! Scroll down to start chatting.")

# ----------------------------
# Chat UI
# ----------------------------
st.subheader("💬 Chat")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your PDFs…")


def stream_answer(user_text: str):
    """Yield tokens from LlamaIndex stream for Streamlit's write_stream."""
    # Ask the chat engine
    response = st.session_state.chat_engine.stream_chat(user_text)
    # Stream tokens to UI
    for token in response.response_gen:
        yield token


if prompt:
    if not require_api_key():
        st.stop()
    if st.session_state.chat_engine is None:
        st.warning("Please upload PDFs and click **Build / Rebuild Index** first.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant streaming response
    with st.chat_message("assistant"):
        try:
            streamed_text = st.write_stream(stream_answer(prompt))
            # Save the final assistant message to history
            st.session_state.messages.append(
                {"role": "assistant", "content": streamed_text}
            )
        except Exception as e:
            st.error(f"Error during chat: {e}")
