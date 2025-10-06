# 📄 PDF Chat (LlamaIndex + OpenAI)

Chat with one or more PDFs locally using **Streamlit**, **LlamaIndex**, and **OpenAI**. Upload PDFs, build an embedding index, and ask natural‑language questions with streaming answers.

---

## 🚀 What this app does

* Ingests uploaded **PDFs** into memory
* Splits text into chunks and creates **OpenAI embeddings**
* Builds an in‑memory **VectorStoreIndex** (LlamaIndex)
* Exposes a **chat UI** where your questions are converted into retrieval queries
* Streams answers back into the chat

> **Note:** By default, nothing is persisted to disk; the index is rebuilt per session unless you add the persistence enhancements below.

---

## 🗺️ High‑level flow

1. **Upload PDFs → Temp Dir**
2. **Load documents** with `SimpleDirectoryReader`
3. **Chunk** via `SentenceSplitter` (size + overlap from UI)
4. **Embed** with `OpenAIEmbedding`
5. **Index** with `VectorStoreIndex.from_documents()`
6. **Chat** via `index.as_chat_engine(chat_mode="condense_question")`
7. **Stream** tokens to Streamlit UI

---

## 📦 Requirements

* Python 3.10+
* Streamlit
* LlamaIndex core + OpenAI integrations
* OpenAI API key

### `requirements.txt`

```txt
streamlit>=1.33.0
llama-index>=0.10.50
llama-index-core>=0.10.50
llama-index-llms-openai>=0.1.25
llama-index-embeddings-openai>=0.1.10
pydantic<3
```

> Pin versions as needed; LlamaIndex evolves quickly.

---

## 🔧 Setup

1. **Clone / copy** this repo and place your `app.py` at the root.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install deps:

   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Use env file:

   ```bash
   echo "OPENAI_API_KEY=sk-..." > .env
   ```

   Or paste your key into the sidebar at runtime.

---

## ▶️ Run

```bash
streamlit run app.py
```

Then open the local URL in your browser.

---

## 🧑‍💻 Usage

1. Enter your **OpenAI API Key** in the sidebar.
2. (Optional) Tune **LLM** and **Embedding** model names. Defaults:

   * LLM: `gpt-4o-mini`
   * Embedding: `text-embedding-3-small`
3. Adjust **Chunk Size** and **Overlap**.
4. **Upload PDFs** and click **Build / Rebuild Index**.
5. Ask questions in the **Chat** section.

---

## ⚙️ Config knobs explained

* **Chunk size:** Larger chunks capture more context but raise token cost and may add noise. 1–2k chars is a good start.
* **Chunk overlap:** Helps preserve continuity across chunk boundaries. 10–15% of chunk size is typical.
* **LLM model:** Controls generation quality/cost/latency.
* **Embedding model:** Affects retrieval quality and index build cost.

---

## 🧱 Current architecture (in‑memory)

```
Streamlit UI ── uploads ─▶ Temp Dir
       │
       ├─▶ LlamaIndex Readers ─▶ Chunk (SentenceSplitter)
       │                       └▶ Embed (OpenAIEmbedding)
       │
       └─▶ VectorStoreIndex (RAM) ──▶ ChatEngine (condense_question)
                                      └─▶ Stream back to UI
```

---

## 🔍 Troubleshooting

* **Stuck on indexing:** Large PDFs can be slow; try smaller `chunk_size` or fewer files.
* **Rate limits:** Reduce simultaneous builds; add retry/backoff (see enhancements).
* **Empty answers:** Inspect PDFs for scanned images; consider OCR ingestion (see enhancements).
* **Memory usage:** Many/large PDFs can exhaust RAM; switch to a disk/vector DB backend (see enhancements).

---

## 🔒 Security & privacy

* API key is stored only in memory for the Streamlit session (unless you add secrets management).
* Uploaded files are written to a temporary folder; clean up after session if persisting is disabled.
* Consider **Streamlit secrets**, **dotenv**, or a server‑side vault for production.

---

## 💸 Cost controls

* Prefer `text-embedding-3-small` for indexing; upgrade only if recall is poor.
* Use smaller `chunk_size` and fewer docs for experimentation.
* Add a per‑session token budget meter (enhancement below).

---

## 🛠️ Suggested improvements (with code pointers)

Below are pragmatic additions you can cherry‑pick. They’re grouped by theme and include minimal snippets where useful.

### 1) Persistence: save & load indices

Avoid rebuilding every session; store to disk.

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

PERSIST_DIR = "./storage"

# Save
index.storage_context.persist(persist_dir=PERSIST_DIR)

# Load later
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
```

Wire this behind a **"Use persisted index"** toggle. Persisting also enables warm‑starts and multi‑session use.

### 2) Show sources & citations

Display top nodes and page numbers alongside each answer.

```python
response = st.session_state.chat_engine.stream_chat(user_text)
# After collecting tokens, access response.source_nodes
for n in response.source_nodes[:3]:
    st.markdown(f"- **Score:** {n.score:.3f} — {n.node.metadata.get('file_name')} p.{n.node.metadata.get('page_label')}")
```

Use `SimpleDirectoryReader(..., filename_as_id=True)` and ensure PDF loaders record page labels.

### 3) Better retrieval: reranking & hybrid search

* **Rerankers:** Add a cross‑encoder reranker (e.g., FlagEmbedding or Cohere Rerank) on the top‑k chunks before final context.
* **Hybrid BM25+Embeddings:** Combine sparse and dense retrieval for robustness.

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank

retriever = VectorIndexRetriever(index=index, similarity_top_k=20)
reranker = SentenceTransformerRerank(top_n=5, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
response = index.as_query_engine(retriever=retriever, node_postprocessors=[reranker]).query(q)
```

> Swap model name as available in your environment.

### 4) Smarter chunking

* Swap `SentenceSplitter` for **semantic chunkers** to avoid splitting mid‑topic.
* Add **title‑aware** or **heading‑aware** chunking (retain section headers in metadata).

### 5) OCR & scanned PDFs

Add OCR for image‑based PDFs (e.g., `pytesseract`, `ocrmypdf`) before ingestion or use loaders with OCR.

### 6) Multi‑format ingestion

Support more than PDFs: `.docx`, `.md`, `.txt`, web URLs. Use LlamaIndex readers like `BeautifulSoupWebReader`, `UnstructuredReader`, etc.

### 7) Multi‑file, per‑doc filters

Tag nodes with `doc_id` and expose a **filter UI**: search within a specific document or subset.

```python
# When loading
docs = SimpleDirectoryReader(input_files=saved_paths, filename_as_id=True).load_data()
# Later during retrieval
from llama_index.core import Document
# ensure metadata: {"file_name": ..., "doc_id": ...}
```

### 8) Observability (traces, tokens, latency)

Add LlamaIndex callback handlers to log token counts and timings; surface in a Streamlit sidebar.

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
cb = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([cb])
```

### 9) Error handling & retries

Wrap OpenAI calls with **exponential backoff** on rate limits; surface friendly messages in the UI.

### 10) Session persistence & export

* Save chat history + index ID to disk; **Export QA** as CSV/JSON.
* Add a **"Download conversation"** button.

### 11) Auth & multi‑user

* Gate the app with **Streamlit Auth** or reverse proxy auth.
* Separate user storage directories by user/session ID.

### 12) Server mode & API

Expose a `/query` endpoint (FastAPI) so other apps can query the same index.

### 13) Caching

Cache embeddings & chunking using `@st.cache_data`/`@st.cache_resource` where appropriate to avoid recomputation.

### 14) Guardrails & "I don't know"

* Add a system prompt to prefer **abstention** over hallucination.
* Configure a **confidence threshold** on similarity; if below, answer: *"I couldn’t find this in your PDFs"*.

### 15) UI polish

* Display progress bars per document during indexing.
* Show **document list** with sizes and page counts.
* Add **source highlights** (show the exact spans used in the answer).

### 16) Vector DB backends

For larger corpora, switch the index to external stores: **FAISS**, **Chroma**, **PGVector**, **Milvus**, etc. LlamaIndex integrates via `VectorStoreIndex.from_vector_store`.

