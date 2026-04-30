# DocMind — Hybrid PDF RAG System

> **Poppulo Graduate ML Engineer — Take-Home Challenge**  
> Built by Gangatharan G

A production-grade Retrieval-Augmented Generation (RAG) system for PDF documents. Combines **dense vector search** (local Hugging Face embeddings) and **BM25 sparse retrieval** via **Reciprocal Rank Fusion** for best-in-class retrieval accuracy — with full source citations in every answer. Powered by **Groq (Llama-3)** for ultra-fast streaming generation.

---

## Live Demo

| | |
|---|---|
| **Web App** | [huggingface.co/spaces/Ganga1005/Hybrid-Search-RAG-Explorer](https://huggingface.co/spaces/Ganga1005/Hybrid-Search-RAG-Explorer) |
| **API Docs** | [.../Hybrid-Search-RAG-Explorer/docs](https://huggingface.co/spaces/Ganga1005/Hybrid-Search-RAG-Explorer) |
| **Repository** | [github.com/Gangatharangurusamy](https://github.com/Gangatharangurusamy) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCMIND RAG PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌───────────────────────────────────────────────────────┐
  │  PDF     │────▶│                INGESTION ENGINE                       │
  │  Upload  │     │  SHA-256 Content Hashing → Duplicate Detection        │
  └──────────┘     │  PyMuPDF block extraction → sentence-aware chunking   │
                   │  + heading detection + bbox metadata                  │
                   └───────────────────────┬───────────────────────────────┘
                                           │  Chunk objects with metadata:
                                           │  doc_id (content hash), page,
                                           │  section_heading, bbox, text
                                           │
                   ┌───────────────────────▼───────────────────────────────┐
                   │              DUAL INDEXING                            │
                   │                                                       │
                   │  ┌───────────────────────┐  ┌──────────────────────┐ │
                   │  │   ChromaDB (Dense)    │  │   BM25 (Sparse)      │ │
                   │  │  HuggingFace Local    │  │  Tokenized keyword   │ │
                   │  │  all-MiniLM-L6-v2     │  │  index (persisted    │ │
                   │  │  cosine similarity    │  │  as pickle)          │ │
                   │  └───────────────────────┘  └──────────────────────┘ │
                   └───────────────────────────────────────────────────────┘

  ┌──────────┐     ┌───────────────────────────────────────────────────────┐
  │  User    │────▶│              HYBRID RETRIEVAL                         │
  │  Query   │     │                                                       │
  └──────────┘     │  Dense Search ──────────────────┐                    │
                   │  (top-15 semantic candidates)    │                    │
                   │                                  ▼                   │
                   │  BM25 Search ──────────────▶  RRF FUSION             │
                   │  (top-15 keyword candidates)  score = Σ 1/(60+rank)  │
                   │                                  │                    │
                   │                                  ▼                   │
                   │                          Top-6 chunks                │
                   │                          (re-ranked by RRF)          │
                   └───────────────────────┬───────────────────────────────┘
                                           │
                   ┌───────────────────────▼───────────────────────────────┐
                   │                  GENERATION                           │
                   │                                                       │
                   │  Numbered context block (text + provenance metadata)  │
                   │         │                                             │
                   │         ▼                                             │
                   │   Groq — Llama-3.3-70B (temp=0.1)                    │
                   │   Streaming via SSE  ◀── System prompt with          │
                   │         │                citation rules               │
                   │         ▼                                             │
                   │  Answer with inline citations:                        │
                   │  "[Source: doc_name, p.N, §section]"                 │
                   └───────────────────────┬───────────────────────────────┘
                                           │
                   ┌───────────────────────▼───────────────────────────────┐
                   │                  RESPONSE                             │
                   │  {                                                    │
                   │    answer: "...with [Source: X, p.N, §Y] citations", │
                   │    sources: [{ doc_name, page_number,                 │
                   │                section_heading, text_snippet,         │
                   │                rrf_score, rank }]                    │
                   │  }                                                    │
                   └───────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | Async, production-ready, built-in SSE support |
| **PDF Parsing** | PyMuPDF (fitz) | Block-level layout access with bounding boxes |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | **Free, local, no rate limits**, runs on CPU |
| **Vector DB** | ChromaDB (persistent) | Lightweight, file-based, no external server needed |
| **Sparse Search** | BM25Okapi (rank-bm25) | Keyword-exact matching, complement to dense |
| **Retrieval** | Hybrid RRF | Best of semantic + keyword search |
| **Generation** | Groq — Llama-3.3-70B | **Ultra-fast** streaming, generous free tier |
| **Frontend** | Vanilla HTML/CSS/JS | No framework needed, single file, SSE streaming |
| **Duplicate Detection** | SHA-256 Content Hash | Same PDF with different name = skipped automatically |

---

## Why Hybrid Retrieval?

| Approach | Good For | Bad For |
|---|---|---|
| **Dense (Vector)** | Semantic meaning, synonyms, paraphrases | Exact technical terms, model names, equations |
| **BM25 (Sparse)** | Exact keyword matches, acronyms, numbers | Semantic similarity, context |
| **Hybrid (RRF)** | ✅ Both | — |

**Reciprocal Rank Fusion** combines both rank lists without needing to normalize scores:

```
RRF_score(doc) = Σ  1 / (60 + rank_i(doc))
                 i
```

In practice, for queries like *"what is GRPO in DeepSeek-R1?"*, BM25 surfaces the exact term while vector search finds semantically related passages — RRF fuses both for better coverage.

---

## Key Design Decisions

### 1. No LangChain / No Abstraction Frameworks
Every component is built from first principles using only official client libraries:
- **PyMuPDF** (`fitz`) for PDF parsing — direct access to block-level layout
- **ChromaDB client** for vector storage — using its Python SDK directly
- **sentence-transformers** for local embeddings — zero API cost, no rate limits
- **groq** client for generation — direct API calls, Llama-3.3-70B
- **rank-bm25** for sparse retrieval — lightweight, no hidden magic

### 2. Content-Based Duplicate Detection (SHA-256 Hashing)
Unlike filename-based detection, the system reads the entire PDF binary and generates a SHA-256 fingerprint as the `doc_id`.
- If you upload `paper.pdf` and then `paper_copy.pdf` (identical content), the system detects the duplicate and skips re-indexing.
- This saves compute time and prevents database bloat.

### 3. Sentence-Aware Chunking with Metadata
Unlike naive fixed-character chunking:
- Chunks respect sentence boundaries to avoid mid-sentence splits
- Sliding window overlap (80 chars) preserves cross-chunk context
- Every chunk carries: page number, paragraph index, section heading, bounding box

### 4. Local Embeddings — Zero API Cost
Instead of calling OpenAI or Google for every chunk, embeddings are generated **locally** using Hugging Face `all-MiniLM-L6-v2`.
- No rate limits when indexing large PDFs
- No cost per API call
- Fast enough to run on a standard CPU

### 5. Streaming with SSE
The `/api/query/stream` endpoint uses Server-Sent Events:
- Groq tokens stream to the UI in real-time as they're generated
- A final `sources` event delivers the retrieved chunk metadata after generation
- The frontend renders citations inline and source chips below the answer

### 6. Citation-Enforced Prompting
The system prompt instructs the LLM to cite every factual claim:
```
[Source: <doc_name>, p.<page_number>, §<section_heading>]
```
The LLM is explicitly instructed to say *"I could not find information"* if the answer is not in context, preventing hallucination.

---

## Project Structure

```
poppulo-rag/
├── app/
│   ├── main.py                  # FastAPI app, all endpoints + duplicate check logic
│   └── core/
│       ├── ingestion.py         # PDF parsing, SHA-256 hashing, sentence chunking
│       ├── vector_store.py      # ChromaDB + HuggingFace local embeddings
│       ├── bm25_retriever.py    # BM25Okapi sparse retriever (persisted)
│       ├── hybrid_retriever.py  # RRF fusion of dense + sparse
│       └── generator.py        # Groq Llama-3 generation + SSE streaming
├── frontend/
│   └── index.html               # Single-file UI (no framework, SSE streaming)
├── data/                        # Persisted indexes (gitignored)
│   ├── chroma_db/
│   ├── bm25_index.pkl
│   └── pdfs/
├── Dockerfile                   # HuggingFace Spaces ready (port 7860)
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload PDF — detects duplicates via content hash |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Remove a document |
| `POST` | `/api/query` | Ask a question (full response) |
| `GET` | `/api/query/stream` | Ask with SSE streaming (Groq real-time) |

Interactive docs: `http://localhost:8000/docs`

---

## Local Setup

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Step-by-Step

```bash
git clone https://github.com/Ganga1005/Hybrid-Search-RAG-Explorer
cd Hybrid-Search-RAG-Explorer

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~80MB). This only happens once and is cached locally.

### Option 2: Docker

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY

docker build -t docmind-rag .
docker run -p 7860:7860 --env-file .env docmind-rag
```

---

## Deployment (Hugging Face Spaces)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repository to the Space's Git remote
4. In Space **Settings → Variables and Secrets**, add:
   - `GROQ_API_KEY` = your Groq key
5. In Space **Settings → Storage Buckets**, attach a bucket and mount it at `/data`
   - This ensures uploaded PDFs and ChromaDB indexes **persist across restarts**
6. The Space builds automatically and serves at your Space URL

> **Storage:** This Space uses a persistent Storage Bucket mounted at `/data`. Uploaded PDFs and the vector index survive server restarts.

---

## Sample Queries

**Attention Is All You Need:**
- *"What is the scaled dot-product attention formula?"*
- *"Why did the authors use positional encoding?"*
- *"What BLEU score did the transformer achieve on WMT 2014?"*

**DeepSeek-R1:**
- *"What is Group Relative Policy Optimization (GRPO)?"*
- *"How does DeepSeek-R1 handle cold-start in reinforcement learning?"*
- *"What benchmarks were used to evaluate DeepSeek-R1?"*

**Cross-document (Multi-PDF):**
- *"Compare the training approaches between Transformer and DeepSeek-R1"*
- *"What base architecture does DeepSeek-R1 use?"*

---

## Author

**Gangatharan G** — GenAI & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/ganga-guru-91339228a/) · [GitHub](https://github.com/Gangatharangurusamy)  
M.Sc. AI/ML, IIIT Lucknow (GPA: 9.11) · IIT JAM AIR 134
