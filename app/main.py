"""
FastAPI Application
===================
REST API for the PDF RAG system.
Now uses Groq + Local Hugging Face Embeddings.
Supports content hashing to skip duplicate PDF processing.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.ingestion import ingest_pdf, get_doc_id
from app.core.vector_store import VectorStore
from app.core.bm25_retriever import BM25Retriever
from app.core.hybrid_retriever import HybridRetriever
from app.core.generator import generate_answer, generate_answer_streamed_sse

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Poppulo PDF RAG (Groq Edition)",
    description="Local HuggingFace + Groq RAG with Content Hashing",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHROMA_DIR = str(DATA_DIR / "chroma_db")
BM25_PATH = str(DATA_DIR / "bm25_index.pkl")
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

_vector_store: Optional[VectorStore] = None
_bm25: Optional[BM25Retriever] = None
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _vector_store, _bm25, _retriever
    if _retriever is None:
        _vector_store = VectorStore(persist_dir=CHROMA_DIR)
        _bm25 = BM25Retriever(persist_path=BM25_PATH)
        _retriever = HybridRetriever(_vector_store, _bm25)
    return _retriever


class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None
    top_k: int = 6


class SourceChunk(BaseModel):
    doc_name: str
    page_number: int
    section_heading: Optional[str]
    text_snippet: str
    rrf_score: float
    rank: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    total_chunks_retrieved: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF. Uses content hashing to skip if the file already exists.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    # Save to a temp file first to calculate hash
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Calculate doc_id (hash)
        doc_id = get_doc_id(tmp_path)
        
        retriever = get_retriever()
        
        # CHECK IF EXISTS
        existing_docs = _vector_store.list_documents()
        is_duplicate = any(d['doc_id'] == doc_id for d in existing_docs)
        
        if is_duplicate:
            tmp_path.unlink()  # delete temp
            return {
                "doc_id": doc_id,
                "doc_name": file.filename,
                "status": "already_exists",
                "message": f"Document '{file.filename}' (or an identical file) is already in the database. Skipped indexing.",
            }

        # Ingest
        chunks = ingest_pdf(tmp_path)
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract text.")

        # Index in vector store
        new_vec = _vector_store.index_chunks(chunks)

        # Index in BM25
        chunk_dicts = [c.to_dict() for c in chunks]
        new_bm25 = _bm25.index_chunks(chunk_dicts)

        # Move to permanent storage
        dest = PDF_DIR / f"{doc_id}.pdf"
        shutil.move(str(tmp_path), str(dest))

        return {
            "doc_id": doc_id,
            "doc_name": Path(file.filename).stem,
            "total_chunks": len(chunks),
            "newly_indexed": new_vec,
            "message": f"Successfully indexed {len(chunks)} chunks.",
        }

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents():
    retriever = get_retriever()
    docs = _vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    retriever = get_retriever()
    vec_removed = _vector_store.delete_document(doc_id)
    bm25_removed = _bm25.delete_document(doc_id)
    # Remove file
    file_path = PDF_DIR / f"{doc_id}.pdf"
    if file_path.exists():
        file_path.unlink()
    return {
        "doc_id": doc_id,
        "chunks_removed": vec_removed,
        "message": f"Removed {vec_removed} chunks.",
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    retriever = get_retriever()

    if _vector_store.chunk_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    chunks = retriever.retrieve(query=req.question, top_k=req.top_k, doc_ids=req.doc_ids)

    if not chunks:
        return QueryResponse(answer="No relevant content found.", sources=[], total_chunks_retrieved=0)

    answer = generate_answer(req.question, chunks, stream=False)

    sources = [
        SourceChunk(
            doc_name=c.get("doc_name", ""),
            page_number=c.get("page_number", 0),
            section_heading=c.get("section_heading") or None,
            text_snippet=c.get("text", "")[:300] + ("..." if len(c.get("text", "")) > 300 else ""),
            rrf_score=c.get("rrf_score", 0.0),
            rank=c.get("rank", 0),
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, total_chunks_retrieved=len(chunks))


@app.get("/api/query/stream")
def query_stream(
    question: str = Query(..., description="The question to ask"),
    doc_ids: Optional[str] = Query(None, description="Comma-separated doc IDs"),
    top_k: int = Query(6, description="Number of chunks"),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    retriever = get_retriever()
    if _vector_store.chunk_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    doc_id_list = [d.strip() for d in doc_ids.split(",")] if doc_ids else None
    chunks = retriever.retrieve(query=question, top_k=top_k, doc_ids=doc_id_list)

    if not chunks:
        def _empty():
            import json
            yield f"data: {json.dumps({'type': 'delta', 'text': 'No relevant content found.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    return StreamingResponse(
        generate_answer_streamed_sse(question, chunks),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
