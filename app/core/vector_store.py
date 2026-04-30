"""
Vector Store
============
Wraps ChromaDB for storing and retrieving chunk embeddings.
NOW USES LOCAL HUGGING FACE EMBEDDINGS (Sentence-Transformers).
No API keys or rate limits for indexing!
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional
from app.core.ingestion import Chunk

# Using a high-quality, lightweight model from Hugging Face
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "poppulo_rag_chunks"


class VectorStore:
    """
    Persistent ChromaDB vector store.
    Uses Sentence-Transformers to run embeddings locally on your machine.
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # Initialize the local embedding model
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings locally using SentenceTransformers."""
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        Index a list of chunks. Skips already-indexed chunks (idempotent).
        Returns number of newly added chunks.
        """
        if not chunks:
            return 0

        # SAFETY FILTER: Ensure chunks in this batch are unique by ID
        unique_chunks_dict = {}
        for c in chunks:
            unique_chunks_dict[c.chunk_id] = c
        chunks = list(unique_chunks_dict.values())

        # Check which chunk_ids already exist in the database
        existing = set(
            self._collection.get(ids=[c.chunk_id for c in chunks])["ids"]
        )
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = self._embed(texts)

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page_number": c.page_number,
                    "paragraph_index": c.paragraph_index,
                    "sentence_start": c.sentence_start,
                    "sentence_end": c.sentence_end,
                    "section_heading": c.section_heading or "",
                    "chunk_index": c.chunk_index,
                    "bbox": str(c.bbox) if c.bbox else "",
                }
                for c in new_chunks
            ],
        )
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of result dicts with text + metadata.
        """
        query_embedding = self._embed([query_text])[0]

        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        # Fix: Re-added the collection.query call correctly
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        hits = []
        if results["documents"] and len(results["documents"]) > 0:
            for text, meta, dist, chunk_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                hits.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": round(1 - dist, 4),
                    **meta,
                })
        return hits

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """Return unique documents currently indexed."""
        if self._collection.count() == 0:
            return []

        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        seen = {}
        for m in all_meta:
            doc_id = m["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": m["doc_name"],
                }
        return list(seen.values())

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document."""
        results = self._collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def chunk_count(self) -> int:
        return self._collection.count()
