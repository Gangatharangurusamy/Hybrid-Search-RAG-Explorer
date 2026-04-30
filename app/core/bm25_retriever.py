"""
BM25 Sparse Retriever
=====================
Keyword-based retrieval using BM25 (Best Match 25).
Complements dense vector search — especially strong for:
  - Exact technical terms (e.g. "multi-head attention", "GRPO")
  - Acronyms and model names
  - Queries where semantic similarity alone misses exact matches

Combined with VectorStore for Hybrid Retrieval (RRF fusion).
"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9][a-z0-9\-\.]*\b', text)
    return tokens


class BM25Retriever:
    """
    In-memory BM25 index over chunks.
    Persisted to disk as a pickle for fast reload.
    """

    def __init__(self, persist_path: str = "./data/bm25_index.pkl"):
        self._persist_path = Path(persist_path)
        self._chunks: list[dict] = []       # list of chunk dicts (text + metadata)
        self._tokenized: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load()

    def _load(self):
        if self._persist_path.exists():
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
                self._chunks = data["chunks"]
                self._tokenized = data["tokenized"]
                if self._tokenized:
                    self._bm25 = BM25Okapi(self._tokenized)

    def _save(self):
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "tokenized": self._tokenized,
            }, f)

    def index_chunks(self, chunks: list[dict]) -> int:
        """
        Add new chunks to BM25 index. Deduplicates by chunk_id.
        Returns number of newly added chunks.
        """
        existing_ids = {c["chunk_id"] for c in self._chunks}
        new = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new:
            return 0

        for chunk in new:
            self._chunks.append(chunk)
            self._tokenized.append(_tokenize(chunk["text"]))

        self._bm25 = BM25Okapi(self._tokenized)
        self._save()
        return len(new)

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        BM25 search. Returns top-k chunks with bm25_score field.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query_text)
        raw_scores = self._bm25.get_scores(tokens)

        # Apply doc_id filter if requested
        results = []
        for i, (chunk, score) in enumerate(zip(self._chunks, raw_scores)):
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            results.append({**chunk, "bm25_score": float(score)})

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> int:
        original_len = len(self._chunks)
        paired = [
            (c, t) for c, t in zip(self._chunks, self._tokenized)
            if c.get("doc_id") != doc_id
        ]
        if paired:
            self._chunks, self._tokenized = zip(*paired)
            self._chunks = list(self._chunks)
            self._tokenized = list(self._tokenized)
        else:
            self._chunks = []
            self._tokenized = []
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._save()
        return original_len - len(self._chunks)

    def chunk_count(self) -> int:
        return len(self._chunks)
