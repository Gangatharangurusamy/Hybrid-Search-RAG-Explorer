"""
Hybrid Retriever — Reciprocal Rank Fusion
==========================================
Combines dense (vector) + sparse (BM25) retrieval via RRF.

Why Hybrid?
  - Dense search: captures semantic meaning, synonyms, paraphrases
  - BM25 search: captures exact technical terms, model names, equations
  - RRF fusion: combines both without needing to tune score scales

RRF Formula:
  score(d) = Σ 1 / (k + rank_in_list_i(d))
  where k=60 (standard constant from the original RRF paper)

This is the same hybrid approach used in production search systems
at scale (e.g., Azure Cognitive Search, Elasticsearch).
"""

from app.core.vector_store import VectorStore
from app.core.bm25_retriever import BM25Retriever
from typing import Optional

RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


class HybridRetriever:
    """
    Orchestrates dense + sparse retrieval and fuses results with RRF.
    Final results are re-ranked by fused score and returned with all metadata.
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever):
        self._vs = vector_store
        self._bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        doc_ids: Optional[list[str]] = None,
        fetch_k: int = 15,          # candidates to fetch from each retriever
    ) -> list[dict]:
        """
        Hybrid retrieval with RRF fusion.

        Returns top_k chunks, each with:
          - text, doc_name, page_number, section_heading
          - dense_score, bm25_score, rrf_score
          - rank (1-indexed)
        """
        # --- Dense retrieval ---
        dense_hits = self._vs.query(query, top_k=fetch_k, doc_ids=doc_ids)
        # --- Sparse retrieval ---
        sparse_hits = self._bm25.query(query, top_k=fetch_k, doc_ids=doc_ids)

        # --- Build per-chunk score maps ---
        dense_rank: dict[str, int] = {h["chunk_id"]: i for i, h in enumerate(dense_hits)}
        sparse_rank: dict[str, int] = {h["chunk_id"]: i for i, h in enumerate(sparse_hits)}

        # --- Collect all unique chunks ---
        all_chunks: dict[str, dict] = {}
        for h in dense_hits:
            all_chunks[h["chunk_id"]] = h
        for h in sparse_hits:
            if h["chunk_id"] not in all_chunks:
                all_chunks[h["chunk_id"]] = h

        # --- Compute RRF scores ---
        scored = []
        for chunk_id, chunk in all_chunks.items():
            d_rank = dense_rank.get(chunk_id, fetch_k + 10)
            s_rank = sparse_rank.get(chunk_id, fetch_k + 10)
            rrf = _rrf_score(d_rank) + _rrf_score(s_rank)

            scored.append({
                **chunk,
                "dense_score": chunk.get("score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
                "rrf_score": round(rrf, 6),
            })

        # --- Sort by RRF score ---
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        top = scored[:top_k]

        # Add rank
        for i, item in enumerate(top):
            item["rank"] = i + 1

        return top
