"""Vector + full-text search fused with Reciprocal Rank Fusion (a $0 reranker; keeps exact tokens such as
error codes and designators competitive against semantic matches)."""
from __future__ import annotations

from app.config import settings
from app.llm.embed import Embedder
from app.retrieval.repository import Hit, ScopedRepository


def rrf_fuse(vector_hits: list[Hit], fts_hits: list[Hit], k: int | None = None) -> list[Hit]:
    k = k or settings.rrf_k
    merged: dict = {}
    for ranked in (vector_hits, fts_hits):
        for rank, h in enumerate(ranked, start=1):
            m = merged.setdefault(h.chunk_id, h)
            if h.vector_distance is not None:
                m.vector_distance = h.vector_distance
            if h.fts_rank is not None:
                m.fts_rank = h.fts_rank
            m.rrf_score += 1.0 / (k + rank)  # a chunk in both lists accumulates both contributions
    return sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)


def hybrid_search(repo: ScopedRepository, embedder: Embedder, query: str, *, doc_ids=None, content_types=None,
                  top_k: int | None = None) -> list[Hit]:
    n = settings.retrieval_candidate_k
    qvec = embedder.embed_query(query)
    fused = rrf_fuse(repo.vector_search(qvec, n, doc_ids, content_types),
                     repo.fts_search(query, n, doc_ids, content_types))
    return fused[: top_k or n]
