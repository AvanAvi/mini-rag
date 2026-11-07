"""Combines dense and BM25 results into one ranking."""

from mini_rag.hybrid_hit import HybridHit
from mini_rag.rank_fuser import RankFuser
from mini_rag.search_hit import SearchHit


class HybridRetriever:
  """Fuses the rankings of a BM25 index and a dense index with Reciprocal Rank Fusion."""

  def __init__(self, obj_bm25: object, obj_dense: object, i_k: int = 60, i_candidates: int = 20) -> None:
    """Store two already-built indexes, the RRF constant i_k and the per-index candidate count."""
    if i_k < 1:
      raise ValueError(f"i_k must be at least 1, got {i_k}")
    if i_candidates < 1:
      raise ValueError(f"i_candidates must be at least 1, got {i_candidates}")
    self.obj_bm25 = obj_bm25
    self.obj_dense = obj_dense
    self.i_k = i_k
    self.i_candidates = i_candidates
    self.obj_fuser = RankFuser()

  def search(self, s_query: str, i_top_k: int) -> list[HybridHit]:
    """Return up to i_top_k fused hits, best RRF score first."""
    if i_top_k < 1:
      raise ValueError(f"i_top_k must be at least 1, got {i_top_k}")
    if not s_query.strip():
      return []

    i_fetch = max(self.i_candidates, i_top_k)
    l_bm25_hits = self.obj_bm25.search(s_query, i_fetch)
    l_dense_hits = self.obj_dense.search(s_query, i_fetch)
    d_bm25_scores = self.score_table(l_bm25_hits)
    d_dense_scores = self.score_table(l_dense_hits)

    # BM25 first, so ties in fused score keep BM25's order
    l_fused = self.obj_fuser.fuse([l_bm25_hits, l_dense_hits], self.i_k)
    return [
      HybridHit(
        obj_chunk=obj_chunk,
        f_rrf_score=f_rrf_score,
        f_dense_score=d_dense_scores.get(self.obj_fuser.key(obj_chunk)),
        f_bm25_score=d_bm25_scores.get(self.obj_fuser.key(obj_chunk)),
      )
      for obj_chunk, f_rrf_score in l_fused[:i_top_k]
    ]

  def score_table(self, l_hits: list[SearchHit]) -> dict[tuple[str, int], float]:
    """Map each chunk id to its score at its best (first) position in a best-first list."""
    d_scores: dict[tuple[str, int], float] = {}
    for obj_hit in l_hits:
      d_scores.setdefault(self.obj_fuser.key(obj_hit.obj_chunk), obj_hit.f_score)
    return d_scores
