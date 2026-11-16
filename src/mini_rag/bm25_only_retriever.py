"""Retriever for [retrieval] mode "bm25": the keyword index alone, without fusion or embedding."""

from mini_rag.hybrid_hit import HybridHit


class Bm25OnlyRetriever:
  """Returns the BM25 index's hits as HybridHit objects with no dense and no fused score."""

  def __init__(self, obj_bm25: object) -> None:
    """Store an already-built BM25 index."""
    self.obj_bm25 = obj_bm25

  def search(self, s_query: str, i_top_k: int) -> list[HybridHit]:
    """Return up to i_top_k hits with a positive BM25 score, best first; the query is never embedded."""
    if i_top_k < 1:
      raise ValueError(f"i_top_k must be at least 1, got {i_top_k}")
    if not s_query.strip():
      return []
    return [
      HybridHit(obj_chunk=obj_hit.obj_chunk, f_rrf_score=None, f_dense_score=None, f_bm25_score=obj_hit.f_score)
      for obj_hit in self.obj_bm25.search(s_query, i_top_k)
    ]
