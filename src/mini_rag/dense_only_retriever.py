"""Retriever for [retrieval] mode "dense": the dense index alone, without fusion."""

from mini_rag.hybrid_hit import HybridHit


class DenseOnlyRetriever:
  """Returns the dense index's hits as HybridHit objects with no BM25 and no fused score."""

  def __init__(self, obj_dense: object) -> None:
    """Store an already-built or loaded dense index."""
    self.obj_dense = obj_dense

  def search(self, s_query: str, i_top_k: int) -> list[HybridHit]:
    """Return up to i_top_k hits, highest cosine score first."""
    if i_top_k < 1:
      raise ValueError(f"i_top_k must be at least 1, got {i_top_k}")
    if not s_query.strip():
      return []
    return [
      HybridHit(obj_chunk=obj_hit.obj_chunk, f_rrf_score=None, f_dense_score=obj_hit.f_score, f_bm25_score=None)
      for obj_hit in self.obj_dense.search(s_query, i_top_k)
    ]
