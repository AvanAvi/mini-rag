"""Combines dense and BM25 results into one ranking."""


class HybridRetriever:
  """Fuses scores from a dense index and a BM25 index."""

  def __init__(self, obj_dense_index: object, obj_bm25_index: object, f_alpha: float) -> None:
    """Store both indexes and the dense weight alpha (0.0 to 1.0)."""
    raise NotImplementedError

  def retrieve(self, s_query: str, i_top_k: int) -> list[dict]:
    """Return the top-k chunks ranked by the fused score."""
    raise NotImplementedError
