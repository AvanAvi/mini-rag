"""The fixed grid of fusion settings tried by eval --fusion-grid."""

from mini_rag.fusion_setting import FusionSetting


class FusionGrid:
  """Crosses the RRF constants with the BM25 weights; the dense weight stays fixed."""

  T_KS = (5, 10, 20, 60)
  T_BM25_WEIGHTS = (1.0, 0.5, 0.25)
  F_DENSE_WEIGHT = 1.0

  def settings(self) -> list[FusionSetting]:
    """Return every setting of the grid, ordered by k and then by falling BM25 weight."""
    return [
      FusionSetting(i_k=i_k, f_bm25_weight=f_bm25_weight, f_dense_weight=self.F_DENSE_WEIGHT)
      for i_k in self.T_KS
      for f_bm25_weight in self.T_BM25_WEIGHTS
    ]
