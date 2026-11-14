"""One combination of fusion parameters for the hybrid retriever."""

from dataclasses import dataclass

from mini_rag.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class FusionSetting:
  """The RRF constant and the two list weights used for one hybrid evaluation."""

  i_k: int
  f_bm25_weight: float
  f_dense_weight: float

  @classmethod
  def default(cls) -> "FusionSetting":
    """Return the setting a HybridRetriever uses when built without fusion options."""
    return cls(
      i_k=HybridRetriever.I_DEFAULT_K,
      f_bm25_weight=HybridRetriever.F_DEFAULT_BM25_WEIGHT,
      f_dense_weight=HybridRetriever.F_DEFAULT_DENSE_WEIGHT,
    )

  @property
  def s_label(self) -> str:
    """Return the name shown in reports, such as "k=60 w=1.0/1.0" (BM25 weight first)."""
    return f"k={self.i_k} w={float(self.f_bm25_weight)}/{float(self.f_dense_weight)}"
