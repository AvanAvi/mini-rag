"""Unit tests for the fusion weights of the HybridRetriever."""

import pytest

from fake_index import FakeIndex
from mini_rag.chunk import Chunk
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.search_hit import SearchHit


class TestHybridWeights:
  """Checks that the weights reach the fuser in BM25-first order and default to today's behaviour."""

  def hits(self, l_names: list[str]) -> list[SearchHit]:
    """Turn names into a best-first SearchHit list; each name is its own document."""
    return [
      SearchHit(obj_chunk=Chunk(s_doc_id=s_name, i_index=0, i_start=0, s_text=s_name), f_score=1.0)
      for s_name in l_names
    ]

  def retriever(self, **d_options: float) -> HybridRetriever:
    """Return a retriever over BM25 = [A, B, C] and dense = [B, D, A] with the given options."""
    return HybridRetriever(FakeIndex(self.hits(["A", "B", "C"])), FakeIndex(self.hits(["B", "D", "A"])), **d_options)

  def names(self, obj_retriever: HybridRetriever) -> list[str]:
    """Return the document ids of the top four fused hits."""
    return [obj_hit.obj_chunk.s_doc_id for obj_hit in obj_retriever.search("q", 4)]

  def test_defaults_are_k_60_and_weights_of_one(self) -> None:
    """A retriever built without options uses k = 60 and a weight of 1.0 for both lists."""
    obj_retriever = self.retriever()
    assert (obj_retriever.i_k, obj_retriever.f_bm25_weight, obj_retriever.f_dense_weight) == (60, 1.0, 1.0)

  def test_default_scores_equal_the_unweighted_formula_exactly(self) -> None:
    """With default weights every fused score is the plain sum of 1.0 / (60 + rank), bit for bit."""
    d_scores = {obj_hit.obj_chunk.s_doc_id: obj_hit.f_rrf_score for obj_hit in self.retriever().search("q", 4)}
    assert d_scores == {
      "A": 0.0 + 1.0 / 61 + 1.0 / 63, "B": 0.0 + 1.0 / 62 + 1.0 / 61, "C": 0.0 + 1.0 / 63, "D": 0.0 + 1.0 / 62,
    }
    assert self.names(self.retriever()) == ["B", "A", "D", "C"]

  def test_weights_apply_in_bm25_then_dense_order(self) -> None:
    """f_bm25_weight scales the BM25 list and f_dense_weight the dense list."""
    l_hits = self.retriever(i_k=1, f_bm25_weight=0.5, f_dense_weight=1.0).search("q", 4)
    # A = 0.5/2 + 1/4, B = 0.5/3 + 1/2, C = 0.5/4, D = 1/3
    assert [obj_hit.obj_chunk.s_doc_id for obj_hit in l_hits] == ["B", "A", "D", "C"]
    assert [obj_hit.f_rrf_score for obj_hit in l_hits] == pytest.approx([0.5 / 3 + 1 / 2, 0.5, 1 / 3, 0.125])

  def test_zero_bm25_weight_gives_the_dense_order(self) -> None:
    """With no weight on BM25 the dense list's chunks come first, in dense order."""
    assert self.names(self.retriever(f_bm25_weight=0.0)) == ["B", "D", "A", "C"]

  def test_zero_dense_weight_gives_the_bm25_order(self) -> None:
    """With no weight on dense the BM25 list's chunks come first, in BM25 order."""
    assert self.names(self.retriever(f_dense_weight=0.0)) == ["A", "B", "C", "D"]

  def test_negative_weight_raises_on_search(self) -> None:
    """A negative weight is rejected by the fuser when a search runs."""
    with pytest.raises(ValueError, match="negative"):
      self.retriever(f_bm25_weight=-1.0).search("q", 4)
