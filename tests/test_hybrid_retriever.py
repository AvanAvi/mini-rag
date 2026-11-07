"""Unit tests for the HybridRetriever."""

import pytest

from fake_embedder import FakeEmbedder
from fake_index import FakeIndex
from mini_rag.chunk import Chunk
from mini_rag.dense_index import DenseIndex
from mini_rag.hybrid_hit import HybridHit
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.search_hit import SearchHit


class TestHybridRetriever:
  """Checks score fields, limits, candidate counts and parameter validation."""

  def chunks(self) -> list[Chunk]:
    """Four chunks of document "d", one per passage text."""
    l_texts = ["north", "east", "north-east", "up"]
    return [
      Chunk(s_doc_id="d", i_index=i_pos, i_start=i_pos * 10, s_text=s_text)
      for i_pos, s_text in enumerate(l_texts)
    ]

  def dense_index(self) -> DenseIndex:
    """A DenseIndex over chunks() using hand-made vectors; query "q" points north."""
    obj_embedder = FakeEmbedder(
      d_passages={
        "north": [1.0, 0.0, 0.0],
        "east": [0.0, 1.0, 0.0],
        "north-east": [1.0, 1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
      },
      d_queries={"q": [1.0, 0.0, 0.0]},
    )
    obj_dense = DenseIndex(obj_embedder=obj_embedder)
    obj_dense.build(self.chunks())
    return obj_dense

  def bm25_fake(self) -> FakeIndex:
    """A hand-made BM25 ranking: "up" (2.5) then "north" (1.5)."""
    l_chunks = self.chunks()
    return FakeIndex([
      SearchHit(obj_chunk=l_chunks[3], f_score=2.5),
      SearchHit(obj_chunk=l_chunks[0], f_score=1.5),
    ])

  def test_score_fields_filled_or_none(self) -> None:
    """Chunks in both lists get both scores; chunks in one list get None for the other."""
    # i_candidates = 2: dense returns north (1.0), north-east (~0.707)
    # bm25 returns up (2.5), north (1.5)
    # RRF: north = 1/62 + 1/61, up = 1/61, north-east = 1/62
    obj_retriever = HybridRetriever(self.bm25_fake(), self.dense_index(), i_candidates=2)
    l_hits = obj_retriever.search("q", 3)

    assert all(isinstance(obj_hit, HybridHit) for obj_hit in l_hits)
    assert [obj_hit.obj_chunk.s_text for obj_hit in l_hits] == ["north", "up", "north-east"]

    obj_north, obj_up, obj_north_east = l_hits
    assert obj_north.f_rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert obj_north.f_bm25_score == pytest.approx(1.5)
    assert obj_north.f_dense_score == pytest.approx(1.0)

    assert obj_up.f_bm25_score == pytest.approx(2.5)
    assert obj_up.f_dense_score is None

    assert obj_north_east.f_bm25_score is None
    assert obj_north_east.f_dense_score == pytest.approx(2 ** -0.5)

  def test_top_k_limits_results(self) -> None:
    """At most i_top_k hits are returned, best first."""
    obj_retriever = HybridRetriever(self.bm25_fake(), self.dense_index(), i_candidates=2)
    l_hits = obj_retriever.search("q", 1)
    assert [obj_hit.obj_chunk.s_text for obj_hit in l_hits] == ["north"]

  def test_candidate_count_is_max_of_candidates_and_top_k(self) -> None:
    """Each index is asked for max(i_candidates, i_top_k) hits."""
    obj_bm25 = FakeIndex([])
    obj_dense = FakeIndex([])
    obj_retriever = HybridRetriever(obj_bm25, obj_dense, i_candidates=5)
    obj_retriever.search("q", 3)
    obj_retriever.search("q", 8)
    assert obj_bm25.l_top_k_calls == [5, 8]
    assert obj_dense.l_top_k_calls == [5, 8]

  def test_invalid_parameters_raise(self) -> None:
    """i_k, i_candidates and i_top_k below 1 raise ValueError."""
    with pytest.raises(ValueError):
      HybridRetriever(FakeIndex([]), FakeIndex([]), i_k=0)
    with pytest.raises(ValueError):
      HybridRetriever(FakeIndex([]), FakeIndex([]), i_candidates=0)
    obj_retriever = HybridRetriever(FakeIndex([]), FakeIndex([]))
    with pytest.raises(ValueError):
      obj_retriever.search("q", 0)

  def test_empty_query_returns_empty(self) -> None:
    """An empty or blank query returns [] without asking either index."""
    obj_bm25 = self.bm25_fake()
    obj_retriever = HybridRetriever(obj_bm25, self.dense_index())
    assert obj_retriever.search("", 3) == []
    assert obj_retriever.search("   ", 3) == []
    assert obj_bm25.l_top_k_calls == []
