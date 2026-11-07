"""Unit tests for the RankFuser."""

import pytest

from mini_rag.chunk import Chunk
from mini_rag.rank_fuser import RankFuser
from mini_rag.search_hit import SearchHit


class TestRankFuser:
  """Checks RRF arithmetic, ordering, ties, empty input and duplicates."""

  def chunk(self, s_name: str, s_text: str = "") -> Chunk:
    """Make a chunk whose doc id is s_name; text defaults to the name."""
    return Chunk(s_doc_id=s_name, i_index=0, i_start=0, s_text=s_text or s_name)

  def hits(self, l_names: list[str]) -> list[SearchHit]:
    """Turn names into a best-first SearchHit list with dummy scores."""
    return [SearchHit(obj_chunk=self.chunk(s_name), f_score=1.0) for s_name in l_names]

  def names(self, l_fused: list[tuple[Chunk, float]]) -> list[str]:
    """Return the doc ids of a fused list in order."""
    return [obj_chunk.s_doc_id for obj_chunk, f_score in l_fused]

  def test_exact_scores_hand_computed(self) -> None:
    """Fuse two small rankings with i_k = 1 and compare against hand arithmetic."""
    # i_k = 1, score = sum of 1 / (1 + rank):
    #   bm25  = [A, B, C]  -> A rank 1, B rank 2, C rank 3
    #   dense = [B, D, A]  -> B rank 1, D rank 2, A rank 3
    #   A = 1/2 + 1/4 = 0.75
    #   B = 1/3 + 1/2 = 0.8333...
    #   C = 1/4       = 0.25
    #   D = 1/3       = 0.3333...
    # order: B, A, D, C
    l_fused = RankFuser().fuse([self.hits(["A", "B", "C"]), self.hits(["B", "D", "A"])], 1)
    assert self.names(l_fused) == ["B", "A", "D", "C"]
    l_scores = [f_score for obj_chunk, f_score in l_fused]
    assert l_scores == pytest.approx([1 / 3 + 1 / 2, 1 / 2 + 1 / 4, 1 / 3, 1 / 4])

  def test_chunk_in_both_lists_outranks_single_list(self) -> None:
    """A at ranks 1 and 2 beats Y (rank 1 only) and X (rank 2 only)."""
    l_fused = RankFuser().fuse([self.hits(["A", "X"]), self.hits(["Y", "A"])], 60)
    assert self.names(l_fused) == ["A", "Y", "X"]
    assert l_fused[0][1] == pytest.approx(1 / 61 + 1 / 62)

  def test_ties_keep_first_appearance_order(self) -> None:
    """Equal fused scores keep first-appearance order, first list first."""
    # A and B both score 1/61; A appears first because it is in the first list
    assert self.names(RankFuser().fuse([self.hits(["A"]), self.hits(["B"])], 60)) == ["A", "B"]
    # A = B = 1/61 + 1/62; A appears first in the first list
    assert self.names(RankFuser().fuse([self.hits(["A", "B"]), self.hits(["B", "A"])], 60)) == ["A", "B"]

  def test_empty_rankings(self) -> None:
    """No rankings, or only empty ones, fuse to an empty list."""
    assert RankFuser().fuse([], 60) == []
    assert RankFuser().fuse([[], []], 60) == []

  def test_one_empty_ranking(self) -> None:
    """With one empty ranking the result is the other ranking alone."""
    l_fused = RankFuser().fuse([self.hits(["A", "B"]), []], 60)
    assert self.names(l_fused) == ["A", "B"]
    assert [f_score for obj_chunk, f_score in l_fused] == pytest.approx([1 / 61, 1 / 62])

  def test_duplicate_ids_counted_once_at_best_rank(self) -> None:
    """A chunk id repeated in one list counts once, at its first (best) rank."""
    # same (s_doc_id, i_index) at ranks 1 and 3, different text
    l_ranking = [
      SearchHit(obj_chunk=self.chunk("A", "first"), f_score=3.0),
      SearchHit(obj_chunk=self.chunk("B"), f_score=2.0),
      SearchHit(obj_chunk=self.chunk("A", "second"), f_score=1.0),
    ]
    l_fused = RankFuser().fuse([l_ranking], 60)
    assert self.names(l_fused) == ["A", "B"]
    assert l_fused[0][0].s_text == "first"
    assert [f_score for obj_chunk, f_score in l_fused] == pytest.approx([1 / 61, 1 / 62])

  def test_duplicate_keeps_later_ranks_in_place(self) -> None:
    """A duplicate is skipped for scoring but still occupies its position in the list."""
    l_ranking = [
      SearchHit(obj_chunk=self.chunk("A", "first"), f_score=4.0),
      SearchHit(obj_chunk=self.chunk("B"), f_score=3.0),
      SearchHit(obj_chunk=self.chunk("A", "second"), f_score=2.0),
      SearchHit(obj_chunk=self.chunk("C"), f_score=1.0),
    ]
    l_fused = RankFuser().fuse([l_ranking], 60)
    assert self.names(l_fused) == ["A", "B", "C"]
    # C keeps rank 4 (1/64, not 1/63): the duplicate A still occupies position 3
    assert [f_score for obj_chunk, f_score in l_fused] == pytest.approx([1 / 61, 1 / 62, 1 / 64])
