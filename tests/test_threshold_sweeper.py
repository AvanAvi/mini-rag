"""Tests for ThresholdSweeper."""

from mini_rag.answerer import Answerer
from mini_rag.chunk import Chunk
from mini_rag.hybrid_hit import HybridHit
from mini_rag.sweep_row import SweepRow
from mini_rag.threshold_sweeper import ThresholdSweeper


class TestThresholdSweeper:
  """Covers the default thresholds, the found rule and agreement with Answerer.is_found."""

  def hits(self, l_dense_scores: list[float | None]) -> list[HybridHit]:
    """Return hybrid hits that carry the given dense scores."""
    return [
      HybridHit(
        obj_chunk=Chunk(s_doc_id="doc", i_index=i_pos, i_start=0, s_text="text"),
        f_rrf_score=0.01, f_dense_score=f_score, f_bm25_score=None,
      )
      for i_pos, f_score in enumerate(l_dense_scores)
    ]

  def best(self, l_dense_scores: list[float | None]) -> float | None:
    """Return the highest non-None score, or None."""
    l_scores = [f_score for f_score in l_dense_scores if f_score is not None]
    return max(l_scores) if l_scores else None

  def test_default_thresholds(self):
    """The default sweep runs from 0.40 to 0.80 in steps of 0.05."""
    assert ThresholdSweeper().l_thresholds == [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

  def test_none_score_is_never_found(self):
    """A question without a dense score is rejected if answerable and not accepted if unanswerable."""
    l_rows = ThresholdSweeper([0.40]).sweep([None, 0.9], [None])
    assert l_rows == [SweepRow(
      f_threshold=0.40, i_answerable_rejected=1, i_answerable_total=2,
      i_unanswerable_accepted=0, i_unanswerable_total=1,
    )]

  def test_score_equal_to_threshold_counts_as_found(self):
    """A best score exactly on the threshold passes the gate."""
    obj_row = ThresholdSweeper([0.50]).sweep([0.50, 0.49], [0.50, 0.49])[0]
    assert obj_row.i_answerable_rejected == 1
    assert obj_row.i_unanswerable_accepted == 1

  def test_counts_change_along_the_sweep(self):
    """Raising the threshold rejects more answerable and accepts fewer unanswerable questions."""
    l_rows = ThresholdSweeper([0.40, 0.60, 0.80]).sweep([0.45, 0.65, 0.85], [0.41, 0.61])
    assert [obj_row.i_answerable_rejected for obj_row in l_rows] == [0, 1, 2]
    assert [obj_row.i_unanswerable_accepted for obj_row in l_rows] == [2, 1, 0]

  def test_agrees_with_answerer_is_found(self):
    """At every default threshold the sweep rejects exactly the hit lists Answerer calls not found."""
    l_score_lists = [[0.5, None, 0.3], [None, None], [0.74, 0.75], [0.8], [], [0.39, 0.4]]
    l_best = [self.best(l_scores) for l_scores in l_score_lists]
    for obj_row in ThresholdSweeper().sweep(l_best, l_best):
      obj_answerer = Answerer(obj_row.f_threshold)
      i_found = sum(1 for l_scores in l_score_lists if obj_answerer.is_found(self.hits(l_scores)))
      assert obj_row.i_answerable_rejected == len(l_score_lists) - i_found
      assert obj_row.i_unanswerable_accepted == i_found
