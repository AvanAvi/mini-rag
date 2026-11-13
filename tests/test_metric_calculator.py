"""Tests for MetricCalculator."""

import pytest

from mini_rag.metric_calculator import MetricCalculator


class TestMetricCalculator:
  """Checks every metric against hand-computed values."""

  def test_hit_at_k(self):
    """A hit needs a first rank at or above the cut-off; None is never a hit."""
    obj_metrics = MetricCalculator()
    assert obj_metrics.hit_at_k(1, 1) is True
    assert obj_metrics.hit_at_k(3, 3) is True
    assert obj_metrics.hit_at_k(4, 3) is False
    assert obj_metrics.hit_at_k(None, 5) is False

  def test_recall_with_two_relevant_chunks(self):
    """A phrase in two overlapping chunks caps recall@1 at 0.5."""
    obj_metrics = MetricCalculator()
    assert obj_metrics.recall_at_k(1, 2) == 0.5
    assert obj_metrics.recall_at_k(2, 2) == 1.0
    assert obj_metrics.recall_at_k(0, 2) == 0.0

  def test_recall_without_relevant_chunks_is_zero(self):
    """No relevant chunk in the chunk set gives recall 0.0 instead of a division error."""
    assert MetricCalculator().recall_at_k(0, 0) == 0.0

  def test_reciprocal_rank(self):
    """The reciprocal rank is 1 / rank, and 0.0 for None."""
    obj_metrics = MetricCalculator()
    assert obj_metrics.reciprocal_rank(1) == 1.0
    assert obj_metrics.reciprocal_rank(4) == 0.25
    assert obj_metrics.reciprocal_rank(None) == 0.0

  def test_mean(self):
    """The mean of 1, 0.5, 0 and 0.25 is 0.4375; an empty list gives 0.0."""
    obj_metrics = MetricCalculator()
    assert obj_metrics.mean([1.0, 0.5, 0.0, 0.25]) == pytest.approx(0.4375)
    assert obj_metrics.mean([]) == 0.0
