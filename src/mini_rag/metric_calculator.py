"""Retrieval metrics for one question and their averages."""


class MetricCalculator:
  """Computes hit@k, recall@k, reciprocal rank and a plain mean."""

  def hit_at_k(self, i_first_rank: int | None, i_k: int) -> bool:
    """Return True if the first relevant chunk sits at rank i_k or better (ranks start at 1)."""
    return i_first_rank is not None and i_first_rank <= i_k

  def recall_at_k(self, i_found: int, i_total: int) -> float:
    """Return the relevant chunks found in the top k divided by all relevant chunks."""
    # a phrase can sit in two overlapping chunks, so recall@1 is at most 0.5 for such a question
    if i_total == 0:
      return 0.0
    return i_found / i_total

  def reciprocal_rank(self, i_first_rank: int | None) -> float:
    """Return 1 / rank of the first relevant chunk, or 0.0 when none was retrieved."""
    if i_first_rank is None:
      return 0.0
    return 1.0 / i_first_rank

  def mean(self, l_values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    if not l_values:
      return 0.0
    return sum(l_values) / len(l_values)
