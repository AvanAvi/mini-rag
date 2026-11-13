"""Sweeps the answer gate's threshold over best dense scores."""

from mini_rag.sweep_row import SweepRow


class ThresholdSweeper:
  """Counts gate errors at each threshold, using the same rule as Answerer.is_found."""

  def __init__(self, l_thresholds: list[float] | None = None) -> None:
    """Store the thresholds; the default is 0.40 to 0.80 in steps of 0.05."""
    if l_thresholds is None:
      l_thresholds = [round(0.40 + 0.05 * i_step, 2) for i_step in range(9)]
    self.l_thresholds = list(l_thresholds)

  def sweep(self, l_best_answerable: list[float | None], l_best_unanswerable: list[float | None]) -> list[SweepRow]:
    """Return one row per threshold from the best dense score of every question."""
    return [
      SweepRow(
        f_threshold=f_threshold,
        i_answerable_rejected=sum(1 for f_best in l_best_answerable if not self.is_found(f_best, f_threshold)),
        i_answerable_total=len(l_best_answerable),
        i_unanswerable_accepted=sum(1 for f_best in l_best_unanswerable if self.is_found(f_best, f_threshold)),
        i_unanswerable_total=len(l_best_unanswerable),
      )
      for f_threshold in self.l_thresholds
    ]

  def is_found(self, f_best: float | None, f_threshold: float) -> bool:
    """Return True if there is a best dense score and it reaches the threshold."""
    return f_best is not None and f_best >= f_threshold
