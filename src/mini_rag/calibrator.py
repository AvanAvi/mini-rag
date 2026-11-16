"""Sweeps the gate threshold for one embedder, chunk setting and fusion setting."""

import math

from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.evaluator import Evaluator
from mini_rag.fusion_setting import FusionSetting
from mini_rag.index_bundle import IndexBundle
from mini_rag.question import Question
from mini_rag.size_result import SizeResult
from mini_rag.threshold_sweeper import ThresholdSweeper


class Calibrator:
  """Runs only the threshold sweep of the evaluation and names the thresholds with the fewest errors."""

  F_STEP = 0.05
  I_MIN_ANSWERABLE = 10
  I_MIN_UNANSWERABLE = 3

  def __init__(
    self, obj_store: EvalIndexStore, i_size: int, i_overlap: int, obj_setting: FusionSetting, i_top_k: int,
    s_gate_mode: str = "hybrid",
  ) -> None:
    """Store the index store, the chunk setting, the fusion setting, the gate's top-k and the gate's hit source."""
    self.obj_setting = obj_setting
    self.obj_evaluator = Evaluator(
      obj_store, [i_size], i_overlap, i_gate_top_k=i_top_k, l_fusion_settings=[obj_setting], s_gate_mode=s_gate_mode,
    )

  def prepare(self, l_docs: list[Document]) -> IndexBundle:
    """Build or load the indexes of the chunk setting."""
    return self.obj_evaluator.prepare(l_docs)[0]

  def run(self, obj_bundle: IndexBundle, l_questions: list[Question]) -> SizeResult:
    """Return the best dense score per question and the sweep rows; no retrieval metric is computed."""
    obj_gate = self.obj_evaluator.gate_method(obj_bundle, self.obj_setting)
    d_best_dense = {
      obj_question.s_id: self.obj_evaluator.best_dense(obj_gate, obj_question) for obj_question in l_questions
    }
    l_scores = [f_best for f_best in d_best_dense.values() if f_best is not None]
    l_sweep_rows = ThresholdSweeper(self.thresholds(l_scores)).sweep(
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if obj_question.b_answerable],
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if not obj_question.b_answerable],
    )
    return SizeResult(
      i_size=obj_bundle.i_size, i_chunk_count=len(obj_bundle.l_chunks),
      d_outcomes={}, d_best_dense=d_best_dense, l_sweep_rows=l_sweep_rows,
    )

  def thresholds(self, l_scores: list[float]) -> list[float] | None:
    """Return thresholds in steps of 0.05 that span the measured scores, or None to use the sweeper's own."""
    if not l_scores:
      return None
    # work in whole steps so the values are exact multiples of 0.05
    i_low = max(math.floor(min(l_scores) / self.F_STEP), round(-1.0 / self.F_STEP))
    i_high = min(math.ceil(max(l_scores) / self.F_STEP), round(1.0 / self.F_STEP))
    return [round(i_step * self.F_STEP, 2) for i_step in range(i_low, i_high + 1)]

  def best_thresholds(self, obj_size: SizeResult) -> list[float]:
    """Return every swept threshold with the lowest total of both error kinds, lowest first."""
    d_errors = {
      obj_row.f_threshold: obj_row.i_answerable_rejected + obj_row.i_unanswerable_accepted
      for obj_row in obj_size.l_sweep_rows
    }
    if not d_errors:
      return []
    i_lowest = min(d_errors.values())
    return sorted(f_threshold for f_threshold, i_errors in d_errors.items() if i_errors == i_lowest)

  def separation_warning(self, obj_size: SizeResult, f_threshold: float) -> str | None:
    """Return a warning when, at f_threshold, every unanswerable question gets through or every answerable one is rejected."""
    for obj_row in obj_size.l_sweep_rows:
      if abs(obj_row.f_threshold - f_threshold) > 1e-9:
        continue
      b_all_accepted = obj_row.i_unanswerable_total > 0 and obj_row.i_unanswerable_accepted == obj_row.i_unanswerable_total
      b_all_rejected = obj_row.i_answerable_total > 0 and obj_row.i_answerable_rejected == obj_row.i_answerable_total
      if b_all_accepted or b_all_rejected:
        s_what = "every unanswerable question is accepted" if b_all_accepted else "every answerable question is rejected"
        return (
          f"warning: at {f_threshold:.2f} {s_what}; the embedder cannot separate the two groups "
          "on this question file, so do not use the suggested value"
        )
    return None

  def warnings(self, l_questions: list[Question]) -> list[str]:
    """Return a warning for each question group that is too small to calibrate on."""
    i_answerable = sum(1 for obj_question in l_questions if obj_question.b_answerable)
    i_unanswerable = len(l_questions) - i_answerable
    l_warnings = []
    if i_answerable < self.I_MIN_ANSWERABLE:
      l_warnings.append(
        f"warning: only {i_answerable} answerable questions (fewer than {self.I_MIN_ANSWERABLE}); the result is unreliable"
      )
    if i_unanswerable < self.I_MIN_UNANSWERABLE:
      l_warnings.append(
        f"warning: only {i_unanswerable} unanswerable questions (fewer than {self.I_MIN_UNANSWERABLE}); the result is unreliable"
      )
    return l_warnings
