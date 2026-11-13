"""Builds the markdown tables of the evaluation report."""

import statistics

from mini_rag.bm25_method import Bm25Method
from mini_rag.dense_method import DenseMethod
from mini_rag.hybrid_method import HybridMethod
from mini_rag.metric_calculator import MetricCalculator
from mini_rag.question import Question
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.size_result import SizeResult


class ReportTables:
  """Turns one SizeResult into table lines; every method returns a list of lines."""

  T_KS = (1, 3, 5)
  F_LOW_OVERLAP = 0.40

  def __init__(self, d_overlap: dict[str, float] | None = None) -> None:
    """Store the per-question overlap values used to split Table 5."""
    self.obj_metrics = MetricCalculator()
    self.d_overlap = dict(d_overlap or {})

  def metrics(self, obj_size: SizeResult) -> list[str]:
    """Table 1: hit@k, recall@k and MRR per method."""
    l_titles = ["Method"] + [f"hit@{i_k}" for i_k in self.T_KS] + [f"recall@{i_k}" for i_k in self.T_KS] + ["MRR"]
    l_lines = self.head(l_titles)
    for s_name, l_outcomes in obj_size.d_outcomes.items():
      l_cells = [self.hit_text(l_outcomes, i_k) for i_k in self.T_KS]
      for i_k in self.T_KS:
        f_recall = self.obj_metrics.mean([
          self.obj_metrics.recall_at_k(obj_outcome.d_found_at_k[i_k], obj_outcome.i_total_relevant)
          for obj_outcome in l_outcomes
        ])
        l_cells.append(f"{f_recall:.2f}")
      l_lines.append(self.row([s_name] + l_cells + [self.mrr_text(l_outcomes)]))
    return l_lines

  def pairwise(self, obj_size: SizeResult) -> list[str]:
    """Table 2: hybrid against BM25 and against dense, by first rank per question."""
    l_lines = self.head(["Comparison", "First better", "Second better", "Ties"])
    l_hybrid = obj_size.d_outcomes[HybridMethod.S_NAME]
    for s_other in (Bm25Method.S_NAME, DenseMethod.S_NAME):
      s_label = f"{HybridMethod.S_NAME} vs {s_other}"
      l_lines.append(self.row([s_label] + self.compare(l_hybrid, obj_size.d_outcomes[s_other])))
    return l_lines

  def across_sizes(self, obj_first: SizeResult, obj_second: SizeResult) -> list[str]:
    """Table 2, last part: hybrid at one chunk size against hybrid at another."""
    s_name = HybridMethod.S_NAME
    s_label = f"{s_name} at {obj_first.i_size} vs {s_name} at {obj_second.i_size}"
    l_counts = self.compare(obj_first.d_outcomes[s_name], obj_second.d_outcomes[s_name])
    return self.head(["Comparison", "First better", "Second better", "Ties"]) + [self.row([s_label] + l_counts)]

  def sweep(self, obj_size: SizeResult) -> list[str]:
    """Table 3: gate errors per threshold, then the thresholds with the fewest errors."""
    l_lines = self.head(["Threshold", "Answerable rejected", "Unanswerable accepted", "Total errors"])
    d_errors: dict[float, int] = {}
    for obj_row in obj_size.l_sweep_rows:
      d_errors[obj_row.f_threshold] = obj_row.i_answerable_rejected + obj_row.i_unanswerable_accepted
      l_lines.append(self.row([
        f"{obj_row.f_threshold:.2f}",
        f"{obj_row.i_answerable_rejected}/{obj_row.i_answerable_total}",
        f"{obj_row.i_unanswerable_accepted}/{obj_row.i_unanswerable_total}",
        str(d_errors[obj_row.f_threshold]),
      ]))
    if d_errors:
      i_lowest = min(d_errors.values())
      s_best = ", ".join(f"{f_threshold:.2f}" for f_threshold, i_errors in d_errors.items() if i_errors == i_lowest)
      l_lines += ["", f"Lowest total errors (observation only): {s_best}"]
    return l_lines

  def scores(self, obj_size: SizeResult, l_questions: list[Question]) -> list[str]:
    """Table 4: spread of the best dense score for answerable and unanswerable questions."""
    l_lines = self.head(["Group", "Min", "Median", "Max", "None"])
    for s_group, b_answerable in (("Answerable", True), ("Unanswerable", False)):
      l_best = [
        obj_size.d_best_dense.get(obj_question.s_id)
        for obj_question in l_questions if obj_question.b_answerable == b_answerable
      ]
      l_scores = [f_best for f_best in l_best if f_best is not None]
      l_cells = ["-", "-", "-"]
      if l_scores:
        l_cells = [f"{min(l_scores):.3f}", f"{statistics.median(l_scores):.3f}", f"{max(l_scores):.3f}"]
      l_lines.append(self.row([s_group] + l_cells + [str(len(l_best) - len(l_scores))]))
    return l_lines

  def overlap_groups(self, obj_size: SizeResult) -> list[str]:
    """Table 5: hit@5 and MRR per method, split by the question's word overlap with its source sentence."""
    l_lines = self.head(["Overlap group", "Questions", "Method", "hit@5", "MRR"])
    for s_group, b_low in ((f"<= {self.F_LOW_OVERLAP:.2f}", True), (f"> {self.F_LOW_OVERLAP:.2f}", False)):
      l_ids = [
        s_id for s_id, f_overlap in self.d_overlap.items()
        if (f_overlap <= self.F_LOW_OVERLAP) == b_low
      ]
      for s_name, l_outcomes in obj_size.d_outcomes.items():
        l_group = [obj_outcome for obj_outcome in l_outcomes if obj_outcome.s_id in l_ids]
        l_lines.append(self.row([s_group, str(len(l_group)), s_name, self.hit_text(l_group, 5), self.mrr_text(l_group)]))
    return l_lines

  def appendix(self, obj_size: SizeResult) -> list[str]:
    """Appendix: first rank per method and best dense score for every answerable question."""
    l_names = list(obj_size.d_outcomes)
    l_lines = self.head(["id"] + l_names + ["Best dense score"])
    d_ranks = {
      s_name: {obj_outcome.s_id: obj_outcome.i_first_rank for obj_outcome in l_outcomes}
      for s_name, l_outcomes in obj_size.d_outcomes.items()
    }
    for obj_outcome in (obj_size.d_outcomes[l_names[0]] if l_names else []):
      s_id = obj_outcome.s_id
      l_cells = ["-" if d_ranks[s_name].get(s_id) is None else str(d_ranks[s_name][s_id]) for s_name in l_names]
      f_best = obj_size.d_best_dense.get(s_id)
      l_lines.append(self.row([s_id] + l_cells + ["-" if f_best is None else f"{f_best:.3f}"]))
    return l_lines

  def compare(self, l_first: list[QuestionOutcome], l_second: list[QuestionOutcome]) -> list[str]:
    """Count the questions where the first list ranks better, where the second does, and ties."""
    d_second = {obj_outcome.s_id: obj_outcome.i_first_rank for obj_outcome in l_second}
    i_first = i_second = i_ties = 0
    for obj_outcome in l_first:
      f_a = self.rank_value(obj_outcome.i_first_rank)
      f_b = self.rank_value(d_second.get(obj_outcome.s_id))
      if f_a < f_b:
        i_first += 1
      elif f_b < f_a:
        i_second += 1
      else:
        i_ties += 1
    return [str(i_first), str(i_second), str(i_ties)]

  def rank_value(self, i_first_rank: int | None) -> float:
    """Return the rank as a number, with None as the worst possible rank."""
    return float("inf") if i_first_rank is None else float(i_first_rank)

  def hit_text(self, l_outcomes: list[QuestionOutcome], i_k: int) -> str:
    """Return hit@k as "n/N"."""
    i_hits = sum(1 for obj_outcome in l_outcomes if self.obj_metrics.hit_at_k(obj_outcome.i_first_rank, i_k))
    return f"{i_hits}/{len(l_outcomes)}"

  def mrr_text(self, l_outcomes: list[QuestionOutcome]) -> str:
    """Return the mean reciprocal rank with three decimals."""
    f_mrr = self.obj_metrics.mean([
      self.obj_metrics.reciprocal_rank(obj_outcome.i_first_rank) for obj_outcome in l_outcomes
    ])
    return f"{f_mrr:.3f}"

  def head(self, l_titles: list[str]) -> list[str]:
    """Return the header row and the separator row of a table."""
    return [self.row(l_titles), "|" + "---|" * len(l_titles)]

  def row(self, l_cells: list[str]) -> str:
    """Return one markdown table row."""
    return "| " + " | ".join(l_cells) + " |"
