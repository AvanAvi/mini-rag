"""Writes the fusion grid report as a markdown file."""

from pathlib import Path

from mini_rag.dense_method import DenseMethod
from mini_rag.eval_result import EvalResult
from mini_rag.fusion_result import FusionResult
from mini_rag.fusion_setting import FusionSetting
from mini_rag.report_tables import ReportTables
from mini_rag.size_result import SizeResult


class FusionGridWriter:
  """Lays out one table per chunk size with one row per fusion setting."""

  T_KS = (1, 3, 5)
  F_GATE_THRESHOLD = 0.60
  S_DEFAULT_MARK = " (default)"
  S_NOTE = (
    "The grid is tuned on the same {i_questions} questions it is measured on, "
    "so a gain of 2 questions or fewer on hit@5 is noise."
  )

  def __init__(self) -> None:
    """Create the table helper that supplies the metric cells."""
    self.obj_tables = ReportTables()

  def markdown(self, obj_result: EvalResult) -> str:
    """Return the grid report as markdown text with LF line endings."""
    l_lines = ["# Fusion grid", ""] + self.header(obj_result)
    for obj_size in obj_result.l_size_results:
      l_lines += ["", f"## Chunk size {obj_size.i_size} ({obj_size.i_chunk_count} chunks)", ""] + self.table(obj_size)
    l_lines += ["", "## How to read this", "", self.S_NOTE.format(i_questions=self.question_count(obj_result))]
    return "\n".join(l_lines) + "\n"

  def write(self, s_path: str, obj_result: EvalResult) -> None:
    """Write the grid report as UTF-8 with LF line endings, creating the folder if needed."""
    obj_path = Path(s_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    # bytes, so Windows does not turn LF into CRLF
    obj_path.write_bytes(self.markdown(obj_result).encode("utf-8"))

  def header(self, obj_result: EvalResult) -> list[str]:
    """Return the run settings and the meaning of the two extra columns as a bullet list."""
    return [
      f"- Embedding model: {obj_result.s_model_name}",
      f"- Overlap: {obj_result.i_overlap}",
      f"- Depth: {obj_result.i_depth}",
      f"- Gate top-k: {obj_result.i_gate_top_k}",
      "- Setting: RRF constant k and the weights w as BM25/dense.",
      "- vs dense: questions where hybrid ranks its first relevant chunk better / worse / the same as dense alone.",
      f"- Unanswerable accepted at {self.F_GATE_THRESHOLD:.2f}: unanswerable questions the gate lets through "
      f"at that threshold, judged on the top {obj_result.i_gate_top_k} hybrid hits of the setting.",
    ]

  def table(self, obj_size: SizeResult) -> list[str]:
    """Return the grid table of one chunk size: hit@k, MRR, the dense comparison and the gate count."""
    l_titles = ["Setting"] + [f"hit@{i_k}" for i_k in self.T_KS] + ["MRR", "vs dense"]
    l_lines = self.obj_tables.head(l_titles + [f"Unanswerable accepted at {self.F_GATE_THRESHOLD:.2f}"])
    l_dense = obj_size.d_outcomes[DenseMethod.S_NAME]
    for obj_fusion in obj_size.l_fusion_results:
      l_cells = [self.obj_tables.hit_text(obj_fusion.l_outcomes, i_k) for i_k in self.T_KS]
      l_cells.append(self.obj_tables.mrr_text(obj_fusion.l_outcomes))
      l_cells.append(" / ".join(self.obj_tables.compare(obj_fusion.l_outcomes, l_dense)))
      l_lines.append(self.obj_tables.row([self.label(obj_fusion.obj_setting)] + l_cells + [self.accepted_text(obj_fusion)]))
    return l_lines

  def label(self, obj_setting: FusionSetting) -> str:
    """Return the setting's label, marked when it equals the retriever's defaults."""
    return obj_setting.s_label + (self.S_DEFAULT_MARK if obj_setting == FusionSetting.default() else "")

  def accepted_text(self, obj_fusion: FusionResult) -> str:
    """Return "n/N" unanswerable questions accepted at the gate threshold, or "-" if it was not swept."""
    for obj_row in obj_fusion.l_sweep_rows:
      if abs(obj_row.f_threshold - self.F_GATE_THRESHOLD) < 1e-9:
        return f"{obj_row.i_unanswerable_accepted}/{obj_row.i_unanswerable_total}"
    return "-"

  def question_count(self, obj_result: EvalResult) -> int:
    """Return the number of answerable questions the grid was measured on."""
    for obj_size in obj_result.l_size_results:
      for obj_fusion in obj_size.l_fusion_results:
        return len(obj_fusion.l_outcomes)
    return 0
