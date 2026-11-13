"""Writes the evaluation report for the console and as a markdown file."""

from datetime import datetime, timezone
from pathlib import Path

from mini_rag.eval_result import EvalResult
from mini_rag.question import Question
from mini_rag.report_tables import ReportTables
from mini_rag.size_result import SizeResult


class ReportWriter:
  """Lays out the tables of an EvalResult; the numbers come from ReportTables."""

  S_HOW_TO_READ = (
    "The question set is small, so a difference of 1 to 2 questions is noise. "
    "Hand-written questions tend to share words with the text, which favours keyword search. "
    "A threshold picked from Table 3 is tuned on the same questions it is measured on "
    "and will look better than it performs on new questions. "
    "The gate was swept at top-k {i_gate_top_k} only."
  )

  def __init__(self, d_overlap: dict[str, float] | None = None) -> None:
    """Store the per-question overlap values (from ValidationResult.d_overlap) for Table 5."""
    self.obj_tables = ReportTables(d_overlap)

  def console_text(self, obj_result: EvalResult, l_questions: list[Question]) -> str:
    """Return the compact report: Table 1 and Table 3 per chunk size."""
    l_lines = [self.count_text(l_questions)]
    for obj_size in obj_result.l_size_results:
      l_lines += ["", f"{self.size_title(obj_size)} - Table 1: retrieval metrics"] + self.obj_tables.metrics(obj_size)
      l_lines += ["", f"{self.size_title(obj_size)} - Table 3: threshold sweep"] + self.obj_tables.sweep(obj_size)
    return "\n".join(l_lines)

  def markdown(self, obj_result: EvalResult, l_questions: list[Question]) -> str:
    """Return the full report as markdown text with LF line endings."""
    l_sizes = obj_result.l_size_results
    l_lines = ["# Evaluation results", ""] + self.header(obj_result, l_questions)
    l_lines += self.section("Table 1: Retrieval metrics", l_sizes, self.obj_tables.metrics)
    l_lines += self.section("Table 2: Pairwise comparison by first rank", l_sizes, self.obj_tables.pairwise)
    if len(l_sizes) >= 2:
      l_lines += ["", "### Across chunk sizes", ""] + self.obj_tables.across_sizes(l_sizes[0], l_sizes[1])
    l_lines += self.section("Table 3: Threshold sweep", l_sizes, self.obj_tables.sweep)
    l_lines += self.section(
      "Table 4: Best dense score", l_sizes, lambda obj_size: self.obj_tables.scores(obj_size, l_questions)
    )
    l_lines += self.section("Table 5: Results by overlap group", l_sizes, self.obj_tables.overlap_groups)
    l_lines += self.section("Appendix: Per-question results", l_sizes, self.obj_tables.appendix)
    l_lines += ["", "## How to read this", "", self.S_HOW_TO_READ.format(i_gate_top_k=obj_result.i_gate_top_k)]
    return "\n".join(l_lines) + "\n"

  def write(self, s_path: str, obj_result: EvalResult, l_questions: list[Question]) -> None:
    """Write the markdown report as UTF-8 with LF line endings, creating the folder if needed."""
    obj_path = Path(s_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    # bytes, so Windows does not turn LF into CRLF
    obj_path.write_bytes(self.markdown(obj_result, l_questions).encode("utf-8"))

  def header(self, obj_result: EvalResult, l_questions: list[Question]) -> list[str]:
    """Return the settings of the run as a bullet list."""
    s_sizes = ", ".join(str(obj_size.i_size) for obj_size in obj_result.l_size_results)
    return [
      f"- Date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
      f"- Corpus fingerprint: {obj_result.s_corpus_fingerprint[:12]}",
      f"- Embedding model: {obj_result.s_model_name}",
      f"- Chunk sizes: {s_sizes}",
      f"- Overlap: {obj_result.i_overlap}",
      f"- Depth: {obj_result.i_depth}",
      f"- Gate top-k: {obj_result.i_gate_top_k}",
      f"- {self.count_text(l_questions)}",
    ]

  def section(self, s_title: str, l_sizes: list[SizeResult], obj_table: object) -> list[str]:
    """Return a titled section with one table per chunk size, built by obj_table(obj_size)."""
    l_lines = ["", f"## {s_title}"]
    for obj_size in l_sizes:
      l_lines += ["", f"### {self.size_title(obj_size)}", ""] + obj_table(obj_size)
    return l_lines

  def size_title(self, obj_size: SizeResult) -> str:
    """Return the heading of one chunk size."""
    return f"Chunk size {obj_size.i_size} ({obj_size.i_chunk_count} chunks)"

  def count_text(self, l_questions: list[Question]) -> str:
    """Return the number of answerable and unanswerable questions as one line."""
    i_answerable = sum(1 for obj_question in l_questions if obj_question.b_answerable)
    return f"Questions: {i_answerable} answerable, {len(l_questions) - i_answerable} unanswerable"
