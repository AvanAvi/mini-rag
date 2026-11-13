"""Tests for ReportWriter and its tables."""

from mini_rag.eval_result import EvalResult
from mini_rag.question import Question
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.report_writer import ReportWriter
from mini_rag.size_result import SizeResult
from mini_rag.sweep_row import SweepRow


class TestReportWriter:
  """Formats a hand-made EvalResult and checks the exact numbers in every table."""

  D_OVERLAP = {"a1": 0.2, "a2": 0.4, "a3": 0.5, "a4": 0.9}

  def outcome(self, s_id: str, i_first_rank: int | None, i_total: int, l_found: list[int]) -> QuestionOutcome:
    """Return an outcome; l_found holds the relevant chunks found at k = 1, 3, 5."""
    return QuestionOutcome(
      s_id=s_id, i_first_rank=i_first_rank, i_total_relevant=i_total,
      d_found_at_k=dict(zip((1, 3, 5), l_found)),
    )

  def size_result(self, i_size: int, l_hybrid: list[QuestionOutcome]) -> SizeResult:
    """Return a SizeResult with fixed BM25, dense, score and sweep values and the given hybrid outcomes."""
    return SizeResult(
      i_size=i_size,
      i_chunk_count=7,
      d_outcomes={
        "BM25": [
          self.outcome("a1", 1, 1, [1, 1, 1]), self.outcome("a2", 2, 2, [0, 1, 2]),
          self.outcome("a3", None, 1, [0, 0, 0]), self.outcome("a4", 4, 1, [0, 0, 1]),
        ],
        "dense": [
          self.outcome("a1", 2, 1, [0, 1, 1]), self.outcome("a2", 1, 2, [1, 1, 1]),
          self.outcome("a3", 3, 1, [0, 1, 1]), self.outcome("a4", None, 1, [0, 0, 0]),
        ],
        "hybrid": l_hybrid,
      },
      d_best_dense={"a1": 0.71, "a2": 0.55, "a3": 0.50, "a4": None, "u1": 0.45, "u2": 0.62},
      l_sweep_rows=[
        SweepRow(0.40, 1, 4, 2, 2), SweepRow(0.50, 1, 4, 1, 2),
        SweepRow(0.55, 2, 4, 0, 2), SweepRow(0.60, 3, 4, 1, 2),
      ],
    )

  def result(self) -> EvalResult:
    """Return a two-size result whose hybrid outcomes differ between the sizes."""
    return EvalResult(
      s_corpus_fingerprint="0123456789abcdef" * 4,
      s_model_name="test-model",
      i_overlap=100,
      i_depth=10,
      i_gate_top_k=5,
      l_size_results=[
        self.size_result(500, [
          self.outcome("a1", 1, 1, [1, 1, 1]), self.outcome("a2", 1, 2, [1, 2, 2]),
          self.outcome("a3", 3, 1, [0, 1, 1]), self.outcome("a4", 4, 1, [0, 0, 1]),
        ]),
        self.size_result(1000, [
          self.outcome("a1", 2, 1, [0, 1, 1]), self.outcome("a2", 1, 2, [1, 2, 2]),
          self.outcome("a3", None, 1, [0, 0, 0]), self.outcome("a4", 2, 1, [0, 1, 1]),
        ]),
      ],
    )

  def questions(self) -> list[Question]:
    """Return four answerable and two unanswerable questions matching the result."""
    l_questions = [
      Question(s_id=s_id, s_text="?", b_answerable=True, s_source_doc="d", s_must_contain="phrase")
      for s_id in ("a1", "a2", "a3", "a4")
    ]
    l_questions += [
      Question(s_id=s_id, s_text="?", b_answerable=False, t_absent_terms=("x",), s_topic="off_topic")
      for s_id in ("u1", "u2")
    ]
    return l_questions

  def markdown(self) -> str:
    """Return the markdown report of the hand-made result."""
    return ReportWriter(self.D_OVERLAP).markdown(self.result(), self.questions())

  def test_sections_come_in_order(self):
    """The header, the five tables, the appendix and the reading note appear in that order."""
    s_markdown = self.markdown()
    l_titles = [
      "# Evaluation results", "## Table 1: Retrieval metrics", "## Table 2: Pairwise comparison by first rank",
      "### Across chunk sizes", "## Table 3: Threshold sweep", "## Table 4: Best dense score",
      "## Table 5: Results by overlap group", "## Appendix: Per-question results", "## How to read this",
    ]
    l_positions = [s_markdown.index(s_title) for s_title in l_titles]
    assert l_positions == sorted(l_positions)
    assert "### Chunk size 500 (7 chunks)" in s_markdown
    assert "### Chunk size 1000 (7 chunks)" in s_markdown

  def test_header_lists_the_run_settings(self):
    """The header shows the date line, the short fingerprint and the settings."""
    s_markdown = self.markdown()
    assert "- Date (UTC): 20" in s_markdown
    for s_line in (
      "- Corpus fingerprint: 0123456789ab\n", "- Embedding model: test-model\n", "- Chunk sizes: 500, 1000\n",
      "- Overlap: 100\n", "- Depth: 10\n", "- Gate top-k: 5\n", "- Questions: 4 answerable, 2 unanswerable\n",
    ):
      assert s_line in s_markdown

  def test_table_1_numbers(self):
    """hit@k is n/N, recall has two decimals and MRR three."""
    s_markdown = self.markdown()
    assert "| Method | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |" in s_markdown
    # BM25: ranks 1, 2, none, 4 -> MRR (1 + 0.5 + 0 + 0.25) / 4; recall@3 (1 + 0.5 + 0 + 0) / 4
    assert "| BM25 | 1/4 | 2/4 | 3/4 | 0.25 | 0.38 | 0.75 | 0.438 |" in s_markdown
    assert "| dense | 1/4 | 3/4 | 3/4 | 0.12 | 0.62 | 0.62 | 0.458 |" in s_markdown
    assert "| hybrid | 2/4 | 3/4 | 4/4 | 0.38 | 0.75 | 1.00 | 0.646 |" in s_markdown

  def test_table_2_counts(self):
    """Pairwise counts treat None as the worst rank and None against None as a tie."""
    s_markdown = self.markdown()
    assert "| Comparison | First better | Second better | Ties |" in s_markdown
    assert "| hybrid vs BM25 | 2 | 0 | 2 |" in s_markdown  # size 500
    assert "| hybrid vs dense | 2 | 0 | 2 |" in s_markdown  # size 500
    assert "| hybrid vs BM25 | 2 | 1 | 1 |" in s_markdown  # size 1000, a3 is None for both
    assert s_markdown.count("| hybrid at 500 vs hybrid at 1000 | 2 | 1 | 1 |") == 1

  def test_table_3_rows_and_lowest_errors(self):
    """The sweep shows n/N counts, the error total and the thresholds with the fewest errors."""
    s_markdown = self.markdown()
    assert "| Threshold | Answerable rejected | Unanswerable accepted | Total errors |" in s_markdown
    assert "| 0.40 | 1/4 | 2/2 | 3 |" in s_markdown
    assert "| 0.50 | 1/4 | 1/2 | 2 |" in s_markdown
    assert "| 0.60 | 3/4 | 1/2 | 4 |" in s_markdown
    assert "Lowest total errors (observation only): 0.50, 0.55" in s_markdown

  def test_table_4_score_spread(self):
    """Min, median and max have three decimals; None scores are counted separately."""
    s_markdown = self.markdown()
    assert "| Group | Min | Median | Max | None |" in s_markdown
    assert "| Answerable | 0.500 | 0.550 | 0.710 | 1 |" in s_markdown
    assert "| Unanswerable | 0.450 | 0.535 | 0.620 | 0 |" in s_markdown

  def test_table_5_overlap_groups(self):
    """Questions are split at overlap 0.40, with 0.40 itself in the low group."""
    s_markdown = self.markdown()
    assert "| Overlap group | Questions | Method | hit@5 | MRR |" in s_markdown
    assert "| <= 0.40 | 2 | BM25 | 2/2 | 0.750 |" in s_markdown
    assert "| <= 0.40 | 2 | hybrid | 2/2 | 1.000 |" in s_markdown
    assert "| > 0.40 | 2 | BM25 | 1/2 | 0.125 |" in s_markdown
    assert "| > 0.40 | 2 | dense | 1/2 | 0.167 |" in s_markdown
    assert "| > 0.40 | 2 | hybrid | 2/2 | 0.292 |" in s_markdown  # size 500: ranks 3 and 4

  def test_table_5_without_overlap_values(self):
    """Without overlap values both groups are empty."""
    s_markdown = ReportWriter().markdown(self.result(), self.questions())
    assert "| <= 0.40 | 0 | BM25 | 0/0 | 0.000 |" in s_markdown

  def test_appendix_rows(self):
    """Each answerable question shows its first ranks, "-" for none, and the best dense score."""
    s_markdown = self.markdown()
    assert "| id | BM25 | dense | hybrid | Best dense score |" in s_markdown
    assert "| a1 | 1 | 2 | 1 | 0.710 |" in s_markdown
    assert "| a3 | - | 3 | 3 | 0.500 |" in s_markdown
    assert "| a4 | 4 | - | 4 | - |" in s_markdown
    assert "| u1 |" not in s_markdown

  def test_how_to_read_names_the_gate_top_k(self):
    """The closing paragraph states the caveats and the swept top-k."""
    s_markdown = self.markdown()
    assert "a difference of 1 to 2 questions is noise" in s_markdown
    assert "tuned on the same questions it is measured on" in s_markdown
    assert s_markdown.endswith("The gate was swept at top-k 5 only.\n")

  def test_console_text_has_tables_1_and_3_only(self):
    """The console version keeps the metrics and the sweep and drops the rest."""
    s_text = ReportWriter(self.D_OVERLAP).console_text(self.result(), self.questions())
    assert "Questions: 4 answerable, 2 unanswerable" in s_text
    assert "Chunk size 500 (7 chunks) - Table 1: retrieval metrics" in s_text
    assert "Chunk size 1000 (7 chunks) - Table 3: threshold sweep" in s_text
    assert "| BM25 | 1/4 | 2/4 | 3/4 | 0.25 | 0.38 | 0.75 | 0.438 |" in s_text
    assert "| 0.50 | 1/4 | 1/2 | 2 |" in s_text
    assert "Comparison" not in s_text
    assert "Best dense score" not in s_text

  def test_write_uses_utf8_and_lf(self, tmp_path):
    """The file is created with its folder, in UTF-8 and without carriage returns."""
    obj_path = tmp_path / "nested" / "results.md"
    ReportWriter(self.D_OVERLAP).write(str(obj_path), self.result(), self.questions())
    arr_bytes = obj_path.read_bytes()
    assert b"\r" not in arr_bytes
    assert arr_bytes.decode("utf-8").startswith("# Evaluation results\n")
