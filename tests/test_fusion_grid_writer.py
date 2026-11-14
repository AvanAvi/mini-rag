"""Unit tests for the FusionGridWriter."""

from pathlib import Path

from mini_rag.eval_result import EvalResult
from mini_rag.fusion_grid_writer import FusionGridWriter
from mini_rag.fusion_result import FusionResult
from mini_rag.fusion_setting import FusionSetting
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.size_result import SizeResult
from mini_rag.sweep_row import SweepRow


class TestFusionGridWriter:
  """Checks the grid rows against hand-made outcomes."""

  def outcomes(self, l_ranks: list[int | None]) -> list[QuestionOutcome]:
    """Return one outcome per first rank, for questions q1, q2, ..."""
    return [
      QuestionOutcome(s_id=f"q{i_number}", i_first_rank=i_rank, i_total_relevant=1, d_found_at_k={1: 0, 3: 0, 5: 0})
      for i_number, i_rank in enumerate(l_ranks, start=1)
    ]

  def sweep_row(self, f_threshold: float, i_accepted: int) -> SweepRow:
    """Return a sweep row that accepts i_accepted of 2 unanswerable questions."""
    return SweepRow(
      f_threshold=f_threshold, i_answerable_rejected=0, i_answerable_total=3,
      i_unanswerable_accepted=i_accepted, i_unanswerable_total=2,
    )

  def result(self) -> EvalResult:
    """Return one chunk size with dense ranks (1, 3, None) and two fusion settings."""
    # default setting: ranks (2, 1, None) -> better than dense on q2, worse on q1, tied on q3
    obj_default = FusionResult(
      obj_setting=FusionSetting.default(), l_outcomes=self.outcomes([2, 1, None]), d_best_dense={},
      l_sweep_rows=[self.sweep_row(0.55, 2), self.sweep_row(0.60, 1)],
    )
    # second setting: ranks (1, 3, 4) -> better on q3, tied on q1 and q2; 0.60 was not swept
    obj_other = FusionResult(
      obj_setting=FusionSetting(i_k=5, f_bm25_weight=0.5, f_dense_weight=1.0),
      l_outcomes=self.outcomes([1, 3, 4]), d_best_dense={}, l_sweep_rows=[self.sweep_row(0.55, 2)],
    )
    obj_size = SizeResult(
      i_size=500, i_chunk_count=7,
      d_outcomes={"dense": self.outcomes([1, 3, None]), "hybrid": obj_default.l_outcomes},
      d_best_dense={}, l_sweep_rows=obj_default.l_sweep_rows, l_fusion_results=[obj_default, obj_other],
    )
    return EvalResult(
      s_corpus_fingerprint="abc", s_model_name="test-model", i_overlap=100, i_depth=10, i_gate_top_k=5,
      l_size_results=[obj_size],
    )

  def test_rows_match_hand_computed_values(self) -> None:
    """Each row shows hit@k, MRR, better / worse / tied against dense and the accepted count."""
    l_lines = FusionGridWriter().table(self.result().l_size_results[0])
    assert l_lines[0] == "| Setting | hit@1 | hit@3 | hit@5 | MRR | vs dense | Unanswerable accepted at 0.60 |"
    # MRR: (1/2 + 1 + 0) / 3 = 0.500 and (1 + 1/3 + 1/4) / 3 = 0.528
    assert l_lines[2] == "| k=60 w=1.0/1.0 (default) | 1/3 | 2/3 | 2/3 | 0.500 | 1 / 1 / 1 | 1/2 |"
    assert l_lines[3] == "| k=5 w=0.5/1.0 | 1/3 | 2/3 | 3/3 | 0.528 | 1 / 0 / 2 | - |"
    assert len(l_lines) == 4

  def test_markdown_has_size_heading_and_reading_note(self) -> None:
    """The report names the chunk size and ends with the noise note for the question count."""
    s_text = FusionGridWriter().markdown(self.result())
    assert "## Chunk size 500 (7 chunks)" in s_text
    assert "tuned on the same 3 questions" in s_text
    assert "a gain of 2 questions or fewer on hit@5 is noise" in s_text
    assert s_text.count("(default)") == 1

  def test_write_uses_lf_line_endings(self, tmp_path: Path) -> None:
    """The file is written as UTF-8 with LF only, in a folder created on demand."""
    obj_path = tmp_path / "out" / "fusion_grid.md"
    FusionGridWriter().write(str(obj_path), self.result())
    obj_bytes = obj_path.read_bytes()
    assert b"\r" not in obj_bytes
    assert obj_bytes.decode("utf-8") == FusionGridWriter().markdown(self.result())
