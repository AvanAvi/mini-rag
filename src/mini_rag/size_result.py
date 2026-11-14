"""Data class for the evaluation results at one chunk size."""

from dataclasses import dataclass, field

from mini_rag.fusion_result import FusionResult
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.sweep_row import SweepRow


@dataclass
class SizeResult:
  """Everything measured with one chunk size."""

  i_size: int
  i_chunk_count: int
  d_outcomes: dict[str, list[QuestionOutcome]]  # method name -> one outcome per answerable question
  d_best_dense: dict[str, float | None]  # question id -> best dense score among the gate hits
  l_sweep_rows: list[SweepRow]
  # hybrid under every evaluated fusion setting; the fields above hold the main setting
  l_fusion_results: list[FusionResult] = field(default_factory=list)
