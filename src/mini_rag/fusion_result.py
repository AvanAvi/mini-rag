"""Results of the hybrid method under one fusion setting."""

from dataclasses import dataclass

from mini_rag.fusion_setting import FusionSetting
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.sweep_row import SweepRow


@dataclass(frozen=True)
class FusionResult:
  """Hybrid outcomes, gate scores and the threshold sweep for one fusion setting at one chunk size."""

  obj_setting: FusionSetting
  l_outcomes: list[QuestionOutcome]  # one outcome per answerable question
  d_best_dense: dict[str, float | None]  # question id -> best dense score among the gate hits
  l_sweep_rows: list[SweepRow]
