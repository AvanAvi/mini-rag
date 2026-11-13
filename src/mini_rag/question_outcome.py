"""Data class for how one method did on one answerable question."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionOutcome:
  """Where the relevant chunks of a question landed in one method's ranking."""

  s_id: str
  i_first_rank: int | None  # 1-based rank of the first relevant chunk, None if not retrieved
  i_total_relevant: int  # relevant chunks in the whole chunk set
  d_found_at_k: dict[int, int]  # k -> relevant chunks found in the top k
