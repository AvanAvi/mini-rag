"""Data class for one threshold of the gate sweep."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepRow:
  """The two kinds of gate error at one threshold."""

  f_threshold: float
  i_answerable_rejected: int  # answerable questions the gate would call "not found"
  i_answerable_total: int
  i_unanswerable_accepted: int  # unanswerable questions the gate would let through
  i_unanswerable_total: int
