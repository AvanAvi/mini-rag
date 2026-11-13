"""Data class for the outcome of validating a question set."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
  """The problems found, plus per-question figures for answerable questions."""

  l_problems: list[str]
  d_overlap: dict[str, float]  # id -> share of question tokens found in the source sentence
  d_phrase_docs: dict[str, int]  # id -> number of documents containing the phrase
