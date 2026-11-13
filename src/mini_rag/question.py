"""Data class for one evaluation question."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
  """An evaluation question; fields that do not apply are "" or an empty tuple."""

  s_id: str
  s_text: str
  b_answerable: bool
  s_source_doc: str = ""  # answerable: the document that holds the answer
  s_must_contain: str = ""  # answerable: phrase a relevant chunk must contain
  t_absent_terms: tuple[str, ...] = ()  # unanswerable: terms a correct answer would need
  s_topic: str = ""  # unanswerable: "on_topic" or "off_topic"
