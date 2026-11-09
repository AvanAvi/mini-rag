"""Data class for the result of answering a question."""

from dataclasses import dataclass

from mini_rag.chunk import Chunk


@dataclass(frozen=True)
class Answer:
  """Answer text, the chunks it draws on, whether the corpus had it, and whether a model wrote it."""

  s_text: str
  l_sources: list[Chunk]
  b_found: bool
  b_generated: bool  # True only when a model wrote s_text
