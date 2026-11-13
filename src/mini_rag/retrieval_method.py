"""Common interface of the retrieval methods the evaluator compares."""

from abc import ABC, abstractmethod

from mini_rag.chunk import Chunk


class RetrievalMethod(ABC):
  """A named way of ranking chunks for a question."""

  @property
  @abstractmethod
  def s_name(self) -> str:
    """Return the name shown in reports."""

  @abstractmethod
  def rank(self, s_question: str, i_depth: int) -> list[Chunk]:
    """Return up to i_depth chunks for the question, best first."""
