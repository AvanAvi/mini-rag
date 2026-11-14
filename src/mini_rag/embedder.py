"""Interface every embedding model wrapper implements."""

from abc import ABC, abstractmethod


class Embedder(ABC):
  """Turns passages and queries into vectors; subclass it to plug in another embedding model."""

  @property
  @abstractmethod
  def s_name(self) -> str:
    """Return a name that identifies the model and its settings; it is saved with the index and checked on load."""

  @abstractmethod
  def embed(self, l_texts: list[str]) -> object:
    """Return one vector per text, as a list of equal-length float lists or a 2-D array."""

  @abstractmethod
  def query_embed(self, s_query: str) -> object:
    """Return the vector of one query, as a float list or a 1-D array of the same length."""
