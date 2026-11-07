"""Loads raw documents from the corpus folder."""


class Loader:
  """Reads text documents from a corpus directory."""

  def __init__(self, s_corpus_dir: str) -> None:
    """Store the corpus directory to read from."""
    raise NotImplementedError

  def load(self) -> list[dict]:
    """Return every document as a dict with its source path and text."""
    raise NotImplementedError
