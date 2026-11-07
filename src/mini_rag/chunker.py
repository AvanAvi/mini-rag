"""Splits documents into overlapping chunks."""


class Chunker:
  """Cuts document text into fixed-size, overlapping chunks."""

  def __init__(self, i_chunk_size: int, i_overlap: int) -> None:
    """Store the chunk size and overlap, both measured in characters."""
    raise NotImplementedError

  def chunk(self, d_document: dict) -> list[dict]:
    """Split one document into chunk dicts that keep a reference to their source."""
    raise NotImplementedError

  def chunk_all(self, l_documents: list[dict]) -> list[dict]:
    """Split every document and return all chunks in one flat list."""
    raise NotImplementedError
