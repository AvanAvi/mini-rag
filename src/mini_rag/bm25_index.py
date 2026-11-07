"""Sparse keyword index built on rank-bm25."""


class Bm25Index:
  """Ranks chunks by BM25 keyword relevance to a query."""

  def __init__(self) -> None:
    """Prepare an empty index."""
    raise NotImplementedError

  def build(self, l_chunks: list[dict]) -> None:
    """Tokenise all chunks and build the BM25 model."""
    raise NotImplementedError

  def search(self, s_query: str, i_top_k: int) -> list[dict]:
    """Return the top-k chunks with their BM25 scores."""
    raise NotImplementedError
