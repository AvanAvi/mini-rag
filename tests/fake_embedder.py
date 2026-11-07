"""Test double that returns hand-made vectors instead of running a model."""

import numpy as np


class FakeEmbedder:
  """Looks up fixed vectors for passages and queries by their exact text."""

  def __init__(self, d_passages: dict[str, list[float]], d_queries: dict[str, list[float]]) -> None:
    """Store the text-to-vector tables and start with no recorded calls."""
    self.d_passages = d_passages
    self.d_queries = d_queries
    self.l_query_calls: list[str] = []

  def embed(self, l_texts: list[str]) -> np.ndarray:
    """Return the passage vector of every text as a 2-D array."""
    return np.asarray([self.d_passages[s_text] for s_text in l_texts], dtype=np.float32)

  def query_embed(self, s_query: str) -> np.ndarray:
    """Record the call and return the query vector as a 1-D array."""
    self.l_query_calls.append(s_query)
    return np.asarray(self.d_queries[s_query], dtype=np.float32)
