"""Test double that counts how often passages are embedded."""

import numpy as np


class CountingEmbedder:
  """Wraps another embedder and counts the embed() calls made through it."""

  def __init__(self, obj_inner: object) -> None:
    """Store the wrapped embedder and start the count at zero."""
    self.obj_inner = obj_inner
    self.i_embed_calls = 0

  def embed(self, l_texts: list[str]) -> np.ndarray:
    """Count the call and return the wrapped embedder's passage vectors."""
    self.i_embed_calls += 1
    return self.obj_inner.embed(l_texts)

  def query_embed(self, s_query: str) -> np.ndarray:
    """Return the wrapped embedder's query vector; queries are not counted."""
    return self.obj_inner.query_embed(s_query)
