"""Test double that embeds text as word counts over a fixed vocabulary."""

import re

import numpy as np


class BagOfWordsEmbedder:
  """Deterministic embedder: one dimension per vocabulary word, valued by its count."""

  def __init__(self, l_vocabulary: list[str]) -> None:
    """Store the vocabulary (lowercase words) and the word pattern."""
    self.l_vocabulary = l_vocabulary
    self.obj_pattern = re.compile(r"[a-z0-9]+")

  def embed(self, l_texts: list[str]) -> np.ndarray:
    """Return the count vector of every text as a 2-D array."""
    return np.asarray([self.vector(s_text) for s_text in l_texts], dtype=np.float32)

  def query_embed(self, s_query: str) -> np.ndarray:
    """Return the count vector of the query as a 1-D array."""
    return np.asarray(self.vector(s_query), dtype=np.float32)

  def vector(self, s_text: str) -> list[float]:
    """Count how often each vocabulary word occurs in the text."""
    l_words = self.obj_pattern.findall(s_text.lower())
    return [float(l_words.count(s_word)) for s_word in self.l_vocabulary]
