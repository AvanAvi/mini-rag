"""Example of a custom embedder you can copy: a hashed bag of words, standard library only.

It is a starting point, not a good embedding model. It only matches texts that share words.
"""

import re
import zlib

from mini_rag.embedder import Embedder


class HashedBagOfWordsEmbedder(Embedder):
  """Counts each word of a text in one of i_dim buckets chosen by a hash of the word."""

  def __init__(self, i_dim: int = 256) -> None:
    """Store the vector length; the [embedder.options] table of the config supplies it."""
    if i_dim < 1:
      raise ValueError(f"i_dim must be at least 1, got {i_dim}")
    self.i_dim = i_dim
    self.obj_pattern = re.compile(r"[a-z0-9]+")

  @property
  def s_name(self) -> str:
    """Return a name that changes with the settings, so an index built with other settings is rejected."""
    return f"hashed-bag-of-words-{self.i_dim}"

  def embed(self, l_texts: list[str]) -> list[list[float]]:
    """Return one count vector per passage."""
    return [self.vector(s_text) for s_text in l_texts]

  def query_embed(self, s_query: str) -> list[float]:
    """Return the count vector of the query; queries and passages are treated alike here."""
    return self.vector(s_query)

  def vector(self, s_text: str) -> list[float]:
    """Count the words of the text into their buckets."""
    l_vector = [0.0] * self.i_dim
    for s_word in self.obj_pattern.findall(s_text.lower()):
      # crc32 gives the same bucket in every run, unlike Python's built-in hash()
      l_vector[zlib.crc32(s_word.encode("utf-8")) % self.i_dim] += 1.0
    return l_vector
