"""Thin wrapper around a fastembed TextEmbedding model."""

import numpy as np
from fastembed import TextEmbedding

from mini_rag.embedder import Embedder


class FastEmbedEmbedder(Embedder):
  """Turns passages and queries into embedding vectors with fastembed."""

  def __init__(self, s_model_name: str) -> None:
    """Store the model name; the model is loaded on first embed, so reading s_name loads nothing."""
    self.s_model_name = s_model_name
    self.obj_model = None

  def model(self) -> object:
    """Return the fastembed model, loading it on first use (downloads into FASTEMBED_CACHE_PATH if needed)."""
    if self.obj_model is None:
      self.obj_model = TextEmbedding(model_name=self.s_model_name)
    return self.obj_model

  @property
  def s_name(self) -> str:
    """Return the fastembed model name, which identifies this embedder in a saved index."""
    return self.s_model_name

  def embed(self, l_texts: list[str]) -> np.ndarray:
    """Return one passage embedding per text as a 2-D float32 array."""
    return np.asarray(list(self.model().embed(l_texts)), dtype=np.float32)

  def query_embed(self, s_query: str) -> np.ndarray:
    """Return the query embedding (with the model's query prefix) as a 1-D float32 array."""
    return np.asarray(next(iter(self.model().query_embed(s_query))), dtype=np.float32)
