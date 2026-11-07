"""Thin wrapper around a fastembed TextEmbedding model."""

import numpy as np
from fastembed import TextEmbedding


class FastEmbedEmbedder:
  """Turns passages and queries into embedding vectors with fastembed."""

  def __init__(self, s_model_name: str) -> None:
    """Load the fastembed model (downloads into FASTEMBED_CACHE_PATH if needed)."""
    self.s_model_name = s_model_name
    self.obj_model = TextEmbedding(model_name=s_model_name)

  def embed(self, l_texts: list[str]) -> np.ndarray:
    """Return one passage embedding per text as a 2-D float32 array."""
    return np.asarray(list(self.obj_model.embed(l_texts)), dtype=np.float32)

  def query_embed(self, s_query: str) -> np.ndarray:
    """Return the query embedding (with the model's query prefix) as a 1-D float32 array."""
    return np.asarray(next(iter(self.obj_model.query_embed(s_query))), dtype=np.float32)
