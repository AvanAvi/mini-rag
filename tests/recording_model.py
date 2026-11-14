"""Test double that stands in for fastembed's TextEmbedding and records every use."""

import numpy as np


class RecordingModel:
  """Called like the TextEmbedding class; records each load, embed and query_embed, and any other attribute read."""

  I_DIM = 4  # length of every vector it returns

  def __init__(self) -> None:
    """Start with no recorded events."""
    self.l_events: list[str] = []

  def __call__(self, **d_kwargs: str) -> "RecordingModel":
    """Record a model load (TextEmbedding(model_name=...)) and return itself as the loaded model."""
    self.l_events.append(f"load {d_kwargs['model_name']}")
    return self

  def embed(self, l_texts: list[str]) -> list[np.ndarray]:
    """Record the call and return one constant vector per text."""
    self.l_events.append("embed")
    return [np.ones(self.I_DIM, dtype=np.float32) for _ in l_texts]

  def query_embed(self, s_query: str) -> list[np.ndarray]:
    """Record the call and return one constant query vector."""
    self.l_events.append("query_embed")
    return [np.ones(self.I_DIM, dtype=np.float32)]

  def __getattr__(self, s_attribute: str) -> object:
    """Record a read of any attribute not defined above, then fail as a missing attribute would."""
    self.__dict__.setdefault("l_events", []).append(f"getattr {s_attribute}")
    raise AttributeError(s_attribute)
