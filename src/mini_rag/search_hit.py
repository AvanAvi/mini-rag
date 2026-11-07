"""Data class for one retrieval result."""

from dataclasses import dataclass

from mini_rag.chunk import Chunk


@dataclass(frozen=True)
class SearchHit:
  """A retrieved chunk together with its relevance score."""

  obj_chunk: Chunk
  f_score: float
