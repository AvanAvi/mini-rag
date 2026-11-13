"""Retrieval method backed by the dense vector index."""

from mini_rag.chunk import Chunk
from mini_rag.dense_index import DenseIndex
from mini_rag.retrieval_method import RetrievalMethod


class DenseMethod(RetrievalMethod):
  """Ranks chunks with a built or loaded DenseIndex."""

  S_NAME = "dense"

  def __init__(self, obj_index: DenseIndex) -> None:
    """Store the already-built index."""
    self.obj_index = obj_index

  @property
  def s_name(self) -> str:
    """Return the name shown in reports."""
    return self.S_NAME

  def rank(self, s_question: str, i_depth: int) -> list[Chunk]:
    """Return the chunks of the top i_depth dense hits, best first."""
    return [obj_hit.obj_chunk for obj_hit in self.obj_index.search(s_question, i_depth)]
