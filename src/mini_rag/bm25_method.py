"""Retrieval method backed by the BM25 keyword index."""

from mini_rag.bm25_index import Bm25Index
from mini_rag.chunk import Chunk
from mini_rag.retrieval_method import RetrievalMethod


class Bm25Method(RetrievalMethod):
  """Ranks chunks with a built Bm25Index."""

  S_NAME = "BM25"

  def __init__(self, obj_index: Bm25Index) -> None:
    """Store the already-built index."""
    self.obj_index = obj_index

  @property
  def s_name(self) -> str:
    """Return the name shown in reports."""
    return self.S_NAME

  def rank(self, s_question: str, i_depth: int) -> list[Chunk]:
    """Return the chunks of the top i_depth BM25 hits, best first."""
    return [obj_hit.obj_chunk for obj_hit in self.obj_index.search(s_question, i_depth)]
