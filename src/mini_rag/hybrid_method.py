"""Retrieval method backed by the hybrid retriever."""

from mini_rag.chunk import Chunk
from mini_rag.hybrid_hit import HybridHit
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.retrieval_method import RetrievalMethod


class HybridMethod(RetrievalMethod):
  """Ranks chunks with a HybridRetriever and can also return its full hits."""

  S_NAME = "hybrid"

  def __init__(self, obj_retriever: HybridRetriever) -> None:
    """Store the retriever, which already holds both indexes."""
    self.obj_retriever = obj_retriever

  @property
  def s_name(self) -> str:
    """Return the name shown in reports."""
    return self.S_NAME

  def rank(self, s_question: str, i_depth: int) -> list[Chunk]:
    """Return the chunks of the top i_depth fused hits, best first."""
    return [obj_hit.obj_chunk for obj_hit in self.hits(s_question, i_depth)]

  def hits(self, s_question: str, i_depth: int) -> list[HybridHit]:
    """Return the top i_depth fused hits with their dense and BM25 scores."""
    return self.obj_retriever.search(s_question, i_depth)
