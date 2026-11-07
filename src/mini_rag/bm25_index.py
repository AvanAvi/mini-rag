"""Sparse keyword index built on rank-bm25."""

from rank_bm25 import BM25Okapi

from mini_rag.chunk import Chunk
from mini_rag.search_hit import SearchHit
from mini_rag.tokenizer import Tokenizer


class Bm25Index:
  """Ranks chunks by BM25 keyword relevance to a query."""

  def __init__(self) -> None:
    """Prepare an empty index with its own tokenizer."""
    self.obj_tokenizer = Tokenizer()
    self.l_chunks: list[Chunk] = []
    self.obj_bm25: BM25Okapi | None = None

  def build(self, l_chunks: list[Chunk]) -> None:
    """Tokenise all chunks and build the BM25 model."""
    if not l_chunks:
      raise ValueError("Cannot build a BM25 index from an empty chunk list")
    l_corpus = [self.obj_tokenizer.tokenize(obj_chunk.s_text) for obj_chunk in l_chunks]
    # BM25Okapi divides by the average document length, which would be 0 here
    if not any(l_corpus):
      raise ValueError("Cannot build a BM25 index: every chunk produced zero tokens")
    self.l_chunks = list(l_chunks)
    self.obj_bm25 = BM25Okapi(l_corpus)

  def search(self, s_query: str, i_top_k: int) -> list[SearchHit]:
    """Return up to i_top_k hits with a positive BM25 score, best first."""
    if self.obj_bm25 is None:
      raise RuntimeError("build() must be called before search()")
    if i_top_k < 1:
      raise ValueError(f"i_top_k must be at least 1, got {i_top_k}")

    l_query = self.obj_tokenizer.tokenize(s_query)
    if not l_query:
      return []

    l_scores = [float(f_score) for f_score in self.obj_bm25.get_scores(l_query)]
    # sorted() is stable, so equal scores keep chunk order
    l_order = sorted(range(len(l_scores)), key=lambda i_pos: -l_scores[i_pos])
    l_hits = [
      SearchHit(obj_chunk=self.l_chunks[i_pos], f_score=l_scores[i_pos])
      for i_pos in l_order if l_scores[i_pos] > 0
    ]
    return l_hits[:i_top_k]
