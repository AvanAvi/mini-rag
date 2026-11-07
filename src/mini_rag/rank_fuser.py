"""Reciprocal Rank Fusion of several rankings."""

from mini_rag.chunk import Chunk
from mini_rag.search_hit import SearchHit


class RankFuser:
  """Merges best-first rankings into one list using Reciprocal Rank Fusion."""

  def fuse(self, l_rankings: list[list[SearchHit]], i_k: int) -> list[tuple[Chunk, float]]:
    """Return (chunk, RRF score) pairs, best first; each ranking adds 1 / (i_k + rank)."""
    d_scores: dict[tuple[str, int], float] = {}
    # insertion order records first appearance across rankings, in the order given
    d_chunks: dict[tuple[str, int], Chunk] = {}

    for l_ranking in l_rankings:
      d_best_rank = self.best_ranks(l_ranking, d_chunks)
      for t_key, i_rank in d_best_rank.items():
        d_scores[t_key] = d_scores.get(t_key, 0.0) + 1.0 / (i_k + i_rank)

    # sorted() is stable, so equal scores keep first-appearance order
    l_ordered = sorted(d_chunks.values(), key=lambda obj_chunk: -d_scores[self.key(obj_chunk)])
    return [(obj_chunk, d_scores[self.key(obj_chunk)]) for obj_chunk in l_ordered]

  def best_ranks(
    self, l_ranking: list[SearchHit], d_chunks: dict[tuple[str, int], Chunk]
  ) -> dict[tuple[str, int], int]:
    """Map each chunk id in one ranking to its best 1-based rank and record unseen chunks."""
    d_best_rank: dict[tuple[str, int], int] = {}
    for i_rank, obj_hit in enumerate(l_ranking, start=1):
      t_key = self.key(obj_hit.obj_chunk)
      # the list is best-first, so the first occurrence holds the best rank
      if t_key not in d_best_rank:
        d_best_rank[t_key] = i_rank
      if t_key not in d_chunks:
        d_chunks[t_key] = obj_hit.obj_chunk
    return d_best_rank

  def key(self, obj_chunk: Chunk) -> tuple[str, int]:
    """Return the identity of a chunk: (document id, chunk index)."""
    return (obj_chunk.s_doc_id, obj_chunk.i_index)
