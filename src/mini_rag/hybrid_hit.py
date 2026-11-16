"""Data class for one fused retrieval result."""

from dataclasses import dataclass

from mini_rag.chunk import Chunk


@dataclass(frozen=True)
class HybridHit:
  """A chunk with its fused RRF score and each retriever's own score (None if it missed the chunk).

  f_rrf_score is None in the "dense" and "bm25" retrieval modes, which fuse nothing.
  """

  obj_chunk: Chunk
  f_rrf_score: float | None
  f_dense_score: float | None
  f_bm25_score: float | None
