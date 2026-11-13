"""Data class for the indexes of one chunk setting."""

from dataclasses import dataclass

from mini_rag.bm25_index import Bm25Index
from mini_rag.chunk import Chunk
from mini_rag.dense_index import DenseIndex


@dataclass
class IndexBundle:
  """The chunks and both indexes for one chunk size and overlap."""

  l_chunks: list[Chunk]
  obj_bm25: Bm25Index
  obj_dense: DenseIndex
  i_size: int
  i_overlap: int
  b_loaded: bool  # True when the dense index came from the cache
