"""Splits documents into overlapping chunks."""

from mini_rag.chunk import Chunk
from mini_rag.document import Document


class Chunker:
  """Cuts document text into fixed-size, overlapping character windows."""

  def __init__(self, i_size: int = 500, i_overlap: int = 100) -> None:
    """Validate and store the window size and overlap, both in characters."""
    if i_size < 1:
      raise ValueError(f"i_size must be at least 1, got {i_size}")
    if i_overlap < 0:
      raise ValueError(f"i_overlap must not be negative, got {i_overlap}")
    if i_overlap >= i_size:
      raise ValueError(f"i_overlap ({i_overlap}) must be smaller than i_size ({i_size})")
    self.i_size = i_size
    self.i_overlap = i_overlap
    self.i_step = i_size - i_overlap

  def chunk_document(self, obj_doc: Document) -> list[Chunk]:
    """Split one document into chunks; stop once a chunk reaches the end of the text."""
    s_text = obj_doc.s_text
    i_length = len(s_text)
    l_chunks = []
    i_start = 0
    while i_start < i_length:
      i_end = min(i_start + self.i_size, i_length)
      l_chunks.append(Chunk(
        s_doc_id=obj_doc.s_doc_id,
        i_index=len(l_chunks),
        i_start=i_start,
        s_text=s_text[i_start:i_end],
      ))
      if i_end == i_length:
        break  # any further window would be pure overlap
      i_start += self.i_step
    return l_chunks

  def chunk_documents(self, l_docs: list[Document]) -> list[Chunk]:
    """Split every document and return all chunks in one flat list."""
    l_chunks = []
    for obj_doc in l_docs:
      l_chunks.extend(self.chunk_document(obj_doc))
    return l_chunks
