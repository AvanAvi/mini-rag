"""Unit tests for the Chunker."""

import pytest

from mini_rag.chunk import Chunk
from mini_rag.chunker import Chunker
from mini_rag.document import Document


class TestChunker:
  """Checks window positions, overlap, edge cases and parameter validation."""

  def chunk(self, s_text: str, i_size: int, i_overlap: int) -> list[Chunk]:
    """Chunk a single document with id "d" using the given parameters."""
    obj_chunker = Chunker(i_size=i_size, i_overlap=i_overlap)
    return obj_chunker.chunk_document(Document(s_doc_id="d", s_text=s_text))

  def test_defaults(self) -> None:
    """The default size is 500 with an overlap of 100."""
    obj_chunker = Chunker()
    assert (obj_chunker.i_size, obj_chunker.i_overlap) == (500, 100)

  def test_short_string_exact_offsets(self) -> None:
    """A 10-char string with size 4, overlap 1 gives windows at 0, 3 and 6."""
    l_chunks = self.chunk("abcdefghij", 4, 1)
    assert l_chunks == [
      Chunk(s_doc_id="d", i_index=0, i_start=0, s_text="abcd"),
      Chunk(s_doc_id="d", i_index=1, i_start=3, s_text="defg"),
      Chunk(s_doc_id="d", i_index=2, i_start=6, s_text="ghij"),
    ]

  def test_overlap_content(self) -> None:
    """The last i_overlap chars of each chunk equal the first i_overlap of the next."""
    i_overlap = 3
    l_chunks = self.chunk("the quick brown fox jumps over", 8, i_overlap)
    assert len(l_chunks) > 2
    for obj_prev, obj_next in zip(l_chunks, l_chunks[1:]):
      assert obj_prev.s_text[-i_overlap:] == obj_next.s_text[:i_overlap]

  def test_text_equal_to_size(self) -> None:
    """Text exactly i_size long gives one full chunk."""
    l_chunks = self.chunk("abcd", 4, 1)
    assert l_chunks == [Chunk(s_doc_id="d", i_index=0, i_start=0, s_text="abcd")]

  def test_text_size_plus_one(self) -> None:
    """Text of i_size + 1 gives a full chunk plus a short trailing chunk."""
    l_chunks = self.chunk("abcde", 4, 1)
    assert l_chunks == [
      Chunk(s_doc_id="d", i_index=0, i_start=0, s_text="abcd"),
      Chunk(s_doc_id="d", i_index=1, i_start=3, s_text="de"),
    ]

  def test_shorter_than_size(self) -> None:
    """Text shorter than i_size gives exactly one chunk with all the text."""
    l_chunks = self.chunk("ab", 4, 1)
    assert l_chunks == [Chunk(s_doc_id="d", i_index=0, i_start=0, s_text="ab")]

  def test_no_pure_overlap_tail(self) -> None:
    """A trailing window lying wholly inside the previous chunk is not emitted."""
    # size 4, overlap 2, length 7: windows 0-4, 2-6, 4-7; a window at 6 would be pure overlap
    l_chunks = self.chunk("abcdefg", 4, 2)
    assert [obj_chunk.i_start for obj_chunk in l_chunks] == [0, 2, 4]
    assert l_chunks[-1].s_text == "efg"

  def test_empty_text(self) -> None:
    """Empty text gives no chunks."""
    assert self.chunk("", 4, 1) == []

  def test_zero_overlap(self) -> None:
    """With i_overlap = 0 the windows are adjacent and do not share characters."""
    l_chunks = self.chunk("abcdefghij", 4, 0)
    assert [(obj_chunk.i_start, obj_chunk.s_text) for obj_chunk in l_chunks] == [
      (0, "abcd"), (4, "efgh"), (8, "ij"),
    ]

  @pytest.mark.parametrize("i_size, i_overlap", [
    (0, 0),    # size below 1
    (-1, 0),   # negative size
    (4, -1),   # negative overlap
    (4, 4),    # overlap equal to size
    (4, 5),    # overlap larger than size
    (1, 1),    # smallest size, overlap equal
  ])
  def test_invalid_parameters(self, i_size: int, i_overlap: int) -> None:
    """Every invalid size/overlap combination raises ValueError."""
    with pytest.raises(ValueError):
      Chunker(i_size=i_size, i_overlap=i_overlap)

  @pytest.mark.parametrize("i_size, i_overlap", [(1, 0), (3, 1), (5, 4), (7, 0), (50, 10)])
  def test_text_matches_slice(self, i_size: int, i_overlap: int) -> None:
    """Every chunk's text equals the original slice starting at i_start."""
    s_text = "Line one.\nLine two, a bit longer.\nThird!"
    l_chunks = self.chunk(s_text, i_size, i_overlap)
    assert l_chunks[-1].i_start + len(l_chunks[-1].s_text) == len(s_text)
    for i_pos, obj_chunk in enumerate(l_chunks):
      assert obj_chunk.i_index == i_pos
      assert obj_chunk.s_text == s_text[obj_chunk.i_start:obj_chunk.i_start + len(obj_chunk.s_text)]

  def test_chunk_documents_flattens(self) -> None:
    """chunk_documents returns all chunks of all documents in order, indexes per document."""
    obj_chunker = Chunker(i_size=4, i_overlap=1)
    l_docs = [Document(s_doc_id="a", s_text="abcde"), Document(s_doc_id="b", s_text="xy")]
    l_chunks = obj_chunker.chunk_documents(l_docs)
    assert [(obj_chunk.s_doc_id, obj_chunk.i_index, obj_chunk.s_text) for obj_chunk in l_chunks] == [
      ("a", 0, "abcd"), ("a", 1, "de"), ("b", 0, "xy"),
    ]
