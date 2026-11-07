"""Unit tests for the Bm25Index."""

import pytest

from mini_rag.bm25_index import Bm25Index
from mini_rag.chunk import Chunk
from mini_rag.search_hit import SearchHit


class TestBm25Index:
  """Checks ranking, filtering, limits and error handling."""

  def make_chunks(self, l_texts: list[str]) -> list[Chunk]:
    """Wrap each text in a Chunk of document "d", indexed by position."""
    return [
      Chunk(s_doc_id="d", i_index=i_pos, i_start=0, s_text=s_text)
      for i_pos, s_text in enumerate(l_texts)
    ]

  def build_index(self, l_texts: list[str]) -> Bm25Index:
    """Return a Bm25Index built over chunks made from the texts."""
    obj_index = Bm25Index()
    obj_index.build(self.make_chunks(l_texts))
    return obj_index

  def corpus(self) -> list[str]:
    """A 6-chunk corpus in which each animal word appears in exactly one chunk."""
    return [
      "the river flows past the old mill",
      "a zebra grazes near the river",
      "the mill grinds wheat into flour",
      "an otter swims in the cold river",
      "the baker sells bread and flour",
      "a heron waits by the quiet water",
    ]

  def ids(self, l_hits: list[SearchHit]) -> list[int]:
    """Return the chunk indexes of the hits, in order."""
    return [obj_hit.obj_chunk.i_index for obj_hit in l_hits]

  def test_rare_term_ranks_first(self) -> None:
    """The only chunk containing a rare query term ranks first."""
    obj_index = self.build_index(self.corpus())
    l_hits = obj_index.search("zebra river", 6)
    assert l_hits[0].obj_chunk.i_index == 1
    assert isinstance(l_hits[0], SearchHit)
    assert isinstance(l_hits[0].f_score, float)

  def test_top_k_limits_results(self) -> None:
    """At most i_top_k hits are returned, even if more chunks match."""
    obj_index = self.build_index(self.corpus())
    s_query = "zebra otter heron"
    assert len(obj_index.search(s_query, 10)) == 3
    assert len(obj_index.search(s_query, 2)) == 2
    assert len(obj_index.search(s_query, 1)) == 1

  def test_zero_score_excluded(self) -> None:
    """Chunks that match no query term are not returned."""
    obj_index = self.build_index(self.corpus())
    l_hits = obj_index.search("otter", 6)
    assert self.ids(l_hits) == [3]
    assert all(obj_hit.f_score > 0 for obj_hit in l_hits)

  def test_unknown_term_returns_empty(self) -> None:
    """A query whose terms appear nowhere returns no hits."""
    obj_index = self.build_index(self.corpus())
    assert obj_index.search("giraffe", 6) == []

  def test_search_before_build_raises(self) -> None:
    """Searching an index that was never built raises RuntimeError."""
    with pytest.raises(RuntimeError):
      Bm25Index().search("river", 3)

  def test_empty_build_raises(self) -> None:
    """Building from an empty list raises ValueError."""
    with pytest.raises(ValueError):
      Bm25Index().build([])

  def test_all_chunks_without_tokens_raises(self) -> None:
    """Building from chunks that contain only punctuation raises ValueError."""
    with pytest.raises(ValueError, match="zero tokens"):
      Bm25Index().build(self.make_chunks(["...", "?!", " -- "]))

  @pytest.mark.parametrize("i_top_k", [0, -1])
  def test_invalid_top_k_raises(self, i_top_k: int) -> None:
    """An i_top_k below 1 raises ValueError."""
    obj_index = self.build_index(self.corpus())
    with pytest.raises(ValueError):
      obj_index.search("river", i_top_k)

  @pytest.mark.parametrize("s_query", ["", "   ", "?!..."])
  def test_empty_query_returns_empty(self, s_query: str) -> None:
    """An empty query, or one with no tokens, returns an empty list."""
    obj_index = self.build_index(self.corpus())
    assert obj_index.search(s_query, 3) == []

  def test_exact_order_three_chunks(self) -> None:
    """On a 3-chunk corpus the hits come back best first in a known order."""
    # "dog" is in one chunk only (high IDF); "cat" is in all three, so its raw IDF is
    # negative and rank-bm25 replaces it with a small positive floor.
    # Chunk 2 has "cat" twice, so it beats chunk 0 on the "cat" term alone.
    obj_index = self.build_index([
      "the cat sat on the mat",
      "a dog and a cat played",
      "the cat chased the cat toy",
    ])
    l_hits = obj_index.search("cat dog", 3)
    assert self.ids(l_hits) == [1, 2, 0]
    assert l_hits[0].f_score > l_hits[1].f_score > l_hits[2].f_score

  def test_ties_keep_chunk_order(self) -> None:
    """Chunks with equal scores keep their original order."""
    obj_index = self.build_index([
      "the cat sat on the mat",
      "a dog and a cat played",
      "the cat chased the cat toy",
    ])
    l_hits = obj_index.search("cat", 3)
    assert self.ids(l_hits) == [2, 0, 1]
    assert l_hits[1].f_score == l_hits[2].f_score

  def test_two_chunk_corpus_gives_zero_idf(self) -> None:
    """Documents BM25Okapi behaviour: in a 2-chunk corpus a term in one chunk has IDF 0.

    IDF is log((N - n + 0.5) / (n + 0.5)); with N = 2 and n = 1 that is log(1) = 0,
    so the matching chunk scores 0 and is dropped. Larger corpora avoid this.
    """
    obj_index = self.build_index(["apple pie", "banana bread"])
    assert obj_index.search("apple", 2) == []
