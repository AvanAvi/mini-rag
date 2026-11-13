"""Tests for RelevanceJudge."""

from mini_rag.chunk import Chunk
from mini_rag.relevance_judge import RelevanceJudge


class TestRelevanceJudge:
  """Covers normalisation and the phrase-in-chunk relevance rule."""

  def chunk(self, s_text: str, i_index: int = 0) -> Chunk:
    """Return a chunk of a made-up document with the given text."""
    return Chunk(s_doc_id="doc", i_index=i_index, i_start=0, s_text=s_text)

  def test_normalise_lowercases_and_collapses_whitespace(self):
    """Tabs, line breaks and repeated spaces become one space; the ends are stripped."""
    assert RelevanceJudge().normalise("  The  Quick\n\tBrown\r\nFox ") == "the quick brown fox"

  def test_relevant_across_a_line_break_and_case(self):
    """A phrase matches even when the chunk breaks the line or uses other case."""
    obj_chunk = self.chunk("It was founded\nIN  1998 by two people.")
    assert RelevanceJudge().is_relevant(obj_chunk, "founded in 1998")

  def test_not_relevant_when_phrase_is_missing(self):
    """A chunk without the phrase is not relevant."""
    assert not RelevanceJudge().is_relevant(self.chunk("It was founded in 1998."), "founded in 1999")

  def test_count_relevant_counts_matching_chunks(self):
    """Only chunks holding the phrase are counted."""
    l_chunks = [self.chunk("red apple", 0), self.chunk("green pear", 1), self.chunk("a RED  apple pie", 2)]
    assert RelevanceJudge().count_relevant(l_chunks, "red apple") == 2
    assert RelevanceJudge().count_relevant([], "red apple") == 0
