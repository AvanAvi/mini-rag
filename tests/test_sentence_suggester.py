"""Unit tests for the SentenceSuggester."""

import pytest

from mini_rag.document import Document
from mini_rag.sentence_suggester import SentenceSuggester


class TestSentenceSuggester:
  """Checks which sentences count as facts, the skipped parts and the two limits."""

  S_FACT = "The reactor first reached full power in 1984 after a long series of delays."

  def fact(self, i_number: int) -> str:
    """Return a distinct fact sentence that mentions i_number."""
    return f"Unit number {i_number} of the plant was connected to the grid after several years of work."

  def test_only_complete_sentences_with_a_digit_are_suggested(self) -> None:
    """Sentences without a digit, too short, too long or not starting with a capital are left out."""
    s_text = "\n".join([
      self.S_FACT,
      "This sentence is long enough but it holds no number at all, so it is not a fact.",
      "Built in 1984.",
      "lowercase start, but the plant was connected to the grid in 1984 after many delays.",
      "A very long sentence about 1984 " + "that goes on and on " * 12 + "until it ends.",
    ])
    assert SentenceSuggester().suggest([Document(s_doc_id="plant", s_text=s_text)]) == [("plant", self.S_FACT)]

  def test_headings_are_skipped_and_reference_sections_end_the_document(self) -> None:
    """Heading lines are never suggested, and nothing after a reference heading is read."""
    s_text = "\n".join([
      "== History of the plant from 1970 to 1990, a heading that looks like a long fact. ==",
      self.fact(1),
      "# A Markdown heading that mentions 1984 and is long enough to pass the length check.",
      self.fact(2),
      "== References ==",
      self.fact(3),
    ])
    l_pairs = SentenceSuggester().suggest([Document(s_doc_id="plant", s_text=s_text)])
    assert l_pairs == [("plant", self.fact(1)), ("plant", self.fact(2))]

  def test_several_sentences_on_one_line_are_split(self) -> None:
    """A paragraph line is split at sentence ends."""
    s_text = f"{self.fact(1)} A short one. {self.fact(2)}"
    l_pairs = SentenceSuggester().suggest([Document(s_doc_id="plant", s_text=s_text)])
    assert [s_sentence for s_doc_id, s_sentence in l_pairs] == [self.fact(1), self.fact(2)]

  def test_at_most_five_per_document_evenly_spaced(self) -> None:
    """A document with many facts gives 5 of them, spread from start to end."""
    s_text = "\n".join(self.fact(i_number) for i_number in range(10))
    l_pairs = SentenceSuggester().suggest([Document(s_doc_id="plant", s_text=s_text)])
    assert [s_sentence for s_doc_id, s_sentence in l_pairs] == [self.fact(i_number) for i_number in (0, 2, 4, 6, 8)]

  def test_limit_is_shared_out_in_rounds(self) -> None:
    """With a limit below the total, every document gets one sentence before any gets a second."""
    l_docs = [
      Document(s_doc_id=s_doc_id, s_text="\n".join(self.fact(i_number) for i_number in range(5)))
      for s_doc_id in ("a", "b", "c")
    ]
    l_pairs = SentenceSuggester(i_limit=4).suggest(l_docs)
    assert [s_doc_id for s_doc_id, s_sentence in l_pairs] == ["a", "a", "b", "c"]
    assert len(SentenceSuggester(i_limit=30).suggest(l_docs)) == 15

  def test_bad_limits_raise(self) -> None:
    """A limit or a per-document count below 1 raises ValueError."""
    with pytest.raises(ValueError):
      SentenceSuggester(i_limit=0)
    with pytest.raises(ValueError):
      SentenceSuggester(i_per_doc=0)
