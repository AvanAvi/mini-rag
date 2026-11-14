"""Picks candidate fact sentences from a corpus to help a user write evaluation questions."""

import re

from mini_rag.document import Document


class SentenceSuggester:
  """Suggests sentences that hold a number or a date; the user writes the questions."""

  I_MIN_LENGTH = 60
  I_MAX_LENGTH = 200
  T_REFERENCE_TITLES = (
    "references", "see also", "notes", "external links", "further reading",
    "bibliography", "sources", "citations", "footnotes",
  )

  def __init__(self, i_limit: int = 30, i_per_doc: int = 5) -> None:
    """Store the total number of sentences to return and the most to take from one document."""
    if i_limit < 1:
      raise ValueError(f"i_limit must be at least 1, got {i_limit}")
    if i_per_doc < 1:
      raise ValueError(f"i_per_doc must be at least 1, got {i_per_doc}")
    self.i_limit = i_limit
    self.i_per_doc = i_per_doc
    # "== Title ==" as in the plain-text extracts, or "# Title" as in Markdown
    self.obj_heading = re.compile(r"^\s*(?:=+\s*(.*?)\s*=+|#+\s+(.*?))\s*$")
    self.obj_sentence_end = re.compile(r"(?<=[.!?])\s+")

  def suggest(self, l_docs: list[Document]) -> list[tuple[str, str]]:
    """Return up to i_limit (doc_id, sentence) pairs, spread over the documents, in document order."""
    l_per_doc = [self.pick(self.candidates(obj_doc.s_text)) for obj_doc in l_docs]
    # one round takes the next pick of every document, so a small limit still covers many documents
    l_taken = [0] * len(l_docs)
    i_total = 0
    for i_round in range(self.i_per_doc):
      for i_doc, l_picked in enumerate(l_per_doc):
        if i_total < self.i_limit and i_round < len(l_picked):
          l_taken[i_doc] += 1
          i_total += 1
    return [
      (obj_doc.s_doc_id, s_sentence)
      for obj_doc, l_picked, i_taken in zip(l_docs, l_per_doc, l_taken)
      for s_sentence in l_picked[:i_taken]
    ]

  def candidates(self, s_text: str) -> list[str]:
    """Return the fact sentences of one document, skipping headings and everything from a reference section on."""
    l_sentences: list[str] = []
    for s_line in s_text.split("\n"):
      obj_match = self.obj_heading.match(s_line)
      if obj_match is not None:
        s_title = (obj_match.group(1) or obj_match.group(2) or "").strip().lower()
        if s_title in self.T_REFERENCE_TITLES:
          break
        continue
      for s_sentence in self.obj_sentence_end.split(s_line.strip()):
        if self.is_fact(s_sentence) and s_sentence not in l_sentences:
          l_sentences.append(s_sentence)
    return l_sentences

  def is_fact(self, s_sentence: str) -> bool:
    """Return True for a complete sentence of usable length that contains a digit."""
    return (
      self.I_MIN_LENGTH <= len(s_sentence) <= self.I_MAX_LENGTH
      and s_sentence[0].isupper()
      and s_sentence.endswith(".")
      and any(s_char.isdigit() for s_char in s_sentence)
    )

  def pick(self, l_sentences: list[str]) -> list[str]:
    """Return at most i_per_doc sentences, evenly spaced through the document."""
    if len(l_sentences) <= self.i_per_doc:
      return list(l_sentences)
    return [l_sentences[i_pick * len(l_sentences) // self.i_per_doc] for i_pick in range(self.i_per_doc)]
