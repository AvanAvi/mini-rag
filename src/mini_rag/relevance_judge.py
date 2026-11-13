"""Decides whether a chunk is relevant to a question."""

from mini_rag.chunk import Chunk


class RelevanceJudge:
  """Calls a chunk relevant when it contains the question's phrase, ignoring case and spacing."""

  def normalise(self, s_text: str) -> str:
    """Lowercase the text, collapse every whitespace run to one space and strip the ends."""
    return " ".join(s_text.lower().split())

  def is_relevant(self, obj_chunk: Chunk, s_phrase: str) -> bool:
    """Return True if the normalised phrase occurs in the normalised chunk text."""
    return self.normalise(s_phrase) in self.normalise(obj_chunk.s_text)

  def count_relevant(self, l_chunks: list[Chunk], s_phrase: str) -> int:
    """Return how many of the chunks are relevant to the phrase."""
    return sum(1 for obj_chunk in l_chunks if self.is_relevant(obj_chunk, s_phrase))
