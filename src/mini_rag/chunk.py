"""Data class for one chunk cut from a document."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
  """A slice of a document's text and where it came from."""

  s_doc_id: str
  i_index: int  # position of this chunk within its document
  i_start: int  # character offset into the document text
  s_text: str
