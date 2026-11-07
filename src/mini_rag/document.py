"""Data class for one loaded source document."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
  """A source document: its id (the filename stem) and its full text."""

  s_doc_id: str
  s_text: str
