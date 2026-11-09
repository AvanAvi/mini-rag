"""Data class for one fetched Wikipedia article."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WikiPage:
  """A fetched Wikipedia article: resolved title (after redirects), plain text, revision id and timestamp."""

  s_title: str
  s_text: str
  i_revision_id: int
  s_timestamp: str
