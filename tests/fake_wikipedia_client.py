"""Test double that returns fixed WikiPages instead of calling Wikipedia."""

from mini_rag.wiki_page import WikiPage


class FakeWikipediaClient:
  """Returns a fixed page per requested title, records each request, and can fail on demand."""

  def __init__(self, d_pages: dict[str, WikiPage], s_failing_title: str | None = None) -> None:
    """Store the pages keyed by requested title and the optional title that raises."""
    self.d_pages = d_pages
    self.s_failing_title = s_failing_title
    self.l_requested: list[str] = []

  def fetch(self, s_title: str) -> WikiPage:
    """Record the title and return its page; raise RuntimeError for the failing or an unknown title."""
    self.l_requested.append(s_title)
    if s_title == self.s_failing_title or s_title not in self.d_pages:
      raise RuntimeError(f"Fake fetch failure for {s_title!r}")
    return self.d_pages[s_title]
