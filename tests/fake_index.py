"""Test double that returns a hand-made ranking instead of searching."""

from mini_rag.search_hit import SearchHit


class FakeIndex:
  """Returns a fixed best-first SearchHit list and records every requested i_top_k."""

  def __init__(self, l_hits: list[SearchHit]) -> None:
    """Store the ranking to return and start with no recorded calls."""
    self.l_hits = l_hits
    self.l_top_k_calls: list[int] = []

  def search(self, s_query: str, i_top_k: int) -> list[SearchHit]:
    """Record i_top_k and return the first i_top_k stored hits."""
    self.l_top_k_calls.append(i_top_k)
    return self.l_hits[:i_top_k]
