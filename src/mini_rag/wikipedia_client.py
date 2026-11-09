"""Fetches plain-text article extracts from the English Wikipedia API."""

import json
import urllib.error
import urllib.parse
import urllib.request

from mini_rag.wiki_page import WikiPage


class WikipediaClient:
  """Requests one article per call and turns the API response into a WikiPage."""

  S_ENDPOINT = "https://en.wikipedia.org/w/api.php"
  S_USER_AGENT = "mini-rag/0.1 (https://github.com/AvanAvi/mini-rag)"

  def __init__(self, f_timeout_seconds: float = 30.0) -> None:
    """Store the request timeout in seconds."""
    self.f_timeout_seconds = f_timeout_seconds

  def fetch(self, s_title: str) -> WikiPage:
    """Request one article and return it as a WikiPage; raise RuntimeError on HTTP or JSON failure."""
    obj_request = urllib.request.Request(
      self.build_url(s_title), headers={"User-Agent": self.S_USER_AGENT}
    )
    try:
      with urllib.request.urlopen(obj_request, timeout=self.f_timeout_seconds) as obj_response:
        obj_raw = obj_response.read()  # bytes; no type prefix exists, so obj_
    except urllib.error.HTTPError as obj_error:
      raise RuntimeError(f"HTTP {obj_error.code} while fetching {s_title!r}") from obj_error
    except (urllib.error.URLError, TimeoutError) as obj_error:
      raise RuntimeError(f"Network error while fetching {s_title!r}: {obj_error}") from obj_error

    try:
      d_json = json.loads(obj_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as obj_error:
      raise RuntimeError(f"Invalid JSON returned for {s_title!r}: {obj_error}") from obj_error
    return self.parse_response(d_json, s_title)

  def build_url(self, s_title: str) -> str:
    """Return the full API URL for one title."""
    d_params = {
      "action": "query",
      "format": "json",
      "formatversion": "2",
      "prop": "extracts|revisions",
      "explaintext": "1",
      "redirects": "1",
      "rvprop": "ids|timestamp",
      "titles": s_title,
    }
    return f"{self.S_ENDPOINT}?{urllib.parse.urlencode(d_params)}"

  def parse_response(self, d_json: dict, s_title: str) -> WikiPage:
    """Turn a formatversion=2 query response into a WikiPage; raise on a missing page or empty extract."""
    if "error" in d_json:
      raise RuntimeError(f"Wikipedia API error for {s_title!r}: {d_json['error']}")

    l_pages = d_json.get("query", {}).get("pages", [])
    if not l_pages:
      raise RuntimeError(f"Unexpected Wikipedia response for {s_title!r}: no pages")
    d_page = l_pages[0]

    if d_page.get("missing") or d_page.get("invalid"):
      raise LookupError(f"Wikipedia page not found: {s_title!r}")

    s_text = d_page.get("extract", "")
    if not s_text.strip():
      raise ValueError(f"Wikipedia page {s_title!r} has an empty extract")

    l_revisions = d_page.get("revisions", [])
    if not l_revisions:
      raise RuntimeError(f"Unexpected Wikipedia response for {s_title!r}: no revision")
    d_revision = l_revisions[0]

    return WikiPage(
      s_title=d_page["title"],
      s_text=s_text,
      i_revision_id=int(d_revision["revid"]),
      s_timestamp=d_revision["timestamp"],
    )
