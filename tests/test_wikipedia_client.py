"""Unit tests for WikipediaClient.parse_response and build_url; no network calls are made."""

import urllib.parse

import pytest

from mini_rag.wiki_page import WikiPage
from mini_rag.wikipedia_client import WikipediaClient


class TestWikipediaClient:
  """Checks how hand-made API responses become WikiPages or clear errors."""

  def make_response(self, d_page: dict, l_redirects: list | None = None) -> dict:
    """Wrap one page dict in a formatversion=2 query response."""
    d_query: dict = {"pages": [d_page]}
    if l_redirects is not None:
      d_query["redirects"] = l_redirects
    return {"batchcomplete": True, "query": d_query}

  def test_normal_page(self) -> None:
    """A normal page yields its title, extract, revision id and timestamp."""
    d_json = self.make_response({
      "pageid": 1, "ns": 0, "title": "Tokamak",
      "extract": "A tokamak is a device.\nSecond line.",
      "revisions": [{"revid": 123456, "parentid": 123455, "timestamp": "2026-10-01T12:00:00Z"}],
    })
    obj_page = WikipediaClient().parse_response(d_json, "Tokamak")
    assert obj_page == WikiPage(
      s_title="Tokamak",
      s_text="A tokamak is a device.\nSecond line.",
      i_revision_id=123456,
      s_timestamp="2026-10-01T12:00:00Z",
    )

  def test_missing_page(self) -> None:
    """A missing page raises LookupError naming the requested title."""
    d_json = self.make_response({"ns": 0, "title": "No such article", "missing": True})
    with pytest.raises(LookupError, match="No such article"):
      WikipediaClient().parse_response(d_json, "No such article")

  def test_empty_extract(self) -> None:
    """A whitespace-only extract raises ValueError."""
    d_json = self.make_response({
      "pageid": 2, "ns": 0, "title": "Empty", "extract": "  \n ",
      "revisions": [{"revid": 1, "timestamp": "2026-10-01T12:00:00Z"}],
    })
    with pytest.raises(ValueError, match="empty extract"):
      WikipediaClient().parse_response(d_json, "Empty")

  def test_redirect_uses_resolved_title(self) -> None:
    """After a redirect the page carries the resolved title, not the requested one."""
    d_json = self.make_response(
      {
        "pageid": 3, "ns": 0, "title": "Pressurized water reactor",
        "extract": "A PWR is a type of reactor.",
        "revisions": [{"revid": 777, "timestamp": "2026-09-30T08:00:00Z"}],
      },
      l_redirects=[{"from": "PWR", "to": "Pressurized water reactor"}],
    )
    obj_page = WikipediaClient().parse_response(d_json, "PWR")
    assert obj_page.s_title == "Pressurized water reactor"
    assert obj_page.i_revision_id == 777

  def test_api_error(self) -> None:
    """An API-level error object raises RuntimeError."""
    d_json = {"error": {"code": "badvalue", "info": "Unrecognized value"}}
    with pytest.raises(RuntimeError, match="API error"):
      WikipediaClient().parse_response(d_json, "Tokamak")

  def test_build_url_parameters(self) -> None:
    """The request URL carries exactly the agreed query parameters."""
    s_url = WikipediaClient().build_url("Plasma (physics)")
    obj_parts = urllib.parse.urlsplit(s_url)
    assert f"{obj_parts.scheme}://{obj_parts.netloc}{obj_parts.path}" == "https://en.wikipedia.org/w/api.php"
    assert dict(urllib.parse.parse_qsl(obj_parts.query)) == {
      "action": "query",
      "format": "json",
      "formatversion": "2",
      "prop": "extracts|revisions",
      "explaintext": "1",
      "redirects": "1",
      "rvprop": "ids|timestamp",
      "titles": "Plasma (physics)",
    }

  def test_user_agent_has_no_email(self) -> None:
    """The User-Agent names the project URL and contains no email address."""
    assert WikipediaClient.S_USER_AGENT == "mini-rag/0.1 (https://github.com/AvanAvi/mini-rag)"
    assert "@" not in WikipediaClient.S_USER_AGENT
