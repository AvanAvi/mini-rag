"""Unit tests for CorpusDownloader using a fake Wikipedia client; no network calls are made."""

import hashlib
from pathlib import Path

import pytest

from fake_wikipedia_client import FakeWikipediaClient
from mini_rag.corpus_downloader import CorpusDownloader
from mini_rag.wiki_page import WikiPage


class TestSlug:
  """Checks the title-to-slug rule used for file names and document ids."""

  @pytest.mark.parametrize(
    "s_title, s_expected",
    [
      ("Tokamak", "tokamak"),
      ("Plasma (physics)", "plasma_physics"),
      ("Loss-of-coolant accident", "loss_of_coolant_accident"),
      ("IEC 61508", "iec_61508"),
      ("Réacteur nucléaire", "reacteur_nucleaire"),
      ("  ...Hello, World!?  ", "hello_world"),
      ("A -- B & C", "a_b_c"),
    ],
  )
  def test_slug(self, s_title: str, s_expected: str) -> None:
    """Lowercase ASCII, one underscore per non-alphanumeric run, none at either end."""
    assert CorpusDownloader.slug(s_title) == s_expected


class TestCorpusDownloader:
  """Checks saving, SOURCES.md, skip/force behaviour and failure handling."""

  @pytest.fixture(autouse=True)
  def setup_dirs(self, tmp_path: Path) -> None:
    """Create a titles file and fixed pages in pytest's temporary directory."""
    self.obj_titles = tmp_path / "titles.txt"
    self.obj_titles.write_text("Tokamak\nPlasma (physics)\n\nPWR\n", encoding="utf-8")
    self.obj_corpus = tmp_path / "corpus"
    self.d_pages = {
      "Tokamak": WikiPage("Tokamak", "A tokamak confines plasma.\nIt is toroidal.", 111, "2026-10-01T00:00:00Z"),
      "Plasma (physics)": WikiPage("Plasma (physics)", "Plasma is ionised gas.\r\nÉlectrons move.", 222, "2026-10-02T00:00:00Z"),
      "PWR": WikiPage("Pressurized water reactor", "A PWR uses pressurised water.", 333, "2026-10-03T00:00:00Z"),
    }

  def make_downloader(self, obj_client: FakeWikipediaClient) -> CorpusDownloader:
    """Return a downloader on the temp paths with no pause between requests."""
    return CorpusDownloader(str(self.obj_titles), str(self.obj_corpus), obj_client=obj_client, f_pause_seconds=0.0)

  def sources(self) -> str:
    """Return the text of the written SOURCES.md."""
    return (self.obj_corpus / "SOURCES.md").read_text(encoding="utf-8")

  def test_files_written_with_exact_content(self) -> None:
    """Each article is saved under its slug as UTF-8 with \\n line endings only."""
    self.make_downloader(FakeWikipediaClient(self.d_pages)).download()
    assert (self.obj_corpus / "tokamak.txt").read_bytes() == "A tokamak confines plasma.\nIt is toroidal.".encode("utf-8")
    assert (self.obj_corpus / "plasma_physics.txt").read_bytes() == "Plasma is ionised gas.\nÉlectrons move.".encode("utf-8")
    assert (self.obj_corpus / "pwr.txt").read_bytes() == "A PWR uses pressurised water.".encode("utf-8")
    for obj_path in self.obj_corpus.glob("*.txt"):
      assert b"\r" not in obj_path.read_bytes()

  def test_sources_md_contents(self) -> None:
    """SOURCES.md has the licence header and each title, revision id, permalink and sha256."""
    self.make_downloader(FakeWikipediaClient(self.d_pages)).download()
    s_sources = self.sources()
    assert "CC BY-SA 4.0" in s_sources
    assert "https://creativecommons.org/licenses/by-sa/4.0/" in s_sources
    for s_file, obj_page in [("tokamak.txt", self.d_pages["Tokamak"]),
                             ("plasma_physics.txt", self.d_pages["Plasma (physics)"]),
                             ("pwr.txt", self.d_pages["PWR"])]:
      obj_saved = (self.obj_corpus / s_file).read_bytes()
      assert obj_page.s_title in s_sources
      assert str(obj_page.i_revision_id) in s_sources
      assert hashlib.sha256(obj_saved).hexdigest() in s_sources
    assert "https://en.wikipedia.org/wiki/Plasma_(physics)" in s_sources
    assert "https://en.wikipedia.org/w/index.php?title=Pressurized_water_reactor&oldid=333" in s_sources

  def test_existing_file_skipped_without_force(self, capsys: pytest.CaptureFixture) -> None:
    """An existing file is not fetched or overwritten, and the output says it was skipped."""
    self.obj_corpus.mkdir()
    (self.obj_corpus / "tokamak.txt").write_bytes(b"old text")
    obj_client = FakeWikipediaClient(self.d_pages)
    self.make_downloader(obj_client).download()
    assert (self.obj_corpus / "tokamak.txt").read_bytes() == b"old text"
    assert "Tokamak" not in obj_client.l_requested
    assert "Tokamak: 8 characters, skipped" in capsys.readouterr().out

  def test_existing_file_overwritten_with_force(self) -> None:
    """With force set, an existing file is fetched again and overwritten."""
    self.obj_corpus.mkdir()
    (self.obj_corpus / "tokamak.txt").write_bytes(b"old text")
    obj_client = FakeWikipediaClient(self.d_pages)
    self.make_downloader(obj_client).download(b_force=True)
    assert (self.obj_corpus / "tokamak.txt").read_bytes() == b"A tokamak confines plasma.\nIt is toroidal."
    assert "Tokamak" in obj_client.l_requested

  def test_skipped_file_keeps_previous_sources_row(self) -> None:
    """A second run that skips everything keeps the revision ids from the first SOURCES.md."""
    self.make_downloader(FakeWikipediaClient(self.d_pages)).download()
    s_first = self.sources()
    obj_client = FakeWikipediaClient(self.d_pages)
    self.make_downloader(obj_client).download()
    assert obj_client.l_requested == []
    assert self.sources() == s_first

  def test_failure_raises_and_keeps_written_files(self) -> None:
    """A failing title raises, and files written before it stay in place."""
    obj_client = FakeWikipediaClient(self.d_pages, s_failing_title="Plasma (physics)")
    with pytest.raises(RuntimeError, match="Plasma"):
      self.make_downloader(obj_client).download()
    assert (self.obj_corpus / "tokamak.txt").read_bytes() == b"A tokamak confines plasma.\nIt is toroidal."
    assert not (self.obj_corpus / "plasma_physics.txt").exists()
    assert not (self.obj_corpus / "pwr.txt").exists()

  def test_summary_line(self, capsys: pytest.CaptureFixture) -> None:
    """The last output line gives the article count and total characters."""
    self.make_downloader(FakeWikipediaClient(self.d_pages)).download()
    i_total = sum(len(obj_page.s_text.replace("\r\n", "\n")) for obj_page in self.d_pages.values())
    l_lines = capsys.readouterr().out.strip().splitlines()
    assert l_lines[-1] == f"3 articles, {i_total} characters in total"
