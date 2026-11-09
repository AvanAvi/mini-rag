"""Downloads the Wikipedia corpus listed in a titles file and records its sources."""

import argparse
import hashlib
import re
import time
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from mini_rag.wiki_page import WikiPage
from mini_rag.wikipedia_client import WikipediaClient


class CorpusDownloader:
  """Fetches each listed article, saves it as <slug>.txt and writes SOURCES.md."""

  S_SOURCES_NAME = "SOURCES.md"
  S_WIKI_BASE = "https://en.wikipedia.org"
  S_URL_SAFE = "()_-.,:'!"  # characters left unescaped in article URLs
  T_COLUMNS = (
    "file", "title", "article URL", "permanent link", "revision id",
    "retrieved (UTC)", "characters", "sha256",
  )

  def __init__(
    self,
    s_titles_path: str,
    s_corpus_dir: str,
    obj_client: object | None = None,
    f_pause_seconds: float = 1.0,
  ) -> None:
    """Store the paths, the client (a real WikipediaClient by default) and the pause between requests."""
    self.s_titles_path = s_titles_path
    self.s_corpus_dir = s_corpus_dir
    self.obj_client = obj_client if obj_client is not None else WikipediaClient()
    self.f_pause_seconds = f_pause_seconds

  @classmethod
  def main(cls) -> None:
    """Command-line entry point: download with the default project paths; --force re-downloads."""
    obj_parser = argparse.ArgumentParser(description="Download the Wikipedia corpus.")
    obj_parser.add_argument("--force", action="store_true", help="overwrite existing files")
    obj_args = obj_parser.parse_args()

    # src/mini_rag/corpus_downloader.py -> project root is two levels above the package
    obj_root = Path(__file__).resolve().parents[2]
    obj_downloader = cls(
      str(obj_root / "data" / "corpus_titles.txt"),
      str(obj_root / "data" / "corpus"),
    )
    obj_downloader.download(b_force=obj_args.force)

  @classmethod
  def slug(cls, s_title: str) -> str:
    """Return the lowercase ASCII slug of a title: runs of non-alphanumerics become one underscore."""
    s_ascii = unicodedata.normalize("NFKD", s_title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", s_ascii.lower()).strip("_")

  def download(self, b_force: bool = False) -> None:
    """Fetch and save every listed article, skipping existing files unless b_force, then write SOURCES.md."""
    obj_dir = Path(self.s_corpus_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    d_previous_rows = self.read_previous_rows()

    l_rows = []
    i_total_chars = 0
    b_fetched_before = False
    for s_title in self.read_titles():
      obj_path = obj_dir / f"{self.slug(s_title)}.txt"

      if obj_path.exists() and not b_force:
        s_text = obj_path.read_bytes().decode("utf-8")
        s_row = d_previous_rows.get(obj_path.name) or self.build_unknown_row(obj_path.name, s_title, s_text)
        s_status = "skipped (file exists, use --force to overwrite)"
      else:
        if b_fetched_before:
          time.sleep(self.f_pause_seconds)  # be polite to the API between requests
        obj_page = self.obj_client.fetch(s_title)
        b_fetched_before = True
        s_text = self.save_page(obj_page, obj_path)
        s_row = self.build_row(obj_path.name, obj_page, s_text)
        s_status = "downloaded"

      l_rows.append(s_row)
      i_total_chars += len(s_text)
      print(f"{s_title}: {len(s_text)} characters, {s_status}")

    (obj_dir / self.S_SOURCES_NAME).write_bytes(self.build_sources(l_rows).encode("utf-8"))
    print(f"{len(l_rows)} articles, {i_total_chars} characters in total")

  def read_titles(self) -> list[str]:
    """Return the non-blank, stripped lines of the titles file."""
    s_raw = Path(self.s_titles_path).read_text(encoding="utf-8")
    return [s_line.strip() for s_line in s_raw.splitlines() if s_line.strip()]

  def save_page(self, obj_page: WikiPage, obj_path: Path) -> str:
    """Write the page text as UTF-8 with \\n line endings and return the text as saved."""
    s_text = obj_page.s_text.replace("\r\n", "\n").replace("\r", "\n")
    obj_path.write_bytes(s_text.encode("utf-8"))
    return s_text

  def sha256(self, s_text: str) -> str:
    """Return the hex sha256 of the UTF-8 encoded text."""
    return hashlib.sha256(s_text.encode("utf-8")).hexdigest()

  def article_url(self, s_title: str) -> str:
    """Return the https://en.wikipedia.org/wiki/<Title> URL of a title."""
    s_quoted = urllib.parse.quote(s_title.replace(" ", "_"), safe=self.S_URL_SAFE)
    return f"{self.S_WIKI_BASE}/wiki/{s_quoted}"

  def permanent_link(self, s_title: str, i_revision_id: int) -> str:
    """Return the index.php permanent link to one revision of a title."""
    s_quoted = urllib.parse.quote(s_title.replace(" ", "_"), safe=self.S_URL_SAFE)
    return f"{self.S_WIKI_BASE}/w/index.php?title={s_quoted}&oldid={i_revision_id}"

  def build_row(self, s_file: str, obj_page: WikiPage, s_text: str) -> str:
    """Return the SOURCES.md table row for a freshly downloaded page."""
    s_retrieved = datetime.now(timezone.utc).date().isoformat()
    return self.format_row((
      s_file,
      obj_page.s_title,
      self.article_url(obj_page.s_title),
      self.permanent_link(obj_page.s_title, obj_page.i_revision_id),
      str(obj_page.i_revision_id),
      s_retrieved,
      str(len(s_text)),
      self.sha256(s_text),
    ))

  def build_unknown_row(self, s_file: str, s_title: str, s_text: str) -> str:
    """Return a row for a skipped file with no earlier SOURCES.md entry; revision data is unknown."""
    return self.format_row((
      s_file, s_title, self.article_url(s_title), "unknown", "unknown", "unknown",
      str(len(s_text)), self.sha256(s_text),
    ))

  def format_row(self, t_cells: tuple) -> str:
    """Join cells into one Markdown table row, escaping any pipe characters."""
    return "| " + " | ".join(s_cell.replace("|", "\\|") for s_cell in t_cells) + " |"

  def read_previous_rows(self) -> dict[str, str]:
    """Return the existing SOURCES.md table rows keyed by file name, so skipped files keep their entry."""
    obj_path = Path(self.s_corpus_dir) / self.S_SOURCES_NAME
    if not obj_path.exists():
      return {}
    d_rows = {}
    for s_line in obj_path.read_text(encoding="utf-8").splitlines():
      l_cells = [s_cell.strip() for s_cell in s_line.strip().strip("|").split("|")]
      if s_line.startswith("|") and l_cells and l_cells[0].endswith(".txt"):
        d_rows[l_cells[0]] = s_line
    return d_rows

  def build_sources(self, l_rows: list[str]) -> str:
    """Return the full SOURCES.md text: licence header followed by the table."""
    l_lines = [
      "# Sources",
      "",
      "The texts in this folder come from English Wikipedia and are licensed under",
      "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).",
      "The authors of each article are listed in that article's page history.",
      "Each file is the unmodified plain-text extract returned by the Wikipedia API",
      "(saved as UTF-8 with \\n line endings).",
      "",
      self.format_row(self.T_COLUMNS),
      "|" + "---|" * len(self.T_COLUMNS),
    ]
    l_lines.extend(l_rows)
    return "\n".join(l_lines) + "\n"


if __name__ == "__main__":
  CorpusDownloader.main()
