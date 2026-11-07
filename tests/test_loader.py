"""Unit tests for the Loader."""

from pathlib import Path

import pytest

from mini_rag.document import Document
from mini_rag.loader import Loader


class TestLoader:
  """Checks file discovery, ordering, normalisation and error handling."""

  @pytest.fixture(autouse=True)
  def setup_dir(self, tmp_path: Path) -> None:
    """Store pytest's temporary directory (fixture name is fixed by pytest)."""
    self.obj_dir = tmp_path

  def write(self, s_name: str, s_text: str) -> None:
    """Write raw text to a file in the temp directory without newline translation."""
    (self.obj_dir / s_name).write_bytes(s_text.encode("utf-8"))

  def test_sorted_order(self) -> None:
    """Documents come back in sorted filename order regardless of write order."""
    self.write("c.txt", "three")
    self.write("a.txt", "one")
    self.write("b.txt", "two")
    l_docs = Loader(str(self.obj_dir)).load()
    assert [obj_doc.s_doc_id for obj_doc in l_docs] == ["a", "b", "c"]
    assert [obj_doc.s_text for obj_doc in l_docs] == ["one", "two", "three"]

  def test_returns_documents(self) -> None:
    """Each result is a Document whose id is the filename stem."""
    self.write("notes.txt", "hello")
    l_docs = Loader(str(self.obj_dir)).load()
    assert l_docs == [Document(s_doc_id="notes", s_text="hello")]

  def test_crlf_normalised(self) -> None:
    """CRLF and lone CR become LF; other characters are left untouched."""
    self.write("a.txt", "line1\r\nline2\rline3\n  tail  ")
    l_docs = Loader(str(self.obj_dir)).load()
    assert l_docs[0].s_text == "line1\nline2\nline3\n  tail  "

  def test_utf8_read(self) -> None:
    """Non-ASCII text is decoded as UTF-8."""
    self.write("a.txt", "café ✓")
    assert Loader(str(self.obj_dir)).load()[0].s_text == "café ✓"

  def test_empty_file_skipped(self) -> None:
    """Empty and whitespace-only files are skipped."""
    self.write("a.txt", "")
    self.write("b.txt", "  \r\n\t ")
    self.write("c.txt", "content")
    l_docs = Loader(str(self.obj_dir)).load()
    assert [obj_doc.s_doc_id for obj_doc in l_docs] == ["c"]

  def test_missing_directory_error(self) -> None:
    """A directory that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
      Loader(str(self.obj_dir / "missing")).load()

  def test_no_usable_files_error(self) -> None:
    """A directory holding only empty or non-.txt files raises ValueError."""
    self.write("a.txt", "   ")
    self.write("b.md", "text")
    with pytest.raises(ValueError, match="No non-empty .txt files"):
      Loader(str(self.obj_dir)).load()

  def test_non_txt_ignored(self) -> None:
    """Files without a .txt extension, and subdirectories, are ignored."""
    self.write("a.txt", "keep")
    self.write("b.md", "skip")
    self.write("c.txt.bak", "skip")
    (self.obj_dir / "d.txt").mkdir()
    l_docs = Loader(str(self.obj_dir)).load()
    assert [obj_doc.s_doc_id for obj_doc in l_docs] == ["a"]
