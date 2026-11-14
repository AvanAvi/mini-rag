"""Loads raw documents from the corpus folder."""

from pathlib import Path

from mini_rag.document import Document


class Loader:
  """Reads UTF-8 text documents (.txt by default, .md on request) from a corpus directory."""

  def __init__(self, s_corpus_dir: str, t_extensions: tuple[str, ...] = (".txt",)) -> None:
    """Store the corpus directory and the file extensions to read, such as (".txt", ".md")."""
    self.s_corpus_dir = s_corpus_dir
    self.t_extensions = tuple(s_extension.lower() for s_extension in t_extensions)

  def load(self) -> list[Document]:
    """Return every non-empty file with a wanted extension as a Document, in sorted filename order."""
    obj_dir = Path(self.s_corpus_dir)
    if not obj_dir.is_dir():
      raise FileNotFoundError(f"Corpus directory not found: {self.s_corpus_dir}")

    l_paths = sorted(
      (obj_path for obj_path in obj_dir.iterdir()
       if obj_path.is_file() and obj_path.suffix.lower() in self.t_extensions),
      key=lambda obj_path: obj_path.name,
    )

    l_docs = []
    l_ids = []
    for obj_path in l_paths:
      s_text = self.read_text(obj_path)
      if not s_text.strip():
        continue  # skip files with no real content
      if obj_path.stem in l_ids:
        # the stem is the document id, so notes.txt and notes.md would collide
        raise ValueError(f"Two corpus files share the name {obj_path.stem!r}: {self.s_corpus_dir}")
      l_ids.append(obj_path.stem)
      l_docs.append(Document(s_doc_id=obj_path.stem, s_text=s_text))

    if not l_docs:
      raise ValueError(f"No non-empty {' or '.join(self.t_extensions)} files found in: {self.s_corpus_dir}")
    return l_docs

  def read_text(self, obj_path: Path) -> str:
    """Read one file as UTF-8 and normalise its line endings to \\n."""
    s_raw = obj_path.read_bytes().decode("utf-8")
    return s_raw.replace("\r\n", "\n").replace("\r", "\n")
