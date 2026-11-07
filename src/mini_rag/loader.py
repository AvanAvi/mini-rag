"""Loads raw documents from the corpus folder."""

from pathlib import Path

from mini_rag.document import Document


class Loader:
  """Reads UTF-8 .txt documents from a corpus directory."""

  def __init__(self, s_corpus_dir: str) -> None:
    """Store the corpus directory to read from."""
    self.s_corpus_dir = s_corpus_dir

  def load(self) -> list[Document]:
    """Return every non-empty .txt file as a Document, in sorted filename order."""
    obj_dir = Path(self.s_corpus_dir)
    if not obj_dir.is_dir():
      raise FileNotFoundError(f"Corpus directory not found: {self.s_corpus_dir}")

    l_paths = sorted(
      (obj_path for obj_path in obj_dir.iterdir()
       if obj_path.is_file() and obj_path.suffix.lower() == ".txt"),
      key=lambda obj_path: obj_path.name,
    )

    l_docs = []
    for obj_path in l_paths:
      s_text = self.read_text(obj_path)
      if not s_text.strip():
        continue  # skip files with no real content
      l_docs.append(Document(s_doc_id=obj_path.stem, s_text=s_text))

    if not l_docs:
      raise ValueError(f"No non-empty .txt files found in: {self.s_corpus_dir}")
    return l_docs

  def read_text(self, obj_path: Path) -> str:
    """Read one file as UTF-8 and normalise its line endings to \\n."""
    s_raw = obj_path.read_bytes().decode("utf-8")
    return s_raw.replace("\r\n", "\n").replace("\r", "\n")
