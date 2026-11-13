"""Builds and caches the evaluation indexes, one folder per chunk setting."""

import hashlib
import json
from pathlib import Path

from mini_rag.bm25_index import Bm25Index
from mini_rag.chunker import Chunker
from mini_rag.dense_index import DenseIndex
from mini_rag.document import Document
from mini_rag.index_bundle import IndexBundle


class EvalIndexStore:
  """Returns the indexes for a chunk setting, embedding only when the cache does not match."""

  S_META_FILE = "meta.json"

  def __init__(self, s_cache_dir: str = ".cache/eval", obj_embedder: object | None = None) -> None:
    """Store the cache folder and the optional embedder passed on to DenseIndex."""
    self.s_cache_dir = s_cache_dir
    self.obj_embedder = obj_embedder
    self.s_model_name = DenseIndex(obj_embedder=obj_embedder).s_model_name

  def get_indexes(self, l_docs: list[Document], i_size: int, i_overlap: int, b_rebuild: bool = False) -> IndexBundle:
    """Load the dense index from the cache if its meta.json matches, otherwise build and save it."""
    obj_chunker = Chunker(i_size, i_overlap)
    s_dir = self.folder(i_size, i_overlap)
    d_expected = {
      "i_size": i_size,
      "i_overlap": i_overlap,
      "s_model_name": self.s_model_name,
      "s_corpus_fingerprint": self.fingerprint(l_docs),
    }
    obj_dense = DenseIndex(self.s_model_name, self.obj_embedder)
    b_loaded = not b_rebuild and self.try_load(obj_dense, s_dir, d_expected)
    if not b_loaded:
      obj_dense.build(obj_chunker.chunk_documents(l_docs))
      obj_dense.save(s_dir)
      self.write_meta(s_dir, dict(d_expected, i_chunk_count=len(obj_dense.l_chunks)))

    # BM25 is cheap, so it is rebuilt from the chunks every time
    obj_bm25 = Bm25Index()
    obj_bm25.build(obj_dense.l_chunks)
    return IndexBundle(
      l_chunks=obj_dense.l_chunks, obj_bm25=obj_bm25, obj_dense=obj_dense,
      i_size=i_size, i_overlap=i_overlap, b_loaded=b_loaded,
    )

  def folder(self, i_size: int, i_overlap: int) -> str:
    """Return the cache folder of one chunk setting."""
    return str(Path(self.s_cache_dir) / f"size_{i_size}_overlap_{i_overlap}")

  def fingerprint(self, l_docs: list[Document]) -> str:
    """Return a SHA-256 over the sorted "doc_id:sha256(text)" lines of the corpus."""
    l_lines = sorted(
      f"{obj_doc.s_doc_id}:{hashlib.sha256(obj_doc.s_text.encode('utf-8')).hexdigest()}"
      for obj_doc in l_docs
    )
    return hashlib.sha256("\n".join(l_lines).encode("utf-8")).hexdigest()

  def try_load(self, obj_dense: DenseIndex, s_dir: str, d_expected: dict) -> bool:
    """Load the saved index if meta.json matches every expected value; return whether it worked."""
    obj_meta_path = Path(s_dir) / self.S_META_FILE
    if not obj_meta_path.is_file():
      return False
    try:
      d_meta = json.loads(obj_meta_path.read_text(encoding="utf-8"))
      if any(d_meta.get(s_key) != obj_value for s_key, obj_value in d_expected.items()):
        return False
      obj_dense.load(s_dir)
    except (FileNotFoundError, ValueError, KeyError, AttributeError):
      return False  # unreadable or incomplete cache: build again
    return len(obj_dense.l_chunks) == d_meta.get("i_chunk_count")

  def write_meta(self, s_dir: str, d_meta: dict) -> None:
    """Write meta.json next to the files saved by DenseIndex."""
    s_json = json.dumps(d_meta, ensure_ascii=False, indent=2)
    (Path(s_dir) / self.S_META_FILE).write_text(s_json, encoding="utf-8")
