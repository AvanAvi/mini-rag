"""Dense vector index built on fastembed embeddings."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mini_rag.chunk import Chunk
from mini_rag.search_hit import SearchHit


class DenseIndex:
  """Embeds chunks and ranks them by cosine similarity to a query."""

  S_VECTORS_FILE = "vectors.npy"
  S_CHUNKS_FILE = "chunks.json"

  def __init__(self, s_model_name: str = "BAAI/bge-small-en-v1.5", obj_embedder: object | None = None) -> None:
    """Store the model name and optional embedder; the real model is loaded lazily."""
    self.s_model_name = s_model_name
    self.obj_embedder = obj_embedder
    self.l_chunks: list[Chunk] = []
    self.arr_vectors: np.ndarray | None = None

  def build(self, l_chunks: list[Chunk]) -> None:
    """Embed all chunks, L2-normalise the vectors and keep them in memory."""
    if not l_chunks:
      raise ValueError("Cannot build a dense index from an empty chunk list")
    l_texts = [obj_chunk.s_text for obj_chunk in l_chunks]
    arr_raw = np.asarray(self.get_embedder().embed(l_texts), dtype=np.float32)
    self.arr_vectors = self.normalise(arr_raw)
    self.l_chunks = list(l_chunks)

  def search(self, s_query: str, i_top_k: int) -> list[SearchHit]:
    """Return up to i_top_k hits by cosine similarity, best first."""
    if self.arr_vectors is None:
      raise RuntimeError("build() or load() must be called before search()")
    if i_top_k < 1:
      raise ValueError(f"i_top_k must be at least 1, got {i_top_k}")
    if not s_query.strip():
      return []

    # query_embed applies the model's query prefix (bge); passages use embed()
    arr_query = np.asarray(self.get_embedder().query_embed(s_query), dtype=np.float32)
    arr_query = self.normalise(arr_query.reshape(1, -1))[0]
    arr_scores = self.arr_vectors @ arr_query
    # a stable sort keeps chunk order for equal scores
    arr_order = np.argsort(-arr_scores, kind="stable")[:i_top_k]
    return [
      SearchHit(obj_chunk=self.l_chunks[i_pos], f_score=float(arr_scores[i_pos]))
      for i_pos in arr_order
    ]

  def save(self, s_dir: str) -> None:
    """Write vectors.npy and chunks.json (chunks plus model name) into s_dir."""
    if self.arr_vectors is None:
      raise RuntimeError("build() or load() must be called before save()")
    obj_dir = Path(s_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    np.save(obj_dir / self.S_VECTORS_FILE, self.arr_vectors, allow_pickle=False)
    d_payload = {
      "s_model_name": self.s_model_name,
      "l_chunks": [asdict(obj_chunk) for obj_chunk in self.l_chunks],
    }
    s_json = json.dumps(d_payload, ensure_ascii=False, indent=2)
    (obj_dir / self.S_CHUNKS_FILE).write_text(s_json, encoding="utf-8")

  def load(self, s_dir: str) -> None:
    """Restore vectors and chunks saved by save(); the model name must match."""
    obj_dir = Path(s_dir)
    for s_name in (self.S_VECTORS_FILE, self.S_CHUNKS_FILE):
      if not (obj_dir / s_name).is_file():
        raise FileNotFoundError(f"Missing {s_name} in {obj_dir}")

    d_payload = json.loads((obj_dir / self.S_CHUNKS_FILE).read_text(encoding="utf-8"))
    if d_payload["s_model_name"] != self.s_model_name:
      raise ValueError(
        f"Saved model {d_payload['s_model_name']!r} differs from {self.s_model_name!r}"
      )
    arr_vectors = np.load(obj_dir / self.S_VECTORS_FILE, allow_pickle=False)
    l_chunks = [Chunk(**d_chunk) for d_chunk in d_payload["l_chunks"]]
    if arr_vectors.ndim != 2:
      raise ValueError(f"Saved vectors must be 2-D, got {arr_vectors.ndim}-D")
    if arr_vectors.shape[0] != len(l_chunks):
      raise ValueError(
        f"Saved vectors have {arr_vectors.shape[0]} rows but there are {len(l_chunks)} chunks"
      )
    self.arr_vectors = arr_vectors
    self.l_chunks = l_chunks

  def get_embedder(self) -> object:
    """Return the embedder, creating the fastembed wrapper on first use."""
    if self.obj_embedder is None:
      # imported here so tests with a fake embedder never import fastembed
      from mini_rag.fastembed_embedder import FastEmbedEmbedder
      self.obj_embedder = FastEmbedEmbedder(self.s_model_name)
    return self.obj_embedder

  def normalise(self, arr_matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 norm; all-zero rows are left as zeros."""
    arr_norms = np.linalg.norm(arr_matrix, axis=1, keepdims=True)
    arr_norms[arr_norms == 0] = 1.0
    return (arr_matrix / arr_norms).astype(np.float32)
