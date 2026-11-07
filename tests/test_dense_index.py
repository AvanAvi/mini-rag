"""Unit tests for the DenseIndex."""

import json
from pathlib import Path

import numpy as np
import pytest

from fake_embedder import FakeEmbedder
from mini_rag.chunk import Chunk
from mini_rag.dense_index import DenseIndex
from mini_rag.search_hit import SearchHit


class TestDenseIndex:
  """Checks normalisation, ranking, limits, persistence and error handling."""

  def passages(self) -> dict[str, list[float]]:
    """Four passages with hand-made, deliberately unnormalised vectors."""
    return {
      "north": [3.0, 0.0, 0.0],
      "east": [0.0, 5.0, 0.0],
      "north-east": [2.0, 2.0, 0.0],
      "up": [0.0, 0.0, 0.5],
    }

  def queries(self) -> dict[str, list[float]]:
    """Query vectors; "tie" is equally close to "north" and "east"."""
    return {
      "mostly north": [4.0, 1.0, 0.0],
      "tie": [1.0, 1.0, 0.0],
      "up": [0.0, 0.0, 9.0],
    }

  def make_chunks(self, l_texts: list[str]) -> list[Chunk]:
    """Wrap each text in a Chunk of document "d", indexed by position."""
    return [
      Chunk(s_doc_id="d", i_index=i_pos, i_start=i_pos * 10, s_text=s_text)
      for i_pos, s_text in enumerate(l_texts)
    ]

  def build_index(self, l_texts: list[str] | None = None) -> DenseIndex:
    """Return a DenseIndex with a FakeEmbedder, built over the given texts."""
    if l_texts is None:
      l_texts = ["north", "east", "north-east", "up"]
    obj_index = DenseIndex(obj_embedder=FakeEmbedder(self.passages(), self.queries()))
    obj_index.build(self.make_chunks(l_texts))
    return obj_index

  def texts(self, l_hits: list[SearchHit]) -> list[str]:
    """Return the chunk texts of the hits, in order."""
    return [obj_hit.obj_chunk.s_text for obj_hit in l_hits]

  def test_default_model_and_lazy_embedder(self) -> None:
    """The default model is bge-small and no embedder is created in __init__."""
    obj_index = DenseIndex()
    assert obj_index.s_model_name == "BAAI/bge-small-en-v1.5"
    assert obj_index.obj_embedder is None

  def test_rows_have_unit_norm(self) -> None:
    """After build every stored vector has L2 norm 1 and dtype float32."""
    obj_index = self.build_index()
    assert obj_index.arr_vectors.dtype == np.float32
    arr_norms = np.linalg.norm(obj_index.arr_vectors, axis=1)
    np.testing.assert_allclose(arr_norms, 1.0, rtol=1e-6)

  def test_exact_ranking_order(self) -> None:
    """Hits come back best first with the expected cosine scores."""
    obj_index = self.build_index()
    l_hits = obj_index.search("mostly north", 4)
    assert self.texts(l_hits) == ["north", "north-east", "east", "up"]
    f_norm = np.sqrt(17.0)
    l_expected = [4.0 / f_norm, 5.0 / (f_norm * np.sqrt(2.0)), 1.0 / f_norm, 0.0]
    np.testing.assert_allclose([obj_hit.f_score for obj_hit in l_hits], l_expected, rtol=1e-5, atol=1e-6)
    assert all(isinstance(obj_hit, SearchHit) for obj_hit in l_hits)
    assert all(isinstance(obj_hit.f_score, float) for obj_hit in l_hits)

  def test_query_uses_query_embed(self) -> None:
    """search() embeds the query through query_embed, not embed."""
    obj_index = self.build_index()
    obj_index.search("up", 1)
    assert obj_index.obj_embedder.l_query_calls == ["up"]

  def test_low_scores_are_kept(self) -> None:
    """Zero-similarity chunks are still returned; thresholding is not done here."""
    obj_index = self.build_index()
    l_hits = obj_index.search("up", 4)
    assert self.texts(l_hits)[0] == "up"
    assert len(l_hits) == 4
    assert l_hits[-1].f_score == pytest.approx(0.0)

  def test_top_k_limits_results(self) -> None:
    """At most i_top_k hits are returned, never more than there are chunks."""
    obj_index = self.build_index()
    assert len(obj_index.search("mostly north", 1)) == 1
    assert len(obj_index.search("mostly north", 3)) == 3
    assert len(obj_index.search("mostly north", 10)) == 4

  def test_ties_keep_chunk_order(self) -> None:
    """Chunks with equal scores keep their original chunk order."""
    obj_index = self.build_index(["east", "north", "north-east"])
    l_hits = obj_index.search("tie", 3)
    # north-east is identical to the query; east and north tie behind it
    assert self.texts(l_hits) == ["north-east", "east", "north"]
    assert l_hits[1].f_score == l_hits[2].f_score

  @pytest.mark.parametrize("s_query", ["", "   ", "\n\t"])
  def test_empty_query_returns_empty(self, s_query: str) -> None:
    """An empty or whitespace-only query returns an empty list without embedding."""
    obj_index = self.build_index()
    assert obj_index.search(s_query, 3) == []
    assert obj_index.obj_embedder.l_query_calls == []

  def test_search_before_build_raises(self) -> None:
    """Searching an index that was never built or loaded raises RuntimeError."""
    obj_index = DenseIndex(obj_embedder=FakeEmbedder(self.passages(), self.queries()))
    with pytest.raises(RuntimeError):
      obj_index.search("up", 3)

  @pytest.mark.parametrize("i_top_k", [0, -1])
  def test_invalid_top_k_raises(self, i_top_k: int) -> None:
    """An i_top_k below 1 raises ValueError."""
    obj_index = self.build_index()
    with pytest.raises(ValueError):
      obj_index.search("up", i_top_k)

  def test_empty_build_raises(self) -> None:
    """Building from an empty list raises ValueError."""
    obj_index = DenseIndex(obj_embedder=FakeEmbedder(self.passages(), self.queries()))
    with pytest.raises(ValueError):
      obj_index.build([])

  def test_save_load_round_trip(self, tmp_path: Path) -> None:
    """A loaded index has identical vectors, chunks and search results."""
    obj_original = self.build_index()
    s_dir = str(tmp_path / "index")
    obj_original.save(s_dir)
    assert (tmp_path / "index" / "vectors.npy").is_file()
    d_saved = json.loads((tmp_path / "index" / "chunks.json").read_text(encoding="utf-8"))
    assert d_saved["s_model_name"] == "BAAI/bge-small-en-v1.5"

    obj_loaded = DenseIndex(obj_embedder=FakeEmbedder(self.passages(), self.queries()))
    obj_loaded.load(s_dir)
    np.testing.assert_array_equal(obj_loaded.arr_vectors, obj_original.arr_vectors)
    assert obj_loaded.arr_vectors.dtype == np.float32
    assert obj_loaded.l_chunks == obj_original.l_chunks
    assert obj_loaded.search("mostly north", 4) == obj_original.search("mostly north", 4)

  @pytest.mark.parametrize("s_missing", ["vectors.npy", "chunks.json"])
  def test_load_missing_file_raises(self, tmp_path: Path, s_missing: str) -> None:
    """Loading when either saved file is missing raises FileNotFoundError."""
    self.build_index().save(str(tmp_path))
    (tmp_path / s_missing).unlink()
    with pytest.raises(FileNotFoundError):
      DenseIndex(obj_embedder=FakeEmbedder({}, {})).load(str(tmp_path))

  def test_load_model_mismatch_raises(self, tmp_path: Path) -> None:
    """Loading an index saved with another model name raises ValueError."""
    self.build_index().save(str(tmp_path))
    obj_other = DenseIndex(s_model_name="some/other-model", obj_embedder=FakeEmbedder({}, {}))
    with pytest.raises(ValueError):
      obj_other.load(str(tmp_path))

  def write_saved_pair(self, obj_dir: Path, arr_vectors: np.ndarray, i_chunks: int) -> None:
    """Write vectors.npy and a chunks.json holding i_chunks chunks into obj_dir by hand."""
    np.save(obj_dir / "vectors.npy", arr_vectors, allow_pickle=False)
    d_payload = {
      "s_model_name": "BAAI/bge-small-en-v1.5",
      "l_chunks": [
        {"s_doc_id": "d", "i_index": i_pos, "i_start": 0, "s_text": f"text {i_pos}"}
        for i_pos in range(i_chunks)
      ],
    }
    (obj_dir / "chunks.json").write_text(json.dumps(d_payload), encoding="utf-8")

  def test_load_non_2d_vectors_raises(self, tmp_path: Path) -> None:
    """Loading a 1-D vectors array raises ValueError and leaves the index unbuilt."""
    self.write_saved_pair(tmp_path, np.ones(3, dtype=np.float32), 3)
    obj_index = DenseIndex(obj_embedder=FakeEmbedder({}, {}))
    with pytest.raises(ValueError, match="2-D"):
      obj_index.load(str(tmp_path))
    assert obj_index.arr_vectors is None
    assert obj_index.l_chunks == []

  def test_load_row_count_mismatch_raises(self, tmp_path: Path) -> None:
    """Loading 3 vector rows with 2 saved chunks raises ValueError and leaves the index unbuilt."""
    self.write_saved_pair(tmp_path, np.ones((3, 4), dtype=np.float32), 2)
    obj_index = DenseIndex(obj_embedder=FakeEmbedder({}, {}))
    with pytest.raises(ValueError, match="rows"):
      obj_index.load(str(tmp_path))
    assert obj_index.arr_vectors is None
    assert obj_index.l_chunks == []

  @pytest.mark.slow
  def test_real_model_ranks_closest_chunk_first(self) -> None:
    """With the real fastembed model the semantically closest chunk ranks first."""
    obj_index = DenseIndex()
    obj_index.build(self.make_chunks([
      "The cat curled up on the sofa and purred.",
      "Interest rates were raised by the central bank.",
      "The rocket launched into orbit around the Earth.",
    ]))
    l_hits = obj_index.search("Which pet was sleeping on the couch?", 3)
    assert l_hits[0].obj_chunk.i_index == 0
    assert l_hits[0].f_score > l_hits[1].f_score
