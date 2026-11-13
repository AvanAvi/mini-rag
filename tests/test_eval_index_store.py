"""Tests for EvalIndexStore."""

import json

from bag_of_words_embedder import BagOfWordsEmbedder
from counting_embedder import CountingEmbedder

from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore


class TestEvalIndexStore:
  """Covers building, loading from the cache and the reasons to rebuild."""

  L_VOCABULARY = ["apple", "banana", "cherry", "date"]

  def docs(self) -> list[Document]:
    """Return three short documents; with fewer, a word in one document gets a BM25 score of zero."""
    return [
      Document(s_doc_id="doc_a", s_text="apple banana apple"),
      Document(s_doc_id="doc_b", s_text="cherry date cherry"),
      Document(s_doc_id="doc_c", s_text="apple apple apple"),
    ]

  def embedder(self) -> CountingEmbedder:
    """Return a counting wrapper around the bag-of-words embedder."""
    return CountingEmbedder(BagOfWordsEmbedder(self.L_VOCABULARY))

  def test_first_call_builds_and_writes_the_cache(self, tmp_path):
    """The first call embeds once and writes the index files plus meta.json."""
    obj_embedder = self.embedder()
    obj_store = EvalIndexStore(str(tmp_path), obj_embedder)
    obj_bundle = obj_store.get_indexes(self.docs(), 50, 10)
    assert obj_bundle.b_loaded is False
    assert obj_embedder.i_embed_calls == 1
    assert (obj_bundle.i_size, obj_bundle.i_overlap, len(obj_bundle.l_chunks)) == (50, 10, 3)

    obj_dir = tmp_path / "size_50_overlap_10"
    assert (obj_dir / "vectors.npy").is_file()
    assert (obj_dir / "chunks.json").is_file()
    assert json.loads((obj_dir / "meta.json").read_text(encoding="utf-8")) == {
      "i_size": 50, "i_overlap": 10, "s_model_name": "BAAI/bge-small-en-v1.5",
      "s_corpus_fingerprint": obj_store.fingerprint(self.docs()), "i_chunk_count": 3,
    }

  def test_second_call_loads_without_embedding(self, tmp_path):
    """A matching cache is loaded and the embedder is not called again."""
    obj_embedder = self.embedder()
    obj_store = EvalIndexStore(str(tmp_path), obj_embedder)
    obj_first = obj_store.get_indexes(self.docs(), 50, 10)
    obj_second = obj_store.get_indexes(self.docs(), 50, 10)
    assert obj_second.b_loaded is True
    assert obj_embedder.i_embed_calls == 1
    assert obj_second.l_chunks == obj_first.l_chunks
    # both indexes work after a load
    assert obj_second.obj_bm25.search("banana", 1)[0].obj_chunk.s_doc_id == "doc_a"
    assert obj_second.obj_dense.search("cherry", 1)[0].obj_chunk.s_doc_id == "doc_b"

  def test_rebuild_flag_embeds_again(self, tmp_path):
    """b_rebuild ignores a matching cache."""
    obj_embedder = self.embedder()
    obj_store = EvalIndexStore(str(tmp_path), obj_embedder)
    obj_store.get_indexes(self.docs(), 50, 10)
    obj_bundle = obj_store.get_indexes(self.docs(), 50, 10, b_rebuild=True)
    assert obj_bundle.b_loaded is False
    assert obj_embedder.i_embed_calls == 2

  def test_changed_corpus_rebuilds(self, tmp_path):
    """A different corpus fingerprint makes the cache stale."""
    obj_embedder = self.embedder()
    obj_store = EvalIndexStore(str(tmp_path), obj_embedder)
    obj_store.get_indexes(self.docs(), 50, 10)
    l_changed = self.docs() + [Document(s_doc_id="doc_d", s_text="date date apple")]
    obj_bundle = obj_store.get_indexes(l_changed, 50, 10)
    assert obj_bundle.b_loaded is False
    assert obj_embedder.i_embed_calls == 2
    assert len(obj_bundle.l_chunks) == 4

  def test_each_setting_has_its_own_folder(self, tmp_path):
    """Another chunk size is built into a second folder and does not disturb the first."""
    obj_embedder = self.embedder()
    obj_store = EvalIndexStore(str(tmp_path), obj_embedder)
    obj_store.get_indexes(self.docs(), 50, 10)
    obj_store.get_indexes(self.docs(), 12, 4)
    assert (tmp_path / "size_12_overlap_4" / "meta.json").is_file()
    assert obj_store.get_indexes(self.docs(), 50, 10).b_loaded is True
    assert obj_embedder.i_embed_calls == 2

  def test_fingerprint_ignores_document_order(self, tmp_path):
    """The fingerprint depends on the documents, not on the order they are given in."""
    obj_store = EvalIndexStore(str(tmp_path), self.embedder())
    l_docs = self.docs()
    assert obj_store.fingerprint(l_docs) == obj_store.fingerprint(list(reversed(l_docs)))
    assert len(obj_store.fingerprint(l_docs)) == 64
