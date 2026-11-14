"""Tests for the example custom embedder and its config, run through the CLI on a tiny corpus."""

import json
from pathlib import Path

import pytest

from mini_rag.cli import Cli
from mini_rag.component_factory import ComponentFactory
from mini_rag.config_loader import ConfigLoader
from mini_rag.embedder import Embedder


class TestExampleCustomEmbedder:
  """Checks that examples/custom.toml builds the example embedder and that ingest and query work with it."""

  S_CONFIG = "examples/custom.toml"
  D_DOCUMENTS = {
    "cats.txt": "Cats sleep most of the day.",
    "dogs.txt": "Dogs bark loudly at night.",
    "tokamak.txt": "A tokamak confines plasma with magnetic fields.",
  }

  @pytest.fixture(autouse=True)
  def project_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from the project root, where the examples folder lives."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)

  def write_corpus(self, obj_tmp: Path) -> str:
    """Write the tiny corpus into obj_tmp and return the folder as a string."""
    obj_corpus = obj_tmp / "corpus"
    obj_corpus.mkdir()
    for s_name, s_text in self.D_DOCUMENTS.items():
      (obj_corpus / s_name).write_text(s_text, encoding="utf-8")
    return str(obj_corpus)

  def test_example_config_builds_the_example_embedder(self) -> None:
    """The factory imports the class named in the example config and applies its options."""
    obj_embedder = ComponentFactory(ConfigLoader().load(self.S_CONFIG)).embedder()
    assert isinstance(obj_embedder, Embedder)
    assert type(obj_embedder).__name__ == "HashedBagOfWordsEmbedder"
    assert obj_embedder.s_name == "hashed-bag-of-words-256"

  def test_vectors_are_deterministic_and_share_buckets_for_shared_words(self) -> None:
    """The same text gives the same vector, of the configured length, with one count per word."""
    obj_embedder = ComponentFactory(ConfigLoader().load(self.S_CONFIG)).embedder()
    l_first = obj_embedder.query_embed("plasma plasma tokamak")
    assert l_first == obj_embedder.embed(["Plasma, plasma: TOKAMAK"])[0]
    assert len(l_first) == 256
    assert sum(l_first) == 3.0
    assert max(l_first) >= 2.0

  def test_ingest_and_query_with_the_example_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """ingest and query run with the example embedder; the corpus and index folders are overridden."""
    s_corpus = self.write_corpus(tmp_path)
    s_index = str(tmp_path / "index")
    obj_cli = Cli()
    assert obj_cli.run(["--config", self.S_CONFIG, "ingest", "--corpus", s_corpus, "--index-dir", s_index]) == 0
    assert "documents: 3" in capsys.readouterr().out
    # the index records the embedder's name
    d_saved = json.loads((tmp_path / "index" / "chunks.json").read_text(encoding="utf-8"))
    assert d_saved["s_model_name"] == "hashed-bag-of-words-256"

    l_args = ["--config", self.S_CONFIG, "query", "what confines plasma", "--index-dir", s_index, "--threshold", "0.2"]
    assert obj_cli.run(l_args) == 0
    s_out = capsys.readouterr().out
    assert s_out.startswith("1. [tokamak#0]")
    assert "found: yes, written by a model: no" in s_out

  def test_index_from_another_embedder_is_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Querying with other embedder settings than the index was built with fails and asks for a new ingest."""
    s_corpus = self.write_corpus(tmp_path)
    s_index = str(tmp_path / "index")
    assert Cli().run(["--config", self.S_CONFIG, "ingest", "--corpus", s_corpus, "--index-dir", s_index]) == 0
    capsys.readouterr()

    obj_other = tmp_path / "other.toml"
    obj_other.write_text(
      '[embedder]\nkind = "custom"\nclass = "examples.custom_embedder:HashedBagOfWordsEmbedder"\n'
      "[embedder.options]\ni_dim = 64\n",
      encoding="utf-8",
    )
    assert Cli().run(["--config", str(obj_other), "query", "plasma", "--index-dir", s_index]) == 2
    s_err = capsys.readouterr().err
    assert "hashed-bag-of-words-256" in s_err and "hashed-bag-of-words-64" in s_err
    assert "run ingest again" in s_err

  def test_eval_cache_records_the_embedder_name(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval --build-only with the example config saves the embedder's name in the cache's meta.json."""
    l_args = [
      "--config", self.S_CONFIG, "eval", "--build-only", "--corpus", self.write_corpus(tmp_path),
      "--sizes", "100", "--overlap", "10", "--cache-dir", str(tmp_path / "cache"),
    ]
    assert Cli().run(l_args) == 0
    capsys.readouterr()
    d_meta = json.loads((tmp_path / "cache" / "size_100_overlap_10" / "meta.json").read_text(encoding="utf-8"))
    assert d_meta["s_model_name"] == "hashed-bag-of-words-256"
