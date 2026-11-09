"""Offline tests for the ingest, query and eval commands of the Cli."""

from pathlib import Path

import pytest

from bag_of_words_embedder import BagOfWordsEmbedder
from fake_llm_client import FakeLlmClient
from mini_rag.cli import Cli


class TestCli:
  """Runs the Cli on a tiny hand-made corpus with a bag-of-words embedder."""

  L_VOCABULARY = ["cats", "sleep", "tokamak", "plasma", "magnetic", "dogs", "bark"]
  D_DOCUMENTS = {
    "cats.txt": "Cats are small furry animals. Cats sleep most of the day.",
    "dogs.txt": "Dogs bark loudly at night.",
    "tokamak.txt": "A tokamak is a device that confines plasma with magnetic fields.",
  }
  S_QUESTION = "What is a tokamak?"

  def make_cli(self, obj_llm_client: object | None = None) -> Cli:
    """Return a Cli with the fake embedder and an optional fake client."""
    return Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY), obj_llm_client=obj_llm_client)

  def write_corpus(self, obj_dir: Path, d_documents: dict[str, str]) -> str:
    """Write each document as a .txt file into obj_dir and return the folder as a string."""
    obj_dir.mkdir()
    for s_name, s_text in d_documents.items():
      (obj_dir / s_name).write_text(s_text, encoding="utf-8")
    return str(obj_dir)

  def ingest(self, obj_tmp: Path) -> str:
    """Ingest the tiny corpus into obj_tmp/index and return the index folder."""
    s_corpus = self.write_corpus(obj_tmp / "corpus", self.D_DOCUMENTS)
    s_index = str(obj_tmp / "index")
    assert self.make_cli().run(["ingest", "--corpus", s_corpus, "--index-dir", s_index]) == 0
    return s_index

  def test_ingest_writes_index_and_prints_counts(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A 50-character document cut 20/5 gives 3 chunks; both index files are written."""
    s_corpus = self.write_corpus(tmp_path / "corpus", {"long.txt": "x" * 50})
    s_index = str(tmp_path / "index")
    l_args = ["ingest", "--corpus", s_corpus, "--index-dir", s_index, "--size", "20", "--overlap", "5"]

    assert self.make_cli().run(l_args) == 0
    s_out = capsys.readouterr().out
    assert "documents: 1" in s_out
    assert "chunks: 3" in s_out
    assert "chunk size: 20" in s_out
    assert "overlap: 5" in s_out
    assert f"index folder: {s_index}" in s_out
    assert "elapsed:" in s_out
    assert (tmp_path / "index" / "vectors.npy").is_file()
    assert (tmp_path / "index" / "chunks.json").is_file()

  def test_ingest_missing_corpus_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing corpus folder is reported as an error line, not raised."""
    l_args = ["ingest", "--corpus", str(tmp_path / "nope"), "--index-dir", str(tmp_path / "index")]
    assert self.make_cli().run(l_args) == 2
    assert capsys.readouterr().err.startswith("error: ")

  def test_ingest_overlap_not_below_size_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """overlap >= size is rejected with an error line and nothing is written."""
    s_corpus = self.write_corpus(tmp_path / "corpus", self.D_DOCUMENTS)
    l_args = ["ingest", "--corpus", s_corpus, "--index-dir", str(tmp_path / "index"),
              "--size", "100", "--overlap", "100"]
    assert self.make_cli().run(l_args) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert not (tmp_path / "index").exists()

  def test_query_before_ingest_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Querying a folder without an index tells the user to run ingest first."""
    s_index = str(tmp_path / "index")
    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index]) == 2
    assert f"error: no index found in {s_index}; run ingest first" in capsys.readouterr().err

  def test_query_prints_hits_and_answer(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The tokamak chunk ranks first with all three scores and the answer is found."""
    s_index = self.ingest(tmp_path)
    capsys.readouterr()

    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index]) == 0
    s_out = capsys.readouterr().out
    s_first = next(s_line for s_line in s_out.splitlines() if s_line.startswith("1. "))
    assert s_first.startswith("1. [tokamak#0]")
    assert "dense 0.577" in s_first
    assert "bm25 " in s_first and "bm25 -" not in s_first
    assert "fused " in s_first
    assert "bm25 -" in s_out  # the other chunks share no word with the question
    assert self.D_DOCUMENTS["tokamak.txt"] in s_out
    assert "found: yes, written by a model: no" in s_out

  def test_query_threshold_one_is_not_found(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With the threshold at 1.0 no hit qualifies and the answer is "not found"."""
    s_index = self.ingest(tmp_path)
    capsys.readouterr()

    l_args = ["query", self.S_QUESTION, "--index-dir", s_index, "--threshold", "1.0"]
    assert self.make_cli().run(l_args) == 0
    s_out = capsys.readouterr().out
    assert "Not found in the corpus." in s_out
    assert "found: no, written by a model: no" in s_out

  def test_query_invalid_top_k_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A top-k below 1 is reported as an error line."""
    s_index = self.ingest(tmp_path)
    capsys.readouterr()
    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index, "--top-k", "0"]) == 2
    assert capsys.readouterr().err.startswith("error: ")

  def test_generate_uses_given_client(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--generate with an injected client prints the client's reply as a model answer."""
    s_index = self.ingest(tmp_path)
    capsys.readouterr()

    obj_cli = self.make_cli(FakeLlmClient("A tokamak confines plasma. [tokamak#0]"))
    assert obj_cli.run(["query", self.S_QUESTION, "--index-dir", s_index, "--generate"]) == 0
    s_out = capsys.readouterr().out
    assert "A tokamak confines plasma. [tokamak#0]" in s_out
    assert "found: yes, written by a model: yes" in s_out

  def test_generate_without_key_falls_back(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """--generate without a client or key says why and prints the extractive answer."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s_index = self.ingest(tmp_path)
    capsys.readouterr()

    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index, "--generate"]) == 0
    s_out = capsys.readouterr().out
    assert "showing the extractive answer" in s_out
    s_answer = s_out.split("Answer:\n", 1)[1]
    assert s_answer.startswith("[tokamak#0]\n")
    assert "found: yes, written by a model: no" in s_out

  def test_eval_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
    """eval is not implemented and says so."""
    assert self.make_cli().run(["eval"]) == 2
    assert "eval is not implemented yet" in capsys.readouterr().err

  def test_no_command_returns_2(self) -> None:
    """Running without a subcommand is a usage error."""
    assert self.make_cli().run([]) == 2
