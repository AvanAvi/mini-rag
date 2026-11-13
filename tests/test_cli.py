"""Offline tests for the ingest, query and eval commands of the Cli."""

import json
from pathlib import Path

import pytest

from bag_of_words_embedder import BagOfWordsEmbedder
from eval_set_builder import EvalSetBuilder
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

  def test_eval_missing_questions_file_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval with a questions file that does not exist reports an error line."""
    l_args = self.eval_args(tmp_path, str(tmp_path / "absent.json"))
    assert self.make_eval_cli().run(l_args) == 2
    assert capsys.readouterr().err.startswith("error: Questions file not found")
    assert not (tmp_path / "results.md").exists()

  def make_eval_cli(self) -> Cli:
    """Return a Cli whose fake embedder knows the words of the eval test corpus."""
    return Cli(obj_embedder=BagOfWordsEmbedder(EvalSetBuilder.L_VOCABULARY))

  def eval_args(self, obj_tmp: Path, s_questions: str) -> list[str]:
    """Write the eval test corpus and return eval arguments that keep every file inside obj_tmp."""
    s_corpus = EvalSetBuilder().write_corpus(obj_tmp / "corpus")
    return [
      "eval", "--questions", s_questions, "--corpus", s_corpus, "--sizes", "60,120", "--overlap", "10",
      "--cache-dir", str(obj_tmp / "cache"), "--out", str(obj_tmp / "results.md"),
    ]

  def test_eval_writes_report_and_returns_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval on a small valid set prints the compact tables and writes the markdown report."""
    obj_builder = EvalSetBuilder()
    s_questions = obj_builder.write_questions(tmp_path / "questions.json", obj_builder.entries())

    assert self.make_eval_cli().run(self.eval_args(tmp_path, s_questions)) == 0
    s_out = capsys.readouterr().out
    assert "Questions: 24 answerable, 6 unanswerable" in s_out
    assert "Table 1: retrieval metrics" in s_out
    assert "Table 3: threshold sweep" in s_out
    assert f"report: {tmp_path / 'results.md'}" in s_out
    s_report = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "- Chunk sizes: 60, 120" in s_report
    assert "- Questions: 24 answerable, 6 unanswerable" in s_report
    assert "## Table 5: Results by overlap group" in s_report
    assert (tmp_path / "cache" / "size_60_overlap_10" / "meta.json").is_file()
    assert (tmp_path / "cache" / "size_120_overlap_10" / "meta.json").is_file()

  def test_eval_build_only_writes_no_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--build-only fills the cache, prints one line per size and needs no questions file."""
    l_args = self.eval_args(tmp_path, str(tmp_path / "absent.json")) + ["--build-only"]

    assert self.make_eval_cli().run(l_args) == 0
    l_lines = capsys.readouterr().out.splitlines()
    assert len(l_lines) == 2
    assert l_lines[0].startswith("size 60: ") and " chunks, built, " in l_lines[0]
    assert l_lines[1].startswith("size 120: 14 chunks, built, ")
    assert not (tmp_path / "results.md").exists()

    # a second run finds the cache
    assert self.make_eval_cli().run(l_args[:1] + l_args[1:]) == 0
    assert all(" chunks, loaded, " in s_line for s_line in capsys.readouterr().out.splitlines())

  def test_eval_invalid_question_set_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A question set the validator rejects prints at most 10 problems and writes no report."""
    obj_builder = EvalSetBuilder()
    l_entries = [dict(d_entry, source_doc="doc_99") for d_entry in obj_builder.entries()[:14]]
    s_questions = obj_builder.write_questions(tmp_path / "questions.json", l_entries)

    assert self.make_eval_cli().run(self.eval_args(tmp_path, s_questions)) == 2
    l_lines = capsys.readouterr().err.splitlines()
    assert l_lines[0].startswith("error: the question set has ")
    assert len(l_lines) == 11  # the summary line plus 10 of the problems
    assert "a00: source_doc 'doc_99' is not a loaded document" in l_lines[1]
    assert not (tmp_path / "results.md").exists()

  def test_eval_bad_sizes_returns_2(self, tmp_path: Path) -> None:
    """A --sizes value that is not a list of whole numbers is a usage error."""
    assert self.make_eval_cli().run(["eval", "--sizes", "500,big"]) == 2

  def test_query_mismatched_model_name_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An index saved under another model name is reported as an error line, not raised."""
    s_index = self.ingest(tmp_path)
    capsys.readouterr()
    obj_chunks_path = Path(s_index) / "chunks.json"
    d_payload = json.loads(obj_chunks_path.read_text(encoding="utf-8"))
    d_payload["s_model_name"] = "some/other-model"
    obj_chunks_path.write_text(json.dumps(d_payload), encoding="utf-8")

    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index]) == 2
    s_err = capsys.readouterr().err
    assert s_err.startswith("error: ")
    assert "some/other-model" in s_err

  def test_query_brief_cuts_long_hits_only(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--brief cuts a hit's text to 160 characters plus "..." and leaves shorter hits alone."""
    s_long = "A tokamak confines plasma. " + "word " * 40
    s_short = "Dogs bark loudly at night."
    s_corpus = self.write_corpus(tmp_path / "corpus", {"dogs.txt": s_short, "tokamak.txt": s_long})
    s_index = str(tmp_path / "index")
    assert self.make_cli().run(["ingest", "--corpus", s_corpus, "--index-dir", s_index]) == 0
    capsys.readouterr()

    assert self.make_cli().run(["query", self.S_QUESTION, "--index-dir", s_index, "--brief"]) == 0
    l_lines = capsys.readouterr().out.split("Answer:\n", 1)[0].splitlines()
    i_first = next(i_pos for i_pos, s_line in enumerate(l_lines) if s_line.startswith("1. [tokamak#0]"))
    assert l_lines[i_first + 1] == s_long[:160] + "..."
    i_second = next(i_pos for i_pos, s_line in enumerate(l_lines) if s_line.startswith("2. [dogs#0]"))
    assert l_lines[i_second + 1] == s_short

  def test_no_command_returns_2(self) -> None:
    """Running without a subcommand is a usage error."""
    assert self.make_cli().run([]) == 2
