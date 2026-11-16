"""Tests for [retrieval] mode: hybrid (default), dense and bm25 in query and calibrate."""

import re
from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from calibration_set_builder import CalibrationSetBuilder
from recording_model import RecordingModel

import mini_rag.fastembed_embedder
from mini_rag.answerer import Answerer
from mini_rag.bm25_index import Bm25Index
from mini_rag.bm25_only_retriever import Bm25OnlyRetriever
from mini_rag.chunk import Chunk
from mini_rag.cli import Cli
from mini_rag.config import RagConfig
from mini_rag.config_loader import ConfigLoader
from mini_rag.dense_method import DenseMethod
from mini_rag.dense_only_retriever import DenseOnlyRetriever
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.evaluator import Evaluator
from mini_rag.hybrid_hit import HybridHit
from mini_rag.hybrid_method import HybridMethod


class TestRetrievalMode:
  """Runs query and calibrate in each mode on tiny corpora whose scores are known."""

  L_VOCABULARY = ["cats", "dogs", "bark", "sleep", "tokamak", "plasma", "magnetic"]
  D_DOCUMENTS = {
    "cats.txt": "Cats sleep most of the day.",
    "dogs.txt": "Dogs bark loudly at night.",
    "tokamak.txt": "A tokamak confines plasma with magnetic fields. Plasma is hot.",
  }
  S_HIT_PATTERN = r"^\d+\. \[(\S+)\]  dense (\S+)  bm25 (\S+)  fused (\S+)$"
  S_BM25_NOTE = (
    "note: bm25 mode has no dense score, so the threshold is not applied; "
    "the answer counts as found when at least one chunk matched a query term"
  )
  S_CALIBRATE_ERROR = "error: calibrate needs a dense score; set [retrieval] mode to hybrid or dense\n"

  def cli(self) -> Cli:
    """Return a Cli whose embedder counts vocabulary words."""
    return Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY))

  def write_config(self, obj_tmp: Path, s_extra: str = "") -> str:
    """Write the corpus and a config pointing at it and at an index folder inside obj_tmp."""
    obj_corpus = obj_tmp / "corpus"
    obj_corpus.mkdir(exist_ok=True)
    for s_name, s_text in self.D_DOCUMENTS.items():
      (obj_corpus / s_name).write_text(s_text, encoding="utf-8")
    obj_path = obj_tmp / "my.toml"
    obj_path.write_text(
      f"[corpus]\ndir = '{obj_corpus.as_posix()}'\n[index]\ndir = '{(obj_tmp / 'index').as_posix()}'\n"
      f"[chunking]\nsize = 200\noverlap = 20\n{s_extra}",
      encoding="utf-8",
    )
    return str(obj_path)

  def ingest(self, obj_tmp: Path, capsys: pytest.CaptureFixture[str], s_extra: str = "") -> str:
    """Write the config, build the index with the bag-of-words embedder and return the config path."""
    s_config = self.write_config(obj_tmp, s_extra)
    assert self.cli().run(["--config", s_config, "ingest"]) == 0
    capsys.readouterr()
    return s_config

  def hit_rows(self, s_out: str) -> list[tuple[str, str, str, str]]:
    """Return (marker, dense, bm25, fused) for every printed hit line."""
    return [obj_match.groups() for obj_match in re.finditer(self.S_HIT_PATTERN, s_out, re.MULTILINE)]

  def spy_on_bm25_build(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count calls of Bm25Index.build while keeping its behaviour; return the list that records them."""
    l_calls: list[int] = []
    obj_original = Bm25Index.build

    def record(obj_self: Bm25Index, l_chunks: list[Chunk]) -> None:
      """Record the call and build the index as before."""
      l_calls.append(len(l_chunks))
      obj_original(obj_self, l_chunks)

    monkeypatch.setattr(Bm25Index, "build", record)
    return l_calls

  # config and precedence

  def test_default_mode_is_hybrid(self) -> None:
    """The built-in config and the shipped mini-rag.toml both use mode "hybrid"."""
    assert RagConfig().s_retrieval_mode == "hybrid"
    assert ConfigLoader().load("mini-rag.toml").s_retrieval_mode == "hybrid"

  def test_invalid_mode_names_the_key_and_the_allowed_values(self) -> None:
    """A mode outside hybrid, dense and bm25 raises ValueError naming [retrieval] mode."""
    with pytest.raises(ValueError, match=r"\[retrieval\] mode must be one of hybrid, dense, bm25, got 'sparse'"):
      ConfigLoader().from_dict({"retrieval": {"mode": "sparse"}})

  @pytest.mark.parametrize("s_command", ["query", "calibrate"])
  def test_option_then_config_then_default(self, s_command: str) -> None:
    """--mode beats the config file, and the config file beats the built-in default."""
    l_args = [s_command, "q"] if s_command == "query" else [s_command]
    assert Cli().build_parser().parse_args(l_args).s_mode == "hybrid"
    obj_dense_config = RagConfig(s_retrieval_mode="dense")
    assert Cli().build_parser(obj_dense_config).parse_args(l_args).s_mode == "dense"
    assert Cli().build_parser(obj_dense_config).parse_args([*l_args, "--mode", "bm25"]).s_mode == "bm25"

  def test_invalid_mode_option_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown --mode value exits with 2."""
    assert Cli().run(["query", "q", "--mode", "sparse"]) == 2
    assert "--mode" in capsys.readouterr().err

  def test_config_mode_is_used_by_query(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """mode = "dense" in the config file gives dense hits without any --mode option."""
    s_config = self.ingest(tmp_path, capsys, '[retrieval]\nmode = "dense"\n')
    assert self.cli().run(["--config", s_config, "query", "tokamak plasma", "--threshold", "0.1"]) == 0
    assert all(t_row[2:] == ("-", "-") for t_row in self.hit_rows(capsys.readouterr().out))

  # query in each mode

  def test_dense_mode_orders_by_cosine_and_builds_no_bm25_index(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Dense mode lists hits by falling cosine score, prints "-" for BM25 and fused, and never builds BM25."""
    s_config = self.ingest(tmp_path, capsys)
    l_builds = self.spy_on_bm25_build(monkeypatch)
    assert self.cli().run(["--config", s_config, "query", "tokamak plasma", "--mode", "dense", "--threshold", "0.1"]) == 0
    s_out = capsys.readouterr().out
    l_rows = self.hit_rows(s_out)
    assert l_rows[0][0] == "tokamak#0"
    l_dense = [float(t_row[1]) for t_row in l_rows]
    assert l_dense == sorted(l_dense, reverse=True)
    assert all(t_row[2:] == ("-", "-") for t_row in l_rows)
    assert l_builds == []
    assert self.S_BM25_NOTE not in s_out
    assert "found: yes, written by a model: no" in s_out

  def test_dense_mode_keeps_the_threshold(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dense mode applies the threshold as hybrid does: 1.0 rejects a best score below 1.0."""
    s_config = self.ingest(tmp_path, capsys)
    assert self.cli().run(["--config", s_config, "query", "tokamak plasma", "--mode", "dense", "--threshold", "1.0"]) == 0
    assert "found: no, written by a model: no" in capsys.readouterr().out

  def test_hybrid_mode_builds_bm25_once_and_prints_fused_scores(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """The default mode builds the BM25 index once and prints a number in the fused column."""
    s_config = self.ingest(tmp_path, capsys)
    l_builds = self.spy_on_bm25_build(monkeypatch)
    assert self.cli().run(["--config", s_config, "query", "tokamak plasma", "--threshold", "0.1"]) == 0
    l_rows = self.hit_rows(capsys.readouterr().out)
    assert len(l_builds) == 1
    assert all(re.fullmatch(r"\d\.\d{4}", t_row[3]) for t_row in l_rows)

  def test_bm25_mode_orders_by_bm25_and_skips_the_threshold(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
  ) -> None:
    """bm25 mode lists hits by falling BM25 score, prints the note before the answer and ignores the threshold."""
    s_config = self.ingest(tmp_path, capsys)
    assert self.cli().run(["--config", s_config, "query", "plasma dogs", "--mode", "bm25", "--threshold", "1.0"]) == 0
    s_out = capsys.readouterr().out
    l_rows = self.hit_rows(s_out)
    assert [t_row[0] for t_row in l_rows] == ["tokamak#0", "dogs#0"]
    l_bm25 = [float(t_row[2]) for t_row in l_rows]
    assert l_bm25 == sorted(l_bm25, reverse=True)
    assert all((t_row[1], t_row[3]) == ("-", "-") for t_row in l_rows)
    assert s_out.index(self.S_BM25_NOTE) < s_out.index("Answer:\n")
    assert "found: yes, written by a model: no" in s_out

  def test_bm25_mode_without_a_matching_term_is_not_found(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
  ) -> None:
    """With no chunk sharing a query term, bm25 mode returns no hit and the answer is not found."""
    s_config = self.ingest(tmp_path, capsys)
    assert self.cli().run(["--config", s_config, "query", "zebra", "--mode", "bm25", "--threshold", "-1.0"]) == 0
    s_out = capsys.readouterr().out
    assert self.hit_rows(s_out) == []
    assert "Not found in the corpus." in s_out
    assert "found: no, written by a model: no" in s_out

  def test_bm25_mode_loads_and_embeds_nothing(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """With the real FastEmbedEmbedder over a recorder, a bm25 query records no load, embed or query_embed."""
    obj_recorder = RecordingModel()
    monkeypatch.setattr(mini_rag.fastembed_embedder, "TextEmbedding", obj_recorder)
    s_config = self.write_config(tmp_path)
    assert Cli().run(["--config", s_config, "ingest"]) == 0
    assert obj_recorder.l_events == ["load BAAI/bge-small-en-v1.5", "embed"]
    obj_recorder.l_events.clear()
    capsys.readouterr()
    assert Cli().run(["--config", s_config, "query", "tokamak plasma", "--mode", "bm25"]) == 0
    assert self.hit_rows(capsys.readouterr().out)[0][0] == "tokamak#0"
    assert obj_recorder.l_events == []

  # gate and retrievers

  def test_answerer_without_dense_gate_needs_one_hit(self) -> None:
    """b_gate_on_dense=False counts any hit as found, whatever the threshold, and no hit as not found."""
    obj_hit = HybridHit(Chunk(s_doc_id="d", i_index=0, i_start=0, s_text="t"), None, None, 3.2)
    assert Answerer(1.0, b_gate_on_dense=False).answer("q", [obj_hit]).b_found is True
    assert Answerer(-1.0, b_gate_on_dense=False).answer("q", []).b_found is False
    assert Answerer(1.0).answer("q", [obj_hit]).b_found is False

  @pytest.mark.parametrize("obj_class", [DenseOnlyRetriever, Bm25OnlyRetriever])
  def test_single_retrievers_check_top_k_and_blank_queries(self, obj_class: type) -> None:
    """Both retrievers reject a top-k below 1 and return no hit for a blank query without searching."""
    obj_retriever = obj_class(None)
    with pytest.raises(ValueError, match="i_top_k must be at least 1"):
      obj_retriever.search("q", 0)
    assert obj_retriever.search("   ", 5) == []

  # calibrate

  def calibrate_args(self, obj_tmp: Path) -> list[str]:
    """Write the calibration corpus and questions into obj_tmp and return the calibrate arguments."""
    obj_builder = CalibrationSetBuilder()
    return [
      "calibrate", "--questions", obj_builder.write_questions(obj_tmp / "q.json"),
      "--corpus", obj_builder.write_corpus(obj_tmp / "corpus"), "--cache-dir", str(obj_tmp / "cache"),
    ]

  def test_calibrate_dense_mode_uses_the_dense_hits(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """calibrate --mode dense takes every best score from DenseMethod.hits and none from the hybrid hits."""
    d_calls = {"dense": 0, "hybrid": 0}
    obj_dense_hits = DenseMethod.hits
    obj_hybrid_hits = HybridMethod.hits

    def dense_hits(obj_self: DenseMethod, s_question: str, i_depth: int) -> list[HybridHit]:
      """Count the call and return the dense hits."""
      d_calls["dense"] += 1
      return obj_dense_hits(obj_self, s_question, i_depth)

    def hybrid_hits(obj_self: HybridMethod, s_question: str, i_depth: int) -> list[HybridHit]:
      """Count the call and return the hybrid hits."""
      d_calls["hybrid"] += 1
      return obj_hybrid_hits(obj_self, s_question, i_depth)

    monkeypatch.setattr(DenseMethod, "hits", dense_hits)
    monkeypatch.setattr(HybridMethod, "hits", hybrid_hits)
    obj_cli = Cli(obj_embedder=BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY))
    assert obj_cli.run([*self.calibrate_args(tmp_path), "--mode", "dense"]) == 0
    s_out = capsys.readouterr().out
    assert d_calls == {"dense": 4, "hybrid": 0}
    # an injected embedder is stored under DenseIndex's default model name
    assert "Threshold sweep for BAAI/bge-small-en-v1.5, dense mode, top-k 5" in s_out

  def test_calibrate_hybrid_mode_uses_the_hybrid_hits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default calibrate takes its best scores from the hybrid hits, as before."""
    l_calls: list[str] = []
    obj_hybrid_hits = HybridMethod.hits

    def hybrid_hits(obj_self: HybridMethod, s_question: str, i_depth: int) -> list[HybridHit]:
      """Record the question and return the hybrid hits."""
      l_calls.append(s_question)
      return obj_hybrid_hits(obj_self, s_question, i_depth)

    monkeypatch.setattr(HybridMethod, "hits", hybrid_hits)
    obj_cli = Cli(obj_embedder=BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY))
    assert obj_cli.run(self.calibrate_args(tmp_path)) == 0
    assert len(l_calls) == 4

  def test_calibrate_bm25_mode_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """calibrate in bm25 mode, from the option or from the config, prints the error to stderr and exits 2."""
    obj_cli = Cli(obj_embedder=BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY))
    l_args = self.calibrate_args(tmp_path)
    assert obj_cli.run([*l_args, "--mode", "bm25"]) == 2
    obj_captured = capsys.readouterr()
    assert (obj_captured.out, obj_captured.err) == ("", self.S_CALIBRATE_ERROR)
    obj_config = tmp_path / "bm25.toml"
    obj_config.write_text('[retrieval]\nmode = "bm25"\n', encoding="utf-8")
    assert obj_cli.run(["--config", str(obj_config), *l_args]) == 2
    assert capsys.readouterr().err == self.S_CALIBRATE_ERROR

  def test_evaluator_refuses_a_gate_mode_without_dense_scores(self, tmp_path: Path) -> None:
    """The Evaluator's gate mode is hybrid or dense; bm25 raises ValueError."""
    obj_store = EvalIndexStore(str(tmp_path), BagOfWordsEmbedder(self.L_VOCABULARY))
    assert Evaluator(obj_store).s_gate_mode == "hybrid"
    with pytest.raises(ValueError, match="s_gate_mode must be one of hybrid, dense"):
      Evaluator(obj_store, s_gate_mode="bm25")
