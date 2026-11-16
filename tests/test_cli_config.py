"""Tests for the --config option and the precedence of CLI option, config file and default."""

from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder

from mini_rag.cli import Cli
from mini_rag.config import RagConfig
from mini_rag.config_loader import ConfigLoader
from mini_rag.fusion_setting import FusionSetting


class TestCliConfig:
  """Checks each precedence level on the parser and once end to end on a tiny corpus."""

  L_VOCABULARY = ["cats", "dogs", "bark", "sleep", "tokamak", "plasma", "magnetic"]
  D_DOCUMENTS = {
    "cats.txt": "Cats sleep most of the day.",
    "dogs.txt": "Dogs bark loudly at night.",
    "tokamak.txt": "A tokamak confines plasma with magnetic fields.",
  }

  def cli(self) -> Cli:
    """Return a Cli whose embedder counts vocabulary words."""
    return Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY))

  def write_config(self, obj_tmp: Path, s_extra: str = "") -> str:
    """Write a corpus and a config that points at it and at an index folder inside obj_tmp."""
    obj_corpus = obj_tmp / "corpus"
    obj_corpus.mkdir()
    for s_name, s_text in self.D_DOCUMENTS.items():
      (obj_corpus / s_name).write_text(s_text, encoding="utf-8")
    obj_path = obj_tmp / "my.toml"
    obj_path.write_text(
      f"[corpus]\ndir = '{obj_corpus}'\n[index]\ndir = '{obj_tmp / 'index'}'\n"
      f"[chunking]\nsize = 200\noverlap = 20\n{s_extra}",
      encoding="utf-8",
    )
    return str(obj_path)

  def hit_count(self, s_out: str) -> int:
    """Count the printed hit lines, which start with a rank and a marker."""
    return len([s_line for s_line in s_out.split("\n") if s_line[:1].isdigit() and ". [" in s_line])

  def test_built_in_defaults_without_a_config(self) -> None:
    """A parser built without a config carries today's option defaults."""
    obj_parser = Cli().build_parser()
    obj_ingest = obj_parser.parse_args(["ingest"])
    assert (obj_ingest.s_corpus, obj_ingest.s_index_dir, obj_ingest.i_size, obj_ingest.i_overlap) == (
      "data/corpus", ".cache/index", 500, 100,
    )
    obj_query = obj_parser.parse_args(["query", "q"])
    assert (obj_query.s_index_dir, obj_query.i_top_k, obj_query.f_threshold) == (".cache/index", 5, 0.55)
    obj_eval = obj_parser.parse_args(["eval"])
    assert (obj_eval.s_questions, obj_eval.s_corpus, obj_eval.l_sizes, obj_eval.i_overlap) == (
      "eval/questions.json", "data/corpus", [500, 1000], 100,
    )
    assert (obj_eval.i_gate_top_k, obj_eval.s_cache_dir, obj_eval.s_out) == (5, ".cache/eval", "eval/results.md")

  def test_config_values_become_the_option_defaults(self) -> None:
    """A value from the config file replaces the built-in default."""
    obj_config = ConfigLoader().from_dict({
      "corpus": {"dir": "docs"}, "index": {"dir": "idx"}, "chunking": {"size": 300, "overlap": 30},
      "retrieval": {"top_k": 3}, "gate": {"threshold": 0.3},
      "eval": {"questions": "q.json", "sizes": [200], "cache_dir": "ec"},
    })
    obj_parser = Cli().build_parser(obj_config)
    obj_ingest = obj_parser.parse_args(["ingest"])
    assert (obj_ingest.s_corpus, obj_ingest.s_index_dir, obj_ingest.i_size, obj_ingest.i_overlap) == ("docs", "idx", 300, 30)
    obj_query = obj_parser.parse_args(["query", "q"])
    assert (obj_query.s_index_dir, obj_query.i_top_k, obj_query.f_threshold) == ("idx", 3, 0.3)
    obj_eval = obj_parser.parse_args(["eval"])
    assert (obj_eval.s_questions, obj_eval.s_corpus, obj_eval.l_sizes, obj_eval.i_overlap) == ("q.json", "docs", [200], 30)
    assert (obj_eval.i_gate_top_k, obj_eval.s_cache_dir) == (3, "ec")

  def test_cli_option_overrides_the_config(self) -> None:
    """An option given on the command line wins over the config file."""
    obj_config = ConfigLoader().from_dict({"retrieval": {"top_k": 3}, "gate": {"threshold": 0.3}, "chunking": {"size": 300}})
    obj_parser = Cli().build_parser(obj_config)
    obj_query = obj_parser.parse_args(["query", "q", "--top-k", "9", "--threshold", "0.8"])
    assert (obj_query.i_top_k, obj_query.f_threshold) == (9, 0.8)
    assert obj_parser.parse_args(["ingest", "--size", "50"]).i_size == 50

  def test_fusion_setting_comes_from_the_config(self) -> None:
    """The [retrieval] fusion values reach the FusionSetting; the defaults give the default setting."""
    assert Cli().fusion_setting(RagConfig()) == FusionSetting.default()
    obj_config = ConfigLoader().from_dict({"retrieval": {"rrf_k": 10, "bm25_weight": 0.5}})
    assert Cli().fusion_setting(obj_config) == FusionSetting(i_k=10, f_bm25_weight=0.5, f_dense_weight=1.0)

  def test_end_to_end_precedence(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """ingest and query read the config; --top-k on the command line overrides it, in either position."""
    s_config = self.write_config(tmp_path, "[retrieval]\ntop_k = 2\n[gate]\nthreshold = 0.1\n")
    assert self.cli().run(["--config", s_config, "ingest"]) == 0
    s_out = capsys.readouterr().out
    assert "documents: 3" in s_out and "chunk size: 200" in s_out and f"index folder: {tmp_path / 'index'}" in s_out
    assert (tmp_path / "index" / "vectors.npy").is_file()

    assert self.cli().run(["--config", s_config, "query", "tokamak plasma"]) == 0
    s_out = capsys.readouterr().out
    assert self.hit_count(s_out) == 2
    assert "found: yes" in s_out

    # the command-line option beats the file; --config is also accepted after the command
    assert self.cli().run(["query", "tokamak plasma", "--config", s_config, "--top-k", "3", "--threshold", "1.0"]) == 0
    s_out = capsys.readouterr().out
    assert self.hit_count(s_out) == 3
    assert "found: no" in s_out

  def test_missing_explicit_config_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A --config path that does not exist is an error, not a silent fall back to the defaults."""
    assert self.cli().run(["--config", str(tmp_path / "absent.toml"), "ingest"]) == 2
    assert "config file not found" in capsys.readouterr().err

  def test_bad_config_value_returns_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A config with a wrong type prints an error line naming the key."""
    obj_path = tmp_path / "bad.toml"
    obj_path.write_text('[chunking]\nsize = "big"\n', encoding="utf-8")
    assert self.cli().run(["--config", str(obj_path), "ingest"]) == 2
    assert "chunking.size" in capsys.readouterr().err

  def test_generator_kind_none_prints_a_note(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With generator kind "none", --generate prints a note and shows the extractive answer."""
    s_config = self.write_config(tmp_path, '[gate]\nthreshold = 0.1\n[generator]\nkind = "none"\n')
    assert self.cli().run(["--config", s_config, "ingest"]) == 0
    capsys.readouterr()
    assert self.cli().run(["--config", s_config, "query", "tokamak plasma", "--generate"]) == 0
    s_out = capsys.readouterr().out
    assert 'note: no generator is configured ([generator] kind is "none"); showing the extractive answer' in s_out
    assert "found: yes, written by a model: no" in s_out
