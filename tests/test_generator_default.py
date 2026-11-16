"""Tests for the shipped generator default "none" and the required [generator] model."""

from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder

from mini_rag.cli import Cli
from mini_rag.config import RagConfig
from mini_rag.config_loader import ConfigLoader


class TestGeneratorDefault:
  """Checks the default kind, the model check per kind and query output with and without --generate."""

  L_VOCABULARY = ["cats", "dogs", "bark", "sleep", "tokamak", "plasma", "magnetic"]
  D_DOCUMENTS = {
    "cats.txt": "Cats sleep most of the day.",
    "dogs.txt": "Dogs bark loudly at night.",
    "tokamak.txt": "A tokamak confines plasma with magnetic fields.",
  }
  S_NOTE = 'note: no generator is configured ([generator] kind is "none"); showing the extractive answer'
  S_QUESTION = "tokamak plasma"

  def cli(self) -> Cli:
    """Return a Cli whose embedder counts vocabulary words."""
    return Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY))

  def write_config(self, obj_tmp: Path, s_extra: str = "") -> str:
    """Write a corpus and a config without a [generator] section, so the generator is the default."""
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

  def ingest(self, obj_tmp: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Write the config, build the index and return the config path."""
    s_config = self.write_config(obj_tmp)
    assert self.cli().run(["--config", s_config, "ingest"]) == 0
    capsys.readouterr()
    return s_config

  def query(self, s_config: str, l_extra: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Run a query with the extra options, check the exit code and return the printed output."""
    assert self.cli().run(["--config", s_config, "query", self.S_QUESTION, *l_extra]) == 0
    return capsys.readouterr().out

  def test_default_kind_is_none(self) -> None:
    """The built-in config and the shipped mini-rag.toml both have kind "none" and an empty model."""
    obj_built_in = RagConfig()
    assert (obj_built_in.s_generator_kind, obj_built_in.s_generator_model) == ("none", "")
    obj_shipped = ConfigLoader().load("mini-rag.toml")
    assert (obj_shipped.s_generator_kind, obj_shipped.s_generator_model) == ("none", "")

  @pytest.mark.parametrize("s_kind", ["anthropic", "openai_compatible"])
  def test_kind_that_sends_requests_needs_a_model(self, s_kind: str) -> None:
    """Kinds "anthropic" and "openai_compatible" without a model raise ValueError naming [generator] model."""
    with pytest.raises(ValueError, match=r"\[generator\] model"):
      ConfigLoader().from_dict({"generator": {"kind": s_kind}})
    with pytest.raises(ValueError, match=r"\[generator\] model"):
      ConfigLoader().from_dict({"generator": {"kind": s_kind, "model": ""}})

  @pytest.mark.parametrize("s_kind", ["none", "custom"])
  def test_kind_none_and_custom_need_no_model(self, s_kind: str) -> None:
    """Kinds "none" and "custom" load without a model."""
    assert ConfigLoader().from_dict({"generator": {"kind": s_kind}}).s_generator_kind == s_kind

  def test_kind_with_model_loads(self) -> None:
    """Kind "anthropic" with a model name loads and keeps the name."""
    obj_config = ConfigLoader().from_dict({"generator": {"kind": "anthropic", "model": "test-model"}})
    assert obj_config.s_generator_model == "test-model"

  def test_generate_with_default_prints_note_and_extractive_answer(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
  ) -> None:
    """query --generate with the default generator prints the note once and the extractive answer."""
    s_config = self.ingest(tmp_path, capsys)
    s_out = self.query(s_config, ["--threshold", "0.1", "--generate"], capsys)
    assert s_out.count(self.S_NOTE) == 1
    assert s_out.split("Answer:\n", 1)[1].startswith("[tokamak#0]\n")
    assert "found: yes, written by a model: no" in s_out

  @pytest.mark.parametrize("s_threshold, s_found", [("0.1", "yes"), ("0.9", "no")])
  def test_results_are_the_same_with_and_without_generate(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], s_threshold: str, s_found: str,
  ) -> None:
    """A found and a not-found query print the same hits and answer with and without --generate."""
    s_config = self.ingest(tmp_path, capsys)
    s_without = self.query(s_config, ["--threshold", s_threshold], capsys)
    s_with = self.query(s_config, ["--threshold", s_threshold, "--generate"], capsys)
    assert self.S_NOTE not in s_without
    assert s_with.replace(self.S_NOTE + "\n", "") == s_without
    assert f"found: {s_found}, written by a model: no" in s_without

  def test_anthropic_without_key_still_names_the_variable(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Kind "anthropic" with a model and no key prints the key note and the extractive answer."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s_config = self.write_config(tmp_path, '[generator]\nkind = "anthropic"\nmodel = "test-model"\n')
    assert self.cli().run(["--config", s_config, "ingest"]) == 0
    capsys.readouterr()
    s_out = self.query(s_config, ["--threshold", "0.1", "--generate"], capsys)
    assert "note: ANTHROPIC_API_KEY is not set; showing the extractive answer" in s_out
    assert "found: yes, written by a model: no" in s_out
