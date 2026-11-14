"""Tests that query compares the saved model name before any fastembed model is loaded."""

from pathlib import Path

import pytest
from recording_model import RecordingModel

import mini_rag.fastembed_embedder
from mini_rag.cli import Cli
from mini_rag.fastembed_embedder import FastEmbedEmbedder


class TestModelCheckOrder:
  """Replaces fastembed's TextEmbedding with a recorder and runs ingest and query through Cli.run."""

  S_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
  S_OTHER_MODEL = "minishlab/potion-base-8M"

  @pytest.fixture
  def obj_recorder(self, monkeypatch: pytest.MonkeyPatch) -> RecordingModel:
    """Patch the TextEmbedding name used by FastEmbedEmbedder with a fresh recorder."""
    obj_recording = RecordingModel()
    monkeypatch.setattr(mini_rag.fastembed_embedder, "TextEmbedding", obj_recording)
    return obj_recording

  def write_config(self, obj_tmp: Path, s_file: str, s_model: str) -> str:
    """Write a config naming s_model, a one-document corpus and a shared index folder in obj_tmp."""
    obj_corpus = obj_tmp / "corpus"
    obj_corpus.mkdir(exist_ok=True)
    (obj_corpus / "cats.txt").write_text("Cats sleep most of the day.", encoding="utf-8")
    obj_path = obj_tmp / s_file
    obj_path.write_text(
      f"[corpus]\ndir = '{obj_corpus.as_posix()}'\n[index]\ndir = '{(obj_tmp / 'index').as_posix()}'\n"
      f"[embedder]\nmodel = '{s_model}'\n",
      encoding="utf-8",
    )
    return str(obj_path)

  def test_reading_name_loads_nothing(self, obj_recorder: RecordingModel) -> None:
    """Building the wrapper and reading s_name records no event."""
    obj_embedder = FastEmbedEmbedder(self.S_DEFAULT_MODEL)
    assert obj_embedder.s_name == self.S_DEFAULT_MODEL
    assert obj_recorder.l_events == []

  def test_mismatched_index_fails_before_load(
    self, obj_recorder: RecordingModel, tmp_path: Path, capsys: pytest.CaptureFixture[str],
  ) -> None:
    """An index from another model is refused with the usual message while no model is loaded or used."""
    s_other = self.write_config(tmp_path, "other.toml", self.S_OTHER_MODEL)
    s_default = self.write_config(tmp_path, "default.toml", self.S_DEFAULT_MODEL)
    assert Cli().run(["ingest", "--config", s_other]) == 0
    assert obj_recorder.l_events == [f"load {self.S_OTHER_MODEL}", "embed"]
    obj_recorder.l_events.clear()
    capsys.readouterr()

    assert Cli().run(["query", "Do cats sleep?", "--config", s_default]) == 2
    s_err = capsys.readouterr().err
    assert f"built with embedder {self.S_OTHER_MODEL!r}" in s_err
    assert f"the configured embedder is {self.S_DEFAULT_MODEL!r}; run ingest again" in s_err
    assert obj_recorder.l_events == []

  def test_matching_index_loads_on_query(
    self, obj_recorder: RecordingModel, tmp_path: Path, capsys: pytest.CaptureFixture[str],
  ) -> None:
    """A matching index is accepted, and only then is the model loaded and the query embedded."""
    s_other = self.write_config(tmp_path, "other.toml", self.S_OTHER_MODEL)
    assert Cli().run(["ingest", "--config", s_other]) == 0
    obj_recorder.l_events.clear()

    assert Cli().run(["query", "Do cats sleep?", "--config", s_other]) == 0
    assert obj_recorder.l_events == [f"load {self.S_OTHER_MODEL}", "query_embed"]
    assert "Answer:" in capsys.readouterr().out
