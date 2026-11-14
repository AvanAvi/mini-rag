"""Tests that a failing generator gives a note with the error and the extractive answer."""

import urllib.error
from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from failing_generator import FailingGenerator
from fake_opener import FakeOpener

from mini_rag.cli import Cli
from mini_rag.openai_compatible_client import OpenAiCompatibleClient


class TestGeneratorFallback:
  """Runs query --generate with clients that fail and checks the printed note."""

  L_VOCABULARY = ["tokamak", "plasma", "magnetic", "dogs", "bark"]
  S_KEY_VARIABLE = "MINI_RAG_TEST_KEY"
  S_KEY_VALUE = "dummy-value-for-test"

  def ingest(self, obj_tmp: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Build an index of two documents in obj_tmp and return the query arguments that use it."""
    obj_corpus = obj_tmp / "corpus"
    obj_corpus.mkdir()
    (obj_corpus / "tokamak.txt").write_text("A tokamak confines plasma with magnetic fields.", encoding="utf-8")
    (obj_corpus / "dogs.txt").write_text("Dogs bark loudly at night.", encoding="utf-8")
    s_index = str(obj_tmp / "index")
    l_args = ["ingest", "--corpus", str(obj_corpus), "--index-dir", s_index]
    assert Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY)).run(l_args) == 0
    capsys.readouterr()
    return ["query", "tokamak plasma", "--index-dir", s_index, "--threshold", "0.1", "--generate"]

  def run_query(self, l_args: list[str], obj_client: object, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the query with obj_client as the generator and return the printed output."""
    assert Cli(obj_embedder=BagOfWordsEmbedder(self.L_VOCABULARY), obj_llm_client=obj_client).run(l_args) == 0
    return capsys.readouterr().out

  def test_api_style_error_gives_a_note_and_the_extractive_answer(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An exception that is not a RuntimeError, as an API client raises, no longer stops the query."""
    l_args = self.ingest(tmp_path, capsys)
    s_out = self.run_query(l_args, FailingGenerator(ConnectionError("server said 529 overloaded")), capsys)
    assert "note: the generator failed (ConnectionError: server said 529 overloaded); showing the extractive answer" in s_out
    assert s_out.split("Answer:\n", 1)[1].startswith("[tokamak#0]\n")
    assert "found: yes, written by a model: no" in s_out

  def test_openai_compatible_failure_note_has_the_error_but_not_the_key(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """A connection failure of the chat-completions client prints its message and never the key value."""
    monkeypatch.setenv(self.S_KEY_VARIABLE, self.S_KEY_VALUE)
    l_args = self.ingest(tmp_path, capsys)
    obj_client = OpenAiCompatibleClient(
      "http://localhost:8080/v1", "local-model", self.S_KEY_VARIABLE,
      obj_opener=FakeOpener(obj_error=urllib.error.URLError("connection refused")),
    )
    s_out = self.run_query(l_args, obj_client, capsys)
    assert "note: the generator failed (RuntimeError: the request to http://localhost:8080/v1 failed" in s_out
    assert "connection refused" in s_out
    assert self.S_KEY_VALUE not in s_out
    assert "found: yes, written by a model: no" in s_out
