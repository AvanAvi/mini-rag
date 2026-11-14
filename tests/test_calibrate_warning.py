"""Tests for the calibrate warning when the embedder cannot separate the two question groups."""

import json
from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from calibration_set_builder import CalibrationSetBuilder

from mini_rag.cli import Cli


class TestCalibrateWarning:
  """Builds question files whose best dense scores are known by hand (see CalibrationSetBuilder)."""

  # "kiwi" scores 0.577 against d3, "apple banana" 0.816 against d1, "papaya" 0.577 against d5
  D_KIWI = {"id": "a1", "question": "kiwi", "answerable": True, "source_doc": "d3", "must_contain": "honeydew"}
  D_APPLE = {"id": "u1", "question": "apple banana", "answerable": False, "topic": "on_topic", "absent_terms": ["zebra"]}
  D_PAPAYA = {"id": "u2", "question": "papaya", "answerable": False, "topic": "off_topic", "absent_terms": ["yak"]}

  def calibrate(self, obj_tmp: Path, l_entries: list[dict], capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Run calibrate on the entries and return the printed lines."""
    obj_questions = obj_tmp / "q.json"
    obj_questions.write_text(json.dumps({"questions": l_entries}), encoding="utf-8")
    l_args = [
      "calibrate", "--questions", str(obj_questions),
      "--corpus", CalibrationSetBuilder().write_corpus(obj_tmp / "corpus"), "--cache-dir", str(obj_tmp / "cache"),
    ]
    assert Cli(obj_embedder=BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY)).run(l_args) == 0
    return capsys.readouterr().out.split("\n")

  def warning_lines(self, l_lines: list[str]) -> list[str]:
    """Return the separation warnings among the printed lines."""
    return [s_line for s_line in l_lines if "cannot separate the two groups" in s_line]

  def test_every_unanswerable_accepted(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """0.55 and 0.85 tie at 1 error; at the suggested 0.55 the one unanswerable question gets through."""
    l_lines = self.calibrate(tmp_path, [self.D_KIWI, self.D_APPLE], capsys)
    assert "threshold = 0.55" in l_lines
    assert self.warning_lines(l_lines) == [
      "warning: at 0.55 every unanswerable question is accepted; the embedder cannot separate the two groups "
      "on this question file, so do not use the suggested value"
    ]
    # the warning comes right after the table and its summary line
    assert l_lines.index(self.warning_lines(l_lines)[0]) > l_lines.index("| 0.85 | 1/1 | 0/1 | 1 |")

  def test_every_answerable_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Only 0.85 has the fewest errors, and it rejects the one answerable question."""
    l_lines = self.calibrate(tmp_path, [self.D_KIWI, self.D_APPLE, self.D_PAPAYA], capsys)
    assert "threshold = 0.85" in l_lines
    assert self.warning_lines(l_lines) == [
      "warning: at 0.85 every answerable question is rejected; the embedder cannot separate the two groups "
      "on this question file, so do not use the suggested value"
    ]

  def test_no_warning_when_the_groups_are_partly_separated(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The four-question set of CalibrationSetBuilder gets no separation warning."""
    l_lines = self.calibrate(tmp_path, CalibrationSetBuilder.L_ENTRIES, capsys)
    assert "threshold = 0.05" in l_lines
    assert self.warning_lines(l_lines) == []
