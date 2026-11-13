"""Tests for Question and QuestionLoader."""

import dataclasses
import json

import pytest

from mini_rag.question import Question
from mini_rag.question_loader import QuestionLoader


class TestQuestionLoader:
  """Covers loading a valid file and every way an entry can be rejected."""

  D_ANSWERABLE = {
    "id": "q01", "question": "Which year was it?", "answerable": True,
    "source_doc": "doc_a", "must_contain": "in 1998", "topic": "", "absent_terms": [],
  }
  D_UNANSWERABLE = {
    "id": "q02", "question": "Who paid for it?", "answerable": False,
    "source_doc": "", "must_contain": "", "topic": "on_topic", "absent_terms": ["zebra"],
  }

  def write(self, tmp_path, l_entries: list[dict]) -> str:
    """Write the entries as a questions file and return its path."""
    obj_path = tmp_path / "questions.json"
    obj_path.write_text(json.dumps({"questions": l_entries}), encoding="utf-8")
    return str(obj_path)

  def changed(self, d_entry: dict, s_key: str, obj_value: object) -> dict:
    """Return a copy of the entry with one value replaced."""
    d_copy = dict(d_entry)
    d_copy[s_key] = obj_value
    return d_copy

  def test_loads_both_kinds_in_file_order(self, tmp_path):
    """A valid file yields Question objects with unused fields left empty."""
    l_questions = QuestionLoader().load(self.write(tmp_path, [self.D_ANSWERABLE, self.D_UNANSWERABLE]))
    assert l_questions == [
      Question(s_id="q01", s_text="Which year was it?", b_answerable=True,
               s_source_doc="doc_a", s_must_contain="in 1998", t_absent_terms=(), s_topic=""),
      Question(s_id="q02", s_text="Who paid for it?", b_answerable=False,
               s_source_doc="", s_must_contain="", t_absent_terms=("zebra",), s_topic="on_topic"),
    ]

  def test_question_is_frozen(self):
    """A Question cannot be changed after creation."""
    obj_question = Question(s_id="q01", s_text="What?", b_answerable=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
      obj_question.s_id = "q02"

  def test_missing_file_raises_file_not_found(self, tmp_path):
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      QuestionLoader().load(str(tmp_path / "absent.json"))

  def test_missing_questions_list_raises(self, tmp_path):
    """A file without a top-level questions list is rejected."""
    obj_path = tmp_path / "questions.json"
    obj_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
      QuestionLoader().load(str(obj_path))

  def test_duplicate_id_raises_and_names_the_id(self, tmp_path):
    """Two entries with the same id are rejected."""
    with pytest.raises(ValueError, match="q01"):
      QuestionLoader().load(self.write(tmp_path, [self.D_ANSWERABLE, self.D_ANSWERABLE]))

  @pytest.mark.parametrize("s_key", ["source_doc", "must_contain"])
  def test_answerable_needs_source_and_phrase(self, tmp_path, s_key):
    """An answerable entry with an empty source_doc or must_contain is rejected by id."""
    d_entry = self.changed(self.D_ANSWERABLE, s_key, "")
    with pytest.raises(ValueError, match=f"q01.*{s_key}"):
      QuestionLoader().load(self.write(tmp_path, [d_entry]))

  def test_unanswerable_needs_a_known_topic(self, tmp_path):
    """An unanswerable entry with an unknown topic is rejected by id."""
    d_entry = self.changed(self.D_UNANSWERABLE, "topic", "sideways")
    with pytest.raises(ValueError, match="q02.*topic"):
      QuestionLoader().load(self.write(tmp_path, [d_entry]))

  def test_unanswerable_needs_absent_terms(self, tmp_path):
    """An unanswerable entry with no absent terms is rejected by id."""
    d_entry = self.changed(self.D_UNANSWERABLE, "absent_terms", [])
    with pytest.raises(ValueError, match="q02.*absent_terms"):
      QuestionLoader().load(self.write(tmp_path, [d_entry]))

  def test_answerable_must_be_a_boolean(self, tmp_path):
    """An entry whose answerable flag is not true or false is rejected by id."""
    d_entry = self.changed(self.D_ANSWERABLE, "answerable", "yes")
    with pytest.raises(ValueError, match="q01.*answerable"):
      QuestionLoader().load(self.write(tmp_path, [d_entry]))

  def test_entry_without_id_raises(self, tmp_path):
    """An entry with no id is rejected by its position."""
    d_entry = self.changed(self.D_ANSWERABLE, "id", "")
    with pytest.raises(ValueError, match="entry 0"):
      QuestionLoader().load(self.write(tmp_path, [d_entry]))
