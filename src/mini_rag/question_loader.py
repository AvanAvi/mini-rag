"""Reads the evaluation question set from a JSON file."""

import json
from pathlib import Path

from mini_rag.question import Question


class QuestionLoader:
  """Parses a questions file into Question objects and rejects malformed entries."""

  T_TOPICS = ("on_topic", "off_topic")

  def load(self, s_path: str) -> list[Question]:
    """Return the questions in file order; raise on a missing file or a bad entry."""
    obj_path = Path(s_path)
    if not obj_path.is_file():
      raise FileNotFoundError(f"Questions file not found: {s_path}")
    d_payload = json.loads(obj_path.read_text(encoding="utf-8"))
    if not isinstance(d_payload, dict) or not isinstance(d_payload.get("questions"), list):
      raise ValueError(f'{s_path} must hold an object with a "questions" list')

    l_questions: list[Question] = []
    l_ids: list[str] = []
    for i_pos, d_entry in enumerate(d_payload["questions"]):
      obj_question = self.parse(d_entry, i_pos)
      if obj_question.s_id in l_ids:
        raise ValueError(f"Duplicate question id: {obj_question.s_id}")
      l_ids.append(obj_question.s_id)
      l_questions.append(obj_question)
    return l_questions

  def parse(self, d_entry: object, i_pos: int) -> Question:
    """Turn one JSON entry into a Question; raise ValueError naming its id."""
    if not isinstance(d_entry, dict) or not self.is_text(d_entry.get("id")):
      raise ValueError(f"Question entry {i_pos} has no usable id")
    s_id = d_entry["id"]
    if not self.is_text(d_entry.get("question")):
      raise ValueError(f"Question {s_id}: question must be a non-empty string")
    if not isinstance(d_entry.get("answerable"), bool):
      raise ValueError(f"Question {s_id}: answerable must be true or false")

    if d_entry["answerable"]:
      for s_key in ("source_doc", "must_contain"):
        if not self.is_text(d_entry.get(s_key)):
          raise ValueError(f"Question {s_id}: {s_key} must be a non-empty string")
      return Question(
        s_id=s_id, s_text=d_entry["question"], b_answerable=True,
        s_source_doc=d_entry["source_doc"], s_must_contain=d_entry["must_contain"],
      )

    if d_entry.get("topic") not in self.T_TOPICS:
      raise ValueError(f"Question {s_id}: topic must be one of {', '.join(self.T_TOPICS)}")
    l_terms = d_entry.get("absent_terms")
    if not isinstance(l_terms, list) or not l_terms or not all(self.is_text(s_term) for s_term in l_terms):
      raise ValueError(f"Question {s_id}: absent_terms must be a non-empty list of strings")
    return Question(
      s_id=s_id, s_text=d_entry["question"], b_answerable=False,
      t_absent_terms=tuple(l_terms), s_topic=d_entry["topic"],
    )

  def is_text(self, obj_value: object) -> bool:
    """Return True for a string with at least one non-blank character."""
    return isinstance(obj_value, str) and bool(obj_value.strip())
