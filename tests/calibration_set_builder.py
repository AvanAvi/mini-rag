"""Test helper that builds a five-document corpus and four questions with known best dense scores."""

import json
from pathlib import Path

from mini_rag.document import Document
from mini_rag.question import Question


class CalibrationSetBuilder:
  """With a BagOfWordsEmbedder over L_VOCABULARY the best dense scores are known by hand.

  q1 "apple banana" against "apple banana cherry": 2 / (sqrt(2) * sqrt(3)) = 0.816 (answerable).
  q2 "papaya" against "orange papaya quince": 1 / sqrt(3) = 0.577 (answerable).
  u1 "kiwi" against "grape honeydew kiwi": 0.577 (unanswerable).
  u2 "zzz" has no vocabulary word, so every score is 0.0 (unanswerable).
  """

  L_VOCABULARY = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince",
  ]
  D_TEXTS = {
    "d1": "apple banana cherry",
    "d2": "date elderberry fig",
    "d3": "grape honeydew kiwi",
    "d4": "lemon mango nectarine",
    "d5": "orange papaya quince",
  }
  L_ENTRIES = [
    {"id": "q1", "question": "apple banana", "answerable": True, "source_doc": "d1", "must_contain": "cherry"},
    {"id": "q2", "question": "papaya", "answerable": True, "source_doc": "d5", "must_contain": "quince"},
    {"id": "u1", "question": "kiwi", "answerable": False, "topic": "on_topic", "absent_terms": ["zebra"]},
    {"id": "u2", "question": "zzz", "answerable": False, "topic": "off_topic", "absent_terms": ["yak"]},
  ]

  def documents(self) -> list[Document]:
    """Return the corpus as Document objects."""
    return [Document(s_doc_id=s_doc_id, s_text=s_text) for s_doc_id, s_text in self.D_TEXTS.items()]

  def questions(self) -> list[Question]:
    """Return the four questions as Question objects."""
    l_questions = []
    for d_entry in self.L_ENTRIES:
      if d_entry["answerable"]:
        l_questions.append(Question(
          s_id=d_entry["id"], s_text=d_entry["question"], b_answerable=True,
          s_source_doc=d_entry["source_doc"], s_must_contain=d_entry["must_contain"],
        ))
      else:
        l_questions.append(Question(
          s_id=d_entry["id"], s_text=d_entry["question"], b_answerable=False,
          t_absent_terms=tuple(d_entry["absent_terms"]), s_topic=d_entry["topic"],
        ))
    return l_questions

  def write_corpus(self, obj_dir: Path, s_suffix: str = ".txt") -> str:
    """Write one file per document into obj_dir and return the folder as a string."""
    obj_dir.mkdir()
    for s_doc_id, s_text in self.D_TEXTS.items():
      (obj_dir / f"{s_doc_id}{s_suffix}").write_text(s_text, encoding="utf-8")
    return str(obj_dir)

  def write_questions(self, obj_path: Path) -> str:
    """Write the questions file and return its path as a string."""
    obj_path.write_text(json.dumps({"questions": self.L_ENTRIES}), encoding="utf-8")
    return str(obj_path)
