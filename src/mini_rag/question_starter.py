"""Writes a starter question file for a user's own evaluation set."""

import json
from pathlib import Path


class QuestionStarter:
  """Creates a questions file with one answerable and one unanswerable example entry."""

  S_DEFAULT_PATH = "eval/my_questions.json"
  # JSON has no comments, so each example explains itself in a "comment" field, which the loader ignores
  L_ENTRIES = [
    {
      "comment": (
        "Answerable example. source_doc is the corpus file name without its extension. "
        "must_contain is a phrase of 6 to 60 characters copied from that file, which a correct answer needs. "
        "Do not repeat the phrase in the question."
      ),
      "id": "q01",
      "question": "Replace this with a question your corpus answers.",
      "answerable": True,
      "source_doc": "your_document_name",
      "must_contain": "phrase copied from the document",
    },
    {
      "comment": (
        "Unanswerable example. topic is on_topic (about your subject, but the corpus has no answer) or off_topic. "
        "absent_terms are words a correct answer would contain; none of them may occur in the corpus."
      ),
      "id": "q02",
      "question": "Replace this with a question your corpus cannot answer.",
      "answerable": False,
      "topic": "off_topic",
      "absent_terms": ["word_not_in_corpus"],
    },
  ]

  def write(self, s_path: str, b_force: bool = False) -> None:
    """Write the starter file as UTF-8 with LF line endings; refuse to replace a file unless b_force."""
    obj_path = Path(s_path)
    if obj_path.exists() and not b_force:
      raise FileExistsError(f"{s_path} already exists; use --force to replace it")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    s_json = json.dumps({"questions": self.L_ENTRIES}, ensure_ascii=False, indent=2) + "\n"
    # bytes, so Windows does not turn LF into CRLF
    obj_path.write_bytes(s_json.encode("utf-8"))
