"""Test helper that writes a small corpus and a question set the validator accepts."""

import json
from pathlib import Path


class EvalSetBuilder:
  """Builds 14 two-sentence documents and 24 answerable plus 6 unanswerable questions."""

  I_DOCS = 14
  L_VOCABULARY = ["widget", "weighs", "grams", "gadget", "invented", "heavy", "came"]

  def documents(self) -> dict[str, str]:
    """Return the corpus as file name -> text; every document holds two facts of its own."""
    return {
      f"doc_{i_doc:02d}.txt": (
        f"The alpha{i_doc:02d} widget weighs {100 + i_doc} grams.\n"
        f"The beta{i_doc:02d} gadget was invented by Person{i_doc:02d} long ago."
      )
      for i_doc in range(self.I_DOCS)
    }

  def entries(self) -> list[dict]:
    """Return the question entries in the questions-file format."""
    l_entries = []
    for i_doc in range(self.I_DOCS):
      l_entries.append(self.answerable(
        f"a{i_doc:02d}", f"How heavy is the alpha{i_doc:02d} widget?", i_doc, f"weighs {100 + i_doc} grams",
      ))
    for i_doc in range(10):
      l_entries.append(self.answerable(
        f"b{i_doc:02d}", "Who came up with something?", i_doc, f"invented by Person{i_doc:02d}",
      ))
    for i_pos in range(6):
      l_entries.append({
        "id": f"u{i_pos:02d}", "question": "What colour is the zebra?", "answerable": False,
        "source_doc": "", "must_contain": "", "topic": "on_topic" if i_pos < 3 else "off_topic",
        "absent_terms": ["zebra"],
      })
    return l_entries

  def answerable(self, s_id: str, s_text: str, i_doc: int, s_phrase: str) -> dict:
    """Return one answerable entry about document i_doc."""
    return {
      "id": s_id, "question": s_text, "answerable": True, "source_doc": f"doc_{i_doc:02d}",
      "must_contain": s_phrase, "topic": "", "absent_terms": [],
    }

  def write_corpus(self, obj_dir: Path) -> str:
    """Write the documents into obj_dir and return the folder as a string."""
    obj_dir.mkdir()
    for s_name, s_text in self.documents().items():
      (obj_dir / s_name).write_text(s_text, encoding="utf-8")
    return str(obj_dir)

  def write_questions(self, obj_path: Path, l_entries: list[dict]) -> str:
    """Write the entries as a questions file and return its path as a string."""
    obj_path.write_text(json.dumps({"questions": l_entries}), encoding="utf-8")
    return str(obj_path)
