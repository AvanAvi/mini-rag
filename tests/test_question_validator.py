"""Tests for QuestionValidator."""

import dataclasses

import pytest

from mini_rag.document import Document
from mini_rag.question import Question
from mini_rag.question_validator import QuestionValidator


class TestQuestionValidator:
  """Starts from a passing hand-made set and breaks one rule per test."""

  I_DOCS = 14

  def docs(self) -> list[Document]:
    """Return 14 two-sentence documents with facts unique to each."""
    return [
      Document(
        s_doc_id=f"doc_{i_doc:02d}",
        s_text=(
          f"The alpha{i_doc:02d} widget weighs {100 + i_doc} grams.\n"
          f"The beta{i_doc:02d} gadget was invented by Person{i_doc:02d} long ago."
        ),
      )
      for i_doc in range(self.I_DOCS)
    ]

  def questions(self) -> list[Question]:
    """Return a passing set: 24 answerable (10 with zero overlap) and 6 unanswerable."""
    l_questions = []
    for i_doc in range(self.I_DOCS):
      l_questions.append(Question(
        s_id=f"a{i_doc:02d}", s_text=f"How heavy is the alpha{i_doc:02d} widget?", b_answerable=True,
        s_source_doc=f"doc_{i_doc:02d}", s_must_contain=f"weighs {100 + i_doc} grams",
      ))
    for i_doc in range(10):
      l_questions.append(Question(
        s_id=f"b{i_doc:02d}", s_text="Who came up with something?", b_answerable=True,
        s_source_doc=f"doc_{i_doc:02d}", s_must_contain=f"invented by Person{i_doc:02d}",
      ))
    for i_pos in range(6):
      l_questions.append(Question(
        s_id=f"u{i_pos:02d}", s_text="What colour is the zebra?", b_answerable=False,
        t_absent_terms=("zebra",), s_topic="on_topic" if i_pos < 3 else "off_topic",
      ))
    return l_questions

  def problems(self, l_questions: list[Question]) -> list[str]:
    """Validate the questions against the hand-made documents and return the problems."""
    return QuestionValidator(self.docs()).validate(l_questions).l_problems

  def with_first_changed(self, **d_changes) -> list[Question]:
    """Return the passing set with fields of its first question replaced."""
    l_questions = self.questions()
    l_questions[0] = dataclasses.replace(l_questions[0], **d_changes)
    return l_questions

  def assert_problem(self, l_problems: list[str], s_fragment: str) -> None:
    """Fail unless one of the problems contains the fragment."""
    assert any(s_fragment in s_problem for s_problem in l_problems), l_problems

  def test_passing_set_has_no_problems(self):
    """The hand-made set passes and reports its overlap and phrase-document figures."""
    obj_result = QuestionValidator(self.docs()).validate(self.questions())
    assert obj_result.l_problems == []
    # "heavy", "alpha00", "widget": two of the three are in the source sentence
    assert obj_result.d_overlap["a00"] == pytest.approx(2 / 3)
    assert obj_result.d_overlap["b00"] == 0.0
    assert obj_result.d_phrase_docs["a00"] == 1
    assert len(obj_result.d_overlap) == 24

  def test_overlap_is_zero_without_long_tokens(self):
    """A question with no token longer than 3 characters has overlap 0.0."""
    obj_result = QuestionValidator(self.docs()).validate(self.with_first_changed(s_text="Is it so?"))
    assert obj_result.d_overlap["a00"] == 0.0

  def test_overlap_uses_only_the_sentence_with_the_phrase(self):
    """Words from the document's other sentence do not raise the overlap."""
    l_questions = self.with_first_changed(s_text="Which gadget was invented long ago?")
    assert QuestionValidator(self.docs()).validate(l_questions).d_overlap["a00"] == 0.0

  def test_unknown_source_doc(self):
    """A source_doc that was not loaded is reported."""
    self.assert_problem(self.problems(self.with_first_changed(s_source_doc="doc_99")), "a00: source_doc 'doc_99'")

  def test_phrase_too_short(self):
    """A phrase under 6 characters is reported."""
    self.assert_problem(self.problems(self.with_first_changed(s_must_contain="grams")), "a00: must_contain is shorter")

  def test_phrase_too_long(self):
    """A phrase over 60 characters is reported."""
    l_questions = self.with_first_changed(s_must_contain="weighs 100 grams" + " and more" * 6)
    self.assert_problem(self.problems(l_questions), "a00: must_contain is longer")

  def test_phrase_with_line_break(self):
    """A phrase that contains a line break is reported."""
    l_questions = self.with_first_changed(s_must_contain="100 grams.\nThe beta00")
    self.assert_problem(self.problems(l_questions), "a00: must_contain contains a line break")

  def test_phrase_not_in_source_doc(self):
    """A phrase missing from its source document is reported."""
    l_questions = self.with_first_changed(s_must_contain="weighs 999 grams")
    self.assert_problem(self.problems(l_questions), "a00: must_contain not found in doc_00")

  def test_phrase_inside_question_text(self):
    """A phrase that the question itself states is reported."""
    l_questions = self.with_first_changed(s_text="Is it true that it WEIGHS  100 grams?")
    self.assert_problem(self.problems(l_questions), "a00: must_contain occurs inside the question text")

  def test_phrase_in_more_than_two_documents(self):
    """A phrase found in three or more documents is reported with the count."""
    obj_result = QuestionValidator(self.docs()).validate(self.with_first_changed(s_must_contain="widget weighs"))
    self.assert_problem(obj_result.l_problems, "a00: must_contain occurs in 14 documents")
    assert obj_result.d_phrase_docs["a00"] == 14

  def test_absent_term_present_in_corpus(self):
    """An absent term that occurs in a document is reported, ignoring case."""
    l_questions = self.questions()
    l_questions[-1] = dataclasses.replace(l_questions[-1], t_absent_terms=("zebra", "BETA05  gadget"))
    self.assert_problem(self.problems(l_questions), "u05: absent term 'BETA05  gadget' occurs in 1 documents")

  def test_too_few_answerable(self):
    """Fewer than 24 answerable questions is reported."""
    l_questions = [obj_question for obj_question in self.questions() if obj_question.s_id != "b09"]
    self.assert_problem(self.problems(l_questions), "set: 23 answerable questions")

  def test_too_many_unanswerable(self):
    """More than 7 unanswerable questions is reported."""
    l_questions = self.questions()
    for i_pos in range(2):
      l_questions.append(dataclasses.replace(l_questions[-1], s_id=f"extra{i_pos}"))
    self.assert_problem(self.problems(l_questions), "set: 8 unanswerable questions")

  def test_too_few_of_one_topic(self):
    """Fewer than 2 unanswerable questions of a topic is reported."""
    l_questions = [obj_question for obj_question in self.questions() if obj_question.s_id not in ("u04", "u05")]
    self.assert_problem(self.problems(l_questions), "set: 1 off_topic questions")

  def test_too_few_source_documents(self):
    """Fewer than 14 distinct source documents is reported."""
    l_questions = self.questions()
    l_questions[13] = dataclasses.replace(l_questions[13], s_source_doc="doc_12")
    self.assert_problem(self.problems(l_questions), "set: 13 distinct source documents")

  def test_too_many_questions_on_one_document(self):
    """More than 3 questions on one document is reported."""
    l_questions = self.questions()
    for i_pos in range(2):
      l_questions.append(dataclasses.replace(l_questions[0], s_id=f"extra{i_pos}"))
    self.assert_problem(self.problems(l_questions), "set: 4 questions on doc_00")

  def test_too_few_low_overlap_questions(self):
    """Fewer than 8 questions with overlap <= 0.40 is reported."""
    l_questions = [
      dataclasses.replace(obj_question, s_text="Which gadget was invented long ago?")
      if obj_question.s_id in ("b00", "b01", "b02") else obj_question
      for obj_question in self.questions()
    ]
    self.assert_problem(self.problems(l_questions), "set: 7 questions with overlap <= 0.40")

  def test_too_few_digit_phrases(self):
    """Fewer than 4 phrases with a digit is reported."""
    l_questions = [
      dataclasses.replace(obj_question, s_must_contain="gadget was invented")
      if obj_question.b_answerable and obj_question.s_id not in ("a00", "a01", "a02") else obj_question
      for obj_question in self.questions()
    ]
    self.assert_problem(self.problems(l_questions), "set: 3 phrases contain a digit")
