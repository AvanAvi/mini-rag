"""Checks a question set against the corpus before it is used for evaluation."""

import re

from mini_rag.document import Document
from mini_rag.question import Question
from mini_rag.relevance_judge import RelevanceJudge
from mini_rag.tokenizer import Tokenizer
from mini_rag.validation_result import ValidationResult


class QuestionValidator:
  """Reports per-question and set-level problems; an empty problem list means the set is usable."""

  I_PHRASE_MIN = 6
  I_PHRASE_MAX = 60
  I_MAX_PHRASE_DOCS = 2
  T_ANSWERABLE_RANGE = (24, 28)
  T_UNANSWERABLE_RANGE = (5, 7)
  T_TOPICS = ("on_topic", "off_topic")
  I_MIN_PER_TOPIC = 2
  I_MIN_SOURCE_DOCS = 14
  I_MAX_PER_DOC = 3
  I_MIN_LOW_OVERLAP = 8
  F_LOW_OVERLAP = 0.40
  I_MIN_DIGIT_PHRASES = 4
  I_MIN_TOKEN_LENGTH = 4  # overlap only counts tokens longer than 3 characters

  def __init__(self, l_docs: list[Document]) -> None:
    """Keep the raw and the normalised text of every document, keyed by document id."""
    self.obj_judge = RelevanceJudge()
    self.obj_tokenizer = Tokenizer()
    # a sentence ends at a full stop followed by whitespace (or the end), or at a line break
    self.obj_boundary = re.compile(r"\.(?=\s|$)|\n")
    self.d_texts = {obj_doc.s_doc_id: obj_doc.s_text for obj_doc in l_docs}
    self.d_normalised = {
      obj_doc.s_doc_id: self.obj_judge.normalise(obj_doc.s_text) for obj_doc in l_docs
    }

  def validate(self, l_questions: list[Question]) -> ValidationResult:
    """Run every check and return the problems with the overlap and phrase-document figures."""
    l_problems: list[str] = []
    d_overlap: dict[str, float] = {}
    d_phrase_docs: dict[str, int] = {}
    for obj_question in l_questions:
      if not obj_question.b_answerable:
        l_problems.extend(self.unanswerable_problems(obj_question))
        continue
      i_docs = self.count_docs(obj_question.s_must_contain)
      d_phrase_docs[obj_question.s_id] = i_docs
      l_problems.extend(self.answerable_problems(obj_question, i_docs))
      f_overlap = self.overlap(obj_question)
      if f_overlap is not None:
        d_overlap[obj_question.s_id] = f_overlap
    l_problems.extend(self.set_problems(l_questions, d_overlap))
    return ValidationResult(l_problems=l_problems, d_overlap=d_overlap, d_phrase_docs=d_phrase_docs)

  def answerable_problems(self, obj_question: Question, i_docs: int) -> list[str]:
    """Return the problems of one answerable question."""
    s_id = obj_question.s_id
    s_raw = obj_question.s_must_contain
    s_phrase = self.obj_judge.normalise(s_raw)
    l_problems = []
    if obj_question.s_source_doc not in self.d_texts:
      l_problems.append(f"{s_id}: source_doc {obj_question.s_source_doc!r} is not a loaded document")
    elif s_phrase not in self.d_normalised[obj_question.s_source_doc]:
      l_problems.append(f"{s_id}: must_contain not found in {obj_question.s_source_doc}")
    if len(s_raw) < self.I_PHRASE_MIN:
      l_problems.append(f"{s_id}: must_contain is shorter than {self.I_PHRASE_MIN} characters")
    if len(s_raw) > self.I_PHRASE_MAX:
      l_problems.append(f"{s_id}: must_contain is longer than {self.I_PHRASE_MAX} characters")
    if "\n" in s_raw or "\r" in s_raw:
      l_problems.append(f"{s_id}: must_contain contains a line break")
    if s_phrase and s_phrase in self.obj_judge.normalise(obj_question.s_text):
      l_problems.append(f"{s_id}: must_contain occurs inside the question text")
    if i_docs > self.I_MAX_PHRASE_DOCS:
      l_problems.append(f"{s_id}: must_contain occurs in {i_docs} documents (at most {self.I_MAX_PHRASE_DOCS})")
    return l_problems

  def unanswerable_problems(self, obj_question: Question) -> list[str]:
    """Return one problem per absent term that does occur somewhere in the corpus."""
    l_problems = []
    for s_term in obj_question.t_absent_terms:
      i_docs = self.count_docs(s_term)
      if i_docs > 0:
        l_problems.append(f"{obj_question.s_id}: absent term {s_term!r} occurs in {i_docs} documents")
    return l_problems

  def set_problems(self, l_questions: list[Question], d_overlap: dict[str, float]) -> list[str]:
    """Return the problems of the set as a whole: counts, spread, overlap and digit phrases."""
    l_answerable = [obj_question for obj_question in l_questions if obj_question.b_answerable]
    l_unanswerable = [obj_question for obj_question in l_questions if not obj_question.b_answerable]
    l_problems = []

    i_low, i_high = self.T_ANSWERABLE_RANGE
    if not i_low <= len(l_answerable) <= i_high:
      l_problems.append(f"set: {len(l_answerable)} answerable questions (need {i_low} to {i_high})")
    i_low, i_high = self.T_UNANSWERABLE_RANGE
    if not i_low <= len(l_unanswerable) <= i_high:
      l_problems.append(f"set: {len(l_unanswerable)} unanswerable questions (need {i_low} to {i_high})")
    for s_topic in self.T_TOPICS:
      i_count = sum(1 for obj_question in l_unanswerable if obj_question.s_topic == s_topic)
      if i_count < self.I_MIN_PER_TOPIC:
        l_problems.append(f"set: {i_count} {s_topic} questions (need at least {self.I_MIN_PER_TOPIC})")

    d_per_doc: dict[str, int] = {}
    for obj_question in l_answerable:
      d_per_doc[obj_question.s_source_doc] = d_per_doc.get(obj_question.s_source_doc, 0) + 1
    if len(d_per_doc) < self.I_MIN_SOURCE_DOCS:
      l_problems.append(f"set: {len(d_per_doc)} distinct source documents (need at least {self.I_MIN_SOURCE_DOCS})")
    for s_doc_id, i_count in d_per_doc.items():
      if i_count > self.I_MAX_PER_DOC:
        l_problems.append(f"set: {i_count} questions on {s_doc_id} (at most {self.I_MAX_PER_DOC})")

    i_low_overlap = sum(1 for f_overlap in d_overlap.values() if f_overlap <= self.F_LOW_OVERLAP)
    if i_low_overlap < self.I_MIN_LOW_OVERLAP:
      l_problems.append(
        f"set: {i_low_overlap} questions with overlap <= {self.F_LOW_OVERLAP:.2f} (need at least {self.I_MIN_LOW_OVERLAP})"
      )
    i_digit = sum(1 for obj_question in l_answerable if re.search(r"\d", obj_question.s_must_contain))
    if i_digit < self.I_MIN_DIGIT_PHRASES:
      l_problems.append(f"set: {i_digit} phrases contain a digit (need at least {self.I_MIN_DIGIT_PHRASES})")
    return l_problems

  def count_docs(self, s_phrase: str) -> int:
    """Return the number of documents whose normalised text contains the normalised phrase."""
    s_normalised = self.obj_judge.normalise(s_phrase)
    if not s_normalised:
      return 0
    return sum(1 for s_text in self.d_normalised.values() if s_normalised in s_text)

  def overlap(self, obj_question: Question) -> float | None:
    """Return the share of the question's longer tokens found in the source sentence, or None if there is no such sentence."""
    s_sentence = self.source_sentence(obj_question)
    if s_sentence is None:
      return None
    l_tokens = sorted({
      s_token for s_token in self.obj_tokenizer.tokenize(obj_question.s_text)
      if len(s_token) >= self.I_MIN_TOKEN_LENGTH
    })
    if not l_tokens:
      return 0.0
    l_sentence_tokens = self.obj_tokenizer.tokenize(s_sentence)
    return sum(1 for s_token in l_tokens if s_token in l_sentence_tokens) / len(l_tokens)

  def source_sentence(self, obj_question: Question) -> str | None:
    """Return the sentence around the first occurrence of the phrase in the source document, or None."""
    s_text = self.d_texts.get(obj_question.s_source_doc)
    s_phrase = self.obj_judge.normalise(obj_question.s_must_contain)
    if s_text is None or not s_phrase:
      return None
    # the phrase may be spaced differently in the raw text, so match any whitespace run
    s_pattern = r"\s+".join(re.escape(s_word) for s_word in s_phrase.split(" "))
    obj_match = re.search(s_pattern, s_text, re.IGNORECASE)
    if obj_match is None:
      return None
    i_start = 0
    i_end = len(s_text)
    for obj_boundary in self.obj_boundary.finditer(s_text):
      if obj_boundary.end() <= obj_match.start():
        i_start = obj_boundary.end()
      elif obj_boundary.start() >= obj_match.end():
        i_end = obj_boundary.start()
        break
    return s_text[i_start:i_end]
