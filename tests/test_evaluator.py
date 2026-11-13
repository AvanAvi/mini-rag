"""Tests for Evaluator and the three retrieval methods."""

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder

from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.evaluator import Evaluator
from mini_rag.question import Question
from mini_rag.question_outcome import QuestionOutcome


class TestEvaluator:
  """Runs the evaluator on a corpus small enough to know every rank by hand."""

  L_VOCABULARY = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince",
  ]
  F_ONE_OF_THREE = 0.57735  # cosine of a one-word query against a three-word chunk

  def docs(self) -> list[Document]:
    """Return five documents that each fit in one chunk and share no words."""
    return [
      Document(s_doc_id="d1", s_text="apple banana cherry"),
      Document(s_doc_id="d2", s_text="date elderberry fig"),
      Document(s_doc_id="d3", s_text="grape honeydew kiwi"),
      Document(s_doc_id="d4", s_text="lemon mango nectarine"),
      Document(s_doc_id="d5", s_text="orange papaya quince"),
    ]

  def questions(self) -> list[Question]:
    """Return one easy question, one whose words point at the wrong document, and one unanswerable."""
    return [
      Question(s_id="q1", s_text="where is the banana", b_answerable=True,
               s_source_doc="d1", s_must_contain="banana  CHERRY"),
      Question(s_id="q2", s_text="papaya", b_answerable=True,
               s_source_doc="d4", s_must_contain="lemon mango"),
      Question(s_id="q3", s_text="zzz", b_answerable=False,
               t_absent_terms=("zebra",), s_topic="off_topic"),
    ]

  def evaluator(self, tmp_path, l_sizes: list[int], i_overlap: int = 10) -> Evaluator:
    """Return an evaluator with a bag-of-words embedder and a cache in tmp_path."""
    obj_store = EvalIndexStore(str(tmp_path), BagOfWordsEmbedder(self.L_VOCABULARY))
    return Evaluator(obj_store, l_sizes=l_sizes, i_overlap=i_overlap)

  def test_end_to_end_ranks(self, tmp_path):
    """Every method's outcome matches the ranks worked out by hand."""
    obj_result = self.evaluator(tmp_path, [50, 100]).run(self.docs(), self.questions())
    assert [obj_size.i_size for obj_size in obj_result.l_size_results] == [50, 100]
    for obj_size in obj_result.l_size_results:
      assert obj_size.i_chunk_count == 5
      assert list(obj_size.d_outcomes) == ["BM25", "dense", "hybrid"]
      # q1: only d1 mentions banana, so every method puts it first
      obj_first = QuestionOutcome(s_id="q1", i_first_rank=1, i_total_relevant=1, d_found_at_k={1: 1, 3: 1, 5: 1})
      # q2: no query word is in d4. BM25 returns only d5; dense and hybrid list d5, then d1 to d4 in order
      obj_missed = QuestionOutcome(s_id="q2", i_first_rank=None, i_total_relevant=1, d_found_at_k={1: 0, 3: 0, 5: 0})
      obj_fifth = QuestionOutcome(s_id="q2", i_first_rank=5, i_total_relevant=1, d_found_at_k={1: 0, 3: 0, 5: 1})
      assert obj_size.d_outcomes["BM25"] == [obj_first, obj_missed]
      assert obj_size.d_outcomes["dense"] == [obj_first, obj_fifth]
      assert obj_size.d_outcomes["hybrid"] == [obj_first, obj_fifth]

  def test_best_dense_scores_and_sweep(self, tmp_path):
    """The best dense score covers all questions and feeds the sweep."""
    obj_result = self.evaluator(tmp_path, [50]).run(self.docs(), self.questions())
    obj_size = obj_result.l_size_results[0]
    assert obj_size.d_best_dense["q1"] == pytest.approx(self.F_ONE_OF_THREE, abs=1e-4)
    assert obj_size.d_best_dense["q2"] == pytest.approx(self.F_ONE_OF_THREE, abs=1e-4)
    assert obj_size.d_best_dense["q3"] == 0.0  # no vocabulary word, so every dense score is zero
    d_rejected = {obj_row.f_threshold: obj_row.i_answerable_rejected for obj_row in obj_size.l_sweep_rows}
    assert d_rejected[0.55] == 0
    assert d_rejected[0.60] == 2
    assert all(obj_row.i_answerable_total == 2 for obj_row in obj_size.l_sweep_rows)
    assert all(
      (obj_row.i_unanswerable_accepted, obj_row.i_unanswerable_total) == (0, 1)
      for obj_row in obj_size.l_sweep_rows
    )

  def test_result_carries_the_run_settings(self, tmp_path):
    """The result records the fingerprint, model name, overlap, depth and gate top-k."""
    obj_evaluator = self.evaluator(tmp_path, [50])
    obj_result = obj_evaluator.run(self.docs(), self.questions())
    assert obj_result.s_corpus_fingerprint == obj_evaluator.obj_store.fingerprint(self.docs())
    assert obj_result.s_model_name == "BAAI/bge-small-en-v1.5"
    assert (obj_result.i_overlap, obj_result.i_depth, obj_result.i_gate_top_k) == (10, 10, 5)

  def test_phrase_in_two_overlapping_chunks(self, tmp_path):
    """A phrase inside the overlap of two chunks gives two relevant chunks."""
    # chunks of 30 with overlap 15: [0, 30), [15, 45), [30, 49); "apple fig" starts at 17
    l_docs = [Document(s_doc_id="d1", s_text="kiwi lemon mango apple fig grape melon peach plum")]
    l_questions = [Question(s_id="q1", s_text="apple", b_answerable=True, s_source_doc="d1", s_must_contain="apple fig")]
    obj_result = self.evaluator(tmp_path, [30], i_overlap=15).run(l_docs, l_questions)
    obj_size = obj_result.l_size_results[0]
    assert obj_size.i_chunk_count == 3
    obj_outcome = obj_size.d_outcomes["dense"][0]
    assert obj_outcome.i_total_relevant == 2
    assert obj_outcome.d_found_at_k[3] == 2

  def test_prepare_builds_then_loads(self, tmp_path):
    """prepare returns one bundle per size, built the first time and loaded the second."""
    obj_evaluator = self.evaluator(tmp_path, [50, 100])
    assert [obj_bundle.b_loaded for obj_bundle in obj_evaluator.prepare(self.docs())] == [False, False]
    assert [obj_bundle.b_loaded for obj_bundle in obj_evaluator.prepare(self.docs())] == [True, True]
    assert [obj_bundle.b_loaded for obj_bundle in obj_evaluator.prepare(self.docs(), b_rebuild=True)] == [False, False]

  def test_rejects_bad_settings(self, tmp_path):
    """No sizes, a depth below 1 or a gate top-k below 1 raise ValueError."""
    obj_store = EvalIndexStore(str(tmp_path), BagOfWordsEmbedder(self.L_VOCABULARY))
    with pytest.raises(ValueError):
      Evaluator(obj_store, l_sizes=[])
    with pytest.raises(ValueError):
      Evaluator(obj_store, i_depth=0)
    with pytest.raises(ValueError):
      Evaluator(obj_store, i_gate_top_k=0)
