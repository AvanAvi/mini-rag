"""Measures retrieval quality and the answer gate against a question set."""

from mini_rag.bm25_method import Bm25Method
from mini_rag.dense_method import DenseMethod
from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.eval_result import EvalResult
from mini_rag.hybrid_method import HybridMethod
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.index_bundle import IndexBundle
from mini_rag.question import Question
from mini_rag.question_outcome import QuestionOutcome
from mini_rag.relevance_judge import RelevanceJudge
from mini_rag.retrieval_method import RetrievalMethod
from mini_rag.size_result import SizeResult
from mini_rag.threshold_sweeper import ThresholdSweeper


class Evaluator:
  """Compares BM25, dense and hybrid retrieval per chunk size and sweeps the gate threshold."""

  T_KS = (1, 3, 5)

  def __init__(
    self, obj_store: EvalIndexStore, l_sizes: list[int] | tuple[int, ...] = (500, 1000), i_overlap: int = 100,
    i_depth: int = 10, i_gate_top_k: int = 5, obj_sweeper: ThresholdSweeper | None = None,
  ) -> None:
    """Store the index store, the chunk sizes, the ranking depth and the gate's top-k."""
    if not l_sizes:
      raise ValueError("l_sizes must name at least one chunk size")
    if i_depth < 1:
      raise ValueError(f"i_depth must be at least 1, got {i_depth}")
    if i_gate_top_k < 1:
      raise ValueError(f"i_gate_top_k must be at least 1, got {i_gate_top_k}")
    self.obj_store = obj_store
    self.l_sizes = list(l_sizes)
    self.i_overlap = i_overlap
    self.i_depth = i_depth
    self.i_gate_top_k = i_gate_top_k
    self.obj_sweeper = obj_sweeper if obj_sweeper is not None else ThresholdSweeper()
    self.obj_judge = RelevanceJudge()

  def prepare(self, l_docs: list[Document], b_rebuild: bool = False) -> list[IndexBundle]:
    """Build or load the indexes for every chunk size and return them in size order."""
    return [
      self.obj_store.get_indexes(l_docs, i_size, self.i_overlap, b_rebuild)
      for i_size in self.l_sizes
    ]

  def run(self, l_docs: list[Document], l_questions: list[Question]) -> EvalResult:
    """Evaluate every chunk size on the questions and return all results."""
    return EvalResult(
      s_corpus_fingerprint=self.obj_store.fingerprint(l_docs),
      s_model_name=self.obj_store.s_model_name,
      i_overlap=self.i_overlap,
      i_depth=self.i_depth,
      i_gate_top_k=self.i_gate_top_k,
      l_size_results=[self.run_size(obj_bundle, l_questions) for obj_bundle in self.prepare(l_docs)],
    )

  def run_size(self, obj_bundle: IndexBundle, l_questions: list[Question]) -> SizeResult:
    """Rank the answerable questions with each method, then sweep the gate over all questions."""
    obj_hybrid = HybridMethod(HybridRetriever(obj_bundle.obj_bm25, obj_bundle.obj_dense))
    l_methods = [Bm25Method(obj_bundle.obj_bm25), DenseMethod(obj_bundle.obj_dense), obj_hybrid]
    l_answerable = [obj_question for obj_question in l_questions if obj_question.b_answerable]

    d_outcomes: dict[str, list[QuestionOutcome]] = {obj_method.s_name: [] for obj_method in l_methods}
    for obj_question in l_answerable:
      i_total = self.obj_judge.count_relevant(obj_bundle.l_chunks, obj_question.s_must_contain)
      for obj_method in l_methods:
        d_outcomes[obj_method.s_name].append(self.outcome(obj_method, obj_question, i_total))

    d_best_dense = {
      obj_question.s_id: self.best_dense(obj_hybrid, obj_question) for obj_question in l_questions
    }
    l_sweep_rows = self.obj_sweeper.sweep(
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if obj_question.b_answerable],
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if not obj_question.b_answerable],
    )
    return SizeResult(
      i_size=obj_bundle.i_size, i_chunk_count=len(obj_bundle.l_chunks),
      d_outcomes=d_outcomes, d_best_dense=d_best_dense, l_sweep_rows=l_sweep_rows,
    )

  def outcome(self, obj_method: RetrievalMethod, obj_question: Question, i_total: int) -> QuestionOutcome:
    """Rank to the evaluation depth and record where the relevant chunks landed."""
    l_ranked = obj_method.rank(obj_question.s_text, self.i_depth)
    # a chunk is relevant when it contains the question's phrase
    l_flags = [self.obj_judge.is_relevant(obj_chunk, obj_question.s_must_contain) for obj_chunk in l_ranked]
    return QuestionOutcome(
      s_id=obj_question.s_id,
      i_first_rank=l_flags.index(True) + 1 if True in l_flags else None,
      i_total_relevant=i_total,
      d_found_at_k={i_k: sum(l_flags[:i_k]) for i_k in self.T_KS},
    )

  def best_dense(self, obj_hybrid: HybridMethod, obj_question: Question) -> float | None:
    """Return the highest dense score among the gate's hybrid hits, or None if none has one."""
    l_scores = [
      obj_hit.f_dense_score for obj_hit in obj_hybrid.hits(obj_question.s_text, self.i_gate_top_k)
      if obj_hit.f_dense_score is not None
    ]
    return max(l_scores) if l_scores else None
