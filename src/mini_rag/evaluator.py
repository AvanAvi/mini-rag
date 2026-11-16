"""Measures retrieval quality and the answer gate against a question set."""

from mini_rag.bm25_method import Bm25Method
from mini_rag.dense_method import DenseMethod
from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.eval_result import EvalResult
from mini_rag.fusion_result import FusionResult
from mini_rag.fusion_setting import FusionSetting
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
  T_GATE_MODES = ("hybrid", "dense")  # retrieval modes whose hits carry a dense score

  def __init__(
    self, obj_store: EvalIndexStore, l_sizes: list[int] | tuple[int, ...] = (500, 1000), i_overlap: int = 100,
    i_depth: int = 10, i_gate_top_k: int = 5, obj_sweeper: ThresholdSweeper | None = None,
    l_fusion_settings: list[FusionSetting] | None = None, s_gate_mode: str = "hybrid",
  ) -> None:
    """Store the index store, the chunk sizes, the ranking depth, the gate's top-k and the fusion settings.

    l_fusion_settings lists the settings hybrid is evaluated with; None means the retriever's defaults only.
    s_gate_mode picks the hits the best dense score is taken from: "hybrid" (fused hits) or "dense".
    """
    if s_gate_mode not in self.T_GATE_MODES:
      raise ValueError(f"s_gate_mode must be one of {', '.join(self.T_GATE_MODES)}, got {s_gate_mode!r}")
    self.s_gate_mode = s_gate_mode
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
    self.l_fusion_settings = list(l_fusion_settings) if l_fusion_settings else [FusionSetting.default()]

  def prepare(self, l_docs: list[Document], b_rebuild: bool = False) -> list[IndexBundle]:
    """Build or load the indexes for every chunk size and return them in size order."""
    return [
      self.obj_store.get_indexes(l_docs, i_size, self.i_overlap, b_rebuild)
      for i_size in self.l_sizes
    ]

  def run(
    self, l_docs: list[Document], l_questions: list[Question], l_bundles: list[IndexBundle] | None = None,
  ) -> EvalResult:
    """Evaluate every chunk size on the questions and return all results.

    l_bundles takes indexes already returned by prepare(); None means prepare them here.
    """
    l_used_bundles = l_bundles if l_bundles is not None else self.prepare(l_docs)
    return EvalResult(
      s_corpus_fingerprint=self.obj_store.fingerprint(l_docs),
      s_model_name=self.obj_store.s_model_name,
      i_overlap=self.i_overlap,
      i_depth=self.i_depth,
      i_gate_top_k=self.i_gate_top_k,
      l_size_results=[self.run_size(obj_bundle, l_questions) for obj_bundle in l_used_bundles],
    )

  def run_size(self, obj_bundle: IndexBundle, l_questions: list[Question]) -> SizeResult:
    """Rank the answerable questions with BM25 and dense once, and with hybrid once per fusion setting."""
    l_answerable = [obj_question for obj_question in l_questions if obj_question.b_answerable]
    l_totals = [
      self.obj_judge.count_relevant(obj_bundle.l_chunks, obj_question.s_must_contain)
      for obj_question in l_answerable
    ]
    d_outcomes: dict[str, list[QuestionOutcome]] = {}
    for obj_method in (Bm25Method(obj_bundle.obj_bm25), DenseMethod(obj_bundle.obj_dense)):
      d_outcomes[obj_method.s_name] = self.outcomes(obj_method, l_answerable, l_totals)

    l_fusion_results = [
      self.run_fusion(obj_bundle, obj_setting, l_questions, l_answerable, l_totals)
      for obj_setting in self.l_fusion_settings
    ]
    # the report's hybrid rows, gate scores and sweep come from the main setting
    obj_main = self.main_fusion(l_fusion_results)
    d_outcomes[HybridMethod.S_NAME] = obj_main.l_outcomes
    return SizeResult(
      i_size=obj_bundle.i_size, i_chunk_count=len(obj_bundle.l_chunks),
      d_outcomes=d_outcomes, d_best_dense=obj_main.d_best_dense, l_sweep_rows=obj_main.l_sweep_rows,
      l_fusion_results=l_fusion_results,
    )

  def run_fusion(
    self, obj_bundle: IndexBundle, obj_setting: FusionSetting, l_questions: list[Question],
    l_answerable: list[Question], l_totals: list[int],
  ) -> FusionResult:
    """Evaluate hybrid under one fusion setting, then sweep the gate over all questions."""
    obj_hybrid = self.hybrid_method(obj_bundle, obj_setting)
    obj_gate = obj_hybrid if self.s_gate_mode == "hybrid" else self.gate_method(obj_bundle, obj_setting)
    d_best_dense = {
      obj_question.s_id: self.best_dense(obj_gate, obj_question) for obj_question in l_questions
    }
    l_sweep_rows = self.obj_sweeper.sweep(
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if obj_question.b_answerable],
      [d_best_dense[obj_question.s_id] for obj_question in l_questions if not obj_question.b_answerable],
    )
    return FusionResult(
      obj_setting=obj_setting, l_outcomes=self.outcomes(obj_hybrid, l_answerable, l_totals),
      d_best_dense=d_best_dense, l_sweep_rows=l_sweep_rows,
    )

  def hybrid_method(self, obj_bundle: IndexBundle, obj_setting: FusionSetting) -> HybridMethod:
    """Return the hybrid method over one bundle's indexes with the setting's fusion parameters."""
    return HybridMethod(HybridRetriever(
      obj_bundle.obj_bm25, obj_bundle.obj_dense, i_k=obj_setting.i_k,
      f_bm25_weight=obj_setting.f_bm25_weight, f_dense_weight=obj_setting.f_dense_weight,
    ))

  def gate_method(self, obj_bundle: IndexBundle, obj_setting: FusionSetting) -> HybridMethod | DenseMethod:
    """Return the method whose hits give the best dense score: hybrid with the setting, or dense alone."""
    if self.s_gate_mode == "dense":
      return DenseMethod(obj_bundle.obj_dense)
    return self.hybrid_method(obj_bundle, obj_setting)

  def main_fusion(self, l_fusion_results: list[FusionResult]) -> FusionResult:
    """Return the result of the retriever's default setting if it was evaluated, otherwise the first one."""
    for obj_fusion in l_fusion_results:
      if obj_fusion.obj_setting == FusionSetting.default():
        return obj_fusion
    return l_fusion_results[0]

  def outcomes(
    self, obj_method: RetrievalMethod, l_answerable: list[Question], l_totals: list[int],
  ) -> list[QuestionOutcome]:
    """Return one outcome per answerable question for one method."""
    return [
      self.outcome(obj_method, obj_question, i_total)
      for obj_question, i_total in zip(l_answerable, l_totals)
    ]

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

  def best_dense(self, obj_method: HybridMethod | DenseMethod, obj_question: Question) -> float | None:
    """Return the highest dense score among the gate's hits from obj_method, or None if none has one."""
    l_scores = [
      obj_hit.f_dense_score for obj_hit in obj_method.hits(obj_question.s_text, self.i_gate_top_k)
      if obj_hit.f_dense_score is not None
    ]
    return max(l_scores) if l_scores else None
