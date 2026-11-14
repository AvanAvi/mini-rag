"""Tests for the fusion settings of the Evaluator."""

from bag_of_words_embedder import BagOfWordsEmbedder
from counting_embedder import CountingEmbedder

from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.evaluator import Evaluator
from mini_rag.fusion_grid import FusionGrid
from mini_rag.fusion_setting import FusionSetting
from mini_rag.question import Question


class TestEvaluatorFusion:
  """Checks that hybrid runs once per setting and that the default run is unchanged."""

  L_VOCABULARY = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince",
  ]

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
    """Return two answerable questions and one unanswerable question."""
    return [
      Question(s_id="q1", s_text="where is the banana", b_answerable=True,
               s_source_doc="d1", s_must_contain="banana cherry"),
      Question(s_id="q2", s_text="papaya", b_answerable=True,
               s_source_doc="d4", s_must_contain="lemon mango"),
      Question(s_id="q3", s_text="zzz", b_answerable=False,
               t_absent_terms=("zebra",), s_topic="off_topic"),
    ]

  def evaluator(self, tmp_path, l_fusion_settings: list[FusionSetting] | None, obj_embedder: object = None) -> Evaluator:
    """Return an evaluator for chunk size 50 with a cache in tmp_path."""
    obj_used = obj_embedder if obj_embedder is not None else BagOfWordsEmbedder(self.L_VOCABULARY)
    obj_store = EvalIndexStore(str(tmp_path), obj_used)
    return Evaluator(obj_store, l_sizes=[50], i_overlap=10, l_fusion_settings=l_fusion_settings)

  def test_default_is_one_setting_equal_to_the_retriever_defaults(self, tmp_path) -> None:
    """Without settings hybrid is evaluated once, with the default setting, and fills the main fields."""
    obj_size = self.evaluator(tmp_path, None).run(self.docs(), self.questions()).l_size_results[0]
    assert [obj_fusion.obj_setting for obj_fusion in obj_size.l_fusion_results] == [FusionSetting.default()]
    obj_fusion = obj_size.l_fusion_results[0]
    assert obj_size.d_outcomes["hybrid"] == obj_fusion.l_outcomes
    assert obj_size.d_best_dense == obj_fusion.d_best_dense
    assert obj_size.l_sweep_rows == obj_fusion.l_sweep_rows

  def test_grid_run_keeps_the_main_results_of_a_default_run(self, tmp_path) -> None:
    """A grid that contains the default setting leaves every main field as in a run without a grid."""
    obj_plain = self.evaluator(tmp_path, None).run(self.docs(), self.questions()).l_size_results[0]
    obj_grid = self.evaluator(tmp_path, FusionGrid().settings()).run(self.docs(), self.questions()).l_size_results[0]
    assert list(obj_grid.d_outcomes) == ["BM25", "dense", "hybrid"]
    assert obj_grid.d_outcomes == obj_plain.d_outcomes
    assert obj_grid.d_best_dense == obj_plain.d_best_dense
    assert obj_grid.l_sweep_rows == obj_plain.l_sweep_rows
    assert [obj_fusion.obj_setting for obj_fusion in obj_grid.l_fusion_results] == FusionGrid().settings()

  def test_main_result_is_the_first_setting_when_the_default_is_absent(self, tmp_path) -> None:
    """Without the default setting in the list, the first setting fills the main fields."""
    l_settings = [FusionSetting(i_k=5, f_bm25_weight=0.5, f_dense_weight=1.0), FusionSetting(10, 0.25, 1.0)]
    obj_size = self.evaluator(tmp_path, l_settings).run(self.docs(), self.questions()).l_size_results[0]
    assert obj_size.d_outcomes["hybrid"] is obj_size.l_fusion_results[0].l_outcomes

  def test_hybrid_method_carries_the_setting(self, tmp_path) -> None:
    """The retriever built for a setting uses its k and weights."""
    obj_evaluator = self.evaluator(tmp_path, None)
    obj_bundle = obj_evaluator.prepare(self.docs())[0]
    obj_retriever = obj_evaluator.hybrid_method(obj_bundle, FusionSetting(5, 0.25, 0.75)).obj_retriever
    assert (obj_retriever.i_k, obj_retriever.f_bm25_weight, obj_retriever.f_dense_weight) == (5, 0.25, 0.75)

  def test_zero_bm25_weight_gives_the_dense_outcomes(self, tmp_path) -> None:
    """With no weight on BM25, hybrid finds the relevant chunks at the dense ranks."""
    l_settings = [FusionSetting(i_k=60, f_bm25_weight=0.0, f_dense_weight=1.0)]
    obj_size = self.evaluator(tmp_path, l_settings).run(self.docs(), self.questions()).l_size_results[0]
    assert obj_size.l_fusion_results[0].l_outcomes == obj_size.d_outcomes["dense"]

  def test_grid_run_on_a_cached_index_embeds_nothing(self, tmp_path) -> None:
    """Once the cache exists, a grid run loads it and makes no embed() call."""
    obj_embedder = CountingEmbedder(BagOfWordsEmbedder(self.L_VOCABULARY))
    self.evaluator(tmp_path, None, obj_embedder).prepare(self.docs())
    i_calls_after_build = obj_embedder.i_embed_calls
    assert i_calls_after_build > 0

    obj_evaluator = self.evaluator(tmp_path, FusionGrid().settings(), obj_embedder)
    l_bundles = obj_evaluator.prepare(self.docs())
    obj_result = obj_evaluator.run(self.docs(), self.questions(), l_bundles)
    assert [obj_bundle.b_loaded for obj_bundle in l_bundles] == [True]
    assert len(obj_result.l_size_results[0].l_fusion_results) == 12
    assert obj_embedder.i_embed_calls == i_calls_after_build
