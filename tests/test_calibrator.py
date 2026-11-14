"""Unit tests for the Calibrator."""

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from calibration_set_builder import CalibrationSetBuilder

from mini_rag.calibrator import Calibrator
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.fusion_setting import FusionSetting
from mini_rag.question import Question


class TestCalibrator:
  """Checks the swept range, the error counts and the warnings on a set with hand-computed scores."""

  def calibrator(self, tmp_path) -> Calibrator:
    """Return a calibrator with a bag-of-words embedder and a cache in tmp_path."""
    obj_store = EvalIndexStore(str(tmp_path), BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY))
    return Calibrator(obj_store, 50, 10, FusionSetting.default(), 5)

  def test_best_dense_scores(self, tmp_path) -> None:
    """Every question gets the best dense score worked out by hand; no retrieval metric is filled in."""
    obj_builder = CalibrationSetBuilder()
    obj_calibrator = self.calibrator(tmp_path)
    obj_size = obj_calibrator.run(obj_calibrator.prepare(obj_builder.documents()), obj_builder.questions())
    assert obj_size.d_best_dense == pytest.approx({"q1": 0.8165, "q2": 0.5774, "u1": 0.5774, "u2": 0.0}, abs=1e-4)
    assert obj_size.d_outcomes == {}
    assert (obj_size.i_size, obj_size.i_chunk_count) == (50, 5)

  def test_sweep_spans_the_scores_and_counts_both_errors(self, tmp_path) -> None:
    """Thresholds run in steps of 0.05 from below the lowest score to above the highest."""
    obj_builder = CalibrationSetBuilder()
    obj_calibrator = self.calibrator(tmp_path)
    obj_size = obj_calibrator.run(obj_calibrator.prepare(obj_builder.documents()), obj_builder.questions())
    l_rows = obj_size.l_sweep_rows
    assert [obj_row.f_threshold for obj_row in l_rows] == [round(i_step * 0.05, 2) for i_step in range(0, 18)]
    d_errors = {
      obj_row.f_threshold: (obj_row.i_answerable_rejected, obj_row.i_unanswerable_accepted) for obj_row in l_rows
    }
    # 0.00 accepts both unanswerable questions; up to 0.55 only u1 (0.577) gets through
    assert d_errors[0.0] == (0, 2)
    assert d_errors[0.55] == (0, 1)
    # from 0.60 on q2 (0.577) is rejected, and at 0.85 q1 (0.816) is too
    assert d_errors[0.6] == (1, 0)
    assert d_errors[0.85] == (2, 0)
    # one error at every threshold from 0.05 to 0.80
    assert obj_calibrator.best_thresholds(obj_size) == [round(i_step * 0.05, 2) for i_step in range(1, 17)]

  def test_thresholds_fall_back_and_stay_in_range(self, tmp_path) -> None:
    """No scores means the sweeper's own thresholds; scores at the limits stay within -1.0 to 1.0."""
    obj_calibrator = self.calibrator(tmp_path)
    assert obj_calibrator.thresholds([]) is None
    assert obj_calibrator.thresholds([-1.0, 1.0])[0] == -1.0
    assert obj_calibrator.thresholds([-1.0, 1.0])[-1] == 1.0
    assert obj_calibrator.thresholds([0.489, 0.890]) == [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

  def test_warnings_for_small_groups(self, tmp_path) -> None:
    """Fewer than 10 answerable or 3 unanswerable questions give one warning each."""
    obj_calibrator = self.calibrator(tmp_path)
    l_warnings = obj_calibrator.warnings(CalibrationSetBuilder().questions())
    assert len(l_warnings) == 2
    assert "only 2 answerable" in l_warnings[0] and "only 2 unanswerable" in l_warnings[1]

    l_enough = [
      Question(s_id=f"a{i_number}", s_text="x", b_answerable=True, s_source_doc="d1", s_must_contain="cherry")
      for i_number in range(10)
    ] + [
      Question(s_id=f"u{i_number}", s_text="x", b_answerable=False, t_absent_terms=("zebra",), s_topic="off_topic")
      for i_number in range(3)
    ]
    assert obj_calibrator.warnings(l_enough) == []
