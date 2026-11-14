"""Unit tests for FusionSetting and FusionGrid."""

import dataclasses

import pytest

from mini_rag.fusion_grid import FusionGrid
from mini_rag.fusion_setting import FusionSetting
from mini_rag.hybrid_retriever import HybridRetriever


class TestFusionSetting:
  """Checks the label, the default, immutability and the contents of the grid."""

  def test_label_shows_k_and_both_weights(self) -> None:
    """The label names k and the weights, BM25 first, always with a decimal point."""
    assert FusionSetting(i_k=60, f_bm25_weight=1.0, f_dense_weight=1.0).s_label == "k=60 w=1.0/1.0"
    assert FusionSetting(i_k=10, f_bm25_weight=0.25, f_dense_weight=1).s_label == "k=10 w=0.25/1.0"

  def test_default_matches_a_retriever_built_without_options(self) -> None:
    """default() carries the values a plain HybridRetriever ends up with."""
    obj_retriever = HybridRetriever(object(), object())
    obj_default = FusionSetting.default()
    assert obj_default == FusionSetting(
      i_k=obj_retriever.i_k, f_bm25_weight=obj_retriever.f_bm25_weight, f_dense_weight=obj_retriever.f_dense_weight,
    )

  def test_setting_is_frozen(self) -> None:
    """A setting cannot be changed after creation."""
    with pytest.raises(dataclasses.FrozenInstanceError):
      FusionSetting.default().i_k = 5

  def test_grid_has_twelve_settings_in_order(self) -> None:
    """The grid crosses k in (5, 10, 20, 60) with BM25 weights (1.0, 0.5, 0.25) at dense weight 1.0."""
    l_settings = FusionGrid().settings()
    assert len(l_settings) == 12
    assert len(set(l_settings)) == 12
    assert [obj_setting.i_k for obj_setting in l_settings] == [5] * 3 + [10] * 3 + [20] * 3 + [60] * 3
    assert [obj_setting.f_bm25_weight for obj_setting in l_settings] == [1.0, 0.5, 0.25] * 4
    assert {obj_setting.f_dense_weight for obj_setting in l_settings} == {1.0}

  def test_grid_contains_k_60_with_equal_weights(self) -> None:
    """The grid includes k = 60 with both weights at 1.0."""
    assert FusionSetting(i_k=60, f_bm25_weight=1.0, f_dense_weight=1.0) in FusionGrid().settings()
