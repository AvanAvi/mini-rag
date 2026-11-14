"""Tests that ComponentFactory changes the import path only for a "custom" kind, and only at the end."""

import os
import sys

import pytest

from mini_rag.component_factory import ComponentFactory
from mini_rag.config import RagConfig


class TestFactoryImportPath:
  """Checks sys.path before and after building components."""

  def test_non_custom_kinds_leave_the_import_path_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building every non-custom generator and checking a fastembed name leave sys.path as it was."""
    monkeypatch.setattr(sys, "path", [s_entry for s_entry in sys.path if s_entry != os.getcwd()])
    l_before = list(sys.path)
    ComponentFactory(RagConfig()).check_fastembed_model(RagConfig().s_embedder_model)
    for obj_config in (
      RagConfig(),
      RagConfig(s_generator_kind="none"),
      RagConfig(s_generator_kind="openai_compatible", s_generator_model="m", s_generator_base_url="http://localhost:1/v1"),
    ):
      ComponentFactory(obj_config).generator()
    assert sys.path == l_before

  def test_custom_kind_appends_the_working_directory_at_the_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom embedder adds the working directory once, as the last entry."""
    monkeypatch.setattr(sys, "path", [s_entry for s_entry in sys.path if s_entry != os.getcwd()])
    l_before = list(sys.path)
    obj_config = RagConfig(
      s_embedder_kind="custom", s_embedder_class="bag_of_words_embedder:BagOfWordsEmbedder",
      d_embedder_options={"l_vocabulary": ["a"]},
    )
    ComponentFactory(obj_config).embedder()
    ComponentFactory(obj_config).embedder()
    assert sys.path == l_before + [os.getcwd()]
