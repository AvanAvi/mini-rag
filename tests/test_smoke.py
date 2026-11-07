"""Smoke test: every module in the package imports cleanly."""

import importlib


class TestSmoke:
  """Checks that the skeleton is importable."""

  l_modules = [
    "mini_rag",
    "mini_rag.loader",
    "mini_rag.chunker",
    "mini_rag.dense_index",
    "mini_rag.bm25_index",
    "mini_rag.hybrid_retriever",
    "mini_rag.answerer",
    "mini_rag.evaluator",
    "mini_rag.cli",
  ]

  def test_all_modules_import(self) -> None:
    """Import each module and fail on the first one that raises."""
    for s_module in self.l_modules:
      obj_module = importlib.import_module(s_module)
      assert obj_module is not None, s_module
