"""Data class for a whole evaluation run."""

from dataclasses import dataclass

from mini_rag.size_result import SizeResult


@dataclass
class EvalResult:
  """The settings of a run and its results per chunk size."""

  s_corpus_fingerprint: str
  s_model_name: str
  i_overlap: int
  i_depth: int
  i_gate_top_k: int
  l_size_results: list[SizeResult]
