"""Settings of one mini-rag setup, with the built-in defaults."""

from dataclasses import dataclass, field

from mini_rag.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class RagConfig:
  """Every configurable value; an instance built without arguments is the built-in setup."""

  T_EMBEDDER_KINDS = ("fastembed", "custom")
  T_GENERATOR_KINDS = ("anthropic", "openai_compatible", "none", "custom")

  # [corpus]
  s_corpus_dir: str = "data/corpus"
  t_extensions: tuple[str, ...] = (".txt",)
  # [chunking]
  i_chunk_size: int = 500
  i_chunk_overlap: int = 100
  # [embedder]
  s_embedder_kind: str = "fastembed"
  s_embedder_model: str = "BAAI/bge-small-en-v1.5"
  s_embedder_class: str = ""  # "package.module:ClassName", for kind "custom"
  d_embedder_options: dict = field(default_factory=dict)
  # [index]
  s_index_dir: str = ".cache/index"
  # [retrieval]
  i_top_k: int = 5
  i_rrf_k: int = HybridRetriever.I_DEFAULT_K
  f_bm25_weight: float = HybridRetriever.F_DEFAULT_BM25_WEIGHT
  f_dense_weight: float = HybridRetriever.F_DEFAULT_DENSE_WEIGHT
  # [gate]; the threshold belongs to the configured embedding model
  f_threshold: float = 0.55
  # [generator]
  s_generator_kind: str = "none"
  s_generator_model: str = ""  # required for kinds "anthropic" and "openai_compatible"
  s_generator_base_url: str = ""  # for kind "openai_compatible"
  s_generator_api_key_env: str = ""  # name of the environment variable holding the key, never the key
  s_generator_class: str = ""  # "package.module:ClassName", for kind "custom"
  d_generator_options: dict = field(default_factory=dict)
  # [eval]
  s_questions: str = "eval/questions.json"
  t_eval_sizes: tuple[int, ...] = (500, 1000)
  s_eval_cache_dir: str = ".cache/eval"
