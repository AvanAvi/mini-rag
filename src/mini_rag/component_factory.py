"""Builds the embedder and the generator named in a RagConfig."""

import importlib
import os
import sys

from mini_rag.bm25_index import Bm25Index
from mini_rag.bm25_only_retriever import Bm25OnlyRetriever
from mini_rag.config import RagConfig
from mini_rag.dense_only_retriever import DenseOnlyRetriever
from mini_rag.embedder import Embedder
from mini_rag.generator import Generator
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.llm_client import LlmClient
from mini_rag.openai_compatible_client import OpenAiCompatibleClient


class ComponentFactory:
  """Turns the [embedder], [retrieval] and [generator] sections of a config into objects."""

  S_LIST_MODELS_HINT = (
    'python -c "from fastembed import TextEmbedding; '
    "print([d_model['model'] for d_model in TextEmbedding.list_supported_models()])\""
  )

  def __init__(self, obj_config: RagConfig) -> None:
    """Store the config to build from."""
    self.obj_config = obj_config

  def embedder(self) -> Embedder:
    """Return the configured embedder: a fastembed model or a custom Embedder subclass."""
    if self.obj_config.s_embedder_kind == "custom":
      return self.custom(
        "embedder", self.obj_config.s_embedder_class, self.obj_config.d_embedder_options, Embedder,
      )
    self.check_fastembed_model(self.obj_config.s_embedder_model)
    # imported here so setups with a custom embedder never import fastembed
    from mini_rag.fastembed_embedder import FastEmbedEmbedder
    return FastEmbedEmbedder(self.obj_config.s_embedder_model)

  def generator(self) -> Generator | None:
    """Return the configured generator, or None for kind "none"."""
    s_kind = self.obj_config.s_generator_kind
    if s_kind == "none":
      return None
    if s_kind == "custom":
      return self.custom(
        "generator", self.obj_config.s_generator_class, self.obj_config.d_generator_options, Generator,
      )
    if s_kind == "openai_compatible":
      for s_key, s_value in (("base_url", self.obj_config.s_generator_base_url), ("model", self.obj_config.s_generator_model)):
        if not s_value:
          raise ValueError(f'config key generator.{s_key} is required for kind "openai_compatible"')
      return OpenAiCompatibleClient(
        self.obj_config.s_generator_base_url, self.obj_config.s_generator_model,
        self.obj_config.s_generator_api_key_env,
      )
    return LlmClient(s_model=self.obj_config.s_generator_model)

  def retriever(self, obj_dense: object) -> object:
    """Return the retriever of the configured [retrieval] mode over a loaded dense index.

    The BM25 index is built from the dense index's chunks in modes "hybrid" and "bm25" only, and
    the fusion settings apply to mode "hybrid" only.
    """
    s_mode = self.obj_config.s_retrieval_mode
    if s_mode == "dense":
      return DenseOnlyRetriever(obj_dense)
    obj_bm25 = Bm25Index()
    obj_bm25.build(obj_dense.l_chunks)
    if s_mode == "bm25":
      return Bm25OnlyRetriever(obj_bm25)
    return HybridRetriever(
      obj_bm25, obj_dense, i_k=self.obj_config.i_rrf_k,
      f_bm25_weight=self.obj_config.f_bm25_weight, f_dense_weight=self.obj_config.f_dense_weight,
    )

  def check_fastembed_model(self, s_model: str) -> None:
    """Raise ValueError when fastembed's list of supported models does not hold the name."""
    from fastembed import TextEmbedding
    l_names = [d_model["model"].lower() for d_model in TextEmbedding.list_supported_models()]
    if s_model.lower() not in l_names:
      raise ValueError(
        f"config key embedder.model: {s_model!r} is not a model fastembed supports; "
        f"list the supported names with: {self.S_LIST_MODELS_HINT}"
      )

  def custom(self, s_section: str, s_spec: str, d_options: dict, obj_base: type) -> object:
    """Import "package.module:ClassName", check its base class and build it with the options."""
    s_full_key = f"{s_section}.class"
    s_module, s_separator, s_class = s_spec.partition(":")
    if not s_module or not s_separator or not s_class:
      raise ValueError(f'config key {s_full_key} must look like "package.module:ClassName", got {s_spec!r}')
    self.add_working_directory()
    try:
      obj_module = importlib.import_module(s_module)
    except ImportError as obj_error:
      raise ValueError(f"config key {s_full_key}: cannot import {s_module!r} ({obj_error})") from obj_error
    obj_class = getattr(obj_module, s_class, None)
    if not isinstance(obj_class, type) or not issubclass(obj_class, obj_base):
      raise ValueError(f"config key {s_full_key}: {s_spec!r} is not a subclass of {obj_base.__name__}")
    try:
      return obj_class(**d_options)
    except TypeError as obj_error:
      raise ValueError(f"config table {s_section}.options does not fit {s_spec!r}: {obj_error}") from obj_error

  def add_working_directory(self) -> None:
    """Append the working directory to the end of the import path, only for a "custom" kind.

    The console script does not put the working directory on the import path, so a class in the
    project folder could not be imported otherwise. Appending keeps installed modules first.
    """
    if os.getcwd() not in sys.path:
      sys.path.append(os.getcwd())
