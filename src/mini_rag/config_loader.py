"""Reads a mini-rag TOML config file into a RagConfig."""

import tomllib
from pathlib import Path

from mini_rag.config import RagConfig


class ConfigLoader:
  """Maps TOML sections and keys onto RagConfig fields and checks every value's type."""

  T_KINDS_NEEDING_MODEL = ("anthropic", "openai_compatible")  # generator kinds that send a model name

  # section -> key -> (RagConfig field, value type)
  D_SCHEMA = {
    "corpus": {"dir": ("s_corpus_dir", "str"), "extensions": ("t_extensions", "str_list")},
    "chunking": {"size": ("i_chunk_size", "int"), "overlap": ("i_chunk_overlap", "int")},
    "embedder": {
      "kind": ("s_embedder_kind", "str"), "model": ("s_embedder_model", "str"),
      "class": ("s_embedder_class", "str"), "options": ("d_embedder_options", "table"),
    },
    "index": {"dir": ("s_index_dir", "str")},
    "retrieval": {
      "mode": ("s_retrieval_mode", "str"), "top_k": ("i_top_k", "int"), "rrf_k": ("i_rrf_k", "int"),
      "bm25_weight": ("f_bm25_weight", "float"), "dense_weight": ("f_dense_weight", "float"),
    },
    "gate": {"threshold": ("f_threshold", "float")},
    "generator": {
      "kind": ("s_generator_kind", "str"), "model": ("s_generator_model", "str"),
      "base_url": ("s_generator_base_url", "str"), "api_key_env": ("s_generator_api_key_env", "str"),
      "class": ("s_generator_class", "str"), "options": ("d_generator_options", "table"),
    },
    "eval": {
      "questions": ("s_questions", "str"), "sizes": ("t_eval_sizes", "int_list"),
      "cache_dir": ("s_eval_cache_dir", "str"),
    },
  }

  def load(self, s_path: str) -> RagConfig:
    """Return the config in s_path; a missing file gives the built-in defaults."""
    obj_path = Path(s_path)
    if not obj_path.is_file():
      return RagConfig()
    try:
      d_toml = tomllib.loads(obj_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as obj_error:
      raise ValueError(f"{s_path} is not valid TOML: {obj_error}") from obj_error
    return self.from_dict(d_toml)

  def from_dict(self, d_toml: dict) -> RagConfig:
    """Build a RagConfig from parsed TOML; keys that are left out keep their defaults."""
    d_fields: dict[str, object] = {}
    for s_section, obj_table in d_toml.items():
      if s_section not in self.D_SCHEMA:
        raise ValueError(f"unknown config section [{s_section}]")
      if not isinstance(obj_table, dict):
        raise ValueError(f"config key {s_section} must be a section, such as [{s_section}]")
      for s_key, obj_value in obj_table.items():
        s_full_key = f"{s_section}.{s_key}"
        if s_key not in self.D_SCHEMA[s_section]:
          raise ValueError(f"unknown config key {s_full_key}")
        s_field, s_type = self.D_SCHEMA[s_section][s_key]
        d_fields[s_field] = self.convert(s_full_key, obj_value, s_type)
    obj_config = RagConfig(**d_fields)
    self.check_kind("embedder.kind", obj_config.s_embedder_kind, RagConfig.T_EMBEDDER_KINDS)
    self.check_kind("generator.kind", obj_config.s_generator_kind, RagConfig.T_GENERATOR_KINDS)
    self.check_generator_model(obj_config)
    self.check_retrieval_mode(obj_config.s_retrieval_mode)
    return obj_config

  def check_retrieval_mode(self, s_mode: str) -> None:
    """Raise ValueError naming [retrieval] mode and the allowed values when the mode is not one of them."""
    if s_mode not in RagConfig.T_RETRIEVAL_MODES:
      raise ValueError(
        f"[retrieval] mode must be one of {', '.join(RagConfig.T_RETRIEVAL_MODES)}, got {s_mode!r}"
      )

  def convert(self, s_full_key: str, obj_value: object, s_type: str) -> object:
    """Return the value in the field's type; raise ValueError naming the key for a wrong type."""
    if s_type == "str" and isinstance(obj_value, str):
      return obj_value
    if s_type == "int" and self.is_int(obj_value):
      return obj_value
    # a whole number is accepted where a decimal is expected, for example threshold = 1
    if s_type == "float" and (isinstance(obj_value, float) or self.is_int(obj_value)):
      return float(obj_value)
    if s_type == "table" and isinstance(obj_value, dict):
      return dict(obj_value)
    if s_type == "int_list" and isinstance(obj_value, list) and obj_value and all(self.is_int(obj_item) for obj_item in obj_value):
      return tuple(obj_value)
    if s_type == "str_list" and isinstance(obj_value, list) and obj_value and all(isinstance(obj_item, str) for obj_item in obj_value):
      return tuple(obj_value)
    d_wanted = {
      "str": "a string", "int": "a whole number", "float": "a number", "table": "a table",
      "int_list": "a non-empty list of whole numbers", "str_list": "a non-empty list of strings",
    }
    raise ValueError(f"config key {s_full_key} must be {d_wanted[s_type]}, got {obj_value!r}")

  def is_int(self, obj_value: object) -> bool:
    """Return True for a whole number; True and False do not count."""
    return isinstance(obj_value, int) and not isinstance(obj_value, bool)

  def check_kind(self, s_full_key: str, s_kind: str, t_kinds: tuple[str, ...]) -> None:
    """Raise ValueError naming the key when the kind is not one of the known kinds."""
    if s_kind not in t_kinds:
      raise ValueError(f"config key {s_full_key} must be one of {', '.join(t_kinds)}, got {s_kind!r}")

  def check_generator_model(self, obj_config: RagConfig) -> None:
    """Raise ValueError naming [generator] model when a kind that sends requests has no model name."""
    s_kind = obj_config.s_generator_kind
    if s_kind in self.T_KINDS_NEEDING_MODEL and not obj_config.s_generator_model:
      raise ValueError(
        f'[generator] model is required for kind "{s_kind}"; set it to the model name from your provider'
      )
