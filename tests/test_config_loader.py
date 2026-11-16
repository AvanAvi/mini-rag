"""Tests for RagConfig and ConfigLoader."""

import dataclasses
from pathlib import Path

import pytest

from mini_rag.answerer import Answerer
from mini_rag.config import RagConfig
from mini_rag.config_loader import ConfigLoader
from mini_rag.dense_index import DenseIndex
from mini_rag.fusion_setting import FusionSetting


class TestConfigLoader:
  """Checks the defaults, the TOML mapping and the error for every kind of bad value."""

  def load(self, tmp_path: Path, s_toml: str) -> RagConfig:
    """Write s_toml to a file and load it."""
    obj_path = tmp_path / "config.toml"
    obj_path.write_text(s_toml, encoding="utf-8")
    return ConfigLoader().load(str(obj_path))

  def test_defaults_match_the_classes_they_configure(self) -> None:
    """The built-in config holds the same defaults as the classes that use them."""
    obj_config = RagConfig()
    assert obj_config.f_threshold == Answerer().f_threshold
    assert obj_config.s_embedder_model == DenseIndex().s_model_name
    assert (obj_config.s_generator_kind, obj_config.s_generator_model) == ("none", "")
    obj_fusion = FusionSetting.default()
    assert (obj_config.i_rrf_k, obj_config.f_bm25_weight, obj_config.f_dense_weight) == (
      obj_fusion.i_k, obj_fusion.f_bm25_weight, obj_fusion.f_dense_weight,
    )
    assert (obj_config.s_corpus_dir, obj_config.t_extensions) == ("data/corpus", (".txt",))
    assert (obj_config.i_chunk_size, obj_config.i_chunk_overlap, obj_config.i_top_k) == (500, 100, 5)
    assert (obj_config.s_embedder_kind, obj_config.s_generator_kind) == ("fastembed", "none")
    assert (obj_config.s_index_dir, obj_config.s_eval_cache_dir) == (".cache/index", ".cache/eval")
    assert (obj_config.s_questions, obj_config.t_eval_sizes) == ("eval/questions.json", (500, 1000))

  def test_config_is_frozen(self) -> None:
    """A config cannot be changed after creation."""
    with pytest.raises(dataclasses.FrozenInstanceError):
      RagConfig().i_top_k = 9

  def test_missing_file_gives_the_defaults(self, tmp_path: Path) -> None:
    """A path without a file returns the built-in config."""
    assert ConfigLoader().load(str(tmp_path / "absent.toml")) == RagConfig()

  def test_shipped_example_equals_the_defaults(self) -> None:
    """mini-rag.toml at the project root reproduces the built-in config exactly."""
    obj_path = Path(__file__).resolve().parent.parent / "mini-rag.toml"
    assert obj_path.is_file()
    assert ConfigLoader().load(str(obj_path)) == RagConfig()

  def test_every_key_is_read(self, tmp_path: Path) -> None:
    """Each section and key lands in its field."""
    obj_config = self.load(tmp_path, """
[corpus]
dir = "my/docs"
extensions = [".txt", ".md"]
[chunking]
size = 300
overlap = 30
[embedder]
kind = "custom"
model = "m"
class = "pkg.mod:Cls"
[embedder.options]
i_dim = 8
[index]
dir = "idx"
[retrieval]
top_k = 3
rrf_k = 10
bm25_weight = 0.5
dense_weight = 2
[gate]
threshold = 0.3
[generator]
kind = "openai_compatible"
model = "local-model"
base_url = "http://localhost:8080/v1"
api_key_env = "MY_KEY"
class = "pkg.mod:Gen"
[generator.options]
s_reply = "x"
[eval]
questions = "q.json"
sizes = [200]
cache_dir = "evalcache"
""")
    assert obj_config == RagConfig(
      s_corpus_dir="my/docs", t_extensions=(".txt", ".md"), i_chunk_size=300, i_chunk_overlap=30,
      s_embedder_kind="custom", s_embedder_model="m", s_embedder_class="pkg.mod:Cls", d_embedder_options={"i_dim": 8},
      s_index_dir="idx", i_top_k=3, i_rrf_k=10, f_bm25_weight=0.5, f_dense_weight=2.0, f_threshold=0.3,
      s_generator_kind="openai_compatible", s_generator_model="local-model",
      s_generator_base_url="http://localhost:8080/v1", s_generator_api_key_env="MY_KEY",
      s_generator_class="pkg.mod:Gen", d_generator_options={"s_reply": "x"},
      s_questions="q.json", t_eval_sizes=(200,), s_eval_cache_dir="evalcache",
    )
    # a whole number given for a weight is stored as a float
    assert isinstance(obj_config.f_dense_weight, float)

  def test_missing_keys_keep_their_defaults(self, tmp_path: Path) -> None:
    """A file with one key changes that field only."""
    obj_config = self.load(tmp_path, "[retrieval]\ntop_k = 7\n")
    assert obj_config == dataclasses.replace(RagConfig(), i_top_k=7)

  @pytest.mark.parametrize("s_toml, s_key", [
    ('[chunking]\nsize = "big"\n', "chunking.size"),
    ("[chunking]\nsize = 1.5\n", "chunking.size"),
    ("[chunking]\noverlap = true\n", "chunking.overlap"),
    ('[gate]\nthreshold = "high"\n', "gate.threshold"),
    ("[corpus]\ndir = 5\n", "corpus.dir"),
    ('[corpus]\nextensions = ".md"\n', "corpus.extensions"),
    ('[eval]\nsizes = [500, "x"]\n', "eval.sizes"),
    ("[eval]\nsizes = []\n", "eval.sizes"),
    ('[embedder]\noptions = "x"\n', "embedder.options"),
  ])
  def test_wrong_type_names_the_key(self, tmp_path: Path, s_toml: str, s_key: str) -> None:
    """A value of the wrong type raises ValueError with the full key name."""
    with pytest.raises(ValueError, match=s_key.replace(".", r"\.")):
      self.load(tmp_path, s_toml)

  @pytest.mark.parametrize("s_toml, s_key", [
    ('[embedder]\nkind = "magic"\n', "embedder.kind"),
    ('[generator]\nkind = "magic"\n', "generator.kind"),
  ])
  def test_unknown_kind_names_the_key(self, tmp_path: Path, s_toml: str, s_key: str) -> None:
    """An unknown embedder or generator kind raises ValueError with the key name."""
    with pytest.raises(ValueError, match=s_key.replace(".", r"\.")):
      self.load(tmp_path, s_toml)

  def test_unknown_key_and_section_are_rejected(self, tmp_path: Path) -> None:
    """A misspelt key or section raises ValueError instead of being ignored."""
    with pytest.raises(ValueError, match=r"retrieval\.topk"):
      self.load(tmp_path, "[retrieval]\ntopk = 3\n")
    with pytest.raises(ValueError, match=r"\[retreival\]"):
      self.load(tmp_path, "[retreival]\ntop_k = 3\n")

  def test_invalid_toml_raises_value_error(self, tmp_path: Path) -> None:
    """A file that is not TOML raises ValueError naming the file."""
    with pytest.raises(ValueError, match="not valid TOML"):
      self.load(tmp_path, "[gate\nthreshold = ")
