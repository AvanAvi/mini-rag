"""Tests for ComponentFactory; no model is loaded and no request is sent."""

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from fake_llm_client import FakeLlmClient

from mini_rag.component_factory import ComponentFactory
from mini_rag.config import RagConfig
from mini_rag.llm_client import LlmClient
from mini_rag.openai_compatible_client import OpenAiCompatibleClient


class TestComponentFactory:
  """Checks every embedder and generator kind and the errors for bad settings."""

  def test_custom_embedder_is_imported_and_built_with_options(self) -> None:
    """Kind "custom" imports the class and passes the options table as keyword arguments."""
    obj_config = RagConfig(
      s_embedder_kind="custom", s_embedder_class="bag_of_words_embedder:BagOfWordsEmbedder",
      d_embedder_options={"l_vocabulary": ["cats", "dogs"]},
    )
    obj_embedder = ComponentFactory(obj_config).embedder()
    assert isinstance(obj_embedder, BagOfWordsEmbedder)
    assert obj_embedder.l_vocabulary == ["cats", "dogs"]

  def test_custom_class_must_subclass_the_interface(self) -> None:
    """A class that is not an Embedder, or a Generator, is rejected with the key name."""
    obj_config = RagConfig(s_embedder_kind="custom", s_embedder_class="fake_index:FakeIndex")
    with pytest.raises(ValueError, match=r"embedder\.class.*not a subclass of Embedder"):
      ComponentFactory(obj_config).embedder()
    obj_config = RagConfig(s_generator_kind="custom", s_generator_class="fake_index:FakeIndex")
    with pytest.raises(ValueError, match=r"generator\.class.*not a subclass of Generator"):
      ComponentFactory(obj_config).generator()

  @pytest.mark.parametrize("s_spec, s_message", [
    ("", "must look like"),
    ("no_colon_here", "must look like"),
    ("module_that_does_not_exist:Thing", "cannot import"),
    ("bag_of_words_embedder:NoSuchClass", "not a subclass"),
  ])
  def test_bad_custom_spec_raises(self, s_spec: str, s_message: str) -> None:
    """A malformed, missing or unknown class reference raises ValueError."""
    obj_config = RagConfig(s_embedder_kind="custom", s_embedder_class=s_spec)
    with pytest.raises(ValueError, match=s_message):
      ComponentFactory(obj_config).embedder()

  def test_options_that_do_not_fit_raise(self) -> None:
    """Options the constructor does not accept raise ValueError naming the options table."""
    obj_config = RagConfig(
      s_embedder_kind="custom", s_embedder_class="bag_of_words_embedder:BagOfWordsEmbedder",
      d_embedder_options={"i_unknown": 1},
    )
    with pytest.raises(ValueError, match=r"embedder\.options"):
      ComponentFactory(obj_config).embedder()

  def test_unknown_fastembed_model_is_rejected_with_a_hint(self) -> None:
    """A name outside fastembed's supported list raises ValueError that says how to see the list."""
    obj_config = RagConfig(s_embedder_model="no/such-model")
    with pytest.raises(ValueError, match=r"embedder\.model.*list_supported_models"):
      ComponentFactory(obj_config).embedder()

  def test_default_fastembed_model_passes_the_name_check(self) -> None:
    """The default model name is in fastembed's list; the check loads no model."""
    ComponentFactory(RagConfig()).check_fastembed_model(RagConfig().s_embedder_model)

  def test_generator_kind_none_gives_no_generator(self) -> None:
    """Kind "none" returns None."""
    assert ComponentFactory(RagConfig(s_generator_kind="none")).generator() is None

  def test_generator_kind_anthropic_uses_the_configured_model(self) -> None:
    """Kind "anthropic" returns an LlmClient for the configured model; no request is made."""
    obj_config = RagConfig(s_generator_kind="anthropic", s_generator_model="test-model")
    obj_generator = ComponentFactory(obj_config).generator()
    assert isinstance(obj_generator, LlmClient)
    assert obj_generator.s_model == "test-model"
    assert ComponentFactory(RagConfig()).generator() is None

  def test_generator_kind_openai_compatible(self) -> None:
    """Kind "openai_compatible" passes the address, the model and the key variable name on."""
    obj_config = RagConfig(
      s_generator_kind="openai_compatible", s_generator_model="local-model",
      s_generator_base_url="http://localhost:8080/v1", s_generator_api_key_env="MY_KEY",
    )
    obj_generator = ComponentFactory(obj_config).generator()
    assert isinstance(obj_generator, OpenAiCompatibleClient)
    assert (obj_generator.s_base_url, obj_generator.s_model, obj_generator.s_api_key_env) == (
      "http://localhost:8080/v1", "local-model", "MY_KEY",
    )

  def test_openai_compatible_needs_a_base_url(self) -> None:
    """Kind "openai_compatible" without base_url raises ValueError naming the key."""
    obj_config = RagConfig(s_generator_kind="openai_compatible", s_generator_model="m")
    with pytest.raises(ValueError, match=r"generator\.base_url"):
      ComponentFactory(obj_config).generator()

  def test_custom_generator(self) -> None:
    """Kind "custom" builds a Generator subclass with its options."""
    obj_config = RagConfig(
      s_generator_kind="custom", s_generator_class="fake_llm_client:FakeLlmClient",
      d_generator_options={"s_reply": "hello"},
    )
    obj_generator = ComponentFactory(obj_config).generator()
    assert isinstance(obj_generator, FakeLlmClient)
    assert obj_generator.generate("s", "u") == "hello"
