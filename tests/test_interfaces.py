"""Tests for the Embedder and Generator interfaces."""

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from counting_embedder import CountingEmbedder
from fake_embedder import FakeEmbedder
from fake_llm_client import FakeLlmClient

from mini_rag.embedder import Embedder
from mini_rag.generator import Generator
from mini_rag.llm_client import LlmClient


class TestInterfaces:
  """Checks that the interfaces are abstract and that the clients and test doubles implement them."""

  def test_embedder_cannot_be_instantiated(self) -> None:
    """Embedder is abstract."""
    with pytest.raises(TypeError):
      Embedder()

  def test_generator_cannot_be_instantiated(self) -> None:
    """Generator is abstract."""
    with pytest.raises(TypeError):
      Generator()

  def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
    """A subclass that leaves out a method or the name stays abstract."""
    obj_class = type("HalfEmbedder", (Embedder,), {"embed": lambda self, l_texts: []})
    with pytest.raises(TypeError):
      obj_class()

  def test_test_doubles_are_embedders_with_names(self) -> None:
    """The embedder test doubles subclass Embedder and report a name."""
    obj_fake = FakeEmbedder({}, {})
    obj_words = BagOfWordsEmbedder(["a"])
    obj_counting = CountingEmbedder(obj_words)
    for obj_embedder in (obj_fake, obj_words, obj_counting):
      assert isinstance(obj_embedder, Embedder)
    assert (obj_fake.s_name, obj_words.s_name, obj_counting.s_name) == ("fake-embedder", "bag-of-words", "bag-of-words")

  def test_generators_have_names(self) -> None:
    """LlmClient and the fake client subclass Generator; LlmClient is named after its model."""
    assert isinstance(FakeLlmClient("x"), Generator)
    assert FakeLlmClient("x").s_name == "fake-generator"
    obj_client = LlmClient(s_model="some-model")
    assert isinstance(obj_client, Generator)
    assert obj_client.s_name == "some-model"

  def test_fastembed_embedder_subclasses_embedder_without_loading_a_model(self) -> None:
    """FastEmbedEmbedder declares Embedder as its base; checked on the source so fastembed is not imported."""
    import ast
    from pathlib import Path

    import mini_rag.embedder

    obj_path = Path(mini_rag.embedder.__file__).with_name("fastembed_embedder.py")
    obj_tree = ast.parse(obj_path.read_text(encoding="utf-8"))
    l_classes = [obj_node for obj_node in obj_tree.body if isinstance(obj_node, ast.ClassDef)]
    assert [obj_base.id for obj_base in l_classes[0].bases] == ["Embedder"]
