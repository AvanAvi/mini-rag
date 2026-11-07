"""Unit tests for the Tokenizer."""

from mini_rag.tokenizer import Tokenizer


class TestTokenizer:
  """Checks lowercasing, splitting and empty inputs."""

  def setup_method(self) -> None:
    """Create a fresh tokenizer for each test."""
    self.obj_tokenizer = Tokenizer()

  def test_lowercasing(self) -> None:
    """Upper and mixed case are lowered."""
    assert self.obj_tokenizer.tokenize("Hello WORLD MiXeD") == ["hello", "world", "mixed"]

  def test_punctuation_split(self) -> None:
    """Punctuation and whitespace separate tokens and are dropped."""
    assert self.obj_tokenizer.tokenize("well-known, isn't it?\nyes") == [
      "well", "known", "isn", "t", "it", "yes",
    ]

  def test_digits_kept(self) -> None:
    """Digits are kept, alone or mixed with letters."""
    assert self.obj_tokenizer.tokenize("Python 3.13 has v2api") == [
      "python", "3", "13", "has", "v2api",
    ]

  def test_empty_string(self) -> None:
    """An empty string gives no tokens."""
    assert self.obj_tokenizer.tokenize("") == []

  def test_only_punctuation(self) -> None:
    """A string of only punctuation and spaces gives no tokens."""
    assert self.obj_tokenizer.tokenize(" ...!?;-- ") == []
