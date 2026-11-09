"""Unit tests for the LlmClient; no network calls are made."""

import sys

import pytest

from mini_rag.llm_client import LlmClient


class TestLlmClient:
  """Checks key detection (presence only) and the missing-extra error."""

  def test_is_available_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable means unavailable."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert LlmClient.is_available() is False

  def test_is_available_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty variable means unavailable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert LlmClient.is_available() is False

  def test_is_available_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty dummy value means available."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-a-key")
    assert LlmClient.is_available() is True

  def test_generate_without_anthropic_raises_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """If anthropic cannot be imported, generate raises RuntimeError naming the uv command."""
    # a None entry in sys.modules makes "import anthropic" raise ImportError
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(RuntimeError, match="uv sync --extra llm"):
      LlmClient().generate("system", "user")
