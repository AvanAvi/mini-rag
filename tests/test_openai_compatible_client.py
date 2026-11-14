"""Tests for OpenAiCompatibleClient; every request goes to a fake opener."""

import json
import urllib.error

import pytest
from fake_opener import FakeOpener

from mini_rag.generator import Generator
from mini_rag.openai_compatible_client import OpenAiCompatibleClient


class TestOpenAiCompatibleClient:
  """Checks the request that is built, the reply parsing and the error handling."""

  S_KEY_VARIABLE = "MINI_RAG_TEST_KEY"
  S_REPLY = json.dumps({"choices": [{"message": {"role": "assistant", "content": "The answer."}}]})

  def client(self, obj_opener: FakeOpener, s_api_key_env: str = "") -> OpenAiCompatibleClient:
    """Return a client for a made-up local address that sends through obj_opener."""
    return OpenAiCompatibleClient(
      "http://localhost:8080/v1/", "local-model", s_api_key_env, i_max_tokens=77, obj_opener=obj_opener,
    )

  def test_is_a_generator_named_after_its_model(self) -> None:
    """The client subclasses Generator and reports the model name."""
    obj_client = self.client(FakeOpener())
    assert isinstance(obj_client, Generator)
    assert obj_client.s_name == "local-model"

  def test_request_url_method_and_body(self) -> None:
    """generate() posts model, messages and max_tokens to <base_url>/chat/completions."""
    obj_opener = FakeOpener(self.S_REPLY.encode("utf-8"))
    assert self.client(obj_opener).generate("be brief", "what is a tokamak?") == "The answer."

    assert len(obj_opener.l_requests) == 1
    obj_request = obj_opener.l_requests[0]
    # the trailing slash of the base address is dropped
    assert obj_request.full_url == "http://localhost:8080/v1/chat/completions"
    assert obj_request.get_method() == "POST"
    assert obj_request.get_header("Content-type") == "application/json"
    assert json.loads(obj_request.data.decode("utf-8")) == {
      "model": "local-model",
      "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what is a tokamak?"},
      ],
      "max_tokens": 77,
    }
    assert obj_opener.l_timeouts == [OpenAiCompatibleClient.I_TIMEOUT_SECONDS]

  def test_no_key_variable_sends_no_authorization(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a key variable, or with one that is not set, no Authorization header is sent."""
    monkeypatch.delenv(self.S_KEY_VARIABLE, raising=False)
    for s_api_key_env in ("", self.S_KEY_VARIABLE):
      obj_opener = FakeOpener(self.S_REPLY.encode("utf-8"))
      self.client(obj_opener, s_api_key_env).generate("s", "u")
      assert obj_opener.l_requests[0].get_header("Authorization") is None

  def test_key_is_read_from_the_named_variable_and_not_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """The key goes into the bearer header at request time and is not kept on the client."""
    monkeypatch.setenv(self.S_KEY_VARIABLE, "dummy-value-for-test")
    obj_opener = FakeOpener(self.S_REPLY.encode("utf-8"))
    obj_client = self.client(obj_opener, self.S_KEY_VARIABLE)
    obj_client.generate("s", "u")
    assert obj_opener.l_requests[0].get_header("Authorization") == "Bearer dummy-value-for-test"
    assert "dummy-value-for-test" not in repr(vars(obj_client))
    assert obj_client.s_api_key_env == self.S_KEY_VARIABLE

  def test_network_error_becomes_runtime_error(self) -> None:
    """A failed connection raises RuntimeError, which the CLI turns into a note."""
    obj_opener = FakeOpener(obj_error=urllib.error.URLError("connection refused"))
    with pytest.raises(RuntimeError, match="failed"):
      self.client(obj_opener).generate("s", "u")

  @pytest.mark.parametrize("s_reply", ["not json", "{}", '{"choices": []}', '{"choices": [{"message": {"content": 5}}]}'])
  def test_unexpected_reply_becomes_runtime_error(self, s_reply: str) -> None:
    """A reply that is not JSON or lacks the first choice's text raises RuntimeError."""
    with pytest.raises(RuntimeError, match="unexpected shape"):
      self.client(FakeOpener(s_reply.encode("utf-8"))).generate("s", "u")

  def test_bad_settings_raise(self) -> None:
    """An address without http or https, or an empty model, raises ValueError."""
    with pytest.raises(ValueError, match="base_url"):
      OpenAiCompatibleClient("localhost:8080", "m", obj_opener=FakeOpener())
    with pytest.raises(ValueError, match="model"):
      OpenAiCompatibleClient("http://localhost:8080", "", obj_opener=FakeOpener())
