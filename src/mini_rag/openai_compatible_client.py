"""Client for servers that accept the commonly used chat-completions request format."""

import json
import os
import urllib.error
import urllib.request

from mini_rag.generator import Generator


class OpenAiCompatibleClient(Generator):
  """Posts a system and a user message to <base_url>/chat/completions and returns the reply text.

  It follows the commonly used chat-completions request format, which many local model servers
  accept. It was not tested against a live server in this project; its tests use a fake opener.
  """

  I_TIMEOUT_SECONDS = 120

  def __init__(
    self, s_base_url: str, s_model: str, s_api_key_env: str = "", i_max_tokens: int = 500,
    obj_opener: object | None = None,
  ) -> None:
    """Store the server address, the model name and the name of the API key variable.

    s_api_key_env names an environment variable; its value is read for each request and never stored.
    Leave it empty for a server that needs no key. obj_opener replaces the urllib opener in tests.
    """
    if not s_base_url.lower().startswith(("http://", "https://")):
      raise ValueError(f"base_url must start with http:// or https://, got {s_base_url!r}")
    if not s_model:
      raise ValueError("model must not be empty")
    self.s_base_url = s_base_url.rstrip("/")
    self.s_model = s_model
    self.s_api_key_env = s_api_key_env
    self.i_max_tokens = i_max_tokens
    self.obj_opener = obj_opener if obj_opener is not None else urllib.request.build_opener()

  @property
  def s_name(self) -> str:
    """Return the model name sent with every request."""
    return self.s_model

  def generate(self, s_system: str, s_user: str) -> str:
    """Send both prompts and return the first choice's message text; failures raise RuntimeError."""
    obj_request = self.build_request(s_system, s_user)
    try:
      with self.obj_opener.open(obj_request, timeout=self.I_TIMEOUT_SECONDS) as obj_response:
        d_reply = json.loads(obj_response.read().decode("utf-8"))
      s_text = d_reply["choices"][0]["message"]["content"]
    except (urllib.error.URLError, OSError) as obj_error:
      raise RuntimeError(f"the request to {self.s_base_url} failed: {obj_error}") from obj_error
    except (ValueError, KeyError, IndexError, TypeError) as obj_error:
      raise RuntimeError(f"the reply from {self.s_base_url} has an unexpected shape") from obj_error
    if not isinstance(s_text, str):
      raise RuntimeError(f"the reply from {self.s_base_url} has an unexpected shape")
    return s_text

  def build_request(self, s_system: str, s_user: str) -> urllib.request.Request:
    """Return the POST request with the JSON body and, if a key variable is set, the bearer header."""
    d_body = {
      "model": self.s_model,
      "messages": [{"role": "system", "content": s_system}, {"role": "user", "content": s_user}],
      "max_tokens": self.i_max_tokens,
    }
    d_headers = {"Content-Type": "application/json"}
    # the key goes from the environment straight into the header and is not kept on the object
    if self.s_api_key_env and os.environ.get(self.s_api_key_env):
      d_headers["Authorization"] = f"Bearer {os.environ[self.s_api_key_env]}"
    return urllib.request.Request(
      f"{self.s_base_url}/chat/completions", data=json.dumps(d_body).encode("utf-8"),
      headers=d_headers, method="POST",
    )
