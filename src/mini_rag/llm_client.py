"""Thin wrapper around the Anthropic Messages API (the optional "llm" extra)."""

import os


class LlmClient:
  """Sends a system and a user prompt to a Claude model and returns the reply text."""

  S_KEY_VARIABLE = "ANTHROPIC_API_KEY"

  def __init__(self, s_model: str = "claude-sonnet-5-5", i_max_tokens: int = 500) -> None:
    """Store the model and token limit; the anthropic client is created on first use."""
    self.s_model = s_model
    self.i_max_tokens = i_max_tokens
    self.obj_client: object | None = None

  @classmethod
  def is_available(cls) -> bool:
    """Return True if the API key variable is set and non-empty; the value is never exposed."""
    return bool(os.environ.get(cls.S_KEY_VARIABLE))

  def generate(self, s_system: str, s_user: str) -> str:
    """Send both prompts and return the model's reply text; API errors propagate."""
    obj_message = self.get_client().messages.create(
      model=self.s_model,
      max_tokens=self.i_max_tokens,
      system=s_system,
      messages=[{"role": "user", "content": s_user}],
    )
    return "".join(obj_block.text for obj_block in obj_message.content if obj_block.type == "text")

  def get_client(self) -> object:
    """Return the anthropic client, importing the package and creating it on first use."""
    if self.obj_client is None:
      try:
        # imported here so importing this module never requires the extra
        import anthropic
      except ImportError as obj_error:
        raise RuntimeError(
          "The anthropic package is not installed; run: uv sync --extra llm"
        ) from obj_error
      # the client reads the API key from the environment itself
      self.obj_client = anthropic.Anthropic()
    return self.obj_client
