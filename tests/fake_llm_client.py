"""Test double that returns a fixed reply instead of calling a model."""


class FakeLlmClient:
  """Records every prompt pair it receives and returns a fixed reply."""

  def __init__(self, s_reply: str) -> None:
    """Store the reply to return and start with no recorded prompts."""
    self.s_reply = s_reply
    self.l_system_prompts: list[str] = []
    self.l_user_prompts: list[str] = []

  def generate(self, s_system: str, s_user: str) -> str:
    """Record both prompts and return the fixed reply."""
    self.l_system_prompts.append(s_system)
    self.l_user_prompts.append(s_user)
    return self.s_reply
