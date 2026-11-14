"""Interface every answer-writing model client implements."""

from abc import ABC, abstractmethod


class Generator(ABC):
  """Writes text from a system prompt and a user prompt; subclass it to plug in another model."""

  @property
  @abstractmethod
  def s_name(self) -> str:
    """Return a name that identifies the model behind this client."""

  @abstractmethod
  def generate(self, s_system: str, s_user: str) -> str:
    """Send both prompts to the model and return its reply text."""
