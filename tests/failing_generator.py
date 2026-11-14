"""Test double for a generator whose call fails, like an API that returns an error."""

from mini_rag.generator import Generator


class FailingGenerator(Generator):
  """Raises a fixed exception from generate()."""

  def __init__(self, obj_error: Exception) -> None:
    """Store the exception to raise."""
    self.obj_error = obj_error

  @property
  def s_name(self) -> str:
    """Return the name of this test double."""
    return "failing-generator"

  def generate(self, s_system: str, s_user: str) -> str:
    """Raise the stored exception."""
    raise self.obj_error
