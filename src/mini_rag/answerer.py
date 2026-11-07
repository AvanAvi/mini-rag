"""Generates an answer from retrieved chunks via the optional "llm" extra."""


class Answerer:
  """Turns a question plus retrieved chunks into an answer."""

  def __init__(self, s_model_name: str) -> None:
    """Store the model name; the client is created only when needed."""
    raise NotImplementedError

  def is_available(self) -> bool:
    """Return True if the "llm" extra is installed and an API key is set in the session."""
    raise NotImplementedError

  def answer(self, s_question: str, l_chunks: list[dict]) -> str:
    """Return an answer to the question, grounded in the given chunks."""
    raise NotImplementedError
