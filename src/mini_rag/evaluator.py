"""Measures retrieval quality against a question set."""


class Evaluator:
  """Scores a retriever on questions with known relevant sources."""

  def __init__(self, obj_retriever: object) -> None:
    """Store the retriever to evaluate."""
    raise NotImplementedError

  def load_questions(self, s_path: str) -> list[dict]:
    """Read evaluation questions and their expected sources from a file."""
    raise NotImplementedError

  def evaluate(self, l_questions: list[dict], i_top_k: int) -> dict:
    """Run every question and return the aggregate metrics."""
    raise NotImplementedError
