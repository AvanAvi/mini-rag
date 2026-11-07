"""Dense vector index built on fastembed embeddings."""


class DenseIndex:
  """Embeds chunks and ranks them by cosine similarity to a query."""

  def __init__(self, s_model_name: str) -> None:
    """Store the fastembed model name; the model is loaded lazily."""
    raise NotImplementedError

  def build(self, l_chunks: list[dict]) -> None:
    """Embed all chunks and keep the vectors in memory."""
    raise NotImplementedError

  def search(self, s_query: str, i_top_k: int) -> list[dict]:
    """Return the top-k chunks with their dense similarity scores."""
    raise NotImplementedError
