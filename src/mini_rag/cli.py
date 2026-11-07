"""Command-line entry point: ingest, query, eval."""


class Cli:
  """Parses arguments and dispatches to the ingest, query and eval commands."""

  @classmethod
  def main(cls) -> None:
    """Console-script entry point; builds a Cli and runs it with sys.argv."""
    raise NotImplementedError

  def run(self, l_args: list[str]) -> int:
    """Parse the arguments, run the chosen command, and return an exit code."""
    raise NotImplementedError

  def ingest(self, s_corpus_dir: str) -> None:
    """Load, chunk and index the corpus."""
    raise NotImplementedError

  def query(self, s_question: str, i_top_k: int) -> None:
    """Retrieve chunks for a question and print them, plus an answer if available."""
    raise NotImplementedError

  def eval(self, s_questions_path: str, i_top_k: int) -> None:
    """Run the evaluator on a question file and print the metrics."""
    raise NotImplementedError
