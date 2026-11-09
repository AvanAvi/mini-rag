"""Command-line entry point: ingest, query, eval."""

import argparse
import sys
import time

from mini_rag.answer import Answer
from mini_rag.answerer import Answerer
from mini_rag.bm25_index import Bm25Index
from mini_rag.chunker import Chunker
from mini_rag.dense_index import DenseIndex
from mini_rag.hybrid_hit import HybridHit
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.llm_client import LlmClient
from mini_rag.loader import Loader


class Cli:
  """Parses arguments and dispatches to the ingest, query and eval commands."""

  I_OK = 0
  I_USAGE_ERROR = 2

  def __init__(self, obj_embedder: object | None = None, obj_llm_client: object | None = None) -> None:
    """Store optional test doubles: an embedder for DenseIndex and a client replacing LlmClient."""
    self.obj_embedder = obj_embedder
    self.obj_llm_client = obj_llm_client

  @classmethod
  def main(cls) -> None:
    """Console-script entry point; builds a Cli, runs it with sys.argv and exits with its code."""
    for obj_stream in (sys.stdout, sys.stderr):
      obj_stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(cls().run(sys.argv[1:]))

  def run(self, l_args: list[str]) -> int:
    """Parse the arguments, run the chosen command, and return an exit code."""
    try:
      obj_args = self.build_parser().parse_args(l_args)
    except SystemExit as obj_exit:
      # argparse exits with 2 on usage errors and 0 after --help
      return obj_exit.code if isinstance(obj_exit.code, int) else self.I_USAGE_ERROR

    if obj_args.s_command == "ingest":
      return self.ingest(obj_args.s_corpus, obj_args.s_index_dir, obj_args.i_size, obj_args.i_overlap)
    if obj_args.s_command == "query":
      return self.query(
        obj_args.s_question, obj_args.s_index_dir, obj_args.i_top_k,
        obj_args.f_threshold, obj_args.b_generate,
      )
    return self.eval()

  def build_parser(self) -> argparse.ArgumentParser:
    """Return the argument parser with the ingest, query and eval subcommands."""
    obj_parser = argparse.ArgumentParser(prog="mini-rag", description="A small hybrid-retrieval RAG pipeline.")
    obj_commands = obj_parser.add_subparsers(dest="s_command", required=True)

    obj_ingest = obj_commands.add_parser("ingest", help="load, chunk, embed and save the corpus")
    obj_ingest.add_argument("--corpus", dest="s_corpus", default="data/corpus")
    obj_ingest.add_argument("--index-dir", dest="s_index_dir", default=".cache/index")
    obj_ingest.add_argument("--size", dest="i_size", type=int, default=500)
    obj_ingest.add_argument("--overlap", dest="i_overlap", type=int, default=100)

    obj_query = obj_commands.add_parser("query", help="retrieve chunks and answer a question")
    obj_query.add_argument("s_question", metavar="question")
    obj_query.add_argument("--index-dir", dest="s_index_dir", default=".cache/index")
    obj_query.add_argument("--top-k", dest="i_top_k", type=int, default=5)
    obj_query.add_argument("--threshold", dest="f_threshold", type=float, default=0.5)
    obj_query.add_argument("--generate", dest="b_generate", action="store_true")

    obj_commands.add_parser("eval", help="not implemented yet")
    return obj_parser

  def ingest(self, s_corpus_dir: str, s_index_dir: str, i_size: int, i_overlap: int) -> int:
    """Load, chunk, embed and save the corpus; print a summary and return an exit code."""
    f_start = time.perf_counter()
    try:
      obj_chunker = Chunker(i_size, i_overlap)
      l_docs = Loader(s_corpus_dir).load()
      l_chunks = obj_chunker.chunk_documents(l_docs)
      obj_dense = DenseIndex(obj_embedder=self.obj_embedder)
      obj_dense.build(l_chunks)
      obj_dense.save(s_index_dir)
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))

    f_elapsed = time.perf_counter() - f_start
    print(f"documents: {len(l_docs)}")
    print(f"chunks: {len(l_chunks)}")
    print(f"chunk size: {i_size}")
    print(f"overlap: {i_overlap}")
    print(f"index folder: {s_index_dir}")
    print(f"elapsed: {f_elapsed:.2f} s")
    return self.I_OK

  def query(self, s_question: str, s_index_dir: str, i_top_k: int, f_threshold: float, b_generate: bool) -> int:
    """Retrieve chunks for a question and print them plus the answer; return an exit code."""
    obj_dense = DenseIndex(obj_embedder=self.obj_embedder)
    try:
      obj_dense.load(s_index_dir)
    except FileNotFoundError:
      return self.fail(f"no index found in {s_index_dir}; run ingest first")

    try:
      obj_bm25 = Bm25Index()
      obj_bm25.build(obj_dense.l_chunks)
      obj_retriever = HybridRetriever(obj_bm25, obj_dense)
      # built before searching so a bad threshold fails early
      obj_extractive = Answerer(f_threshold)
      l_hits = obj_retriever.search(s_question, i_top_k)
    except ValueError as obj_error:
      return self.fail(str(obj_error))

    for i_rank, obj_hit in enumerate(l_hits, start=1):
      self.print_hit(i_rank, obj_hit)
    self.print_answer(self.answer(s_question, l_hits, obj_extractive, b_generate))
    return self.I_OK

  def eval(self) -> int:
    """Placeholder for the evaluator command; always reports that it is not implemented."""
    print("eval is not implemented yet", file=sys.stderr)
    return self.I_USAGE_ERROR

  def answer(self, s_question: str, l_hits: list[HybridHit], obj_extractive: Answerer, b_generate: bool) -> Answer:
    """Answer with a model client when asked and possible, otherwise with the extractive Answerer."""
    obj_client = self.make_client() if b_generate else None
    if obj_client is not None:
      try:
        return Answerer(obj_extractive.f_threshold, obj_client).answer(s_question, l_hits)
      except RuntimeError as obj_error:
        # e.g. the anthropic package is not installed
        print(f"note: {obj_error}; showing the extractive answer")
    return obj_extractive.answer(s_question, l_hits)

  def make_client(self) -> object | None:
    """Return the injected client, a real LlmClient if a key is set, or None after saying why."""
    if self.obj_llm_client is not None:
      return self.obj_llm_client
    if LlmClient.is_available():
      return LlmClient()
    print(f"note: {LlmClient.S_KEY_VARIABLE} is not set; showing the extractive answer")
    return None

  def print_hit(self, i_rank: int, obj_hit: HybridHit) -> None:
    """Print one hit: rank, marker, dense, BM25 and fused scores, then the chunk text."""
    obj_chunk = obj_hit.obj_chunk
    print(
      f"{i_rank}. [{obj_chunk.s_doc_id}#{obj_chunk.i_index}]"
      f"  dense {self.format_score(obj_hit.f_dense_score)}"
      f"  bm25 {self.format_score(obj_hit.f_bm25_score)}"
      f"  fused {obj_hit.f_rrf_score:.4f}"
    )
    print(obj_chunk.s_text)
    print()

  def print_answer(self, obj_answer: Answer) -> None:
    """Print the answer text and whether it was found and whether a model wrote it."""
    print("Answer:")
    print(obj_answer.s_text)
    s_found = "yes" if obj_answer.b_found else "no"
    s_generated = "yes" if obj_answer.b_generated else "no"
    print(f"found: {s_found}, written by a model: {s_generated}")

  def format_score(self, f_score: float | None) -> str:
    """Return a score with three decimals, or "-" when the retriever missed the chunk."""
    return "-" if f_score is None else f"{f_score:.3f}"

  def fail(self, s_message: str) -> int:
    """Print an error line to stderr and return the usage-error exit code."""
    print(f"error: {s_message}", file=sys.stderr)
    return self.I_USAGE_ERROR
