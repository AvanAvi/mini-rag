"""Command-line entry point: ingest, query, eval and the eval kit commands."""

import argparse
import sys
import time
from pathlib import Path

from mini_rag.answer import Answer
from mini_rag.answerer import Answerer
from mini_rag.bm25_index import Bm25Index
from mini_rag.chunker import Chunker
from mini_rag.component_factory import ComponentFactory
from mini_rag.config import RagConfig
from mini_rag.config_loader import ConfigLoader
from mini_rag.dense_index import DenseIndex
from mini_rag.document import Document
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.eval_kit import EvalKit
from mini_rag.evaluator import Evaluator
from mini_rag.fusion_grid import FusionGrid
from mini_rag.fusion_grid_writer import FusionGridWriter
from mini_rag.fusion_setting import FusionSetting
from mini_rag.hybrid_hit import HybridHit
from mini_rag.hybrid_retriever import HybridRetriever
from mini_rag.index_bundle import IndexBundle
from mini_rag.llm_client import LlmClient
from mini_rag.loader import Loader
from mini_rag.question_loader import QuestionLoader
from mini_rag.question_starter import QuestionStarter
from mini_rag.question_validator import QuestionValidator
from mini_rag.report_writer import ReportWriter


class Cli:
  """Parses arguments and dispatches to the ingest, query and eval commands."""

  I_OK = 0
  I_USAGE_ERROR = 2
  I_MAX_PROBLEMS = 10  # question-set problems printed before giving up
  I_BRIEF_LENGTH = 160  # characters of hit text shown with --brief
  S_GRID_FILE = "fusion_grid.md"  # written next to the eval report by --fusion-grid
  S_CONFIG_FILE = "mini-rag.toml"  # read from the working directory when --config is not given

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
    """Load the config, parse the arguments, run the chosen command, and return an exit code."""
    try:
      obj_config = self.load_config(l_args)
      obj_args = self.build_parser(obj_config).parse_args(l_args)
    except SystemExit as obj_exit:
      # argparse exits with 2 on usage errors and 0 after --help
      return obj_exit.code if isinstance(obj_exit.code, int) else self.I_USAGE_ERROR
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))

    if obj_args.s_command == "ingest":
      return self.ingest(obj_args.s_corpus, obj_args.s_index_dir, obj_args.i_size, obj_args.i_overlap, obj_config)
    if obj_args.s_command == "query":
      return self.query(
        obj_args.s_question, obj_args.s_index_dir, obj_args.i_top_k,
        obj_args.f_threshold, obj_args.b_generate, obj_args.b_brief, obj_config,
      )
    if obj_args.s_command == "eval-init":
      return EvalKit().init(obj_args.s_out, obj_args.b_force)
    if obj_args.s_command == "eval-suggest":
      return EvalKit().suggest(obj_args.s_corpus, obj_config.t_extensions, obj_args.i_limit)
    if obj_args.s_command == "eval-check":
      return EvalKit().check(obj_args.s_questions, obj_args.s_corpus, obj_config.t_extensions, obj_args.b_lenient)
    if obj_args.s_command == "calibrate":
      return self.calibrate(obj_args.s_questions, obj_args.s_corpus, obj_args.s_cache_dir, obj_config)
    return self.eval(
      obj_args.s_questions, obj_args.s_corpus, obj_args.l_sizes, obj_args.i_overlap, obj_args.i_gate_top_k,
      obj_args.s_cache_dir, obj_args.s_out, obj_args.b_rebuild, obj_args.b_build_only, obj_args.b_fusion_grid,
      obj_config,
    )

  def load_config(self, l_args: list[str]) -> RagConfig:
    """Find --config among the arguments and load it; without the option, mini-rag.toml or the defaults."""
    obj_finder = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    obj_finder.add_argument("--config", dest="s_config", default=None)
    s_config = obj_finder.parse_known_args(l_args)[0].s_config
    if s_config is None:
      # the default file is optional: without it every value is the built-in default
      return ConfigLoader().load(self.S_CONFIG_FILE)
    if not Path(s_config).is_file():
      raise FileNotFoundError(f"config file not found: {s_config}")
    return ConfigLoader().load(s_config)

  def build_parser(self, obj_config: RagConfig | None = None) -> argparse.ArgumentParser:
    """Return the argument parser; option defaults come from the config, or the built-in defaults."""
    obj_used = obj_config if obj_config is not None else RagConfig()
    obj_parser = argparse.ArgumentParser(prog="mini-rag", description="A small hybrid-retrieval RAG pipeline.")
    obj_parser.add_argument("--config", dest="s_config", default=self.S_CONFIG_FILE)
    obj_commands = obj_parser.add_subparsers(dest="s_command", required=True)

    obj_ingest = obj_commands.add_parser("ingest", help="load, chunk, embed and save the corpus")
    obj_ingest.add_argument("--corpus", dest="s_corpus", default=obj_used.s_corpus_dir)
    obj_ingest.add_argument("--index-dir", dest="s_index_dir", default=obj_used.s_index_dir)
    obj_ingest.add_argument("--size", dest="i_size", type=int, default=obj_used.i_chunk_size)
    obj_ingest.add_argument("--overlap", dest="i_overlap", type=int, default=obj_used.i_chunk_overlap)

    obj_query = obj_commands.add_parser("query", help="retrieve chunks and answer a question")
    obj_query.add_argument("s_question", metavar="question")
    obj_query.add_argument("--index-dir", dest="s_index_dir", default=obj_used.s_index_dir)
    obj_query.add_argument("--top-k", dest="i_top_k", type=int, default=obj_used.i_top_k)
    obj_query.add_argument("--threshold", dest="f_threshold", type=float, default=obj_used.f_threshold)
    obj_query.add_argument("--generate", dest="b_generate", action="store_true")
    obj_query.add_argument("--brief", dest="b_brief", action="store_true")

    obj_eval = obj_commands.add_parser("eval", help="compare the retrieval methods and sweep the answer gate")
    obj_eval.add_argument("--questions", dest="s_questions", default=obj_used.s_questions)
    obj_eval.add_argument("--corpus", dest="s_corpus", default=obj_used.s_corpus_dir)
    obj_eval.add_argument("--sizes", dest="l_sizes", type=self.parse_sizes, default=list(obj_used.t_eval_sizes))
    obj_eval.add_argument("--overlap", dest="i_overlap", type=int, default=obj_used.i_chunk_overlap)
    obj_eval.add_argument("--gate-top-k", dest="i_gate_top_k", type=int, default=obj_used.i_top_k)
    obj_eval.add_argument("--cache-dir", dest="s_cache_dir", default=obj_used.s_eval_cache_dir)
    obj_eval.add_argument("--out", dest="s_out", default="eval/results.md")
    obj_eval.add_argument("--rebuild", dest="b_rebuild", action="store_true")
    obj_eval.add_argument("--build-only", dest="b_build_only", action="store_true")
    obj_eval.add_argument("--fusion-grid", dest="b_fusion_grid", action="store_true")

    obj_init = obj_commands.add_parser("eval-init", help="write a starter question file")
    obj_init.add_argument("--out", dest="s_out", default=QuestionStarter.S_DEFAULT_PATH)
    obj_init.add_argument("--force", dest="b_force", action="store_true")

    obj_suggest = obj_commands.add_parser("eval-suggest", help="print candidate fact sentences from the corpus")
    obj_suggest.add_argument("--corpus", dest="s_corpus", default=obj_used.s_corpus_dir)
    obj_suggest.add_argument("--limit", dest="i_limit", type=int, default=30)

    obj_check = obj_commands.add_parser("eval-check", help="check a question file against the corpus")
    obj_check.add_argument("--questions", dest="s_questions", default=obj_used.s_questions)
    obj_check.add_argument("--corpus", dest="s_corpus", default=obj_used.s_corpus_dir)
    obj_check.add_argument("--lenient", dest="b_lenient", action="store_true")

    obj_calibrate = obj_commands.add_parser("calibrate", help="sweep the gate threshold for the configured embedder")
    obj_calibrate.add_argument("--questions", dest="s_questions", default=obj_used.s_questions)
    obj_calibrate.add_argument("--corpus", dest="s_corpus", default=obj_used.s_corpus_dir)
    obj_calibrate.add_argument("--cache-dir", dest="s_cache_dir", default=obj_used.s_eval_cache_dir)

    # --config is also accepted after the command; load_config() has already read its value
    for obj_command in (obj_ingest, obj_query, obj_eval, obj_init, obj_suggest, obj_check, obj_calibrate):
      obj_command.add_argument("--config", dest="s_config", default=argparse.SUPPRESS)
    return obj_parser

  def parse_sizes(self, s_sizes: str) -> list[int]:
    """Turn a comma-separated --sizes value such as "500,1000" into a list of ints."""
    try:
      return [int(s_size) for s_size in s_sizes.split(",")]
    except ValueError:
      raise argparse.ArgumentTypeError(f"sizes must be comma-separated whole numbers, got {s_sizes!r}")

  def dense_index(self, obj_config: RagConfig) -> DenseIndex:
    """Return an empty DenseIndex over the injected embedder, or over the configured one under its name."""
    if self.obj_embedder is not None:
      return DenseIndex(obj_embedder=self.obj_embedder)
    obj_embedder = ComponentFactory(obj_config).embedder()
    return DenseIndex(obj_embedder.s_name, obj_embedder)

  def index_store(self, s_cache_dir: str, obj_config: RagConfig) -> EvalIndexStore:
    """Return the eval index cache for the injected embedder, or for the configured one under its name."""
    if self.obj_embedder is not None:
      return EvalIndexStore(s_cache_dir, self.obj_embedder)
    obj_embedder = ComponentFactory(obj_config).embedder()
    return EvalIndexStore(s_cache_dir, obj_embedder, obj_embedder.s_name)

  def ingest(
    self, s_corpus_dir: str, s_index_dir: str, i_size: int, i_overlap: int, obj_config: RagConfig | None = None,
  ) -> int:
    """Load, chunk, embed and save the corpus; print a summary and return an exit code."""
    obj_used = obj_config if obj_config is not None else RagConfig()
    f_start = time.perf_counter()
    try:
      obj_chunker = Chunker(i_size, i_overlap)
      l_docs = Loader(s_corpus_dir, obj_used.t_extensions).load()
      l_chunks = obj_chunker.chunk_documents(l_docs)
      obj_dense = self.dense_index(obj_used)
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

  def query(
    self, s_question: str, s_index_dir: str, i_top_k: int, f_threshold: float, b_generate: bool,
    b_brief: bool = False, obj_config: RagConfig | None = None,
  ) -> int:
    """Retrieve chunks for a question and print them plus the answer; return an exit code."""
    obj_used = obj_config if obj_config is not None else RagConfig()
    try:
      obj_dense = self.dense_index(obj_used)
      obj_dense.load(s_index_dir)
    except FileNotFoundError:
      return self.fail(f"no index found in {s_index_dir}; run ingest first")
    except ValueError as obj_error:
      # a saved index from another embedder, damaged index files, or a bad embedder setting
      return self.fail(str(obj_error))

    try:
      obj_bm25 = Bm25Index()
      obj_bm25.build(obj_dense.l_chunks)
      obj_retriever = HybridRetriever(
        obj_bm25, obj_dense, i_k=obj_used.i_rrf_k,
        f_bm25_weight=obj_used.f_bm25_weight, f_dense_weight=obj_used.f_dense_weight,
      )
      # built before searching so a bad threshold fails early
      obj_extractive = Answerer(f_threshold)
      l_hits = obj_retriever.search(s_question, i_top_k)
    except ValueError as obj_error:
      return self.fail(str(obj_error))

    for i_rank, obj_hit in enumerate(l_hits, start=1):
      self.print_hit(i_rank, obj_hit, b_brief)
    self.print_answer(self.answer(s_question, l_hits, obj_extractive, b_generate, obj_used))
    return self.I_OK

  def eval(
    self, s_questions_path: str, s_corpus_dir: str, l_sizes: list[int], i_overlap: int, i_gate_top_k: int,
    s_cache_dir: str, s_out_path: str, b_rebuild: bool, b_build_only: bool, b_fusion_grid: bool = False,
    obj_config: RagConfig | None = None,
  ) -> int:
    """Run the evaluation and write the report, or only build the index caches; return an exit code.

    With b_fusion_grid, hybrid is also evaluated on the FusionGrid settings and fusion_grid.md is
    written next to the report.
    """
    obj_used = obj_config if obj_config is not None else RagConfig()
    try:
      if b_build_only:
        l_docs = Loader(s_corpus_dir, obj_used.t_extensions).load()
        return self.build_only(self.index_store(s_cache_dir, obj_used), l_docs, l_sizes, i_overlap, b_rebuild)

      l_questions = QuestionLoader().load(s_questions_path)
      l_docs = Loader(s_corpus_dir, obj_used.t_extensions).load()
      obj_validation = QuestionValidator(l_docs).validate(l_questions)
      if obj_validation.l_problems:
        return self.fail_problems(obj_validation.l_problems)

      obj_store = self.index_store(s_cache_dir, obj_used)
      # the grid replaces the configured fusion setting for the extra rows only
      l_settings = FusionGrid().settings() if b_fusion_grid else [self.fusion_setting(obj_used)]
      obj_evaluator = Evaluator(
        obj_store, l_sizes, i_overlap, i_gate_top_k=i_gate_top_k, l_fusion_settings=l_settings,
      )
      if b_rebuild:
        obj_evaluator.prepare(l_docs, b_rebuild=True)
      l_bundles = obj_evaluator.prepare(l_docs)
      if b_fusion_grid:
        self.print_origins(l_bundles)
      obj_result = obj_evaluator.run(l_docs, l_questions, l_bundles)
      obj_writer = ReportWriter(obj_validation.d_overlap)
      print(obj_writer.console_text(obj_result, l_questions))
      obj_writer.write(s_out_path, obj_result, l_questions)
      if b_fusion_grid:
        s_grid_path = str(Path(s_out_path).with_name(self.S_GRID_FILE))
        FusionGridWriter().write(s_grid_path, obj_result)
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))

    print(f"report: {s_out_path}")
    if b_fusion_grid:
      print(f"fusion grid: {s_grid_path}")
    return self.I_OK

  def calibrate(self, s_questions_path: str, s_corpus_dir: str, s_cache_dir: str, obj_config: RagConfig) -> int:
    """Sweep the gate threshold with the configured embedder, chunking, fusion and top-k; return an exit code."""
    try:
      obj_store = self.index_store(s_cache_dir, obj_config)
    except ValueError as obj_error:
      return self.fail(str(obj_error))
    return EvalKit().calibrate(
      obj_store, s_questions_path, s_corpus_dir, obj_config.t_extensions,
      obj_config.i_chunk_size, obj_config.i_chunk_overlap, self.fusion_setting(obj_config), obj_config.i_top_k,
    )

  def fusion_setting(self, obj_config: RagConfig) -> FusionSetting:
    """Return the fusion setting named in the config's [retrieval] section."""
    return FusionSetting(
      i_k=obj_config.i_rrf_k, f_bm25_weight=obj_config.f_bm25_weight, f_dense_weight=obj_config.f_dense_weight,
    )

  def print_origins(self, l_bundles: list[IndexBundle]) -> None:
    """Print, per chunk size, the chunk count and whether the index was loaded from the cache or built."""
    for obj_bundle in l_bundles:
      s_origin = "loaded" if obj_bundle.b_loaded else "built"
      print(f"size {obj_bundle.i_size}: {len(obj_bundle.l_chunks)} chunks, {s_origin}")

  def build_only(
    self, obj_store: EvalIndexStore, l_docs: list[Document], l_sizes: list[int], i_overlap: int, b_rebuild: bool,
  ) -> int:
    """Build or load the index cache of every size and print its chunk count, origin and time."""
    for i_size in l_sizes:
      f_start = time.perf_counter()
      # one evaluator per size, so each size gets its own timing
      obj_bundle = Evaluator(obj_store, [i_size], i_overlap).prepare(l_docs, b_rebuild)[0]
      f_elapsed = time.perf_counter() - f_start
      s_origin = "loaded" if obj_bundle.b_loaded else "built"
      print(f"size {i_size}: {len(obj_bundle.l_chunks)} chunks, {s_origin}, {f_elapsed:.2f} s")
    return self.I_OK

  def fail_problems(self, l_problems: list[str]) -> int:
    """Print the first question-set problems to stderr and return the usage-error exit code."""
    print(f"error: the question set has {len(l_problems)} problems", file=sys.stderr)
    for s_problem in l_problems[:self.I_MAX_PROBLEMS]:
      print(f"  {s_problem}", file=sys.stderr)
    return self.I_USAGE_ERROR

  def answer(
    self, s_question: str, l_hits: list[HybridHit], obj_extractive: Answerer, b_generate: bool,
    obj_config: RagConfig | None = None,
  ) -> Answer:
    """Answer with a model client when asked and possible, otherwise with the extractive Answerer.

    If building or calling the client fails for any reason, a note with the error's type and message
    is printed and the extractive answer is shown.
    """
    try:
      obj_client = self.make_client(obj_config) if b_generate else None
      if obj_client is not None:
        return Answerer(obj_extractive.f_threshold, obj_client).answer(s_question, l_hits)
    except Exception as obj_error:
      # e.g. the anthropic package is missing, the API returned an error, a server did not reply,
      # or a bad generator setting; the clients never put a key value into their messages
      print(f"note: the generator failed ({type(obj_error).__name__}: {obj_error}); showing the extractive answer")
    return obj_extractive.answer(s_question, l_hits)

  def make_client(self, obj_config: RagConfig | None = None) -> object | None:
    """Return the injected client or the configured generator, or None after saying why there is none."""
    if self.obj_llm_client is not None:
      return self.obj_llm_client
    obj_used = obj_config if obj_config is not None else RagConfig()
    if obj_used.s_generator_kind == "none":
      print('note: the generator kind is "none"; showing the extractive answer')
      return None
    if obj_used.s_generator_kind == "anthropic" and not LlmClient.is_available():
      print(f"note: {LlmClient.S_KEY_VARIABLE} is not set; showing the extractive answer")
      return None
    return ComponentFactory(obj_used).generator()

  def print_hit(self, i_rank: int, obj_hit: HybridHit, b_brief: bool = False) -> None:
    """Print one hit: rank, marker, dense, BM25 and fused scores, then the chunk text (cut short if b_brief)."""
    obj_chunk = obj_hit.obj_chunk
    print(
      f"{i_rank}. [{obj_chunk.s_doc_id}#{obj_chunk.i_index}]"
      f"  dense {self.format_score(obj_hit.f_dense_score)}"
      f"  bm25 {self.format_score(obj_hit.f_bm25_score)}"
      f"  fused {obj_hit.f_rrf_score:.4f}"
    )
    s_text = obj_chunk.s_text
    if b_brief and len(s_text) > self.I_BRIEF_LENGTH:
      s_text = s_text[:self.I_BRIEF_LENGTH] + "..."
    print(s_text)
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
