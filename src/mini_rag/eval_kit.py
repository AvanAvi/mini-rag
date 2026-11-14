"""The commands that help a user build and calibrate an evaluation set of their own."""

import sys

from mini_rag.calibrator import Calibrator
from mini_rag.eval_index_store import EvalIndexStore
from mini_rag.fusion_setting import FusionSetting
from mini_rag.loader import Loader
from mini_rag.question_loader import QuestionLoader
from mini_rag.question_starter import QuestionStarter
from mini_rag.question_validator import QuestionValidator
from mini_rag.report_tables import ReportTables
from mini_rag.sentence_suggester import SentenceSuggester


class EvalKit:
  """Runs eval-init, eval-suggest, eval-check and calibrate; each method prints and returns an exit code."""

  I_OK = 0
  I_USAGE_ERROR = 2

  def init(self, s_path: str, b_force: bool) -> int:
    """Write the starter question file and say where it is."""
    try:
      QuestionStarter().write(s_path, b_force)
    except FileExistsError as obj_error:
      return self.fail(str(obj_error))
    print(f"wrote {s_path} with 2 example entries; replace them with your own questions")
    return self.I_OK

  def suggest(self, s_corpus_dir: str, t_extensions: tuple[str, ...], i_limit: int) -> int:
    """Print candidate fact sentences from the corpus as "doc_id | sentence"."""
    try:
      l_pairs = SentenceSuggester(i_limit).suggest(Loader(s_corpus_dir, t_extensions).load())
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))
    for s_doc_id, s_sentence in l_pairs:
      print(f"{s_doc_id} | {s_sentence}")
    return self.I_OK

  def check(self, s_questions_path: str, s_corpus_dir: str, t_extensions: tuple[str, ...], b_lenient: bool) -> int:
    """Validate a question file against the corpus; print the counts and every problem."""
    try:
      l_questions = QuestionLoader().load(s_questions_path)
      l_docs = Loader(s_corpus_dir, t_extensions).load()
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))
    l_problems = QuestionValidator(l_docs).validate(l_questions, b_lenient).l_problems
    i_answerable = sum(1 for obj_question in l_questions if obj_question.b_answerable)
    print(f"questions: {i_answerable} answerable, {len(l_questions) - i_answerable} unanswerable")
    print(f"checks: {'per-question only (--lenient)' if b_lenient else 'per-question and set-level'}")
    print(f"problems: {len(l_problems)}")
    for s_problem in l_problems:
      print(f"  {s_problem}")
    return self.I_OK if not l_problems else self.I_USAGE_ERROR

  def calibrate(
    self, obj_store: EvalIndexStore, s_questions_path: str, s_corpus_dir: str, t_extensions: tuple[str, ...],
    i_size: int, i_overlap: int, obj_setting: FusionSetting, i_top_k: int,
  ) -> int:
    """Sweep the gate threshold on a question file and print the line to put in the config file."""
    try:
      l_questions = QuestionLoader().load(s_questions_path)
      l_docs = Loader(s_corpus_dir, t_extensions).load()
      l_problems = QuestionValidator(l_docs).validate(l_questions, b_lenient=True).l_problems
      if l_problems:
        print(f"error: the question set has {len(l_problems)} problems; run eval-check to see them", file=sys.stderr)
        return self.I_USAGE_ERROR
      obj_calibrator = Calibrator(obj_store, i_size, i_overlap, obj_setting, i_top_k)
      obj_bundle = obj_calibrator.prepare(l_docs)
      obj_size = obj_calibrator.run(obj_bundle, l_questions)
    except (FileNotFoundError, ValueError) as obj_error:
      return self.fail(str(obj_error))

    for s_warning in obj_calibrator.warnings(l_questions):
      print(s_warning)
    s_origin = "loaded" if obj_bundle.b_loaded else "built"
    print(f"size {obj_bundle.i_size}: {len(obj_bundle.l_chunks)} chunks, {s_origin}")
    print(f"Threshold sweep for {obj_store.s_model_name}, {obj_setting.s_label}, top-k {i_top_k}")
    for s_line in ReportTables().sweep(obj_size):
      print(s_line)
    l_best = obj_calibrator.best_thresholds(obj_size)
    if l_best:
      s_warning = obj_calibrator.separation_warning(obj_size, l_best[0])
      if s_warning is not None:
        print(s_warning)
      print("To use the lowest of these thresholds, put this line under [gate] in your config file:")
      print(f"threshold = {l_best[0]:.2f}")
      print("The value is tuned on the questions it was measured on, so expect more errors on new questions.")
    return self.I_OK

  def fail(self, s_message: str) -> int:
    """Print an error line to stderr and return the usage-error exit code."""
    print(f"error: {s_message}", file=sys.stderr)
    return self.I_USAGE_ERROR
