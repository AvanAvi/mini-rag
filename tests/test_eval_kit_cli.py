"""Tests for the eval-init, eval-suggest, eval-check and calibrate commands and for .md corpus files."""

from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from calibration_set_builder import CalibrationSetBuilder

from mini_rag.cli import Cli
from mini_rag.loader import Loader
from mini_rag.question_loader import QuestionLoader
from mini_rag.question_validator import QuestionValidator


class TestEvalKitCli:
  """Runs each eval kit command on tiny inputs."""

  def cli(self) -> Cli:
    """Return a Cli whose embedder counts the calibration vocabulary."""
    return Cli(obj_embedder=BagOfWordsEmbedder(CalibrationSetBuilder.L_VOCABULARY))

  def test_eval_init_writes_a_loadable_starter_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval-init writes one answerable and one unanswerable example, each with a comment, using LF."""
    obj_path = tmp_path / "eval" / "my_questions.json"
    assert self.cli().run(["eval-init", "--out", str(obj_path)]) == 0
    assert str(obj_path) in capsys.readouterr().out
    l_questions = QuestionLoader().load(str(obj_path))
    assert [obj_question.b_answerable for obj_question in l_questions] == [True, False]
    s_text = obj_path.read_bytes().decode("utf-8")
    assert "\r" not in s_text
    assert s_text.count('"comment":') == 2

  def test_eval_init_default_path(self) -> None:
    """Without --out the starter file goes to eval/my_questions.json."""
    assert Cli().build_parser().parse_args(["eval-init"]).s_out == "eval/my_questions.json"

  def test_eval_init_refuses_to_overwrite_without_force(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A second eval-init leaves the file alone and fails; --force replaces it."""
    obj_path = tmp_path / "my_questions.json"
    obj_path.write_text("my own work", encoding="utf-8")
    assert self.cli().run(["eval-init", "--out", str(obj_path)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert obj_path.read_text(encoding="utf-8") == "my own work"
    assert self.cli().run(["eval-init", "--out", str(obj_path), "--force"]) == 0
    assert len(QuestionLoader().load(str(obj_path))) == 2

  def test_eval_suggest_prints_doc_id_and_sentence(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval-suggest prints "doc_id | sentence" lines and respects --limit."""
    obj_corpus = tmp_path / "corpus"
    obj_corpus.mkdir()
    s_first = "The plant was connected to the grid in 1984 after several years of construction work."
    s_second = "A second unit followed in 1991 and ran until the whole site was closed for good."
    (obj_corpus / "plant.txt").write_text(f"== History ==\n{s_first}\n{s_second}\n== References ==\n{s_first[:-1]} again in 1999.\n", encoding="utf-8")
    assert self.cli().run(["eval-suggest", "--corpus", str(obj_corpus)]) == 0
    assert capsys.readouterr().out.split("\n")[:-1] == [f"plant | {s_first}", f"plant | {s_second}"]
    assert self.cli().run(["eval-suggest", "--corpus", str(obj_corpus), "--limit", "1"]) == 0
    assert capsys.readouterr().out.split("\n")[:-1] == [f"plant | {s_first}"]
    assert Cli().build_parser().parse_args(["eval-suggest"]).i_limit == 30

  def test_eval_check_is_strict_by_default_and_lenient_on_request(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Four good questions fail the set-level rules by default and pass with --lenient."""
    obj_builder = CalibrationSetBuilder()
    l_args = [
      "eval-check", "--questions", obj_builder.write_questions(tmp_path / "q.json"),
      "--corpus", obj_builder.write_corpus(tmp_path / "corpus"),
    ]
    assert self.cli().run(l_args) == 2
    s_out = capsys.readouterr().out
    assert "questions: 2 answerable, 2 unanswerable" in s_out
    assert "set: 2 answerable questions (need 24 to 28)" in s_out
    assert "problems: 0" not in s_out

    assert self.cli().run(l_args + ["--lenient"]) == 0
    s_out = capsys.readouterr().out
    assert "problems: 0" in s_out and "set:" not in s_out

  def test_lenient_still_reports_per_question_problems(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The untouched starter file fails eval-check --lenient, because its source document does not exist."""
    obj_builder = CalibrationSetBuilder()
    s_questions = str(tmp_path / "starter.json")
    assert self.cli().run(["eval-init", "--out", s_questions]) == 0
    capsys.readouterr()
    l_args = ["eval-check", "--lenient", "--questions", s_questions, "--corpus", obj_builder.write_corpus(tmp_path / "corpus")]
    assert self.cli().run(l_args) == 2
    assert "q01: source_doc 'your_document_name' is not a loaded document" in capsys.readouterr().out

  def test_validator_lenient_flag_only_drops_set_problems(self) -> None:
    """validate() with b_lenient returns the same result minus the problems that start with "set:"."""
    obj_builder = CalibrationSetBuilder()
    obj_validator = QuestionValidator(obj_builder.documents())
    obj_strict = obj_validator.validate(obj_builder.questions())
    obj_lenient = obj_validator.validate(obj_builder.questions(), b_lenient=True)
    assert obj_strict.l_problems and all(s_problem.startswith("set:") for s_problem in obj_strict.l_problems)
    assert obj_lenient.l_problems == []
    assert obj_lenient.d_overlap == obj_strict.d_overlap

  def test_calibrate_prints_the_sweep_and_the_config_line(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """calibrate warns about the small set, prints the table and the line for the config, and edits nothing."""
    obj_builder = CalibrationSetBuilder()
    obj_config = Path(__file__).resolve().parent.parent / "mini-rag.toml"
    obj_before = obj_config.read_bytes()
    l_args = [
      "calibrate", "--questions", obj_builder.write_questions(tmp_path / "q.json"),
      "--corpus", obj_builder.write_corpus(tmp_path / "corpus"), "--cache-dir", str(tmp_path / "cache"),
    ]
    assert self.cli().run(l_args) == 0
    l_lines = capsys.readouterr().out.split("\n")
    assert l_lines[0].startswith("warning: only 2 answerable questions")
    assert l_lines[1].startswith("warning: only 2 unanswerable questions")
    assert l_lines[2] == "size 500: 5 chunks, built"
    assert l_lines[3].endswith("k=60 w=1.0/1.0, top-k 5")
    assert "| Threshold | Answerable rejected | Unanswerable accepted | Total errors |" in l_lines
    assert "| 0.55 | 0/2 | 1/2 | 1 |" in l_lines
    assert "| 0.60 | 1/2 | 0/2 | 1 |" in l_lines
    # the lowest of the tied thresholds is the one offered
    assert l_lines[l_lines.index("threshold = 0.05") - 1].endswith("under [gate] in your config file:")

    assert self.cli().run(l_args) == 0
    assert "size 500: 5 chunks, loaded" in capsys.readouterr().out
    assert obj_config.read_bytes() == obj_before

  def test_calibrate_stops_on_question_problems(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """calibrate refuses a question file with per-question problems."""
    obj_builder = CalibrationSetBuilder()
    s_questions = str(tmp_path / "starter.json")
    assert self.cli().run(["eval-init", "--out", s_questions]) == 0
    l_args = ["calibrate", "--questions", s_questions, "--corpus", obj_builder.write_corpus(tmp_path / "corpus"), "--cache-dir", str(tmp_path / "cache")]
    assert self.cli().run(l_args) == 2
    assert "run eval-check" in capsys.readouterr().err

  def test_loader_reads_markdown_when_asked(self, tmp_path: Path) -> None:
    """With ".md" among the extensions, .md files load like .txt files; by default they are skipped."""
    (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "b.md").write_bytes("# Title\r\nmarkdown text".encode("utf-8"))
    l_docs = Loader(str(tmp_path), (".txt", ".md")).load()
    assert [(obj_doc.s_doc_id, obj_doc.s_text) for obj_doc in l_docs] == [("a", "plain text"), ("b", "# Title\nmarkdown text")]
    assert [obj_doc.s_doc_id for obj_doc in Loader(str(tmp_path)).load()] == ["a"]

  def test_loader_rejects_two_files_with_one_name(self, tmp_path: Path) -> None:
    """notes.txt and notes.md would share a document id, so loading both raises ValueError."""
    (tmp_path / "notes.txt").write_text("one", encoding="utf-8")
    (tmp_path / "notes.md").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="share the name 'notes'"):
      Loader(str(tmp_path), (".txt", ".md")).load()

  def test_markdown_corpus_through_the_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A config with extensions = [".md"] makes ingest read a folder of .md files."""
    s_corpus = CalibrationSetBuilder().write_corpus(tmp_path / "corpus", ".md")
    obj_config = tmp_path / "md.toml"
    obj_config.write_text(f"[corpus]\ndir = '{s_corpus}'\nextensions = ['.md']\n[index]\ndir = '{tmp_path / 'index'}'\n", encoding="utf-8")
    assert self.cli().run(["--config", str(obj_config), "ingest"]) == 0
    assert "documents: 5" in capsys.readouterr().out
