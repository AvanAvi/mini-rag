"""Tests for the --fusion-grid option of the eval command."""

from pathlib import Path

import pytest
from bag_of_words_embedder import BagOfWordsEmbedder
from counting_embedder import CountingEmbedder
from eval_set_builder import EvalSetBuilder

from mini_rag.cli import Cli


class TestCliFusionGrid:
  """Runs eval with and without the grid on the small eval set."""

  def base_args(self, obj_tmp: Path) -> list[str]:
    """Write the corpus and the questions once and return the eval arguments without --out."""
    obj_builder = EvalSetBuilder()
    s_corpus = obj_builder.write_corpus(obj_tmp / "corpus")
    s_questions = obj_builder.write_questions(obj_tmp / "questions.json", obj_builder.entries())
    return [
      "eval", "--questions", s_questions, "--corpus", s_corpus, "--sizes", "60,120", "--overlap", "10",
      "--cache-dir", str(obj_tmp / "cache"),
    ]

  def report_lines(self, obj_path: Path) -> list[str]:
    """Return the report's lines without the date line."""
    l_lines = obj_path.read_text(encoding="utf-8").split("\n")
    return [s_line for s_line in l_lines if not s_line.startswith("- Date (UTC)")]

  def test_grid_run_loads_the_cache_and_keeps_the_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """After a plain run, a grid run embeds nothing, says "loaded" and writes the same results.md."""
    obj_embedder = CountingEmbedder(BagOfWordsEmbedder(EvalSetBuilder.L_VOCABULARY))
    obj_cli = Cli(obj_embedder=obj_embedder)
    l_args = self.base_args(tmp_path)

    assert obj_cli.run(l_args + ["--out", str(tmp_path / "plain" / "results.md")]) == 0
    s_plain_out = capsys.readouterr().out
    i_calls_after_build = obj_embedder.i_embed_calls
    assert i_calls_after_build > 0

    assert obj_cli.run(l_args + ["--out", str(tmp_path / "grid" / "results.md"), "--fusion-grid"]) == 0
    s_grid_out = capsys.readouterr().out
    assert obj_embedder.i_embed_calls == i_calls_after_build
    assert s_grid_out.count(", loaded") == 2
    assert "built" not in s_grid_out
    assert f"fusion grid: {tmp_path / 'grid' / 'fusion_grid.md'}" in s_grid_out

    # the main report does not depend on the grid
    assert self.report_lines(tmp_path / "grid" / "results.md") == self.report_lines(tmp_path / "plain" / "results.md")
    # the plain run prints no cache or grid lines and writes no grid file
    assert "loaded" not in s_plain_out and "fusion grid" not in s_plain_out
    assert not (tmp_path / "plain" / "fusion_grid.md").exists()

  def test_grid_file_has_twelve_rows_per_size(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """fusion_grid.md holds one table per chunk size with 12 settings and one marked default."""
    obj_cli = Cli(obj_embedder=BagOfWordsEmbedder(EvalSetBuilder.L_VOCABULARY))
    assert obj_cli.run(self.base_args(tmp_path) + ["--out", str(tmp_path / "results.md"), "--fusion-grid"]) == 0
    capsys.readouterr()

    obj_bytes = (tmp_path / "fusion_grid.md").read_bytes()
    assert b"\r" not in obj_bytes
    l_lines = obj_bytes.decode("utf-8").split("\n")
    assert len([s_line for s_line in l_lines if s_line.startswith("## Chunk size ")]) == 2
    assert len([s_line for s_line in l_lines if s_line.startswith("| k=")]) == 24
    assert len([s_line for s_line in l_lines if s_line.startswith("| k=60 w=1.0/1.0 (default) |")]) == 2
    assert any("a gain of 2 questions or fewer on hit@5 is noise" in s_line for s_line in l_lines)
