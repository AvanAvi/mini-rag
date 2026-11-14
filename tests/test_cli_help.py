"""Tests that every command's --help is readable: help on every option and no upper-case dest names."""

import argparse
import re

import pytest

from mini_rag.cli import Cli


class TestCliHelp:
  """Runs --help for each command through Cli.run and checks the parser's option actions."""

  L_COMMANDS = ["ingest", "query", "eval", "eval-init", "eval-suggest", "eval-check", "calibrate"]
  S_DEST_PATTERN = r"^[A-Z]_[A-Z_]+$"  # a raw dest name such as S_INDEX_DIR shown as a metavar

  @pytest.mark.parametrize("s_command", L_COMMANDS)
  def test_help_output(self, s_command: str, capsys: pytest.CaptureFixture[str]) -> None:
    """--help exits with 0 and prints no token that looks like a raw dest name."""
    assert Cli().run([s_command, "--help"]) == 0
    s_out = capsys.readouterr().out
    assert "usage: mini-rag " + s_command in s_out
    l_bad = [s_token for s_token in s_out.split() if re.match(self.S_DEST_PATTERN, s_token.strip("[],."))]
    assert l_bad == []

  @pytest.mark.parametrize("s_command", L_COMMANDS)
  def test_every_option_has_help(self, s_command: str) -> None:
    """Every action of the command's parser, including positionals, has a non-empty help text."""
    obj_parser = self.subparser(Cli().build_parser(), s_command)
    for obj_action in obj_parser._actions:
      assert obj_action.help, f"{s_command}: {obj_action.option_strings or obj_action.dest} has no help"

  def test_top_level_help(self, capsys: pytest.CaptureFixture[str]) -> None:
    """The top-level --help exits with 0, lists the commands and shows no raw dest name."""
    assert Cli().run(["--help"]) == 0
    s_out = capsys.readouterr().out
    assert "eval-suggest" in s_out
    assert not [s_token for s_token in s_out.split() if re.match(self.S_DEST_PATTERN, s_token.strip("[],."))]

  def subparser(self, obj_parser: argparse.ArgumentParser, s_command: str) -> argparse.ArgumentParser:
    """Return the subparser registered for s_command."""
    for obj_action in obj_parser._actions:
      if isinstance(obj_action, argparse._SubParsersAction):
        return obj_action.choices[s_command]
    raise AssertionError("the parser has no subcommands")
