"""Simple, deterministic word tokeniser."""

import re


class Tokenizer:
  """Lowercases text and splits it into runs of ASCII letters and digits."""

  def __init__(self) -> None:
    """Compile the token pattern once."""
    self.obj_pattern = re.compile(r"[a-z0-9]+")

  def tokenize(self, s_text: str) -> list[str]:
    """Return the lowercase letter/digit tokens of the text, in order."""
    return self.obj_pattern.findall(s_text.lower())
