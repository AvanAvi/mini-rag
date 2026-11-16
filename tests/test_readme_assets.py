"""Tests for the README hero SVGs and the README's image references."""

from pathlib import Path

import pytest
from readme_asset_checker import ReadmeAssetChecker


class TestReadmeAssets:
  """Checks the real assets and README, then one hand-made bad case per rule."""

  S_ROOT = str(Path(__file__).resolve().parents[1])
  S_GOOD_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="360" viewBox="0 0 1280 360" role="img">'
    "<title>name</title><desc>a test picture</desc>"
    '<rect width="1280" height="360" fill="#ffffff"/>'
    "<text x=\"64\" y=\"116\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\" font-size=\"20\">"
    "name</text></svg>"
  )

  def write(self, obj_tmp: Path, s_svg: str) -> str:
    """Write s_svg to a file in obj_tmp and return its path."""
    obj_path = obj_tmp / "hero.svg"
    obj_path.write_text(s_svg, encoding="utf-8")
    return str(obj_path)

  def problems(self, obj_tmp: Path, s_old: str, s_new: str) -> list[str]:
    """Return the problems of the good SVG with s_old replaced by s_new."""
    assert s_old in self.S_GOOD_SVG
    return ReadmeAssetChecker().check_svg(self.write(obj_tmp, self.S_GOOD_SVG.replace(s_old, s_new)))

  def write_readme(self, obj_tmp: Path, s_readme: str, l_assets: list[str]) -> str:
    """Write a README and empty asset files under obj_tmp and return the README path."""
    obj_assets = obj_tmp / "assets" / "readme"
    obj_assets.mkdir(parents=True)
    for s_name in l_assets:
      (obj_assets / s_name).write_text("<svg/>", encoding="utf-8")
    obj_readme = obj_tmp / "README.md"
    obj_readme.write_text(s_readme, encoding="utf-8")
    return str(obj_readme)

  @pytest.mark.parametrize("s_name", ["hero.svg", "hero-dark.svg"])
  def test_real_svgs_pass(self, s_name: str) -> None:
    """Both shipped hero files meet every limit."""
    assert ReadmeAssetChecker().check_svg(str(Path(self.S_ROOT) / "assets" / "readme" / s_name)) == []

  def test_real_readme_passes(self) -> None:
    """Every image reference in README.md resolves and every asset is referenced."""
    assert ReadmeAssetChecker().check_readme_images(str(Path(self.S_ROOT) / "README.md"), self.S_ROOT) == []

  def test_good_svg_passes(self, tmp_path: Path) -> None:
    """The hand-made base case used by the negative tests has no problem."""
    assert ReadmeAssetChecker().check_svg(self.write(tmp_path, self.S_GOOD_SVG)) == []

  def test_script_element(self, tmp_path: Path) -> None:
    """A <script> element is reported."""
    l_problems = self.problems(tmp_path, "</svg>", "<script>x()</script></svg>")
    assert "contains a forbidden <script> element" in l_problems

  def test_external_href(self, tmp_path: Path) -> None:
    """An attribute holding an external URL is reported."""
    l_problems = self.problems(tmp_path, '<rect width', '<rect href="https://example.org/x.png" width')
    assert any("holds an external URL" in s_problem for s_problem in l_problems)

  def test_missing_title(self, tmp_path: Path) -> None:
    """An SVG whose first child is not a title is reported."""
    assert "child 1 is not a title element" in self.problems(tmp_path, "<title>name</title>", "")

  def test_oversized_file(self, tmp_path: Path) -> None:
    """A file above 20 KB is reported."""
    s_padding = '<rect width="1" height="1" fill="#ffffff"/>' * 600
    assert any("larger than" in s_problem for s_problem in self.problems(tmp_path, "</svg>", s_padding + "</svg>"))

  def test_small_font(self, tmp_path: Path) -> None:
    """A font-size below 18 is reported."""
    assert "font-size 16 is below 18" in self.problems(tmp_path, 'font-size="20"', 'font-size="16"')

  def test_absolute_path(self, tmp_path: Path) -> None:
    """A Windows or POSIX absolute path in the file is reported."""
    l_problems = self.problems(tmp_path, "a test picture", "made in /home/someone/work")
    assert any("absolute path" in s_problem for s_problem in l_problems)

  def test_digit_in_text(self, tmp_path: Path) -> None:
    """A digit inside a <text> element is reported."""
    assert "a <text> element contains a digit" in self.problems(tmp_path, ">name</text>", ">name 2</text>")

  def test_missing_viewbox(self, tmp_path: Path) -> None:
    """An SVG without the 0 0 1280 360 viewBox is reported."""
    assert "viewBox is not '0 0 1280 360'" in self.problems(tmp_path, ' viewBox="0 0 1280 360"', "")

  def test_unreferenced_asset(self, tmp_path: Path) -> None:
    """A file in assets/readme that the README doesn't reference is reported."""
    s_readme = self.write_readme(
      tmp_path, '<img src="assets/readme/hero.svg" alt="the hero">\n', ["hero.svg", "spare.svg"],
    )
    assert ReadmeAssetChecker().check_readme_images(s_readme, str(tmp_path)) == [
      "assets/readme/spare.svg is not referenced by the README",
    ]

  def test_missing_alt(self, tmp_path: Path) -> None:
    """An <img> or Markdown image without alt text is reported."""
    s_readme = self.write_readme(
      tmp_path, '<img src="assets/readme/hero.svg">\n![](assets/readme/hero.svg)\n', ["hero.svg"],
    )
    l_problems = ReadmeAssetChecker().check_readme_images(s_readme, str(tmp_path))
    assert len(l_problems) == 2
    assert all("without alt text" in s_problem for s_problem in l_problems)

  def test_missing_image_file(self, tmp_path: Path) -> None:
    """A srcset or src that names a file that doesn't exist is reported."""
    s_readme = self.write_readme(
      tmp_path, '<source srcset="assets/readme/gone.svg">\n<img src="assets/readme/hero.svg" alt="x">\n', ["hero.svg"],
    )
    assert ReadmeAssetChecker().check_readme_images(s_readme, str(tmp_path)) == [
      "image reference 'assets/readme/gone.svg' is not an existing file under the project",
    ]
