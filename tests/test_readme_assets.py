"""Tests for the README banner and badge SVGs and the README's image references."""

from pathlib import Path

import pytest
from readme_asset_checker import ReadmeAssetChecker


class TestReadmeAssets:
  """Checks the real assets and README, then one hand-made bad case per rule."""

  S_ROOT = str(Path(__file__).resolve().parents[1])
  S_GOOD_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="520" viewBox="0 0 1280 520" role="img">'
    "<title>name</title><desc>a test picture</desc>"
    '<rect width="1280" height="520" fill="#ffffff"/>'
    "<text x=\"64\" y=\"116\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\" font-size=\"20\">"
    "name</text></svg>"
  )

  def write(self, obj_tmp: Path, s_svg: str, s_name: str = "banner.svg") -> str:
    """Write s_svg to a file named s_name in obj_tmp and return its path."""
    obj_path = obj_tmp / s_name
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

  def asset(self, s_name: str) -> str:
    """Return the path of a shipped file in assets/readme."""
    return str(Path(self.S_ROOT) / "assets" / "readme" / s_name)

  def write_badges(self, obj_tmp: Path, l_pairs: list[tuple[str, str]]) -> str:
    """Write a badges.svg holding one label and value text per pair and return its path."""
    s_texts = "".join(f"<text>{s_label}</text><text>{s_value}</text>" for s_label, s_value in l_pairs)
    s_svg = self.S_GOOD_SVG.replace('viewBox="0 0 1280 520"', 'viewBox="0 0 1280 96"')
    return self.write(obj_tmp, s_svg.split("<text ")[0] + s_texts + "</svg>", "badges.svg")

  def table_pairs(self) -> list[tuple[str, str]]:
    """Return the (label, value) pairs of the provenance table."""
    return [(t_row[0], t_row[1]) for t_row in ReadmeAssetChecker.T_BADGE_PROVENANCE]

  @pytest.mark.parametrize("s_name", ["banner.svg", "badges.svg"])
  def test_real_svgs_pass(self, s_name: str) -> None:
    """Both shipped files meet every limit."""
    assert ReadmeAssetChecker().check_svg(self.asset(s_name)) == []

  def test_real_banner_text_is_traceable(self) -> None:
    """Every text string in banner.svg occurs in a repository source file."""
    assert ReadmeAssetChecker().check_traceability(self.asset("banner.svg"), self.S_ROOT) == []

  def test_tagline_tspans_join_to_the_description(self) -> None:
    """The four tagline tspans, joined in order, equal the 93-character description."""
    l_parts = ReadmeAssetChecker().tspan_texts(self.asset("banner.svg"), "tagline")
    s_description = "answers questions from a local corpus by combining keyword search with dense embedding search"
    assert len(l_parts) == 4
    assert "".join(l_parts) == s_description
    assert len(s_description) == 93

  def test_pipeline_follows_the_flow_diagram(self) -> None:
    """The banner's stages follow the order of the Mermaid diagram in docs/flow.md."""
    s_flow = str(Path(self.S_ROOT) / "docs" / "flow.md")
    obj_checker = ReadmeAssetChecker()
    assert obj_checker.check_stage_order(self.asset("banner.svg"), s_flow) == []
    l_stages = [s_part.strip() for s_part in obj_checker.tspan_texts(self.asset("banner.svg"), "pipeline")[0::2]]
    assert l_stages == ["Text files", "Chunker", "BM25 search", "Dense search", "RankFuser", "Answerer"]

  def test_pipeline_out_of_order(self, tmp_path: Path) -> None:
    """Swapping two stages joined by › is reported."""
    s_svg = Path(self.asset("banner.svg")).read_text(encoding="utf-8")
    s_svg = s_svg.replace("<tspan>RankFuser</tspan>", "<tspan>@</tspan>").replace("<tspan>Answerer</tspan>", "<tspan>RankFuser</tspan>")
    s_path = self.write(tmp_path, s_svg.replace("<tspan>@</tspan>", "<tspan>Answerer</tspan>"))
    assert ReadmeAssetChecker().check_stage_order(s_path, str(Path(self.S_ROOT) / "docs" / "flow.md")) == [
      "'Answerer' does not come before 'RankFuser' in the diagram",
    ]

  def test_real_badges_match_the_provenance_table(self) -> None:
    """The badge pairs equal the provenance table and every source file holds its substring."""
    assert ReadmeAssetChecker().check_provenance(self.asset("badges.svg"), self.S_ROOT) == []

  def test_real_readme_passes(self) -> None:
    """Every image reference in README.md resolves and every asset is referenced."""
    assert ReadmeAssetChecker().check_readme_images(str(Path(self.S_ROOT) / "README.md"), self.S_ROOT) == []

  def test_no_reference_to_the_old_hero(self) -> None:
    """README.md and the docs no longer name hero.svg or hero-dark.svg."""
    l_files = [Path(self.S_ROOT) / "README.md", *sorted((Path(self.S_ROOT) / "docs").glob("*.md"))]
    for obj_file in l_files:
      s_text = obj_file.read_text(encoding="utf-8")
      assert "hero.svg" not in s_text and "hero-dark.svg" not in s_text, obj_file.name

  def test_string_absent_from_every_file(self, tmp_path: Path) -> None:
    """A text string found in no repository file is reported; separator strings are exempt."""
    s_path = self.write(tmp_path, self.S_GOOD_SVG.replace(">name</text>", "><tspan>zebra quartz</tspan><tspan> › </tspan></text>"))
    assert ReadmeAssetChecker().check_traceability(s_path, self.S_ROOT) == [
      "text 'zebra quartz' occurs in no repository source file",
    ]

  def test_badge_missing_from_the_table(self, tmp_path: Path) -> None:
    """A badge pair that isn't in the provenance table is reported."""
    s_path = self.write_badges(tmp_path, self.table_pairs() + [("stars", "many")])
    assert any("differ from the provenance table" in s_problem for s_problem in ReadmeAssetChecker().check_provenance(s_path, self.S_ROOT))

  def test_table_substring_missing(self, tmp_path: Path) -> None:
    """A table entry whose substring is missing from its source file is reported."""
    s_path = self.write_badges(tmp_path, [("license", "MIT")])
    (tmp_path / "LICENSE").write_text("Some other licence", encoding="utf-8")
    t_table = (("license", "MIT", "LICENSE", "MIT License"),)
    assert ReadmeAssetChecker().check_provenance(s_path, str(tmp_path), t_table) == [
      "badge license MIT: LICENSE does not contain 'MIT License'",
    ]

  def test_animate_element(self, tmp_path: Path) -> None:
    """An <animate> element is reported."""
    l_problems = self.problems(tmp_path, "</svg>", '<animate attributeName="x" dur="1s"/></svg>')
    assert "contains a forbidden <animate> element" in l_problems

  def test_css_import(self, tmp_path: Path) -> None:
    """A CSS @import is reported."""
    assert "contains a CSS @import" in self.problems(tmp_path, "</svg>", "<style>@import 'x.css';</style></svg>")

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
    """A file above 40 KB is reported."""
    s_padding = '<rect width="1" height="1" fill="#ffffff"/>' * 1000
    assert any("larger than" in s_problem for s_problem in self.problems(tmp_path, "</svg>", s_padding + "</svg>"))

  def test_small_font(self, tmp_path: Path) -> None:
    """A font-size below 18 is reported."""
    assert "font-size 17 is below 18" in self.problems(tmp_path, 'font-size="20"', 'font-size="17"')

  def test_absolute_path(self, tmp_path: Path) -> None:
    """A Windows or POSIX absolute path in the file is reported."""
    l_problems = self.problems(tmp_path, "a test picture", "made in /home/someone/work")
    assert any("absolute path" in s_problem for s_problem in l_problems)

  def test_missing_viewbox(self, tmp_path: Path) -> None:
    """A banner without the 0 0 1280 520 viewBox is reported."""
    assert "viewBox is not '0 0 1280 520'" in self.problems(tmp_path, ' viewBox="0 0 1280 520"', "")

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
