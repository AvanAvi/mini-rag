"""Test helper that checks the README banner and badge SVGs and the image references in README.md."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path


class ReadmeAssetChecker:
  """Returns a list of problems for an SVG asset or for the images a README references; empty means it passes."""

  S_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
  D_VIEWBOXES = {"banner.svg": "0 0 1280 520", "badges.svg": "0 0 1280 96"}
  I_MAX_BYTES = 40 * 1024
  F_MIN_FONT_SIZE = 18.0
  T_FORBIDDEN_ELEMENTS = (
    "script", "foreignObject", "image", "a", "metadata", "animate", "set", "animateMotion", "animateTransform",
  )
  S_EXEMPT_CHARACTERS = "$>›+ "  # a string made only of these is a separator, not a claim
  T_SOURCE_FILES = ("README.md", "CONTRIBUTING.md", "mini-rag.toml", "pyproject.toml", ".python-version", "LICENSE")
  S_SOURCE_GLOB = "docs/*.md"
  # badge label, badge value, file the value comes from, substring that must be in that file
  T_BADGE_PROVENANCE = (
    ("python", "3.13", ".python-version", "3.13"),
    ("license", "MIT", "LICENSE", "MIT License"),
    ("embeddings", "fastembed", "pyproject.toml", "fastembed"),
    ("keyword", "rank-bm25", "pyproject.toml", "rank-bm25"),
    ("vectors", "numpy", "pyproject.toml", "numpy"),
    ("model", "bge-small-en-v1.5", "mini-rag.toml", "bge-small-en-v1.5"),
    ("config", "TOML", "README.md", "mini-rag.toml"),
    ("platform", "Windows", "README.md", "Windows PowerShell 5.1"),
  )
  T_FONT_STACKS = (
    "system-ui, -apple-system, 'Segoe UI', sans-serif",
    "ui-monospace, Consolas, monospace",
  )
  T_ABSOLUTE_PATTERNS = (r"[A-Za-z]:\\", r"/Users/", r"/home/")
  S_ASSET_DIR = "assets/readme"

  def check_svg(self, s_path: str) -> list[str]:
    """Return every way the SVG at s_path breaks the README hero limits."""
    obj_path = Path(s_path)
    l_problems: list[str] = []
    if obj_path.stat().st_size > self.I_MAX_BYTES:
      l_problems.append(f"file is larger than {self.I_MAX_BYTES} bytes")
    s_raw = obj_path.read_text(encoding="utf-8")
    l_problems.extend(self.raw_text_problems(s_raw))
    try:
      obj_root = ET.fromstring(s_raw)
    except ET.ParseError as obj_error:
      return l_problems + [f"not well-formed XML: {obj_error}"]
    l_problems.extend(self.root_problems(obj_root, self.D_VIEWBOXES.get(obj_path.name)))
    for obj_element in obj_root.iter():
      l_problems.extend(self.element_problems(obj_element))
    return l_problems

  def raw_text_problems(self, s_raw: str) -> list[str]:
    """Return the problems visible in the file text: comments, @import, CSS animation, absolute paths, rasters, fonts."""
    l_problems = []
    if "<!--" in s_raw:
      l_problems.append("contains an XML comment")
    if "@import" in s_raw:
      l_problems.append("contains a CSS @import")
    if re.search(r"@keyframes|\b(animation|transition)(-[a-z]+)?\s*:", s_raw):
      l_problems.append("contains a CSS animation or transition")
    if "data:image" in s_raw:
      l_problems.append("embeds a raster image")
    for s_pattern in self.T_ABSOLUTE_PATTERNS:
      if re.search(s_pattern, s_raw):
        l_problems.append(f"contains an absolute path matching {s_pattern}")
    for s_size in re.findall(r"font-size\s*(?:=\s*[\"']|:\s*)([0-9.]+)", s_raw):
      if float(s_size) < self.F_MIN_FONT_SIZE:
        l_problems.append(f"font-size {s_size} is below {self.F_MIN_FONT_SIZE:g}")
    l_families = re.findall(r"font-family\s*=\s*\"([^\"]*)\"", s_raw)
    l_families += re.findall(r"font-family\s*:\s*([^;\"}]+)", s_raw)
    for s_family in l_families:
      if s_family.strip() not in self.T_FONT_STACKS:
        l_problems.append(f"font-family {s_family.strip()!r} is not an allowed stack")
    return l_problems

  def root_problems(self, obj_root: ET.Element, s_viewbox: str | None) -> list[str]:
    """Return the problems of the root element and of its first two children; s_viewbox is the expected viewBox."""
    l_problems = []
    if obj_root.tag != f"{{{self.S_SVG_NAMESPACE}}}svg":
      l_problems.append("root is not an svg element in the SVG namespace (missing xmlns)")
    if s_viewbox is None:
      l_problems.append(f"no expected viewBox for this file name; known names: {', '.join(self.D_VIEWBOXES)}")
    elif obj_root.get("viewBox") != s_viewbox:
      l_problems.append(f"viewBox is not {s_viewbox!r}")
    if obj_root.get("role") != "img":
      l_problems.append('role is not "img"')
    l_children = list(obj_root)
    for i_pos, s_name in enumerate(("title", "desc")):
      if len(l_children) <= i_pos or self.local_name(l_children[i_pos]) != s_name:
        l_problems.append(f"child {i_pos + 1} is not a {s_name} element")
      elif not "".join(l_children[i_pos].itertext()).strip():
        l_problems.append(f"{s_name} is empty")
    return l_problems

  def element_problems(self, obj_element: ET.Element) -> list[str]:
    """Return the problems of one element: a forbidden kind or an external URL."""
    l_problems = []
    s_name = self.local_name(obj_element)
    if s_name in self.T_FORBIDDEN_ELEMENTS:
      l_problems.append(f"contains a forbidden <{s_name}> element")
    for s_attribute, s_value in obj_element.attrib.items():
      if re.search(r"https?:|^//|url\(\s*['\"]?(https?:|//)", s_value):
        l_problems.append(f"attribute {self.local_name_of(s_attribute)} on <{s_name}> holds an external URL")
      elif s_name == "use" and self.local_name_of(s_attribute) == "href" and not s_value.startswith("#"):
        l_problems.append("a <use> element points outside the file")
    return l_problems

  def text_nodes(self, obj_root: ET.Element) -> list[str]:
    """Return every string inside a text or tspan element, whitespace runs collapsed, empty ones left out.

    A string is one text node: an element's own text, or the text after a child tspan inside it.
    """
    l_strings = []
    for obj_element in obj_root.iter():
      if self.local_name(obj_element) in ("text", "tspan"):
        for s_node in [obj_element.text] + [obj_child.tail for obj_child in obj_element]:
          s_collapsed = self.collapse(s_node or "")
          if s_collapsed:
            l_strings.append(s_collapsed)
    return l_strings

  def collapse(self, s_text: str) -> str:
    """Return s_text with every whitespace run turned into one space and the ends stripped."""
    return re.sub(r"\s+", " ", s_text).strip()

  def check_traceability(self, s_svg_path: str, s_root: str) -> list[str]:
    """Return every SVG text string that occurs in none of the repository source files."""
    obj_root = Path(s_root)
    l_files = [obj_root / s_name for s_name in self.T_SOURCE_FILES] + sorted(obj_root.glob(self.S_SOURCE_GLOB))
    s_sources = "\n".join(self.collapse(obj_file.read_text(encoding="utf-8")) for obj_file in l_files if obj_file.is_file())
    l_problems = []
    for s_string in self.text_nodes(ET.parse(s_svg_path).getroot()):
      if all(s_char in self.S_EXEMPT_CHARACTERS for s_char in s_string):
        continue
      if s_string not in s_sources:
        l_problems.append(f"text {s_string!r} occurs in no repository source file")
    return l_problems

  def tspan_texts(self, s_svg_path: str, s_text_id: str) -> list[str]:
    """Return the raw text of each tspan inside the text element with id s_text_id, in document order."""
    for obj_element in ET.parse(s_svg_path).getroot().iter():
      if self.local_name(obj_element) == "text" and obj_element.get("id") == s_text_id:
        return [obj_child.text or "" for obj_child in obj_element if self.local_name(obj_child) == "tspan"]
    return []

  def mermaid_graph(self, s_flow_path: str) -> tuple[dict, dict]:
    """Return (first label line -> node id, node id -> set of successor ids) from the first Mermaid block.

    Solid (-->) and dotted (-.->) edges both count; the flow guide says the dotted ones mean that a
    search reads the saved index, so the index comes before the search.
    """
    s_text = Path(s_flow_path).read_text(encoding="utf-8")
    s_block = re.search(r"```mermaid\n(.*?)```", s_text.replace("\r\n", "\n"), re.DOTALL).group(1)
    d_labels = {s_label.split("<br/>")[0]: s_id for s_id, s_label in re.findall(r"(\w+)\[\"([^\"]*)\"\]", s_block)}
    d_edges: dict[str, set] = {}
    for s_from, s_to in re.findall(r"(\w+)(?:\[\"[^\"]*\"\])?\s*(?:-->|-\.->)\s*(\w+)", s_block):
      d_edges.setdefault(s_from, set()).add(s_to)
    return d_labels, d_edges

  def reaches(self, d_edges: dict, s_from: str, s_to: str) -> bool:
    """Return True when a path of one or more edges leads from s_from to s_to."""
    l_open = list(d_edges.get(s_from, ()))
    obj_seen: set = set()
    while l_open:
      s_node = l_open.pop()
      if s_node == s_to:
        return True
      if s_node not in obj_seen:
        obj_seen.add(s_node)
        l_open.extend(d_edges.get(s_node, ()))
    return False

  def check_stage_order(self, s_svg_path: str, s_flow_path: str) -> list[str]:
    """Return problems when the banner's pipeline line disagrees with the order of the flow guide's diagram.

    Stages joined by " › " must follow each other in the diagram; stages joined by " + " must be parallel.
    """
    d_labels, d_edges = self.mermaid_graph(s_flow_path)
    l_parts = [self.collapse(s_part) for s_part in self.tspan_texts(s_svg_path, "pipeline")]
    l_problems = []
    for i_pos in range(1, len(l_parts) - 1, 2):
      s_before, s_separator, s_after = l_parts[i_pos - 1], l_parts[i_pos], l_parts[i_pos + 1]
      if s_before not in d_labels or s_after not in d_labels:
        l_problems.append(f"stage {s_before!r} or {s_after!r} is not a node label in the diagram")
        continue
      s_from, s_to = d_labels[s_before], d_labels[s_after]
      if s_separator == "›" and not self.reaches(d_edges, s_from, s_to):
        l_problems.append(f"{s_before!r} does not come before {s_after!r} in the diagram")
      if s_separator == "+" and (self.reaches(d_edges, s_from, s_to) or self.reaches(d_edges, s_to, s_from)):
        l_problems.append(f"{s_before!r} and {s_after!r} are not parallel in the diagram")
    return l_problems

  def check_provenance(self, s_svg_path: str, s_root: str, t_table: tuple | None = None) -> list[str]:
    """Return problems when the badge (label, value) pairs differ from the table or a source substring is missing."""
    t_used = t_table if t_table is not None else self.T_BADGE_PROVENANCE
    obj_svg = ET.parse(s_svg_path).getroot()
    l_texts = [
      self.collapse("".join(obj_element.itertext())) for obj_element in obj_svg.iter()
      if self.local_name(obj_element) == "text"
    ]
    l_problems = []
    if len(l_texts) % 2:
      l_problems.append(f"badge texts come in pairs, found {len(l_texts)} texts")
    l_pairs = list(zip(l_texts[0::2], l_texts[1::2]))
    l_expected = [(t_row[0], t_row[1]) for t_row in t_used]
    if l_pairs != l_expected:
      l_problems.append(f"badge pairs {l_pairs} differ from the provenance table {l_expected}")
    for s_label, s_value, s_file, s_needle in t_used:
      obj_file = Path(s_root) / s_file
      if not obj_file.is_file() or s_needle not in obj_file.read_text(encoding="utf-8"):
        l_problems.append(f"badge {s_label} {s_value}: {s_file} does not contain {s_needle!r}")
    return l_problems

  def local_name(self, obj_element: ET.Element) -> str:
    """Return an element's tag without its namespace."""
    return self.local_name_of(obj_element.tag)

  def local_name_of(self, s_name: str) -> str:
    """Return a tag or attribute name without its {namespace} prefix."""
    return s_name.rsplit("}", 1)[-1]

  def check_readme_images(self, s_readme_path: str, s_root: str) -> list[str]:
    """Return problems with the README's image references and with unreferenced files in assets/readme."""
    s_text = Path(s_readme_path).read_text(encoding="utf-8")
    obj_root = Path(s_root).resolve()
    l_problems: list[str] = []
    l_references: list[str] = []
    for s_tag in re.findall(r"<img\b[^>]*>", s_text):
      obj_src = re.search(r"\bsrc=\"([^\"]*)\"", s_tag)
      obj_alt = re.search(r"\balt=\"([^\"]*)\"", s_tag)
      if obj_src:
        l_references.append(obj_src.group(1))
      else:
        l_problems.append(f"<img> without src: {s_tag}")
      if not obj_alt or not obj_alt.group(1).strip():
        l_problems.append(f"<img> without alt text: {s_tag}")
    for s_srcset in re.findall(r"\bsrcset=\"([^\"]*)\"", s_text):
      l_references.extend(s_part.strip().split()[0] for s_part in s_srcset.split(",") if s_part.strip())
    for s_alt, s_target in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)", s_text):
      l_references.append(s_target)
      if not s_alt.strip():
        l_problems.append(f"Markdown image without alt text: {s_target}")
    d_resolved = {s_reference: self.resolve(obj_root, s_reference) for s_reference in l_references}
    for s_reference, obj_file in d_resolved.items():
      if obj_file is None:
        l_problems.append(f"image reference {s_reference!r} is not an existing file under the project")
    obj_assets = obj_root / self.S_ASSET_DIR
    t_used = tuple(obj_file for obj_file in d_resolved.values() if obj_file is not None)
    if obj_assets.is_dir():
      for obj_file in sorted(obj_assets.rglob("*")):
        if obj_file.is_file() and obj_file.resolve() not in t_used:
          l_problems.append(f"{obj_file.relative_to(obj_root).as_posix()} is not referenced by the README")
    return l_problems

  def resolve(self, obj_root: Path, s_reference: str) -> Path | None:
    """Return the resolved file a relative reference points to inside obj_root, or None."""
    if re.match(r"^[a-z]+:|^//|^/", s_reference, re.IGNORECASE):
      return None
    obj_file = (obj_root / s_reference.split("#", 1)[0]).resolve()
    if not obj_file.is_file() or not obj_file.is_relative_to(obj_root):
      return None
    return obj_file
