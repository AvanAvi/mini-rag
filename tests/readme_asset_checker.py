"""Test helper that checks the README hero SVGs and the image references in README.md."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path


class ReadmeAssetChecker:
  """Returns a list of problems for an SVG asset or for the images a README references; empty means it passes."""

  S_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
  S_VIEWBOX = "0 0 1280 360"
  I_MAX_BYTES = 20 * 1024
  F_MIN_FONT_SIZE = 18.0
  T_FORBIDDEN_ELEMENTS = ("script", "foreignObject", "image", "a", "metadata")
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
    l_problems.extend(self.root_problems(obj_root))
    for obj_element in obj_root.iter():
      l_problems.extend(self.element_problems(obj_element))
    return l_problems

  def raw_text_problems(self, s_raw: str) -> list[str]:
    """Return the problems visible in the file text: comments, @import, absolute paths, rasters, fonts."""
    l_problems = []
    if "<!--" in s_raw:
      l_problems.append("contains an XML comment")
    if "@import" in s_raw:
      l_problems.append("contains a CSS @import")
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

  def root_problems(self, obj_root: ET.Element) -> list[str]:
    """Return the problems of the root element and of its first two children."""
    l_problems = []
    if obj_root.tag != f"{{{self.S_SVG_NAMESPACE}}}svg":
      l_problems.append("root is not an svg element in the SVG namespace (missing xmlns)")
    if obj_root.get("viewBox") != self.S_VIEWBOX:
      l_problems.append(f"viewBox is not {self.S_VIEWBOX!r}")
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
    """Return the problems of one element: forbidden kind, external URL, or a digit in its text."""
    l_problems = []
    s_name = self.local_name(obj_element)
    if s_name in self.T_FORBIDDEN_ELEMENTS:
      l_problems.append(f"contains a forbidden <{s_name}> element")
    for s_attribute, s_value in obj_element.attrib.items():
      if re.search(r"https?:|^//|url\(\s*['\"]?(https?:|//)", s_value):
        l_problems.append(f"attribute {self.local_name_of(s_attribute)} on <{s_name}> holds an external URL")
      elif s_name == "use" and self.local_name_of(s_attribute) == "href" and not s_value.startswith("#"):
        l_problems.append("a <use> element points outside the file")
    if s_name in ("text", "tspan") and re.search(r"\d", "".join(obj_element.itertext())):
      l_problems.append(f"a <{s_name}> element contains a digit")
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
