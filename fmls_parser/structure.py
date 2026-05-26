"""Structural-assembly layer: builds relationships between flat blocks.

Parsers (MinerU, Docling, PyMuPDF) emit flat block lists. The actual structure
of a protocol — list parent/child, section nesting, caption→table pairing,
marker→definition references — has to be reconstructed from generic signals:

  * x0 indentation              → list parent/child within a page
  * heading text patterns       → section tree (1, 1.1, 1.1.1, …)
  * list markers                → "this is item N at level L"
  * bbox proximity              → caption pairs with the nearest table/figure
  * inline reference patterns   → "Section 5", "Table 2", "footnote a" → graph

All rules are corpus-generic (no document-specific tuning) and operate only
on the block stream we already extract. Output is attached as metadata on
each block, never replaces the raw output:

  block.metadata["indent_level"]      0-based, 0 = leftmost column
  block.metadata["parent_block_id"]   order_index of the block this is a child of (within page)
  block.metadata["list_marker"]       e.g. "1", "(a)", "•", or None
  block.metadata["section_path"]      list[str] heading chain at this block
  block.metadata["references_to"]     list[str] e.g. ["Section 5", "Table 2"]
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .models import BlockType, ExtractedBlock, PageResult

# ---- list-marker patterns ----
# Order matters: more-specific patterns first so e.g. "(a)" doesn't match "a ".
_LIST_MARKER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("paren_lower",   re.compile(r"^\(([a-z])\)\s+")),
    ("paren_upper",   re.compile(r"^\(([A-Z])\)\s+")),
    ("paren_num",     re.compile(r"^\((\d{1,3})\)\s+")),
    ("dot_num",       re.compile(r"^(\d{1,3})\.\s+")),
    ("bare_num",      re.compile(r"^(\d{1,3})\s+(?=[A-Z])")),  # require a capital after
    ("dash_letter",   re.compile(r"^([a-z])\)\s+")),
    ("dash_letter2",  re.compile(r"^([a-z])\.\s+")),
    ("bullet",        re.compile(r"^([•‣●◦⁃∙])\s+")),
    ("dash",          re.compile(r"^([‐‑‒–—−\-])\s+")),
    ("arrow",         re.compile(r"^([→⇒])\s+")),  # incl. PUA bullets
]

# ---- heading numeric pattern ----
# Examples: "5.1", "5.1.2", "1.2.3.4"  — used to build section tree.
_HEADING_NUM = re.compile(r"^(\d+(?:\.\d+)*)\s+")

# ---- inline reference patterns ----
# Examples: "Section 5", "Section 5.1", "Table 2", "Figure 1", "Appendix A".
_REFERENCE_PATTERNS = [
    re.compile(r"\bSection\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
    re.compile(r"\bTable\s+(\d+[A-Za-z]?)", re.IGNORECASE),
    re.compile(r"\bFigure\s+(\d+[A-Za-z]?)", re.IGNORECASE),
    re.compile(r"\bAppendix\s+([A-Z](?:\.\d+)*)", re.IGNORECASE),
]


def detect_list_marker(text: str) -> Optional[tuple[str, str]]:
    """Return (kind, value) if `text` starts with a list marker, else None."""
    if not text:
        return None
    for kind, pat in _LIST_MARKER_PATTERNS:
        m = pat.match(text)
        if m:
            return kind, m.group(1)
    return None


def detect_heading_number(text: str) -> Optional[str]:
    """If text starts with `1`, `1.1`, `1.1.2` etc., return that prefix."""
    if not text:
        return None
    m = _HEADING_NUM.match(text)
    return m.group(1) if m else None


def extract_references(text: str) -> list[str]:
    """Find inline references like 'Section 5', 'Table 2', 'Figure 1'."""
    if not text:
        return []
    found: list[str] = []
    for pat in _REFERENCE_PATTERNS:
        for m in pat.finditer(text):
            # Whole match text, normalized
            ref = m.group(0).strip()
            if ref not in found:
                found.append(ref)
    return found


def _cluster_x0(x0_values: list[float], tol: float = 5.0) -> list[float]:
    """Cluster x0 values into representative indent stops.

    Two x0s within `tol` points are considered the same column. We return the
    sorted list of column-anchor x0 values for the page.
    """
    if not x0_values:
        return []
    sorted_x = sorted(x0_values)
    clusters: list[list[float]] = [[sorted_x[0]]]
    for x in sorted_x[1:]:
        if x - clusters[-1][-1] <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [min(c) for c in clusters]


def _indent_level_for_x0(x0: float, anchors: list[float], tol: float = 5.0) -> int:
    """Map x0 to a 0-based indent level using the cluster anchors."""
    for i, a in enumerate(anchors):
        if abs(x0 - a) <= tol:
            return i
    # Larger than all anchors — return the deepest level.
    return len(anchors) - 1 if anchors else 0


# ---- per-page hierarchy ----

# Blocks we generally don't want as parents/children in the document body.
_CHROME_TYPES = {BlockType.HEADER, BlockType.FOOTER, BlockType.OTHER}


def enrich_blocks_structure(pages: list[PageResult]) -> int:
    """Annotate every block with indent_level, parent_block_id, list_marker,
    section_path, references_to.

    Returns the number of blocks updated (for logging)."""
    if not pages:
        return 0

    section_stack: list[tuple[str, str]] = []  # [(number, text)] e.g. [("5", "STUDY POPULATION"), ("5.1", "Inclusion Criteria")]
    total_updated = 0

    for page in pages:
        # ---- per-page indent anchors ----
        body_blocks = [
            b for b in page.blocks
            if b.block_type not in _CHROME_TYPES and b.bbox is not None
        ]
        x0_values = [b.bbox.x0 for b in body_blocks]
        anchors = _cluster_x0(x0_values)

        # ---- walk blocks in reading order, building hierarchy ----
        # last_at_level[level] = order_index of the most recent block at that level
        last_at_level: dict[int, int] = {}

        for b in page.blocks:
            meta = dict(b.metadata or {})

            # Update section stack on headings (works across pages).
            if b.block_type == BlockType.HEADING:
                num = detect_heading_number(b.text or "")
                # Pop the stack to where this heading slots in.
                if num:
                    depth = num.count(".") + 1
                    while section_stack and (section_stack[-1][0].count(".") + 1) >= depth:
                        section_stack.pop()
                    section_stack.append((num, b.text or ""))
                else:
                    # Heading without numeric prefix — treat as depth 1.
                    if section_stack:
                        section_stack.pop()
                    section_stack.append(("", b.text or ""))

            meta["section_path"] = [t for _, t in section_stack]

            if b.block_type in _CHROME_TYPES or b.bbox is None:
                b.metadata = meta
                total_updated += 1
                continue

            # Indent level for this block.
            level = _indent_level_for_x0(b.bbox.x0, anchors)
            meta["indent_level"] = level

            # List marker.
            marker = detect_list_marker(b.text or "")
            if marker:
                meta["list_marker"] = marker[1]
                meta["list_marker_kind"] = marker[0]

            # Parent: most-recent block at level-1.
            if level > 0:
                # Find the closest previous level.
                for prev_level in range(level - 1, -1, -1):
                    if prev_level in last_at_level:
                        meta["parent_block_id"] = last_at_level[prev_level]
                        break

            last_at_level[level] = b.order_index
            # Clear deeper levels (a same-or-shallower block resets descendants).
            for deeper in list(last_at_level.keys()):
                if deeper > level:
                    del last_at_level[deeper]

            # Inline cross-references.
            refs = extract_references(b.text or "")
            if refs:
                meta["references_to"] = refs

            b.metadata = meta
            total_updated += 1

    return total_updated
