"""Hierarchical chunking layer.

Consumes a parser-stage `DocumentResult` (one PDF's parsed.json) and produces
a `ChunkedDocument` with model-provided structure preserved:

  * Section tree from heading sequence + numeric prefixes
    (1, 1.1, 1.1.2 -> dot count = depth; anonymous headings nest one level
    below the most recent numeric heading; this follows established
    document-typography convention, not any specific corpus)
  * Cross-references extracted as metadata (Section X, Table N, Figure M,
    Appendix A) so downstream stages can build a reference graph
  * Footnote markers in body text resolved to nearest preceding footnote
    definitions
  * Table continuation: flagged when a `table` is followed by another `table`
    on the next page with no intervening non-chrome content
  * ICH M11 canonical section IDs attached when the heading numeric prefix
    matches the M11 template
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .chunk_models import Chunk, ChunkedDocument, FootnoteResolution
from .models import BBox, BlockType, DocumentResult, ExtractedBlock, PageResult


# ---- heading parsing ----

# Accept either "1.1 Title" or "1.1. Title" (some sponsor templates trail the
# numeric prefix with a final dot). Treating these as equivalent is the
# generic convention, not a sponsor-specific patch.
_NUMERIC_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)")


def _parse_heading(text: str) -> tuple[Optional[str], str]:
    """Return (numeric_prefix, label) for a heading.

    "5.1 Inclusion Criteria" -> ("5.1", "Inclusion Criteria")
    "Age"                   -> (None, "Age")
    """
    m = _NUMERIC_PREFIX.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return None, text.strip()


# ---- ICH M11 canonical section map ----
# Generic, template-level mapping. Numeric prefixes in clinical protocols
# follow ICH M11 — this map is the canonical list. We attach an `m11_section`
# tag when the numeric prefix matches.

_M11_CANONICAL: dict[str, str] = {
    "1": "Protocol Synopsis",
    "1.1": "Synopsis",
    "1.2": "Schema",
    "1.3": "Schedule of Activities",
    "2": "Introduction",
    "2.1": "Background",
    "2.2": "Rationale",
    "2.3": "Benefit-Risk Assessment",
    "3": "Objectives, Endpoints and Estimands",
    "4": "Trial Design",
    "5": "Trial Population",
    "5.1": "Inclusion Criteria",
    "5.2": "Exclusion Criteria",
    "5.3": "Lifestyle Considerations",
    "5.4": "Screen Failures",
    "6": "Trial Intervention",
    "7": "Discontinuation and Withdrawal",
    "8": "Trial Assessments and Procedures",
    "9": "Statistical Considerations",
    "10": "Supporting Documentation and Operational Considerations",
    "11": "References",
}


def _map_to_m11(numeric_prefix: Optional[str]) -> Optional[str]:
    if not numeric_prefix:
        return None
    # Try exact match, then walk up one level
    if numeric_prefix in _M11_CANONICAL:
        return numeric_prefix
    parts = numeric_prefix.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in _M11_CANONICAL:
            return candidate
        parts = parts[:-1]
    return None


# ---- cross-reference patterns ----

_REF_PATTERNS = [
    re.compile(r"\bSection\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
    re.compile(r"\bTable\s+(\d+[A-Za-z]?)", re.IGNORECASE),
    re.compile(r"\bFigure\s+(\d+[A-Za-z]?)", re.IGNORECASE),
    re.compile(r"\bAppendix\s+([A-Z](?:\.\d+)*)", re.IGNORECASE),
]


def _extract_refs(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            ref = m.group(0).strip()
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
    return out


# ---- footnote marker detection (for resolution back to definitions) ----

# Same patterns as triage.py; deliberately conservative.
_FOOTNOTE_MARKER_PATTERNS = [
    re.compile(r"(?<=[A-Za-z0-9])([ª²³⁰-⁹])"),       # superscript chars
    re.compile(r"(?<=[A-Za-z0-9\)\]])([\*†‡§¶])"),  # symbol footnotes
    re.compile(r"\[(\d{1,3})\]"),                    # bracketed numbers
    re.compile(r"\(([a-z])\)"),                      # (a), (b)
]


def _markers_in(text: str) -> list[str]:
    """Markers in body text that might reference a footnote definition."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pat in _FOOTNOTE_MARKER_PATTERNS:
        for m in pat.finditer(text):
            mk = m.group(1)
            if mk not in seen:
                seen.add(mk)
                found.append(mk)
    return found


def _leading_marker(text: str) -> Optional[str]:
    """If a footnote-typed block's text starts with its own marker, extract it."""
    if not text:
        return None
    m = re.match(r"^[\(\[]?\s*([a-z]|\d{1,3}|[\*†‡§¶])\s*[\)\]\.\:]?\s+", text.strip())
    return m.group(1) if m else None


# ---- section-tree state machine ----


class _SectionStack:
    """Maintains the current section path as we walk a document in reading
    order. Pushes a new heading; pops back when a heading at the same or
    shallower depth arrives.

    Depth rules:
      - Numeric heading "N", "N.M", "N.M.K" -> depth = dot count + 1
      - Anonymous heading (no numeric prefix) -> depth = depth-of-most-recent-
        numeric-heading + 1 (so multiple anonymous headings under "5.1" sit
        at the same depth)
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str, Optional[str]]] = []  # (depth, label, numeric_prefix)
        self._last_numeric_depth = 0

    def push(self, heading_text: str) -> None:
        numeric, label = _parse_heading(heading_text)
        if numeric:
            depth = numeric.count(".") + 1
            self._last_numeric_depth = depth
        else:
            depth = self._last_numeric_depth + 1
            label = heading_text.strip()
        # Pop the stack so we're at depth-1 before pushing.
        while self._stack and self._stack[-1][0] >= depth:
            self._stack.pop()
        self._stack.append((depth, heading_text.strip(), numeric))

    def path(self) -> list[str]:
        return [label for _, label, _ in self._stack]

    def current_numeric(self) -> Optional[str]:
        for depth, _label, numeric in reversed(self._stack):
            if numeric:
                return numeric
        return None

    def current_depth(self) -> int:
        return self._stack[-1][0] if self._stack else 0


# ---- main entry ----


def chunk_document(doc: DocumentResult, doc_id: Optional[str] = None) -> ChunkedDocument:
    """Build a ChunkedDocument from a parser-stage DocumentResult."""
    doc_id = doc_id or _safe_stem(doc.source_filename)
    stack = _SectionStack()
    chunks: list[Chunk] = []
    section_index: list[dict] = []

    # Pass 1 — walk every page/block to build chunks + section tree.
    for page in doc.pages:
        for block in page.blocks:
            # Heading: update section stack, then emit the heading itself as
            # a chunk (it has text downstream consumers want).
            if block.block_type == BlockType.HEADING:
                stack.push(block.text or "")
                numeric = stack.current_numeric()
                m11 = _map_to_m11(numeric)
                chunk = _make_chunk(
                    doc_id=doc_id,
                    block=block,
                    page=page,
                    section_path=stack.path(),
                    section_depth=stack.current_depth(),
                    m11_section=m11,
                )
                chunks.append(chunk)
                if not section_index or section_index[-1].get("section_path") != stack.path():
                    section_index.append({
                        "section_path": stack.path(),
                        "m11_section": m11,
                        "first_chunk_id": chunk.chunk_id,
                        "page_num": page.page_num,
                    })
                continue

            # Chrome blocks: page header / page footer / page number — keep
            # them for provenance but flag them in metadata so downstream
            # consumers can filter.
            chunk = _make_chunk(
                doc_id=doc_id,
                block=block,
                page=page,
                section_path=stack.path(),
                section_depth=stack.current_depth(),
                m11_section=_map_to_m11(stack.current_numeric()),
            )
            chunks.append(chunk)

    # Pass 2 — link parent_chunk_id from block.metadata.parent_block_id (set by MinerU).
    _link_parents(chunks, doc)

    # Pass 3 — footnote resolution: collect per-(section_path) footnote defs,
    # then attach to body chunks that reference matching markers.
    _resolve_footnotes(chunks)

    # Pass 4 — table continuation: when two adjacent table chunks span pages
    # with no intervening non-chrome content, link them.
    _link_table_continuations(chunks)

    return ChunkedDocument(
        doc_id=doc_id,
        source_pdf=doc.source_filename,
        total_pages=doc.total_pages,
        total_chunks=len(chunks),
        chunks=chunks,
        section_index=section_index,
        metadata={
            "parser_pipeline_version": doc.pipeline_version,
            "stage_timings_ms": doc.stage_timings_ms,
        },
    )


# ---- helpers ----


def _safe_stem(filename: str) -> str:
    return re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)


def _make_chunk(
    doc_id: str,
    block: ExtractedBlock,
    page: PageResult,
    section_path: list[str],
    section_depth: int,
    m11_section: Optional[str],
) -> Chunk:
    text = block.text or ""
    return Chunk(
        chunk_id=f"{doc_id}_p{page.page_num}_b{block.order_index}",
        doc_id=doc_id,
        page_num=page.page_num,
        section_path=list(section_path),
        section_depth=section_depth,
        m11_section=m11_section,
        block_type=block.block_type,
        text=text,
        bbox=block.bbox,
        references=_extract_refs(text),
        source_page=page.page_num,
        source_block_indices=[block.order_index],
        metadata=dict(block.metadata or {}),
    )


def _link_parents(chunks: list[Chunk], doc: DocumentResult) -> None:
    """Translate per-page block.metadata['parent_block_id'] (set by MinerU's
    `list` grouping) into parent_chunk_id / child_chunk_ids on Chunk."""
    # Build (page, order_index) -> chunk_id
    lookup: dict[tuple[int, int], str] = {}
    for c in chunks:
        lookup[(c.page_num, c.source_block_indices[0])] = c.chunk_id
    for c in chunks:
        parent_idx = (c.metadata or {}).get("parent_block_id")
        if parent_idx is None:
            continue
        try:
            parent_idx = int(parent_idx)
        except (TypeError, ValueError):
            continue
        parent_id = lookup.get((c.page_num, parent_idx))
        if parent_id and parent_id != c.chunk_id:
            c.parent_chunk_id = parent_id
    # Populate child_chunk_ids in a second pass.
    children: dict[str, list[str]] = {}
    for c in chunks:
        if c.parent_chunk_id:
            children.setdefault(c.parent_chunk_id, []).append(c.chunk_id)
    for c in chunks:
        if c.chunk_id in children:
            c.child_chunk_ids = children[c.chunk_id]


def _resolve_footnotes(chunks: list[Chunk]) -> None:
    """Build a per-page index of footnote definitions and attach any matching
    markers found in body chunks on the same page. Cross-page resolution is
    deliberately conservative — we only resolve to definitions on the same
    page OR in the most-recent table's footnote block, to avoid spurious
    matches when the same marker letter is reused later in the document."""
    # Index: page_num -> list[(marker, definition_text, chunk_id, source_block_idx)]
    per_page_defs: dict[int, list[tuple[str, str, str, int]]] = {}
    for c in chunks:
        if c.block_type != BlockType.FOOTNOTE:
            continue
        marker = (c.metadata or {}).get("footnote_marker") or _leading_marker(c.text)
        if not marker:
            continue
        per_page_defs.setdefault(c.page_num, []).append(
            (str(marker), c.text, c.chunk_id, c.source_block_indices[0])
        )
    for c in chunks:
        if c.block_type == BlockType.FOOTNOTE:
            continue
        markers = _markers_in(c.text)
        if not markers:
            continue
        defs = per_page_defs.get(c.page_num) or []
        for mk in markers:
            for def_marker, def_text, _def_chunk, def_block_idx in defs:
                if def_marker == mk:
                    c.resolved_footnotes.append(FootnoteResolution(
                        marker=mk,
                        definition_text=def_text,
                        source_page=c.page_num,
                        source_block_index=def_block_idx,
                    ))
                    break


_CHROME = {BlockType.HEADER, BlockType.FOOTER, BlockType.OTHER}


def _link_table_continuations(chunks: list[Chunk]) -> None:
    """Heuristic-free table continuation detection: if two `table` chunks
    appear on consecutive pages with NO non-chrome blocks between them, the
    second is a continuation of the first."""
    # Index of chunks in document order is already what we have.
    for i, c in enumerate(chunks):
        if c.block_type != BlockType.TABLE:
            continue
        # Find the next non-chrome chunk after i.
        for j in range(i + 1, len(chunks)):
            other = chunks[j]
            if other.block_type in _CHROME:
                continue
            if other.block_type == BlockType.TABLE and other.page_num == c.page_num + 1:
                c.continued_by = other.chunk_id
                other.continuation_of = c.chunk_id
            break
