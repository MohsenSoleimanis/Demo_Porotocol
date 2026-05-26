"""Pydantic models for the chunking layer.

A `Chunk` is the atomic unit downstream stages consume — RAG retrieval,
USDM mapping, knowledge-graph construction. Each chunk carries its full
section path, provenance back to source pages/blocks, resolved cross-
references, and (where applicable) attached footnote definitions.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import BBox, BlockType


class FootnoteResolution(BaseModel):
    """A footnote marker referenced inside a chunk's text, resolved to its
    definition text + provenance (which block on which page)."""

    marker: str
    definition_text: str
    source_page: int
    source_block_index: int


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str  # NCT or filename stem
    page_num: int
    section_path: list[str] = Field(default_factory=list)  # outermost -> innermost heading text
    section_depth: int = 0
    m11_section: Optional[str] = None  # e.g. "5.1" — canonical ICH M11 section id when recognized

    block_type: BlockType
    text: str
    text_for_embedding: str = ""  # cleaned/condensed; empty means use `text`

    bbox: Optional[BBox] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: list[str] = Field(default_factory=list)

    # Cross-references found inside `text`. e.g. ["Section 6", "Table 2", "Figure 1"]
    references: list[str] = Field(default_factory=list)
    # Footnote markers in body text resolved to their definitions on this page
    # or in the nearest preceding `footnote`-typed block.
    resolved_footnotes: list[FootnoteResolution] = Field(default_factory=list)

    # Multi-page table stitching candidates: when this chunk is a `table` and
    # there's a likely continuation on the next page, list that chunk_id.
    continuation_of: Optional[str] = None     # this chunk continues a previous table
    continued_by: Optional[str] = None        # this table is continued by another chunk

    # Provenance — which source blocks from parsed.json fed this chunk
    source_page: int
    source_block_indices: list[int] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    doc_id: str
    source_pdf: str
    total_pages: int
    total_chunks: int
    chunks: list[Chunk] = Field(default_factory=list)
    # Top-level section index: ordered list of (section_path, first_chunk_id).
    section_index: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
