"""Pydantic data models for the parser pipeline.

Every extracted unit carries provenance: source page, bbox, and which parser
produced it. This is non-negotiable for downstream regulatory use.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ParserRoute(str, Enum):
    PYMUPDF = "pymupdf"               # plain prose, clean text layer
    PDFPLUMBER = "pdfplumber"         # local fallback for ruled tables
    MINERU = "mineru"                 # MinerU 2.5 (1.2B) via vLLM — primary doc parser
    DOCLING = "docling"               # legacy / fallback for structured pages
    QWEN_VL = "qwen_vl"               # legacy / fallback for scanned pages
    QWEN_TABLE = "qwen_table"         # legacy / fallback for SoA-style tables


class ParseStatus(str, Enum):
    OK = "ok"
    FALLBACK = "fallback"
    ERROR = "error"
    SKIPPED = "skipped"


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    OTHER = "other"


class BBox(BaseModel):
    """Page-space bounding box, top-left origin, in PDF points."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class PageFeatures(BaseModel):
    """Cheap signals computed before parsing. Used by the triage router.

    Kept deliberately generic — these are corpus-level features, not tuned
    to any single document's layout.
    """

    page_num: int = Field(..., description="0-indexed page number")
    width: float
    height: float
    char_count: int
    char_density: float = Field(..., description="chars per (width*height) in PDF points")
    image_count: int
    image_coverage: float = Field(..., description="fraction of page area covered by image objects (0-1)")
    has_text_layer: bool
    text_extraction_confidence: float = Field(..., description="0-1, low = likely OCR garbage or scanned")
    likely_table_count: int = Field(..., description="heuristic count of likely table regions")
    column_count_estimate: int = Field(..., description="1, 2, or 3+")
    is_likely_scanned: bool
    raw_signals: dict[str, Any] = Field(default_factory=dict, description="debug: underlying measurements")


class TriageDecision(BaseModel):
    """Per-page routing decision with human-readable rationale."""

    page_num: int
    features: PageFeatures
    primary_route: ParserRoute
    fallback_routes: list[ParserRoute] = Field(default_factory=list)
    reason: str = Field(..., description="why this route was chosen — surfaced in the UI")
    remote_required: bool = False


class ExtractedBlock(BaseModel):
    """A unit of extracted content with full provenance."""

    block_type: BlockType
    text: str
    page_num: int
    bbox: Optional[BBox] = None
    parser_used: ParserRoute
    order_index: int = Field(..., description="reading order within the page")
    metadata: dict[str, Any] = Field(default_factory=dict, description="parser-specific extras (heading level, table cells, font, ...)")


class PageResult(BaseModel):
    page_num: int
    triage: TriageDecision
    parser_used: ParserRoute
    parse_status: ParseStatus
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    parse_duration_ms: float = 0.0
    error: Optional[str] = None
    attempts: list[dict[str, Any]] = Field(default_factory=list, description="audit trail of routes tried and their outcomes")


class DocumentResult(BaseModel):
    source_path: str
    source_filename: str
    total_pages: int
    pages: list[PageResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    remote_configured: bool = False
    pipeline_version: Literal["0.2.0"] = "0.2.0"
    # Per-stage breakdown so we can profile where time goes.
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

    def all_blocks(self) -> list[ExtractedBlock]:
        return [b for p in self.pages for b in p.blocks]

    def route_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pages:
            out[p.parser_used.value] = out.get(p.parser_used.value, 0) + 1
        return out
