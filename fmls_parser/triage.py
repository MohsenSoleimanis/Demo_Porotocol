"""Per-page feature extraction and parser routing.

Design constraint: every signal here is computed from generic structural
properties of the page (text density, image coverage, ruled lines, column
estimate). No regex or heuristic is tied to any specific document's wording,
template, or layout — the router must generalize across a 50k+ corpus of
heterogeneous clinical protocols.

Thresholds are explicit dataclass fields and exposed in the UI so they can
be tuned against a labeled corpus rather than guessed at per-document.
"""

from __future__ import annotations

import os
import statistics
import string
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from .models import PageFeatures, ParserRoute, TriageDecision

PRINTABLE = set(string.printable)


@dataclass
class TriageConfig:
    """Tunable thresholds. Defaults are conservative starting points — they
    should be calibrated on a held-out labeled subset of the real corpus."""

    # A page with very low char density per area is likely scanned/image-only.
    scanned_char_density_max: float = 0.002  # chars per (pt^2)
    # If image coverage is this high AND char density is low, treat as scanned.
    scanned_image_coverage_min: float = 0.4
    # Below this fraction of printable/dictionary-ish characters, treat the
    # text layer as unreliable (e.g. OCR garbage, exotic encodings).
    text_confidence_min: float = 0.7
    # Any page with at least this many detected ruled-line table regions
    # gets routed to a structure-aware parser.
    table_count_min: int = 1
    # >1 columns triggers layout-aware parser.
    multicolumn_min: int = 2
    # COMPLEX-table signals: when both MinerU and the legacy stack are
    # available, these signals can be used to prefer a specific route. With
    # MinerU as primary they're informational only — MinerU handles them all.
    complex_table_min_cols: int = 6
    complex_table_area_ratio: float = 0.5
    # Remote parser routing is only available when a base URL is configured.
    remote_configured: bool = False
    # When True, route every "structured" page (tables / multi-column / scanned)
    # through MinerU 2.5. Legacy Docling + Qwen-VL routes remain available as
    # fallbacks. Default ON.
    use_mineru: bool = True

    @classmethod
    def from_env(cls, remote_configured: bool = False) -> "TriageConfig":
        def _f(name: str, default: float) -> float:
            v = os.getenv(name)
            return float(v) if v else default

        return cls(
            scanned_char_density_max=_f("FMLS_SCANNED_CHAR_DENSITY_MAX", 0.002),
            scanned_image_coverage_min=_f("FMLS_SCANNED_IMG_COVERAGE_MIN", 0.4),
            text_confidence_min=_f("FMLS_TEXT_CONFIDENCE_MIN", 0.7),
            table_count_min=int(_f("FMLS_TABLE_COUNT_MIN", 1)),
            multicolumn_min=int(_f("FMLS_MULTICOLUMN_MIN", 2)),
            remote_configured=remote_configured,
        )


# ---------- feature extraction ----------


def _text_confidence(text: str) -> float:
    """Cheap quality proxy: fraction of printable + plausibly-word-shaped characters.

    Not a language model — just a sanity check that the text layer isn't
    obviously broken (mis-encoded glyphs, OCR garbage). Languages other than
    English are handled because we only penalize non-printable / control
    characters, not specific alphabets.
    """
    if not text:
        return 0.0
    printable = sum(1 for c in text if c in PRINTABLE or c.isalpha())
    return printable / len(text)


def _estimate_columns(text_blocks: list[tuple[float, float, float, float]], page_width: float) -> int:
    """Estimate column count from the distribution of block left edges.

    A single-column page has x0 values clustered around one value (the left
    margin). Two-column has two clusters. We use a simple bin-and-count.
    """
    if not text_blocks or page_width <= 0:
        return 1
    # Normalize x0 to [0, 1].
    x0s = sorted((b[0] / page_width) for b in text_blocks if b[2] > b[0])
    if len(x0s) < 4:
        return 1
    # 10 bins across the page; count bins with at least 2 blocks and gaps between them.
    bins = [0] * 10
    for x in x0s:
        idx = min(int(x * 10), 9)
        bins[idx] += 1
    populated = [i for i, c in enumerate(bins) if c >= 2]
    if not populated:
        return 1
    # Count distinct clusters separated by an empty bin.
    clusters = 1
    for prev, cur in zip(populated, populated[1:]):
        if cur - prev > 1:
            clusters += 1
    return clusters


def _table_signals_from_page(plumber_page) -> tuple[int, float, int]:
    """(likely_table_count, max_table_area_ratio, max_cols) from an already-open plumber page.

    Critical: don't access `t.rows` — that triggers full table extraction.
    Use `t.cells` (raw cell rectangles) which is a cheap O(cells) walk.
    """
    try:
        tables = plumber_page.find_tables() or []
    except Exception:
        return 0, 0.0, 0
    if not tables:
        return 0, 0.0, 0
    page_area = max(float(plumber_page.width) * float(plumber_page.height), 1.0)
    max_area_ratio = 0.0
    max_cols = 0
    for t in tables:
        try:
            bx = t.bbox
            area = max(float(bx[2] - bx[0]) * float(bx[3] - bx[1]), 0.0)
            max_area_ratio = max(max_area_ratio, area / page_area)
            # Cheap column count: cluster cell x-edges and count distinct columns.
            cells = getattr(t, "cells", None) or []
            if cells:
                x_edges = sorted({round(float(c[0]), 1) for c in cells if c and len(c) >= 4})
                max_cols = max(max_cols, len(x_edges))
        except Exception:
            continue
    return len(tables), max_area_ratio, max_cols


def extract_features_from_handles(
    fitz_doc: "fitz.Document",
    plumber_pdf: "pdfplumber.PDF",
    page_index: int,
) -> PageFeatures:
    """Extract triage features using already-open document handles.

    Single text-extraction pass via `get_text("dict")`, single pdfplumber page
    access. No re-opening of the PDF.
    """
    page = fitz_doc[page_index]
    width = float(page.rect.width)
    height = float(page.rect.height)
    page_area = max(width * height, 1.0)

    # One text-extraction pass — derive both plain text and block bboxes from
    # the same dict result instead of running two separate scans.
    text_dict = page.get_text("dict") or {}
    raw_blocks = text_dict.get("blocks", []) or []
    text_parts: list[str] = []
    block_bboxes: list[tuple[float, float, float, float]] = []
    for b in raw_blocks:
        if b.get("type") != 0:
            continue
        bb = b.get("bbox") or ()
        if len(bb) >= 4:
            block_bboxes.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
        for ln in b.get("lines", []) or []:
            for sp in ln.get("spans", []) or []:
                t = sp.get("text") or ""
                if t:
                    text_parts.append(t)
    text = "\n".join(text_parts)
    char_count = len(text.strip())
    char_density = char_count / page_area

    images = page.get_images(full=True)
    image_count = len(images)
    image_area = 0.0
    for img in images:
        xref = img[0]
        try:
            for rect in page.get_image_rects(xref):
                image_area += float(rect.width * rect.height)
        except Exception:
            continue
    image_coverage = min(image_area / page_area, 1.0)

    column_count = _estimate_columns(block_bboxes, width)
    text_conf = _text_confidence(text)
    has_text = char_count > 0

    try:
        plumber_page = plumber_pdf.pages[page_index]
        table_count, max_table_area_ratio, max_table_cols = _table_signals_from_page(plumber_page)
    except Exception:
        table_count, max_table_area_ratio, max_table_cols = 0, 0.0, 0

    is_scanned = (not has_text) or (
        char_density < 0.0005 and image_coverage > 0.2
    )

    return PageFeatures(
        page_num=page_index,
        width=width,
        height=height,
        char_count=char_count,
        char_density=char_density,
        image_count=image_count,
        image_coverage=image_coverage,
        has_text_layer=has_text,
        text_extraction_confidence=text_conf,
        likely_table_count=table_count,
        column_count_estimate=column_count,
        is_likely_scanned=is_scanned,
        raw_signals={
            "text_blocks": len(block_bboxes),
            "image_area_pts2": image_area,
            "page_area_pts2": page_area,
            "max_table_area_ratio": max_table_area_ratio,
            "max_table_cols": max_table_cols,
        },
    )


def extract_features(pdf_path: str, page_index: int) -> PageFeatures:
    """Backward-compat: extract features by opening the PDF once per call.

    Prefer `extract_features_from_handles` in hot paths.
    """
    with fitz.open(pdf_path) as fitz_doc:
        with pdfplumber.open(pdf_path) as plumber_pdf:
            return extract_features_from_handles(fitz_doc, plumber_pdf, page_index)


# ---------- routing ----------


def decide(features: PageFeatures, config: Optional[TriageConfig] = None) -> TriageDecision:
    """Pick a parser for a page.

    Default policy (MinerU mode + remote configured):
      EVERY page -> MinerU 2.5. PyMuPDF only as last-resort fallback.

    Why no fast-lane to PyMuPDF for "plain prose":
      - PyMuPDF mis-handles a long tail of layouts: 2-column bullet lists,
        intra-paragraph indented sub-items, captions adjacent to tables, etc.
      - Triage cannot reliably tell a "trivial" page from one of those cases.
      - MinerU is fast enough at batch=10+ pages (~3-5 s/page) that we don't
        need the optimization.
      - Consistency >> small per-page savings.

    Legacy mode (`use_mineru=False`) preserves the old Docling/Qwen routing
    for evaluation comparisons.
    """
    cfg = config or TriageConfig()

    has_tables = features.likely_table_count >= cfg.table_count_min
    is_multicol = features.column_count_estimate >= cfg.multicolumn_min
    needs_vlm = (
        features.is_likely_scanned
        or not features.has_text_layer
        or features.text_extraction_confidence < cfg.text_confidence_min
    )

    if cfg.use_mineru and cfg.remote_configured:
        # Every page goes to MinerU. The triage features are still recorded
        # so we can audit later, but they don't change the route.
        why_bits: list[str] = []
        if needs_vlm:
            why_bits.append("scanned/garbage-text")
        if has_tables:
            why_bits.append(f"{features.likely_table_count} table region(s)")
        if is_multicol:
            why_bits.append(f"{features.column_count_estimate}-column")
        if not why_bits:
            why_bits.append("plain prose")
        return TriageDecision(
            page_num=features.page_num,
            features=features,
            primary_route=ParserRoute.MINERU,
            fallback_routes=[ParserRoute.PYMUPDF],
            reason="route-all-to-MinerU: " + "; ".join(why_bits),
            remote_required=True,
        )

    # ---- Legacy path (use_mineru=False or remote down): Docling/Qwen split. ----
    if needs_vlm:
        return TriageDecision(
            page_num=features.page_num,
            features=features,
            primary_route=ParserRoute.QWEN_VL,
            fallback_routes=[],
            reason="VLM required (scanned/garbage-text)" + ("" if cfg.remote_configured else " — remote not configured, page will be skipped"),
            remote_required=True,
        )

    if cfg.remote_configured:
        return TriageDecision(
            page_num=features.page_num,
            features=features,
            primary_route=ParserRoute.DOCLING,
            fallback_routes=[ParserRoute.PDFPLUMBER, ParserRoute.PYMUPDF],
            reason="structure-aware (Docling legacy path)",
            remote_required=True,
        )
    return TriageDecision(
        page_num=features.page_num,
        features=features,
        primary_route=ParserRoute.PDFPLUMBER,
        fallback_routes=[ParserRoute.PYMUPDF],
        reason="structure-aware locally (no remote available)",
        remote_required=False,
    )


def triage_document(pdf_path: str, config: Optional[TriageConfig] = None) -> list[TriageDecision]:
    """Run feature extraction + routing for every page. Opens the PDF once."""
    cfg = config or TriageConfig()
    decisions: list[TriageDecision] = []
    with fitz.open(pdf_path) as fitz_doc:
        with pdfplumber.open(pdf_path) as plumber_pdf:
            for i in range(fitz_doc.page_count):
                feats = extract_features_from_handles(fitz_doc, plumber_pdf, i)
                decisions.append(decide(feats, cfg))
    return decisions
