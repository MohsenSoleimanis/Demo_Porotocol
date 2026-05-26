"""Local post-processing of parser blocks.

Things parsers (Docling, MinerU) miss because they're cell-based or layout-
based: vector annotations like duration/span arrows that sit between cells
or describe relationships between visual elements. We detect these from PDF
drawing operations and attach them as metadata to the surrounding block
(usually a table).

All operations here are corpus-generic, no document-specific tuning.
"""

from __future__ import annotations

from typing import Iterable, Optional

import fitz

from .arrows import detect_arrows_on_page, Arrow
from .models import BlockType, ExtractedBlock


def _bbox_contains(bbox, px: float, py: float) -> bool:
    return bbox is not None and bbox.x0 <= px <= bbox.x1 and bbox.y0 <= py <= bbox.y1


def _bbox_intersects(bbox, x0: float, y0: float, x1: float, y1: float) -> bool:
    return bbox is not None and not (
        bbox.x1 < x0 or bbox.x0 > x1 or bbox.y1 < y0 or bbox.y0 > y1
    )


def _arrow_to_dict(a: Arrow) -> dict:
    return {
        "start": [round(a.start[0], 2), round(a.start[1], 2)],
        "end": [round(a.end[0], 2), round(a.end[1], 2)],
        "axis": a.axis,
        "two_headed": a.two_headed,
        "span_pt": round(
            abs(a.end[0] - a.start[0]) if a.axis == "horizontal" else abs(a.end[1] - a.start[1]),
            2,
        ),
    }


def enrich_page_with_arrows(
    fitz_doc: fitz.Document,
    page_index: int,
    blocks: list[ExtractedBlock],
) -> int:
    """Detect arrows on a page and attach them to the appropriate block's metadata.

    Returns the number of arrows attached (for logging / progress).
    """
    try:
        page = fitz_doc[page_index]
    except IndexError:
        return 0
    arrows = detect_arrows_on_page(page)
    if not arrows:
        return 0

    # Prefer attaching each arrow to the TABLE block whose bbox contains its midpoint.
    table_blocks = [b for b in blocks if b.block_type == BlockType.TABLE and b.bbox is not None]
    figure_blocks = [b for b in blocks if b.block_type == BlockType.FIGURE and b.bbox is not None]
    page_level: list[dict] = []
    n_attached = 0

    for ar in arrows:
        mx = (ar.start[0] + ar.end[0]) / 2.0
        my = (ar.start[1] + ar.end[1]) / 2.0
        target = None
        for b in table_blocks:
            if _bbox_contains(b.bbox, mx, my):
                target = b
                break
        if target is None:
            for b in figure_blocks:
                if _bbox_contains(b.bbox, mx, my):
                    target = b
                    break
        if target is None:
            page_level.append(_arrow_to_dict(ar))
            continue
        meta = dict(target.metadata or {})
        arr_list = list(meta.get("arrows") or [])
        arr_list.append(_arrow_to_dict(ar))
        meta["arrows"] = arr_list
        meta["has_arrows"] = True
        target.metadata = meta
        n_attached += 1

    # Page-level orphans go onto the first block (so they're visible somewhere).
    if page_level and blocks:
        b0 = blocks[0]
        meta = dict(b0.metadata or {})
        meta["page_level_arrows"] = page_level
        b0.metadata = meta

    return n_attached + len(page_level)
