"""FastAPI server for heavy-model parsing — runs on Lightning AI, not locally.

Copy this file + ../models.py-relevant schemas to the Lightning AI machine,
install requirements-remote.txt, then start with:

    uvicorn fmls_parser.remote.server:app --host 0.0.0.0 --port 8000

(or run as a script: `python -m fmls_parser.remote.server`)

From your laptop:
    ssh -L 8000:localhost:8000 <lightning-host>
    export FMLS_REMOTE_URL=http://localhost:8000

Models are loaded lazily on first request so server startup is fast and
GPU memory only gets used when needed.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .schemas import HealthResponse, ParseResponse, RemoteBlock

app = FastAPI(title="FMLS heavy-parser service", version="0.3.0")

# Mount the MinerU 2.5 router (primary structured-content extractor).
try:
    from .mineru_server import router as _mineru_router
    app.include_router(_mineru_router)
    _MINERU_AVAILABLE = True
except ImportError as _e:
    _MINERU_AVAILABLE = False
    _MINERU_IMPORT_ERROR = str(_e)

# Lazy globals — populated on first use.
_docling_converter = None
_qwen_model = None
_qwen_processor = None

# Cache of (sha256 -> {page_index -> list[RemoteBlock]}).
# A single Docling convert produces all pages at once; we serve subsequent
# per-page requests for the same PDF from this cache.
DOCLING_CACHE_MAX_DOCS = int(os.getenv("FMLS_DOCLING_CACHE_MAX_DOCS", "4"))
_docling_cache: "OrderedDict[str, dict[int, list[RemoteBlock]]]" = OrderedDict()
_docling_cache_lock = threading.Lock()


class DoclingDocumentResponse(BaseModel):
    """Bulk response: every page's blocks from one Docling conversion."""

    parser: str = Field(default="docling")
    sha256: str
    n_pages: int
    pages: dict[int, list[RemoteBlock]]
    convert_duration_ms: float
    cached: bool


# ---------- health ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = []
    notes = []
    if _MINERU_AVAILABLE:
        try:
            import vllm  # noqa: F401
            from mineru_vl_utils import MinerUClient  # noqa: F401
            available.append("mineru")
        except Exception as e:
            notes.append(f"mineru import failed at runtime: {e}")
    else:
        notes.append(f"mineru router not mounted: {_MINERU_IMPORT_ERROR}")
    try:
        import docling  # noqa: F401
        available.append("docling")
    except Exception as e:
        notes.append(f"docling unavailable: {e}")
    try:
        import transformers  # noqa: F401
        import torch
        if torch.cuda.is_available():
            available.append("qwen_vl")
        else:
            notes.append("qwen_vl: torch present but no CUDA device")
    except Exception as e:
        notes.append(f"qwen_vl unavailable: {e}")
    return HealthResponse(
        status="ok" if available else "degraded",
        available_parsers=available,
        notes=notes,
    )


# ---------- Docling ----------


def _get_docling():
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _docling_block_type(label: str) -> str:
    """Map Docling labels to our block types. Keep mapping conservative."""
    label = (label or "").lower()
    if "title" in label or "heading" in label or "section" in label:
        return "heading"
    if "table" in label:
        return "table"
    if "list" in label:
        return "list"
    if "caption" in label:
        return "caption"
    if "figure" in label or "picture" in label:
        return "figure"
    # Order matters: check footnote BEFORE page-footer (substring overlap).
    if "footnote" in label:
        return "footnote"
    if "page-header" in label or label == "header":
        return "header"
    if "page-footer" in label or label == "footer":
        return "footer"
    if "text" in label or "paragraph" in label:
        return "paragraph"
    return "other"


import re

# Generic footnote-marker patterns. We capture the marker tokens that ANY block
# references; resolving each marker to its definition is a downstream
# (chunking-stage) concern because definitions may live on a different page.
_FOOTNOTE_MARKER_PATTERNS = [
    # superscript-style letter after a word: word^a or word^a,
    re.compile(r"(?<=[A-Za-z0-9])([ª²³⁰-⁹])"),  # unicode superscripts
    # daggers, asterisks, section signs in body
    re.compile(r"(?<=[A-Za-z0-9\)\]])([\*†‡§¶])"),
    # bracketed numeric: [1], [12]
    re.compile(r"\[(\d{1,3})\]"),
    # parenthesized small letter: (a), (b)
    re.compile(r"\(([a-z])\)"),
]


def _cells_to_html(cells: list[dict], n_rows: int, n_cols: int) -> str:
    """Render structured cells (with row_span/col_span) as an HTML table.

    Cells are indexed by their (row, col) top-left position. Spanned cells
    are emitted once with rowspan/colspan attributes; spanned-over positions
    are skipped because the spanning cell already covers them.
    """
    if not cells or n_rows <= 0 or n_cols <= 0:
        return ""

    # Build occupancy map so we know which positions are covered by a span
    # initiated at a different (row, col).
    by_origin: dict[tuple[int, int], dict] = {(c["row"], c["col"]): c for c in cells}
    occupied: set[tuple[int, int]] = set()
    for c in cells:
        r0, c0 = c["row"], c["col"]
        for rr in range(r0, r0 + c["row_span"]):
            for cc in range(c0, c0 + c["col_span"]):
                if (rr, cc) != (r0, c0):
                    occupied.add((rr, cc))

    import html as _html

    out: list[str] = ["<table border='1' style='border-collapse:collapse'>"]
    for r in range(n_rows):
        out.append("<tr>")
        for c in range(n_cols):
            if (r, c) in occupied:
                continue  # covered by an earlier-emitted span
            cell = by_origin.get((r, c))
            if cell is None:
                out.append("<td></td>")
                continue
            tag = "th" if cell.get("is_header") else "td"
            attrs = []
            if cell["row_span"] > 1:
                attrs.append(f"rowspan='{cell['row_span']}'")
            if cell["col_span"] > 1:
                attrs.append(f"colspan='{cell['col_span']}'")
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            out.append(f"<{tag}{attr_str}>{_html.escape(cell['text'])}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _extract_footnote_markers(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for pat in _FOOTNOTE_MARKER_PATTERNS:
        for m in pat.finditer(text):
            found.append(m.group(1))
    # dedupe but preserve order
    seen: set[str] = set()
    out: list[str] = []
    for m in found:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _docling_item_to_block(item, order: int, page_num: int, page_heights: dict[int, float]) -> Optional[RemoteBlock]:
    """Convert a Docling NodeItem into a RemoteBlock with full metadata.

    Returns None if the item has no usable text content.
    Normalizes bbox to top-left origin (matching PyMuPDF / pdfplumber / UI).
    """
    label = getattr(item, "label", "") or getattr(item, "self_ref", "")
    label_str = str(label.value) if hasattr(label, "value") else str(label)
    block_type = _docling_block_type(label_str)

    # Bounding box from prov — Docling uses PDF-native coords (origin bottom-left,
    # Y-up). Flip Y so downstream consumers get top-left origin.
    bbox = None
    prov = getattr(item, "prov", None)
    if prov and len(prov) > 0:
        p0 = prov[0]
        b = getattr(p0, "bbox", None)
        if b is not None:
            try:
                page_h = page_heights.get(page_num)
                if page_h is not None:
                    # b.t is the top edge in PDF coords (larger Y),
                    # b.b is the bottom edge (smaller Y). After flip:
                    # new_top = page_h - b.t, new_bottom = page_h - b.b.
                    x0 = float(b.l)
                    x1 = float(b.r)
                    y0 = float(page_h - b.t)
                    y1 = float(page_h - b.b)
                    if y0 > y1:
                        y0, y1 = y1, y0
                    bbox = (x0, y0, x1, y1)
                else:
                    bbox = (float(b.l), float(b.t), float(b.r), float(b.b))
            except Exception:
                bbox = None

    # Heading level metadata, if any
    meta: dict = {"docling_label": label_str}
    level = getattr(item, "level", None)
    if level is not None:
        meta["heading_level"] = int(level)

    # Text content depends on block type
    if block_type == "table":
        # Docling tables expose richer structure than plain text. We capture:
        #   - markdown (for retrieval/embedding — lossy on merges)
        #   - cells with row_span/col_span (lossless — downstream uses this)
        #   - HTML (for visual rendering with merges preserved)
        text = ""
        try:
            text = item.export_to_markdown()  # type: ignore[attr-defined]
        except Exception:
            text = getattr(item, "text", "") or ""

        data = getattr(item, "data", None)
        if data is not None:
            try:
                meta["n_rows"] = int(getattr(data, "num_rows", 0))
                meta["n_cols"] = int(getattr(data, "num_cols", 0))
            except Exception:
                pass

            # Capture every cell with its true span info. Skip auto-generated
            # "spanned" duplicate references so we keep one logical cell per merge.
            cells_meta: list[dict] = []
            try:
                table_cells = getattr(data, "table_cells", None) or []
                # Track which (row, col) positions we've already emitted via a
                # multi-cell span to avoid double-counting.
                emitted: set[tuple[int, int]] = set()
                for c in table_cells:
                    r0 = int(getattr(c, "start_row_offset_idx", 0))
                    r1 = int(getattr(c, "end_row_offset_idx", r0 + 1))
                    c0 = int(getattr(c, "start_col_offset_idx", 0))
                    c1 = int(getattr(c, "end_col_offset_idx", c0 + 1))
                    if (r0, c0) in emitted:
                        continue
                    emitted.add((r0, c0))
                    cell_text = getattr(c, "text", "") or ""
                    cell_text = str(cell_text).strip()
                    is_header = bool(
                        getattr(c, "column_header", False)
                        or getattr(c, "row_header", False)
                        or getattr(c, "row_section", False)
                    )
                    cells_meta.append(
                        {
                            "text": cell_text,
                            "row": r0,
                            "col": c0,
                            "row_span": max(1, r1 - r0),
                            "col_span": max(1, c1 - c0),
                            "is_header": is_header,
                        }
                    )
                if cells_meta:
                    meta["cells"] = cells_meta
                    meta["has_merges"] = any(
                        c["row_span"] > 1 or c["col_span"] > 1 for c in cells_meta
                    )
            except Exception as e:
                meta["cells_error"] = str(e)

            # Fallback: docling 2.x also exposes a flattened grid for compatibility.
            try:
                grid = getattr(data, "grid", None)
                if grid and "cells" not in meta:
                    meta["grid"] = [[(c.text if hasattr(c, "text") else str(c)) for c in row] for row in grid]
            except Exception:
                pass

        # Build HTML representation that preserves merges so the UI can render it.
        if meta.get("cells"):
            meta["html"] = _cells_to_html(
                meta["cells"], meta.get("n_rows", 0), meta.get("n_cols", 0)
            )
    elif block_type == "figure":
        # Figures have no inherent text but we MUST surface them so chunking +
        # downstream visual extraction (VLM description) can find them.
        text = getattr(item, "text", None) or ""
        if not text.strip():
            text = "[figure]"  # placeholder so the block isn't dropped downstream
        meta["is_visual"] = True
    else:
        text = getattr(item, "text", None) or ""

    if block_type != "figure" and (not text or not text.strip()):
        return None

    # Capture any footnote-marker references in the block's text so the
    # chunking stage can later resolve each to its definition (which may
    # live on a different page).
    markers = _extract_footnote_markers(text)
    if markers:
        meta["footnote_markers"] = markers

    # If this block IS a footnote, extract its leading marker (if any) so it
    # can be matched against references elsewhere.
    if block_type == "footnote":
        stripped = text.strip()
        # patterns: "a foo", "(a) foo", "* foo", "1 foo", "1. foo"
        m = re.match(r"^[\(\[]?\s*([a-z]|\d{1,3}|[\*†‡§¶])\s*[\)\]\.\:]?\s+", stripped)
        if m:
            meta["footnote_marker"] = m.group(1)

    return RemoteBlock(
        block_type=block_type,  # type: ignore[arg-type]
        text=text.strip(),
        page_num=page_num,
        bbox=bbox,
        order_index=order,
        metadata=meta,
    )


def _contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Group sorted 0-indexed page numbers into contiguous (inclusive) ranges."""
    if not pages:
        return []
    pages = sorted(set(pages))
    ranges: list[tuple[int, int]] = []
    start = pages[0]
    prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))
    return ranges


def _convert_range(pdf_path: str, range_1indexed: tuple[int, int]) -> tuple[dict[int, list[RemoteBlock]], dict[int, float]]:
    """Run Docling on a single contiguous 1-indexed page range. Returns
    (per-page block index, per-page height map)."""
    converter = _get_docling()
    result = converter.convert(pdf_path, page_range=range_1indexed)
    doc = result.document

    page_heights: dict[int, float] = {}
    pages_attr = getattr(doc, "pages", {}) or {}
    try:
        iterable = pages_attr.items() if hasattr(pages_attr, "items") else enumerate(pages_attr)
        for key, pg in iterable:
            size = getattr(pg, "size", None)
            if size is not None:
                h = float(getattr(size, "height", 0) or 0)
                if h > 0:
                    page_heights[int(key) - 1] = h
    except Exception:
        pass

    order_per_page: dict[int, int] = {}
    index: dict[int, list[RemoteBlock]] = {}
    items_iter = doc.iterate_items() if hasattr(doc, "iterate_items") else []
    for entry in items_iter:
        item = entry[0] if isinstance(entry, tuple) else entry
        prov = getattr(item, "prov", None)
        page_no = None
        if prov and len(prov) > 0:
            page_no = getattr(prov[0], "page_no", None)
        if page_no is None:
            page_no = getattr(item, "page_no", None)
        if page_no is None:
            continue
        zero_idx = page_no - 1 if page_no >= 1 else page_no
        order = order_per_page.get(zero_idx, 0)
        block = _docling_item_to_block(item, order=order, page_num=zero_idx, page_heights=page_heights)
        if block is None:
            continue
        index.setdefault(zero_idx, []).append(block)
        order_per_page[zero_idx] = order + 1
    return index, page_heights


def _convert_and_index(
    pdf_bytes: bytes,
    pages: Optional[list[int]] = None,
) -> tuple[str, dict[int, list[RemoteBlock]], float, bool]:
    """Convert a PDF with Docling and index its blocks by zero-indexed page.

    If `pages` is provided (list of 0-indexed page numbers), only those pages
    are converted, batched into contiguous ranges so Docling can skip work.
    A range_request like [17, 18, 50, 51] becomes two convert calls
    (page_range=(18,19) and (51,52)) — much cheaper than converting the whole PDF.

    Cache key = (sha256, frozenset(requested_pages_or_ALL)). Pages not requested
    in the current call remain unconverted. Cache is bounded by
    DOCLING_CACHE_MAX_DOCS (LRU).
    """
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    cache_key_pages = frozenset(pages) if pages is not None else None  # None = full doc
    cache_key = (sha, cache_key_pages)

    with _docling_cache_lock:
        # Exact-match cache hit.
        if cache_key in _docling_cache:
            _docling_cache.move_to_end(cache_key)
            return sha, _docling_cache[cache_key], 0.0, True
        # Partial-match: if we have a full-doc cache for this PDF, use it.
        full_key = (sha, None)
        if full_key in _docling_cache:
            _docling_cache.move_to_end(full_key)
            full_index = _docling_cache[full_key]
            if pages is None:
                return sha, full_index, 0.0, True
            return sha, {p: full_index.get(p, []) for p in pages}, 0.0, True

    import tempfile as _tempfile

    t0 = time.perf_counter()
    with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        merged: dict[int, list[RemoteBlock]] = {}
        if pages is None:
            # Full doc convert.
            idx, _ = _convert_range(tmp_path, (1, 9223372036854775807))
            merged = idx
        else:
            ranges = _contiguous_ranges(pages)
            for (a, b) in ranges:
                idx, _ = _convert_range(tmp_path, (a + 1, b + 1))
                merged.update(idx)
            # Ensure every requested page has an entry (even if empty).
            for p in pages:
                merged.setdefault(p, [])
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    convert_ms = (time.perf_counter() - t0) * 1000.0

    with _docling_cache_lock:
        _docling_cache[cache_key] = merged
        _docling_cache.move_to_end(cache_key)
        while len(_docling_cache) > DOCLING_CACHE_MAX_DOCS:
            _docling_cache.popitem(last=False)

    return sha, merged, convert_ms, False


@app.post("/parse/docling", response_model=ParseResponse)
async def parse_docling(
    page_index: int = Query(..., ge=0),
    file: UploadFile = File(...),
) -> ParseResponse:
    """Return Docling-extracted blocks for a single page of an uploaded PDF.

    On cache miss, the entire PDF is converted once (~60s for a 130-page doc)
    and indexed by page. Subsequent calls for any page of the same PDF return
    in milliseconds. Cache is keyed by PDF SHA-256.
    """
    start = time.perf_counter()
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    warnings: list[str] = []
    try:
        sha, index, convert_ms, was_cached = _convert_and_index(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docling convert failed: {e}")

    blocks = index.get(page_index, [])
    if not blocks and was_cached:
        warnings.append(f"page {page_index} present in cached doc but produced no blocks")
    if convert_ms > 0:
        warnings.append(f"cold convert took {convert_ms:.0f} ms (cached for subsequent pages)")

    return ParseResponse(
        parser="docling",
        page_num=page_index,
        blocks=blocks,
        duration_ms=(time.perf_counter() - start) * 1000.0,
        warnings=warnings,
    )


@app.post("/parse/docling/document", response_model=DoclingDocumentResponse)
async def parse_docling_document(
    file: UploadFile = File(...),
    pages: Optional[str] = Query(None, description="comma-separated 0-indexed page numbers; if omitted, converts entire PDF"),
) -> DoclingDocumentResponse:
    """Convert specific pages (or the whole PDF) with Docling in one call.

    Passing `pages=17,18,50,51` is much cheaper than converting the whole PDF
    because Docling only processes the requested pages (grouped into
    contiguous ranges server-side).
    """
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    page_list: Optional[list[int]] = None
    if pages is not None and pages.strip():
        try:
            page_list = sorted({int(x) for x in pages.split(",") if x.strip()})
        except ValueError:
            raise HTTPException(status_code=400, detail="pages must be comma-separated integers")
    try:
        sha, index, convert_ms, was_cached = _convert_and_index(pdf_bytes, pages=page_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docling convert failed: {e}")
    n_pages = max(index.keys(), default=-1) + 1
    return DoclingDocumentResponse(
        sha256=sha,
        n_pages=n_pages,
        pages=index,
        convert_duration_ms=convert_ms,
        cached=was_cached,
    )


@app.post("/parse/docling/cache/clear")
def clear_docling_cache() -> dict:
    """Wipe the in-memory Docling cache. Use after deploying new server code."""
    with _docling_cache_lock:
        n = len(_docling_cache)
        _docling_cache.clear()
    return {"cleared": n}


# ---------- Qwen2.5-VL ----------

QWEN_DEFAULT_MODEL = os.getenv("FMLS_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

QWEN_PROMPT = (
    "You are a precise document layout extractor. The image is a single page of a "
    "clinical-trial protocol PDF. Extract every block of content in reading order. "
    "For each block return JSON with fields: block_type (one of paragraph, heading, "
    "table, list, figure, caption, header, footer, other), text (the literal text; "
    "for tables emit GitHub markdown), and bbox (x0,y0,x1,y1 in pixel coords relative "
    "to the input image, or null). Return ONLY a JSON array, nothing else."
)

QWEN_TABLE_PROMPT = (
    "The image is a single page of a clinical-trial protocol containing a complex "
    "table — typically a Schedule of Activities (SoA), a Schedule of Events, or a "
    "biomarker/visit matrix. Such tables have nested column-group headers, merged "
    "cells, rotated text, and footnote markers (a, b, *, †) tied to legend lines.\n\n"
    "Extract the table as an HTML <table> preserving structure: use rowspan / "
    "colspan attributes for merged cells; use <th> for header cells; preserve any "
    "footnote markers exactly as they appear (including superscripts). After the "
    "table, list any footnote definitions that appear on the page.\n\n"
    "Return ONLY a single JSON object with these fields:\n"
    "  table_html: string, the HTML table\n"
    "  footnote_lines: array of strings, each a footnote definition line found on the page\n"
    "  notes: array of strings, any caveats about uncertainty\n"
    "No prose outside the JSON object."
)


# Per HF model card: cap visual tokens to keep prefill + CPU preprocessing fast.
# 1280 * 28 * 28 = 1,003,520 max pixels is the recommended ceiling.
QWEN_MIN_PIXELS = int(os.getenv("FMLS_QWEN_MIN_PIXELS", str(256 * 28 * 28)))
QWEN_MAX_PIXELS = int(os.getenv("FMLS_QWEN_MAX_PIXELS", str(1280 * 28 * 28)))


def _get_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        _qwen_processor = AutoProcessor.from_pretrained(
            QWEN_DEFAULT_MODEL,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
        )

        # Try flash_attention_2 first (much faster); fall back to sdpa if not
        # installed (flash-attn must be compiled against the CUDA toolchain).
        try:
            _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_DEFAULT_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError, RuntimeError):
            _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_DEFAULT_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )
    return _qwen_model, _qwen_processor


@app.post("/parse/qwen", response_model=ParseResponse)
async def parse_qwen(
    page_index: int = Query(..., ge=0),
    file: UploadFile = File(...),
) -> ParseResponse:
    """Run Qwen2.5-VL on a single rasterized page image (PNG)."""
    import json

    start = time.perf_counter()
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    warnings: list[str] = []
    blocks: list[RemoteBlock] = []
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        model, processor = _get_qwen()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QWEN_PROMPT},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        # Reduce max_new_tokens: HF model card recommends 128 for document tasks.
        # We give it more headroom (1024) for multi-block reading-order output
        # but cap to avoid the 4096-token tail we were paying before.
        out_ids = model.generate(**inputs, max_new_tokens=int(os.getenv("FMLS_QWEN_MAX_NEW_TOKENS", "1024")))
        out_ids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        raw = processor.batch_decode(out_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Strip code fences if the model wrapped its JSON.
        raw_stripped = raw.strip()
        if raw_stripped.startswith("```"):
            raw_stripped = raw_stripped.strip("`")
            if raw_stripped.lower().startswith("json"):
                raw_stripped = raw_stripped[4:]
        try:
            parsed = json.loads(raw_stripped)
        except json.JSONDecodeError as e:
            warnings.append(f"qwen output not valid JSON, falling back to single text block: {e}")
            parsed = [{"block_type": "other", "text": raw_stripped, "bbox": None}]

        if not isinstance(parsed, list):
            parsed = [parsed]
        for order, item in enumerate(parsed):
            bt = (item.get("block_type") or "paragraph").lower()
            if bt not in {
                "paragraph", "heading", "table", "list", "figure", "caption", "header", "footer", "other"
            }:
                bt = "other"
            text = (item.get("text") or "").strip()
            if not text:
                continue
            bb = item.get("bbox")
            bbox = None
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                except Exception:
                    bbox = None
            blocks.append(
                RemoteBlock(
                    block_type=bt,  # type: ignore[arg-type]
                    text=text,
                    page_num=page_index,
                    bbox=bbox,
                    order_index=order,
                    metadata={"source": "qwen_vl"},
                )
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qwen parse failed: {e}")

    return ParseResponse(
        parser="qwen_vl",
        page_num=page_index,
        blocks=blocks,
        duration_ms=(time.perf_counter() - start) * 1000.0,
        warnings=warnings,
    )


@app.post("/parse/qwen/table", response_model=ParseResponse)
async def parse_qwen_table(
    page_index: int = Query(..., ge=0),
    file: UploadFile = File(...),
) -> ParseResponse:
    """Run Qwen2.5-VL with a TABLE-specific prompt on a rasterized page image.

    Designed for SoA / Schedule of Events tables with nested headers, merged
    cells, and footnote markers — cases where Docling's layout model loses
    structure. Returns a single TABLE block whose `metadata["html"]` is the
    Qwen-generated HTML table, plus zero or more FOOTNOTE blocks.
    """
    import json

    start = time.perf_counter()
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    warnings: list[str] = []
    blocks: list[RemoteBlock] = []
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        model, processor = _get_qwen()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QWEN_TABLE_PROMPT},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        # Tables need more tokens than plain reading-order extraction; default
        # 2048 covers most SoA tables. Tunable via env if you see truncation.
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("FMLS_QWEN_TABLE_MAX_NEW_TOKENS", "2048")),
        )
        out_ids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        raw = processor.batch_decode(out_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        raw_stripped = raw.strip()
        # Strip optional code fences.
        if raw_stripped.startswith("```"):
            raw_stripped = raw_stripped.strip("`")
            if raw_stripped.lower().startswith("json"):
                raw_stripped = raw_stripped[4:].strip()
        # Find the first { and last } and parse between.
        try:
            obj_start = raw_stripped.index("{")
            obj_end = raw_stripped.rindex("}")
            payload = json.loads(raw_stripped[obj_start : obj_end + 1])
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(f"qwen table JSON parse failed: {e}; returning raw text block")
            blocks.append(RemoteBlock(
                block_type="other", text=raw_stripped, page_num=page_index,
                order_index=0, metadata={"source": "qwen_table", "parse_error": True},
            ))
            return ParseResponse(
                parser="qwen_vl", page_num=page_index, blocks=blocks,
                duration_ms=(time.perf_counter() - start) * 1000.0, warnings=warnings,
            )

        html = (payload.get("table_html") or "").strip()
        if html:
            blocks.append(RemoteBlock(
                block_type="table",
                text=html,  # keep HTML in text; rendering layer treats as HTML
                page_num=page_index,
                order_index=0,
                metadata={"source": "qwen_table", "html": html, "has_merges": "rowspan" in html.lower() or "colspan" in html.lower()},
            ))
        for i, fn_line in enumerate(payload.get("footnote_lines") or []):
            line = str(fn_line).strip()
            if not line:
                continue
            marker_match = re.match(r"^[\(\[]?\s*([a-z]|\d{1,3}|[\*†‡§¶])\s*[\)\]\.\:]?\s+", line)
            meta = {"source": "qwen_table"}
            if marker_match:
                meta["footnote_marker"] = marker_match.group(1)
            blocks.append(RemoteBlock(
                block_type="footnote",
                text=line,
                page_num=page_index,
                order_index=len(blocks),
                metadata=meta,
            ))
        for note in payload.get("notes") or []:
            warnings.append(str(note))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qwen table parse failed: {e}")

    return ParseResponse(
        parser="qwen_vl",
        page_num=page_index,
        blocks=blocks,
        duration_ms=(time.perf_counter() - start) * 1000.0,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
