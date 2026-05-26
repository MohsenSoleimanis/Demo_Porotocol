"""FastAPI viewer for parser results.

Replaces the Streamlit UI for inspecting parsed protocols. Streamlit re-runs
the entire script on every interaction and holds heavy data in session state,
which crashes on 30+ MB result trees. This viewer is stateless and
server-rendered — the browser only fetches what changes.

Routes:
  GET /                                index: list parsed docs from corpus/results/
  GET /docs/{stem}                     doc overview: triage table + route distribution
  GET /docs/{stem}/p/{page}            single-page detail: image + bbox overlay + blocks
  GET /docs/{stem}/image/{page}.png    rendered page image (PIL, cached in memory)
  GET /docs/{stem}/raw                 raw JSON download

Start:
  .venv/Scripts/python.exe -m uvicorn viewer.app:app --reload --port 8600
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fitz
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = CORPUS_DIR / "results"
VIEWER_DIR = Path(__file__).resolve().parent

app = FastAPI(title="FMLS Parser Viewer", version="0.1.0")
templates = Jinja2Templates(directory=str(VIEWER_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(VIEWER_DIR / "static")), name="static")

BLOCK_COLORS = {
    "heading": "#1f77b4",
    "paragraph": "#2ca02c",
    "table": "#d62728",
    "list": "#ff7f0e",
    "figure": "#9467bd",
    "caption": "#8c564b",
    "header": "#7f7f7f",
    "footer": "#7f7f7f",
    "footnote": "#17becf",
    "other": "#bcbd22",
}


# ---------- Data loading (cached) ----------


def _remap_mineru_block_types(payload: dict) -> dict:
    """Backward-compat fix: when block_type=='table' but metadata.mineru_type is
    `table_caption` / `table_footnote` / `page_number`, remap to the correct
    type. This corrects results saved before the mineru_server.py mapping fix.
    """
    mapping = {
        "table_caption": "caption",
        "table_title": "caption",
        "table_footnote": "footnote",
        "page_number": "other",
        "page-number": "other",
    }
    for page in payload.get("pages") or []:
        for b in page.get("blocks") or []:
            meta = b.get("metadata") or {}
            ml = (meta.get("mineru_type") or "").lower()
            if not ml:
                continue
            new_type = mapping.get(ml)
            if new_type and b.get("block_type") != new_type:
                b["block_type"] = new_type
    return payload


# Cache parsed payloads keyed by (path, mtime) so the viewer auto-reloads
# when a pipeline re-run rewrites the file. Without this, lru_cache holds the
# stale payload forever and the user sees old data after a re-parse.
_RESULT_CACHE: dict[str, tuple[float, dict]] = {}


def _load_result(stem: str) -> dict:
    """Load a parsed result. Search the eval-dataset layout first
    (dataset/{stem}/parsed.json), fall back to legacy corpus/results/{stem}.json.
    """
    candidates = [
        ROOT / "dataset" / stem / "parsed.json",
        RESULTS_DIR / f"{stem}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        key = str(path)
        cached = _RESULT_CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = _remap_mineru_block_types(payload)
        _RESULT_CACHE[key] = (mtime, payload)
        # The page-image overlay + figure-crop caches embed the OLD payload's
        # bboxes — invalidate them all when the underlying parsed.json changes.
        try:
            _render_with_overlay.cache_clear()
            _render_crop.cache_clear()
        except (NameError, AttributeError):
            pass
        return payload
    raise HTTPException(status_code=404, detail=f"no result for {stem}")


def _find_pdf_for_stem(stem: str) -> Optional[Path]:
    """Find the source PDF for a result file.

    Priority:
      0. dataset/{stem}/{stem}.pdf (new eval-dataset layout)
      1. The `source_filename` recorded in the result JSON itself.
      2. Manifest entry whose stem matches.
      3. <stem>.pdf in corpus/ or project root.
      4. Any PDF in the project tree whose stem matches.
    """
    candidates: list[Path] = []
    # 0. Eval-dataset layout.
    dataset_pdf = ROOT / "dataset" / stem / f"{stem}.pdf"
    if dataset_pdf.exists():
        return dataset_pdf
    # Some dataset entries might have a different inner filename
    dataset_dir = ROOT / "dataset" / stem
    if dataset_dir.exists():
        for p in dataset_dir.glob("*.pdf"):
            return p

    # 1. Use source_filename from the result JSON.
    try:
        payload = _load_result(stem)
        src_fn = payload.get("source_filename")
        if src_fn:
            candidates.append(CORPUS_DIR / src_fn)
            candidates.append(ROOT / src_fn)
        src_path = payload.get("source_path")
        if src_path:
            candidates.append(Path(src_path))
    except HTTPException:
        pass

    # 2. Manifest match.
    manifest_path = CORPUS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            for entry in json.loads(manifest_path.read_text()):
                if Path(entry.get("filename", "")).stem == stem:
                    candidates.append(CORPUS_DIR / entry["filename"])
        except Exception:
            pass

    # 3. Direct stem match.
    candidates.append(CORPUS_DIR / f"{stem}.pdf")
    candidates.append(ROOT / f"{stem}.pdf")

    # 4. Any matching-stem PDF anywhere in the tree (cheap glob).
    for p in ROOT.glob("*.pdf"):
        if p.stem == stem:
            candidates.append(p)
    for p in CORPUS_DIR.glob("*.pdf"):
        if p.stem == stem:
            candidates.append(p)

    for c in candidates:
        if c.exists():
            return c
    return None


@lru_cache(maxsize=16)
def _pdf_handle(pdf_path: str) -> fitz.Document:
    return fitz.open(pdf_path)


def _list_results() -> list[dict]:
    """Scan both legacy `corpus/results/*.json` AND the eval-dataset layout
    `dataset/{NCT}/parsed.json`. Newer eval-dataset entries take precedence
    when stems collide."""
    out = []
    seen_stems: set[str] = set()
    manifest_by_stem: dict[str, dict] = {}

    # --- eval dataset (dataset/{NCT}/parsed.json) ---
    dataset_dir = ROOT / "dataset"
    ds_manifest = dataset_dir / "manifest.json"
    ds_manifest_by_nct: dict[str, dict] = {}
    if ds_manifest.exists():
        try:
            for entry in json.loads(ds_manifest.read_text()):
                ds_manifest_by_nct[entry.get("nct_id", "")] = entry
        except Exception:
            pass
    if dataset_dir.exists():
        for jf in sorted(dataset_dir.glob("*/parsed.json")):
            nct = jf.parent.name
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = ds_manifest_by_nct.get(nct, {})
            out.append({
                "stem": nct,                # use NCT as stem
                "source": "dataset",
                "filename": payload.get("source_filename", f"{nct}.pdf"),
                "total_pages": payload.get("total_pages", 0),
                "stratum": meta.get("stratum", ""),
                "title": (meta.get("title") or "")[:90],
                "sponsor": meta.get("sponsor", ""),
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
                "routes": _routes(payload),
                "size_kb": jf.stat().st_size // 1024,
            })
            seen_stems.add(nct)

    # --- legacy corpus/results/*.json ---
    if RESULTS_DIR.exists():
        mpath = CORPUS_DIR / "manifest.json"
        if mpath.exists():
            try:
                for entry in json.loads(mpath.read_text()):
                    manifest_by_stem[Path(entry.get("filename", "")).stem] = entry
            except Exception:
                pass
        for jf in sorted(RESULTS_DIR.glob("*.json")):
            if jf.stem == "summary":
                continue
            if jf.stem in seen_stems:
                continue
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = manifest_by_stem.get(jf.stem, {})
            out.append({
                "stem": jf.stem,
                "source": "corpus",
                "filename": payload.get("source_filename", jf.stem + ".pdf"),
                "total_pages": payload.get("total_pages", 0),
                "stratum": meta.get("stratum", ""),
                "title": (meta.get("title") or "")[:90],
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
                "routes": _routes(payload),
                "size_kb": jf.stat().st_size // 1024,
            })
    return out


def _routes(payload: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in payload.get("pages", []):
        r = p.get("parser_used") or "unknown"
        out[r] = out.get(r, 0) + 1
    return out


# ---------- Routes ----------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    items = _list_results()
    return templates.TemplateResponse(
        request, "index.html", {"items": items, "n": len(items)},
    )


def _load_chunks(stem: str) -> Optional[dict]:
    """Load chunks.json for a doc if present (eval-dataset layout only)."""
    path = ROOT / "dataset" / stem / "chunks.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    key = f"chunks::{path}"
    cached = _RESULT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    _RESULT_CACHE[key] = (mtime, payload)
    return payload


# Canonical M11 top-level sections — we use this to compute coverage.
M11_TOP_SECTIONS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]


def _build_doc_tree(chunks_doc: dict) -> dict:
    """From a ChunkedDocument, build:
      - tree: nested {section_path[-1]: {m11, page, chunks_in_section, children}}
      - m11_coverage: {present: list, missing: list, percent: float}
      - per-section chunk count (only countable content blocks)
    """
    chunks = chunks_doc.get("chunks") or []
    # Per-section chunk counts (countable content types only).
    countable = {"paragraph", "heading", "list", "table", "figure", "caption", "footnote"}
    counts: dict[tuple, int] = {}
    first_page_for: dict[tuple, int] = {}
    first_chunk_for: dict[tuple, str] = {}
    for c in chunks:
        if c.get("block_type") not in countable:
            continue
        path = tuple(c.get("section_path") or [])
        counts[path] = counts.get(path, 0) + 1
        if path not in first_page_for:
            first_page_for[path] = c.get("page_num", 0)
            first_chunk_for[path] = c.get("chunk_id")
    # Build tree by walking section_index in order.
    root: dict = {"children": [], "_index": {}}
    for entry in chunks_doc.get("section_index") or []:
        path = entry.get("section_path") or []
        m11 = entry.get("m11_section")
        page = entry.get("page_num", 0)
        # Walk path, inserting nodes.
        cur = root
        for i, label in enumerate(path):
            key = tuple(path[: i + 1])
            if key not in cur["_index"]:
                node = {
                    "label": label,
                    "path": list(key),
                    "m11_section": m11 if i == len(path) - 1 else None,
                    "page_num": page,
                    "first_chunk_id": first_chunk_for.get(key),
                    "chunk_count": counts.get(key, 0),
                    "children": [],
                    "_index": {},
                }
                cur["_index"][key] = node
                cur["children"].append(node)
            else:
                node = cur["_index"][key]
                # If this entry assigns an m11 to the leaf, keep it.
                if i == len(path) - 1 and m11 and not node.get("m11_section"):
                    node["m11_section"] = m11
            cur = node
    # Drop internal `_index` before returning.
    def _strip(node):
        node.pop("_index", None)
        for ch in node.get("children", []):
            _strip(ch)
    _strip(root)

    # M11 coverage: which top-level §1..§11 are present?
    present_top = set()
    for entry in chunks_doc.get("section_index") or []:
        m = entry.get("m11_section")
        if m:
            top = m.split(".")[0]
            present_top.add(top)
    present = [t for t in M11_TOP_SECTIONS if t in present_top]
    missing = [t for t in M11_TOP_SECTIONS if t not in present_top]
    coverage = {
        "present": present,
        "missing": missing,
        "percent": round(100.0 * len(present) / len(M11_TOP_SECTIONS), 1),
    }
    return {"tree": root["children"], "m11_coverage": coverage}


@app.get("/docs/{stem}", response_class=HTMLResponse)
def doc_overview(stem: str, request: Request):
    payload = _load_result(stem)
    pages = payload.get("pages", [])
    rows = []
    for p in pages:
        triage = p.get("triage", {}) or {}
        feats = triage.get("features", {}) or {}
        type_counts: dict[str, int] = {}
        for b in p.get("blocks", []):
            type_counts[b.get("block_type", "other")] = type_counts.get(b.get("block_type", "other"), 0) + 1
        rows.append({
            "page_num": p.get("page_num"),
            "route": p.get("parser_used"),
            "status": p.get("parse_status"),
            "blocks": len(p.get("blocks", [])),
            "duration_ms": p.get("parse_duration_ms", 0),
            "reason": (triage.get("reason") or "")[:120],
            "n_tables": feats.get("likely_table_count", 0),
            "n_cols": feats.get("column_count_estimate", 0),
            "scanned": feats.get("is_likely_scanned", False),
            "type_counts": type_counts,
        })
    # Load chunks.json if present so the overview can show the document tree.
    chunks_doc = _load_chunks(stem)
    tree_data = _build_doc_tree(chunks_doc) if chunks_doc else None
    return templates.TemplateResponse(
        request,
        "doc.html",
        {
            "stem": stem,
            "filename": payload.get("source_filename", stem + ".pdf"),
            "total_pages": payload.get("total_pages", 0),
            "stage_timings_ms": payload.get("stage_timings_ms", {}),
            "rows": rows,
            "routes": _routes(payload),
            "block_colors": BLOCK_COLORS,
            "tree_data": tree_data,
        },
    )


@app.get("/docs/{stem}/p/{page_idx}", response_class=HTMLResponse)
def page_detail(stem: str, page_idx: int, request: Request):
    payload = _load_result(stem)
    pages = payload.get("pages", [])
    if page_idx < 0 or page_idx >= len(pages):
        raise HTTPException(status_code=404, detail="page out of range")
    page = pages[page_idx]
    # NB: hierarchy was deliberately NOT inferred here — MinerU's
    # content_list.json gives a model-provided text_level instead. See pipeline.
    # Render the page image once (lazy) so the template can <img src=...>
    pdf_path = _find_pdf_for_stem(stem)
    image_available = pdf_path is not None
    prev_idx = page_idx - 1 if page_idx > 0 else None
    next_idx = page_idx + 1 if page_idx + 1 < len(pages) else None
    return templates.TemplateResponse(
        request,
        "page.html",
        {
            "stem": stem,
            "page": page,
            "page_idx": page_idx,
            "total_pages": len(pages),
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "image_available": image_available,
            "block_colors": BLOCK_COLORS,
        },
    )


@app.get("/docs/{stem}/image/{page_idx}.png")
def page_image(stem: str, page_idx: int, dpi: int = 110, overlay: bool = True):
    """Render the page as a PNG with optional bbox overlays.

    Cached in-memory by (stem, page_idx, dpi, overlay) so repeated nav is fast.
    """
    pdf_path = _find_pdf_for_stem(stem)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"no source PDF found for {stem}")
    img_bytes = _render_with_overlay(stem, str(pdf_path), page_idx, dpi, overlay)
    return Response(content=img_bytes, media_type="image/png")


ARROW_COLOR = "#ef4444"  # red — pops against block colors


@lru_cache(maxsize=128)
def _render_with_overlay(stem: str, pdf_path: str, page_idx: int, dpi: int, overlay: bool) -> bytes:
    doc = _pdf_handle(pdf_path)
    page = doc[page_idx]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    if overlay:
        try:
            payload = _load_result(stem)
            page_obj = payload["pages"][page_idx]
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            sx = img.width / page_w
            sy = img.height / page_h
            draw = ImageDraw.Draw(img, "RGBA")
            for b in page_obj.get("blocks", []):
                bb = b.get("bbox")
                if not bb:
                    continue
                color = BLOCK_COLORS.get(b.get("block_type", "other"), "#666")
                x0, y0, x1, y1 = bb["x0"] * sx, bb["y0"] * sy, bb["x1"] * sx, bb["y1"] * sy
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                draw.text((x0 + 2, y0 + 2), f"{b.get('order_index')}:{b.get('block_type')}", fill=color)
                # Detected arrows attached to this block
                meta = b.get("metadata") or {}
                for ar in (meta.get("arrows") or []) + (meta.get("page_level_arrows") or []):
                    _draw_arrow(draw, ar, sx, sy)
        except Exception:
            pass
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _draw_arrow(draw: ImageDraw.ImageDraw, ar: dict, sx: float, sy: float) -> None:
    """Draw a horizontal/vertical arrow with one or two arrowheads."""
    s = ar.get("start") or [0, 0]
    e = ar.get("end") or [0, 0]
    x0, y0 = s[0] * sx, s[1] * sy
    x1, y1 = e[0] * sx, e[1] * sy
    draw.line([x0, y0, x1, y1], fill=ARROW_COLOR, width=2)
    head_len = 10 * sx if ar.get("axis") == "horizontal" else 10 * sy
    # head at endpoint e (always drawn). Direction: from start toward end.
    _draw_head(draw, x1, y1, dx=(x1 - x0), dy=(y1 - y0), head_len=head_len)
    if ar.get("two_headed"):
        # second head at start, pointing back
        _draw_head(draw, x0, y0, dx=-(x1 - x0), dy=-(y1 - y0), head_len=head_len)


def _draw_head(draw, x: float, y: float, dx: float, dy: float, head_len: float) -> None:
    import math
    angle = math.atan2(dy, dx)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    spread = math.radians(25)
    # Two legs of the arrowhead, going BACKWARD from the tip
    for sign in (-1, 1):
        a = angle + math.pi + sign * spread
        lx = x + head_len * math.cos(a)
        ly = y + head_len * math.sin(a)
        draw.line([x, y, lx, ly], fill=ARROW_COLOR, width=2)


@app.get("/docs/{stem}/crop/{page_idx}/{block_idx}.png")
def block_crop(stem: str, page_idx: int, block_idx: int, dpi: int = 150):
    """Crop the rendered page image to the bbox of one block (e.g. a figure)."""
    pdf_path = _find_pdf_for_stem(stem)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"no source PDF for {stem}")
    payload = _load_result(stem)
    pages = payload.get("pages", [])
    if page_idx < 0 or page_idx >= len(pages):
        raise HTTPException(status_code=404, detail="page out of range")
    blocks = pages[page_idx].get("blocks") or []
    target = None
    for b in blocks:
        if b.get("order_index") == block_idx:
            target = b
            break
    if target is None or not target.get("bbox"):
        raise HTTPException(status_code=404, detail="block or bbox not found")
    bb = target["bbox"]
    img_bytes = _render_crop(stem, str(pdf_path), page_idx, dpi,
                              bb["x0"], bb["y0"], bb["x1"], bb["y1"])
    return Response(content=img_bytes, media_type="image/png")


@lru_cache(maxsize=256)
def _render_crop(stem: str, pdf_path: str, page_idx: int, dpi: int,
                 x0: float, y0: float, x1: float, y1: float) -> bytes:
    doc = _pdf_handle(pdf_path)
    page = doc[page_idx]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    sx = img.width / float(page.rect.width)
    sy = img.height / float(page.rect.height)
    pad = 4
    crop = img.crop((
        max(0, int(x0 * sx) - pad),
        max(0, int(y0 * sy) - pad),
        min(img.width, int(x1 * sx) + pad),
        min(img.height, int(y1 * sy) + pad),
    ))
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@app.get("/docs/{stem}/raw")
def raw_json(stem: str):
    return JSONResponse(_load_result(stem))
