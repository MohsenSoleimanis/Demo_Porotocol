"""MinerU 2.5 + vLLM serving — replaces Docling + Qwen-VL for structured pages.

This module is a drop-in addition to server.py. We keep the legacy
/parse/docling, /parse/qwen, /parse/qwen/table endpoints reachable as
fallbacks, but the primary structured-content route is /parse/mineru.

Architecture decisions:
  - MinerU takes IMAGES, not PDFs. We render each requested page as PNG
    server-side using PyMuPDF so we don't pay the wire cost of uploading
    the PDF repeatedly.
  - The vLLM-engine backend uses synchronous batching. We send all
    requested pages as one `batch_two_step_extract` call so vLLM can
    schedule them efficiently.
  - Output is normalized into our RemoteBlock schema (type, text, bbox,
    page_num, order_index) so downstream code doesn't care which engine
    produced the page.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .schemas import RemoteBlock

# ---- module-level lazy globals ----

_mineru_client = None
_mineru_lock = threading.Lock()

MINERU_MODEL = os.getenv("FMLS_MINERU_MODEL", "opendatalab/MinerU2.5-2509-1.2B")
# Page-render DPI. 144 is plenty for MinerU; it does its own downsampling
# in the global-layout stage.
MINERU_RENDER_DPI = int(os.getenv("FMLS_MINERU_RENDER_DPI", "144"))

router = APIRouter(prefix="/parse/mineru", tags=["mineru"])


class MineruDocumentResponse(BaseModel):
    parser: str = Field(default="mineru")
    sha256: str
    n_pages_requested: int
    pages: dict[int, list[RemoteBlock]]
    convert_duration_ms: float


def _get_mineru_client():
    """Lazy-load the MinerU + vLLM client. Holds the GPU."""
    global _mineru_client
    if _mineru_client is not None:
        return _mineru_client
    with _mineru_lock:
        if _mineru_client is not None:
            return _mineru_client
        from vllm import LLM
        from mineru_vl_utils import MinerUClient

        try:
            from mineru_vl_utils import MinerULogitsProcessor
            logits_processors = [MinerULogitsProcessor]
        except ImportError:
            logits_processors = None

        llm_kwargs = {
            "model": MINERU_MODEL,
            # Skip CUDA graph capture + some Triton paths that fail on managed
            # GPUs without nvcc / dev headers. ~10-20% slower but reliable.
            "enforce_eager": True,
            # 1.2B model on L4 has plenty of headroom; cap memory share so we
            # don't accidentally OOM other workloads on the box.
            "gpu_memory_utilization": float(os.getenv("FMLS_MINERU_GPU_UTIL", "0.6")),
            # Single page at a time as a baseline; batching handled via batch_two_step_extract.
            "max_model_len": int(os.getenv("FMLS_MINERU_MAX_LEN", "8192")),
            "trust_remote_code": True,
        }
        if logits_processors is not None:
            llm_kwargs["logits_processors"] = logits_processors
        llm = LLM(**llm_kwargs)
        _mineru_client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=llm,
            # Enable MinerU's built-in image/chart analysis pass so figure
            # blocks come back with descriptive content (arrows, labels,
            # relationships) instead of an empty `[figure]` placeholder.
            image_analysis=True,
        )
        return _mineru_client


def _normalize_block_type(t: str) -> str:
    """Map MinerU's ContentBlock.type strings to our BlockType vocabulary.

    Order matters: MinerU uses *compound* labels like `table_caption` and
    `table_footnote`. We must match the most specific compound first;
    matching plain "table" first would swallow them and lose the structure.
    """
    t = (t or "").lower()
    # Compound table labels — match before the generic "table" rule.
    if "table_caption" in t or "table_title" in t:
        return "caption"
    if "table_footnote" in t:
        return "footnote"
    # Generic
    if "table" in t:
        return "table"
    if "title" in t or "heading" in t or "section" in t:
        return "heading"
    if "list" in t:
        return "list"
    if "caption" in t:
        return "caption"
    if "figure" in t or "image" in t or "picture" in t:
        return "figure"
    if "footnote" in t:
        return "footnote"
    if "page_number" in t or "page-number" in t:
        return "other"  # could be its own type later
    if "header" in t:
        return "header"
    if "footer" in t:
        return "footer"
    if "formula" in t or "equation" in t:
        return "other"
    if "text" in t or "paragraph" in t:
        return "paragraph"
    return "other"


def _content_block_to_remote(cb, page_num: int, order: int, image_width: int, image_height: int) -> Optional[RemoteBlock]:
    """Convert a MinerU ContentBlock into our RemoteBlock schema.

    MinerU bboxes are NORMALIZED to [0, 1]. We rescale them to PDF-point
    space using the rendered image dimensions + DPI so downstream UI overlays
    line up with the original PDF.
    """
    block_type = _normalize_block_type(str(getattr(cb, "type", "")))
    content = getattr(cb, "content", None)
    text = "" if content is None else str(content).strip()
    if block_type == "figure" and not text:
        # image_analysis=True should fill content; fall back only if it didn't.
        text = "[figure: no description from image_analysis]"
    if not text:
        return None

    bbox = None
    nbb = getattr(cb, "bbox", None)
    if nbb is not None and len(nbb) == 4:
        # Convert normalized [0,1] → pixel coords → PDF points.
        try:
            pt_per_px = 72.0 / MINERU_RENDER_DPI
            x0 = float(nbb[0]) * image_width * pt_per_px
            y0 = float(nbb[1]) * image_height * pt_per_px
            x1 = float(nbb[2]) * image_width * pt_per_px
            y1 = float(nbb[3]) * image_height * pt_per_px
            bbox = (x0, y0, x1, y1)
        except Exception:
            bbox = None

    meta = {"source": "mineru", "mineru_type": str(getattr(cb, "type", ""))}
    angle = getattr(cb, "angle", None)
    if angle:
        meta["rotation"] = int(angle)
    if block_type == "table":
        # MinerU emits HTML for tables; surface as both `text` (markdown-ish
        # for retrieval) and `metadata["html"]` (lossless for rendering).
        meta["html"] = text
        meta["has_merges"] = "rowspan" in text.lower() or "colspan" in text.lower()

    return RemoteBlock(
        block_type=block_type,  # type: ignore[arg-type]
        text=text,
        page_num=page_num,
        bbox=bbox,
        order_index=order,
        metadata=meta,
    )


RICH_PROMPT = """You are extracting structured content from a single page of a clinical-trial protocol.

Return ONLY a JSON object (no prose, no code fences) with this exact schema:

{
  "blocks": [
    {
      "id": "B1",
      "type": "heading" | "paragraph" | "list_item" | "table" | "figure" | "caption" | "footnote" | "header" | "footer" | "other",
      "text": "<verbatim text; for tables emit HTML with rowspan/colspan>",
      "level": <integer >= 0; 0 for the leftmost column, +1 for each indent level>,
      "parent_id": "<id of the block this is a child of>" | null,
      "list_marker": "<the literal marker if this is a list item, e.g. '1', '(a)', '•'>" | null,
      "paired_with": "<id of the block this is a caption for, OR id of the marker block this footnote defines>" | null,
      "visual_notes": "<for figures only: describe arrows, labels, flow, relationships>" | null
    }
  ]
}

Rules:
- Preserve EVERY visible piece of content; do not skip headers, page numbers, or footnotes.
- For nested lists, set `level` based on indentation and `parent_id` to the block that started the parent item.
- For category sub-headings inside a list (like "Age", "Type of Participant"), use type "heading", not "list_item".
- For table captions ("Table 3 Schedule of Activities ..."), set type="caption" and paired_with=<id of the table block>.
- For footnotes that define markers (e.g. "a Not a study site visit..."), set type="footnote" and set list_marker to the marker (e.g. "a").
- For figures, ALWAYS set visual_notes describing arrows (what they connect), labels, and relationships shown. Be specific.
- Use top-left page-point coordinates if you include bboxes (optional).
"""


@router.post("/rich")
async def parse_mineru_rich(
    file: UploadFile = File(...),
    page_index: int = Query(..., ge=0),
):
    """Bypass MinerUClient's fixed schema; call the underlying vLLM with a
    custom hierarchy-aware prompt. Single-page only (used for experiments
    and the hard-case route, not the bulk happy path).
    """
    import json as _json
    import time as _time
    import fitz
    from PIL import Image as _Image

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    client = _get_mineru_client()  # ensures the LLM is loaded
    # MinerUClient wraps a VllmEngineVlmClient as `.client`, which holds `.vllm_llm`
    inner = getattr(client, "client", None)
    llm = getattr(inner, "vllm_llm", None) if inner is not None else None
    if llm is None:
        raise HTTPException(status_code=500, detail="vllm_llm not reachable via client.client.vllm_llm")

    # Render the requested page
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise HTTPException(status_code=404, detail="page out of range")
        zoom = MINERU_RENDER_DPI / 72.0
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = _Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()

    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0.0, max_tokens=int(os.getenv("FMLS_MINERU_RICH_MAX_TOKENS", "3000")))

    # vLLM .chat() understands the multimodal message format
    t0 = _time.perf_counter()
    try:
        outputs = llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": RICH_PROMPT},
                    ],
                }
            ],
            sampling_params=sampling,
        )
    except (TypeError, AttributeError):
        # Older vLLM versions: pass as plain generate() with multi_modal_data
        outputs = llm.generate(
            [{"prompt": RICH_PROMPT, "multi_modal_data": {"image": image}}],
            sampling,
        )
    duration_ms = (_time.perf_counter() - t0) * 1000.0

    raw = outputs[0].outputs[0].text if outputs else ""
    # Try to parse JSON; preserve raw on failure for inspection.
    parsed = None
    parse_error = None
    rs = raw.strip()
    if rs.startswith("```"):
        rs = rs.strip("`")
        if rs.lower().startswith("json"):
            rs = rs[4:].lstrip()
    try:
        # Tolerant of trailing junk after the closing brace.
        first = rs.index("{")
        last = rs.rindex("}")
        parsed = _json.loads(rs[first:last+1])
    except (ValueError, _json.JSONDecodeError) as e:
        parse_error = str(e)

    return {
        "page_num": page_index,
        "duration_ms": duration_ms,
        "raw_output": raw,
        "parsed": parsed,
        "parse_error": parse_error,
    }


def _mineru_type_to_block_type(t: str) -> str:
    """Map full-mineru `type` field to our BlockType vocabulary."""
    t = (t or "").lower()
    if t == "title":
        return "heading"
    if t == "table":
        return "table"
    if t in ("image", "chart"):
        return "figure"
    if t == "code":
        return "other"
    if t == "list":
        return "other"  # the list itself is a container; children are emitted as paragraphs
    if t == "header":
        return "header"
    if t == "footer":
        return "footer"
    if t == "page_number":
        return "other"
    if t in ("page_footnote", "footnote"):
        return "footnote"
    if t == "equation":
        return "other"
    if t == "table_caption" or t == "image_caption":
        return "caption"
    if t == "table_footnote" or t == "image_footnote":
        return "footnote"
    return "paragraph"


def _block_text(b: dict) -> str:
    """Concatenate the visible text/HTML inside a middle_json block.

    For most blocks the content is in `lines[].spans[].content`. For
    composite blocks (table, image, chart), the body content sits in a
    NESTED sub-block (e.g. table_body) — handled by _extract_composite_data
    instead.
    """
    if b.get("text"):
        return str(b["text"])
    parts: list[str] = []
    for ln in b.get("lines") or []:
        for sp in ln.get("spans") or []:
            c = sp.get("content") or sp.get("html") or ""
            if c:
                parts.append(c)
    return "".join(parts)


def _extract_composite_data(para_block: dict) -> dict:
    """For a composite para_block (table, image, chart) walk its nested
    sub-blocks and return body html/content, caption text, footnote text.

    Shape from `mineru.backend.vlm.vlm_middle_json_mkcontent`:

        para_block.type      == "table" | "image" | "chart"
        para_block.blocks    == [
            {type: "table_body",     lines: [{spans: [{html: "<table>..."}]}]},
            {type: "table_caption",  lines: [{spans: [{content: "..."}]}]},
            {type: "table_footnote", lines: [{spans: [{content: "..."}]}]},
        ]
    """
    out: dict = {"html": "", "content": "", "captions": [], "footnotes": [], "image_path": ""}
    for sub in para_block.get("blocks") or []:
        t = (sub.get("type") or "").lower()
        text = _block_text(sub).strip()
        if t.endswith("_body") or t in ("table_body", "image_body", "chart_body"):
            # Walk spans for html / content / image_path
            for ln in sub.get("lines") or []:
                for sp in ln.get("spans") or []:
                    if sp.get("html"):
                        out["html"] = sp["html"]
                    if sp.get("content"):
                        out["content"] = (out["content"] + sp["content"]) if out["content"] else sp["content"]
                    if sp.get("image_path"):
                        out["image_path"] = sp["image_path"]
        elif t in ("table_caption", "image_caption", "chart_caption"):
            if text:
                out["captions"].append(text)
        elif t in ("table_footnote", "image_footnote", "chart_footnote"):
            if text:
                out["footnotes"].append(text)
    return out


def _bbox_tuple(b: dict, page_w: float, page_h: float) -> Optional[tuple[float, float, float, float]]:
    """Normalize MinerU bbox to PDF-point top-left coords.

    middle_json bboxes are in PDF-point coords already (not the 0-1000 normalized
    scale used by content_list.json). We just clamp and return.
    """
    bbox = b.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None


def _pdf_info_to_pages(pdf_info: list[dict]) -> dict[int, list[RemoteBlock]]:
    """Convert full-mineru pdf_info into our flat list of RemoteBlock,
    preserving parent_block_id for nested list children and tagging chrome
    blocks distinctly."""
    out: dict[int, list[RemoteBlock]] = {}
    for page in pdf_info:
        page_idx = int(page.get("page_idx", 0))
        size = page.get("page_size") or [0, 0]
        page_w = float(size[0] or 0)
        page_h = float(size[1] or 0)
        blocks_out: list[RemoteBlock] = []

        def emit(b: dict, parent_id: Optional[int]) -> None:
            order = len(blocks_out)
            t = b.get("type", "")
            block_type = _mineru_type_to_block_type(t)

            # ---- composite blocks (table / image / chart): consolidate the
            # nested sub-blocks (body, caption, footnote) into one emitted block. ----
            if t in ("table", "image", "chart"):
                data = _extract_composite_data(b)
                bbox = _bbox_tuple(b, page_w, page_h)
                meta: dict = {
                    "source": "mineru-v2",
                    "mineru_type": t,
                    "mineru_index": b.get("index"),
                }
                if b.get("sub_type"):
                    meta["sub_type"] = b["sub_type"]
                if b.get("angle"):
                    meta["angle"] = b["angle"]
                if parent_id is not None:
                    meta["parent_block_id"] = parent_id
                if data["captions"]:
                    if t == "table":
                        meta["table_caption"] = data["captions"]
                    else:
                        meta["image_caption"] = data["captions"]
                if data["footnotes"]:
                    if t == "table":
                        meta["table_footnote"] = data["footnotes"]
                    else:
                        meta["image_footnote"] = data["footnotes"]
                if data["image_path"]:
                    meta["image_path"] = data["image_path"]

                if t == "table":
                    html = data["html"]
                    meta["html"] = html
                    meta["has_merges"] = ("rowspan" in html.lower() or "colspan" in html.lower())
                    # `text` is the markdown-ish render; keep the HTML for the UI to render.
                    text_out = html
                elif t in ("image", "chart"):
                    # Figure: use content/description if MinerU produced one, otherwise [figure].
                    text_out = data["content"] or "[figure]"
                else:
                    text_out = data["content"] or ""

                blocks_out.append(RemoteBlock(
                    block_type=block_type,  # type: ignore[arg-type]
                    text=text_out,
                    page_num=page_idx,
                    bbox=BBoxTuple(bbox) if bbox else None,  # type: ignore[arg-type]
                    order_index=order,
                    metadata=meta,
                ))
                # Composite handled — do NOT recurse into its sub-blocks (they
                # were already consolidated above).
                return

            text = _block_text(b)
            # For list-container blocks, skip emitting an empty wrapper.
            if t == "list" and not text.strip():
                lead_parent = blocks_out[-1].order_index if blocks_out else None
                for child in (b.get("blocks") or []):
                    emit(child, parent_id=lead_parent)
                return
            if not text.strip():
                return  # skip empty non-composite blocks
            bbox = _bbox_tuple(b, page_w, page_h)
            meta = {
                "source": "mineru-v2",
                "mineru_type": t,
                "mineru_index": b.get("index"),
            }
            if b.get("sub_type"):
                meta["sub_type"] = b["sub_type"]
            if b.get("merge_prev"):
                meta["merge_prev"] = True
            if b.get("angle"):
                meta["angle"] = b["angle"]
            if parent_id is not None:
                meta["parent_block_id"] = parent_id
            blocks_out.append(RemoteBlock(
                block_type=block_type,  # type: ignore[arg-type]
                text=text.strip(),
                page_num=page_idx,
                bbox=BBoxTuple(bbox) if bbox else None,  # type: ignore[arg-type]
                order_index=order,
                metadata=meta,
            ))
            for child in (b.get("blocks") or []):
                emit(child, parent_id=order)

        # Chrome first (so it appears in the audit but is clearly typed).
        for b in page.get("discarded_blocks") or []:
            emit(b, parent_id=None)
        # Main content in reading order.
        for b in page.get("para_blocks") or []:
            emit(b, parent_id=None)

        out[page_idx] = blocks_out
    return out


# Pydantic doesn't accept tuples for bbox; we keep the optional tuple form
# via a tiny alias so the helper above compiles cleanly.
def BBoxTuple(t):  # type: ignore[no-redef]
    return tuple(t) if t is not None else None


@router.post("/v2/document", response_model=MineruDocumentResponse)
async def parse_mineru_v2(
    file: UploadFile = File(...),
    pages: Optional[str] = Query(None, description="comma-separated 0-indexed pages; if omitted, whole PDF"),
):
    """FULL mineru pipeline -> RemoteBlock list with parent_block_id +
    real heading types (`title` -> heading) + paired caption/footnote arrays."""
    import tempfile
    import hashlib

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    client = _get_mineru_client()
    try:
        from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
        from mineru.data.data_reader_writer import FileBasedDataWriter
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"full mineru not available: {e}")

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        image_writer = FileBasedDataWriter(tmp)
        try:
            middle_json, _infer = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                predictor=client,
                image_analysis=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"vlm_doc_analyze failed: {type(e).__name__}: {e}")
    duration_ms = (time.perf_counter() - t0) * 1000.0

    pdf_info = middle_json.get("pdf_info", []) or []
    page_blocks = _pdf_info_to_pages(pdf_info)

    if pages is not None and pages.strip():
        try:
            wanted = {int(x) for x in pages.split(",") if x.strip()}
        except ValueError:
            raise HTTPException(status_code=400, detail="pages must be comma-separated integers")
        page_blocks = {k: v for k, v in page_blocks.items() if k in wanted}

    return MineruDocumentResponse(
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        n_pages_requested=len(page_blocks),
        pages=page_blocks,
        convert_duration_ms=duration_ms,
    )


@router.post("/document", response_model=MineruDocumentResponse)
async def parse_mineru_document(
    file: UploadFile = File(...),
    pages: Optional[str] = Query(None, description="comma-separated 0-indexed page numbers; if omitted, processes all pages"),
) -> MineruDocumentResponse:
    """Render the requested PDF pages to images and run MinerU 2.5 on them.

    Returns blocks indexed by 0-based page number. One round-trip per
    document — pages are batched server-side into a single vLLM call.
    """
    import fitz  # PyMuPDF — local-only, fast
    from PIL import Image

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    page_list: Optional[list[int]] = None
    if pages is not None and pages.strip():
        try:
            page_list = sorted({int(x) for x in pages.split(",") if x.strip()})
        except ValueError:
            raise HTTPException(status_code=400, detail="pages must be comma-separated integers")

    sha = hashlib.sha256(pdf_bytes).hexdigest()
    t0 = time.perf_counter()

    # ---- 1. Render the requested pages to PIL Images. ----
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_list is None:
            page_list = list(range(doc.page_count))
        else:
            page_list = [p for p in page_list if 0 <= p < doc.page_count]
        if not page_list:
            return MineruDocumentResponse(
                sha256=sha, n_pages_requested=0, pages={}, convert_duration_ms=0.0,
            )
        zoom = MINERU_RENDER_DPI / 72.0
        images: list[Image.Image] = []
        dims: list[tuple[int, int]] = []
        for p in page_list:
            pix = doc[p].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
            dims.append((img.width, img.height))
    finally:
        doc.close()

    # ---- 2. Run MinerU's batched two-step extract. ----
    client = _get_mineru_client()
    try:
        per_page_blocks = client.batch_two_step_extract(images)
    except AttributeError:
        # Older versions exposed only the single-image API.
        per_page_blocks = [client.two_step_extract(img) for img in images]

    # ---- 3. Normalize into RemoteBlock per page. ----
    out: dict[int, list[RemoteBlock]] = {}
    for page_num, cbs, (w, h) in zip(page_list, per_page_blocks, dims):
        page_blocks: list[RemoteBlock] = []
        for order, cb in enumerate(cbs or []):
            blk = _content_block_to_remote(cb, page_num=page_num, order=order,
                                           image_width=w, image_height=h)
            if blk is not None:
                page_blocks.append(blk)
        out[page_num] = page_blocks

    return MineruDocumentResponse(
        sha256=sha,
        n_pages_requested=len(page_list),
        pages=out,
        convert_duration_ms=(time.perf_counter() - t0) * 1000.0,
    )
