"""End-to-end orchestrator: triage every page, dispatch to the right parser,
fall back gracefully, and assemble a DocumentResult with full audit trail.

Production hot paths in this file:
  - The PDF is opened ONCE per document with both `fitz` and `pdfplumber`
    handles. All per-page work reuses those handles. Re-opening per page
    was the dominant cost (50-500 ms × N pages with pdfplumber).
  - Stage-level timings are recorded so we know where time goes.
  - Remote calls are retried on transient errors only (network / 5xx / 429).
  - Exception handling is narrow: ParseError + RemoteUnavailable + network
    errors, NOT every Exception. Programming bugs propagate as they should.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import fitz
import httpx
import pdfplumber

from ._retry import with_retry
from .models import (
    DocumentResult,
    ExtractedBlock,
    PageResult,
    ParserRoute,
    ParseStatus,
    TriageDecision,
)
from .parsers import PdfPlumberParser, PyMuPDFParser
from .parsers.base import ParseError
from .remote.client import RemoteClient, RemoteUnavailable
from .remote.client import to_extracted_block as _remote_to_block
from .triage import TriageConfig, decide, extract_features_from_handles

log = logging.getLogger("fmls.pipeline")

ProgressCallback = Callable[[int, int, str], None]  # (page_index, total, message)


# Transient errors we retry on (anything from _retry.is_transient).
_TRANSIENT_HTTP = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.HTTPStatusError,
)


def _render_page_to_png(fitz_doc: fitz.Document, page_index: int, dpi: int = 144) -> bytes:
    """Rasterize a single page to PNG bytes (for VLM input).

    144 DPI on a US-letter page (8.5x11) gives ~1224x1584 px ≈ 1.9 M pixels,
    which is comfortably above Qwen-VL's 1 M max_pixels cap (the model
    downsamples anyway). Rendering at higher DPI just wastes CPU + bandwidth.
    """
    page = fitz_doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def _try_local(
    route: ParserRoute,
    fitz_doc: fitz.Document,
    plumber_pdf: pdfplumber.PDF,
    page_index: int,
) -> list[ExtractedBlock]:
    if route == ParserRoute.PYMUPDF:
        return PyMuPDFParser().parse_page_with_handle(fitz_doc, page_index)
    if route == ParserRoute.PDFPLUMBER:
        return PdfPlumberParser().parse_page_with_handle(plumber_pdf, page_index)
    raise ParseError(f"{route} is not a local parser")


def _try_remote(
    route: ParserRoute,
    client: RemoteClient,
    fitz_doc: fitz.Document,
    pdf_bytes: bytes,
    page_index: int,
    filename: str,
) -> list[ExtractedBlock]:
    if route == ParserRoute.MINERU:
        # Single-page fallback path. Hot path goes through the bulk prefetch
        # below — this is only hit when prefetch wasn't done.
        bulk = with_retry(
            lambda: client.parse_with_mineru_document(pdf_bytes, filename=filename, pages=[page_index]),
            op_name=f"mineru page {page_index}",
        )
        return [_remote_to_block(rb, ParserRoute.MINERU) for rb in bulk.pages.get(page_index, [])]
    if route == ParserRoute.DOCLING:
        resp = with_retry(
            lambda: client.parse_with_docling(pdf_bytes, page_index, filename=filename),
            op_name=f"docling page {page_index}",
        )
        return [_remote_to_block(rb, ParserRoute.DOCLING) for rb in resp.blocks]
    if route == ParserRoute.QWEN_VL:
        png = _render_page_to_png(fitz_doc, page_index)
        resp = with_retry(
            lambda: client.parse_with_qwen_vl(png, page_index),
            op_name=f"qwen_vl page {page_index}",
        )
        return [_remote_to_block(rb, ParserRoute.QWEN_VL) for rb in resp.blocks]
    if route == ParserRoute.QWEN_TABLE:
        png = _render_page_to_png(fitz_doc, page_index)
        resp = with_retry(
            lambda: client.parse_with_qwen_table(png, page_index),
            op_name=f"qwen_table page {page_index}",
        )
        return [_remote_to_block(rb, ParserRoute.QWEN_TABLE) for rb in resp.blocks]
    raise ParseError(f"{route} is not a remote parser")


def _is_remote(route: ParserRoute) -> bool:
    return route in (
        ParserRoute.MINERU,
        ParserRoute.DOCLING,
        ParserRoute.QWEN_VL,
        ParserRoute.QWEN_TABLE,
    )


def _parse_one_page(
    triage: TriageDecision,
    fitz_doc: fitz.Document,
    plumber_pdf: pdfplumber.PDF,
    pdf_bytes: bytes,
    filename: str,
    remote: RemoteClient,
    docling_prefetch: dict[int, list],
    mineru_prefetch: dict[int, list],
) -> PageResult:
    start = time.perf_counter()
    attempts: list[dict] = []
    routes_to_try: list[ParserRoute] = [triage.primary_route, *triage.fallback_routes]

    last_err: Optional[str] = None
    for i, route in enumerate(routes_to_try):
        if _is_remote(route) and not remote.enabled:
            attempts.append({"route": route.value, "outcome": "skipped", "reason": "remote not configured"})
            last_err = "remote not configured"
            continue
        try:
            t0 = time.perf_counter()
            # Fast path: use prefetched MinerU blocks if we have them.
            if route == ParserRoute.MINERU and triage.page_num in mineru_prefetch:
                prefetched = mineru_prefetch[triage.page_num]
                blocks = [_remote_to_block(rb, ParserRoute.MINERU) for rb in prefetched]
                attempts.append({
                    "route": route.value, "outcome": "ok",
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "n_blocks": len(blocks), "source": "prefetch",
                })
                status = ParseStatus.OK if i == 0 else ParseStatus.FALLBACK
                return PageResult(
                    page_num=triage.page_num, triage=triage,
                    parser_used=route, parse_status=status, blocks=blocks,
                    parse_duration_ms=(time.perf_counter() - start) * 1000.0,
                    attempts=attempts,
                )
            # Fast path: use prefetched Docling blocks if we have them.
            if route == ParserRoute.DOCLING and triage.page_num in docling_prefetch:
                prefetched = docling_prefetch[triage.page_num]
                blocks = [_remote_to_block(rb, ParserRoute.DOCLING) for rb in prefetched]
                attempts.append({
                    "route": route.value,
                    "outcome": "ok",
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "n_blocks": len(blocks),
                    "source": "prefetch",
                })
                status = ParseStatus.OK if i == 0 else ParseStatus.FALLBACK
                return PageResult(
                    page_num=triage.page_num,
                    triage=triage,
                    parser_used=route,
                    parse_status=status,
                    blocks=blocks,
                    parse_duration_ms=(time.perf_counter() - start) * 1000.0,
                    attempts=attempts,
                )
            if _is_remote(route):
                blocks = _try_remote(route, remote, fitz_doc, pdf_bytes, triage.page_num, filename)
            else:
                blocks = _try_local(route, fitz_doc, plumber_pdf, triage.page_num)
            attempts.append({
                "route": route.value,
                "outcome": "ok",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
                "n_blocks": len(blocks),
            })
            status = ParseStatus.OK if i == 0 else ParseStatus.FALLBACK
            return PageResult(
                page_num=triage.page_num,
                triage=triage,
                parser_used=route,
                parse_status=status,
                blocks=blocks,
                parse_duration_ms=(time.perf_counter() - start) * 1000.0,
                attempts=attempts,
            )
        except (ParseError, RemoteUnavailable, *_TRANSIENT_HTTP) as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning("page %d route %s failed: %s", triage.page_num, route.value, last_err)
            attempts.append({"route": route.value, "outcome": "error", "error": last_err})
            continue

    return PageResult(
        page_num=triage.page_num,
        triage=triage,
        parser_used=triage.primary_route,
        parse_status=ParseStatus.SKIPPED if triage.remote_required and not remote.enabled else ParseStatus.ERROR,
        blocks=[],
        parse_duration_ms=(time.perf_counter() - start) * 1000.0,
        error=last_err or "no parser succeeded",
        attempts=attempts,
    )


def parse_document(
    pdf_path: str,
    triage_config: Optional[TriageConfig] = None,
    remote_url: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> DocumentResult:
    """Parse a PDF end-to-end.

    The PDF is opened once with `fitz` and `pdfplumber`; both handles are
    reused for triage and for every local parser call. Remote calls share
    a single `RemoteClient` (one httpx.Client). Per-stage timings go into
    the returned `DocumentResult.stage_timings_ms`.
    """
    pdf_path = str(Path(pdf_path).resolve())
    remote = RemoteClient(base_url=remote_url) if remote_url is not None else RemoteClient()
    config = triage_config or TriageConfig.from_env(remote_configured=remote.enabled)
    config.remote_configured = remote.enabled

    stage_timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    log.info("parse_document start path=%s remote=%s", pdf_path, remote.enabled)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    try:
        with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_pdf:
            total_pages = fitz_doc.page_count

            # ---- 1. Triage every page, reusing the open handles. ----
            t_triage = time.perf_counter()
            triage_decisions: list[TriageDecision] = []
            for page_index in range(total_pages):
                if progress:
                    progress(page_index, total_pages, "extracting features")
                feats = extract_features_from_handles(fitz_doc, plumber_pdf, page_index)
                triage_decisions.append(decide(feats, config))
            stage_timings["triage_ms"] = (time.perf_counter() - t_triage) * 1000.0
            log.info("triage done %d pages in %.0f ms", total_pages, stage_timings["triage_ms"])

            # ---- 2a. Bulk-prefetch pages routed to MinerU. ----
            t_mineru = time.perf_counter()
            mineru_prefetch: dict[int, list] = {}
            mineru_pages = [t.page_num for t in triage_decisions if t.primary_route == ParserRoute.MINERU]
            if mineru_pages and remote.enabled:
                if progress:
                    progress(0, total_pages, f"MinerU batch on {len(mineru_pages)} pages")
                try:
                    mineru_bulk = with_retry(
                        lambda: remote.parse_with_mineru_document(
                            pdf_bytes, filename=Path(pdf_path).name, pages=mineru_pages,
                        ),
                        op_name="mineru_bulk",
                    )
                    mineru_prefetch = mineru_bulk.pages
                except _TRANSIENT_HTTP + (RemoteUnavailable,) as e:
                    log.warning("mineru bulk prefetch failed; per-page fallback will be attempted: %s", e)
            stage_timings["mineru_bulk_ms"] = (time.perf_counter() - t_mineru) * 1000.0
            if mineru_pages:
                log.info("mineru bulk %d pages in %.0f ms", len(mineru_pages), stage_timings["mineru_bulk_ms"])

            # ---- 2b. Bulk-prefetch pages routed to Docling (legacy fallback path). ----
            t_docling = time.perf_counter()
            docling_prefetch: dict[int, list] = {}
            docling_pages = [t.page_num for t in triage_decisions if t.primary_route == ParserRoute.DOCLING]
            if docling_pages and remote.enabled:
                if progress:
                    progress(0, total_pages, f"fetching Docling for {len(docling_pages)} pages")
                try:
                    bulk = with_retry(
                        lambda: remote.parse_with_docling_document(
                            pdf_bytes, filename=Path(pdf_path).name, pages=docling_pages,
                        ),
                        op_name="docling_bulk",
                    )
                    docling_prefetch = bulk.pages
                except _TRANSIENT_HTTP + (RemoteUnavailable,) as e:
                    log.warning("docling bulk prefetch failed; per-page fallback will be attempted: %s", e)
            stage_timings["docling_bulk_ms"] = (time.perf_counter() - t_docling) * 1000.0
            if docling_pages:
                log.info("docling bulk %d pages in %.0f ms (cached or selective convert)",
                         len(docling_pages), stage_timings["docling_bulk_ms"])

            # ---- 3. Per-page parse, reusing handles. ----
            t_parse = time.perf_counter()
            pages: list[PageResult] = []
            for page_index, triage in enumerate(triage_decisions):
                if progress:
                    progress(page_index, total_pages, f"parsing via {triage.primary_route.value}")
                page_result = _parse_one_page(
                    triage=triage,
                    fitz_doc=fitz_doc,
                    plumber_pdf=plumber_pdf,
                    pdf_bytes=pdf_bytes,
                    filename=Path(pdf_path).name,
                    remote=remote,
                    docling_prefetch=docling_prefetch,
                    mineru_prefetch=mineru_prefetch,
                )
                pages.append(page_result)
            stage_timings["parse_ms"] = (time.perf_counter() - t_parse) * 1000.0

            # ---- 4. Local post-processing: detect vector arrows (duration / span
            # annotations between table cells) and attach them to the table /
            # figure block they belong to. Cheap, local, no remote needed.
            t_arrows = time.perf_counter()
            try:
                from .postprocess import enrich_page_with_arrows
                total_arrows = 0
                for pr in pages:
                    total_arrows += enrich_page_with_arrows(fitz_doc, pr.page_num, pr.blocks)
                if total_arrows:
                    log.info("arrow detection: %d arrows attached across %d pages", total_arrows, len(pages))
            except Exception as e:
                log.warning("arrow detection failed: %s", e)
            stage_timings["arrows_ms"] = (time.perf_counter() - t_arrows) * 1000.0

            # NOTE: a heuristic structural-assembly layer was prototyped in
            # fmls_parser/structure.py but disabled by default — the rules
            # introduce their own edge cases (e.g., category sub-headings get
            # typed like list items; cross-page parents are missed). The
            # correct fix is to get the VLM to emit hierarchy directly via a
            # richer prompt, not to bolt heuristics on top of flat output.
    finally:
        remote.close()

    total_ms = (time.perf_counter() - pipeline_start) * 1000.0
    stage_timings["total_ms"] = total_ms

    result = DocumentResult(
        source_path=pdf_path,
        source_filename=Path(pdf_path).name,
        total_pages=total_pages,
        pages=pages,
        total_duration_ms=total_ms,
        remote_configured=remote.enabled,
        stage_timings_ms=stage_timings,
    )
    log.info(
        "parse_document done path=%s pages=%d total=%.0f ms (triage=%.0f docling=%.0f parse=%.0f) routes=%s",
        Path(pdf_path).name, total_pages, total_ms,
        stage_timings.get("triage_ms", 0),
        stage_timings.get("docling_bulk_ms", 0),
        stage_timings.get("parse_ms", 0),
        result.route_distribution(),
    )
    return result
