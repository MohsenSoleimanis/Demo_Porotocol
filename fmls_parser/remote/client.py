"""Local HTTP client for the remote heavy-parser service.

Behavior:
  - If FMLS_REMOTE_URL is unset, .enabled is False and every parse_* call
    raises a clear RemoteUnavailable. The pipeline checks .enabled first
    and falls back to local parsers / skips, so unconfigured runs never
    silently produce wrong output.
  - Reaches Lightning AI through an SSH tunnel: typically
        ssh -L 8000:localhost:8000 <lightning-host>
    then leave FMLS_REMOTE_URL=http://localhost:8000.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

from ..models import BBox, BlockType, ExtractedBlock, ParserRoute
from .schemas import HealthResponse, ParseResponse, RemoteBlock


@dataclass
class DoclingDocumentBlocks:
    sha256: str
    n_pages: int
    pages: dict[int, list[RemoteBlock]]
    convert_duration_ms: float
    cached: bool


@dataclass
class MineruDocumentBlocks:
    sha256: str
    n_pages_requested: int
    pages: dict[int, list[RemoteBlock]]
    convert_duration_ms: float


class RemoteUnavailable(RuntimeError):
    """Raised when remote heavy parsers are requested but not reachable."""


class RemoteClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 180.0):
        self.base_url = (base_url or os.getenv("FMLS_REMOTE_URL") or "").rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _http(self) -> httpx.Client:
        if not self.enabled:
            raise RemoteUnavailable("FMLS_REMOTE_URL is not set — remote parsers unavailable")
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- API methods ----

    def health(self) -> HealthResponse:
        r = self._http().get("/health")
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())

    def parse_with_docling(self, pdf_bytes: bytes, page_index: int, filename: str = "doc.pdf") -> ParseResponse:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        r = self._http().post("/parse/docling", files=files, params={"page_index": page_index})
        r.raise_for_status()
        return ParseResponse.model_validate(r.json())

    def parse_with_docling_document(
        self,
        pdf_bytes: bytes,
        filename: str = "doc.pdf",
        pages: Optional[list[int]] = None,
    ) -> "DoclingDocumentBlocks":
        """Convert specific pages (or full PDF) with Docling in one round-trip.

        If `pages` is provided, server only converts those pages (much cheaper).
        Otherwise the whole PDF is converted.
        """
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        params = {}
        if pages is not None:
            params["pages"] = ",".join(str(p) for p in sorted(set(pages)))
        r = self._http().post("/parse/docling/document", files=files, params=params)
        r.raise_for_status()
        payload = r.json()
        # JSON keys come back as strings — coerce page numbers to int.
        pages_raw = payload.get("pages", {}) or {}
        pages: dict[int, list[RemoteBlock]] = {
            int(k): [RemoteBlock.model_validate(b) for b in v] for k, v in pages_raw.items()
        }
        return DoclingDocumentBlocks(
            sha256=payload["sha256"],
            n_pages=int(payload["n_pages"]),
            pages=pages,
            convert_duration_ms=float(payload["convert_duration_ms"]),
            cached=bool(payload.get("cached", False)),
        )

    def parse_with_qwen_vl(self, page_png: bytes, page_index: int) -> ParseResponse:
        files = {"file": (f"page_{page_index}.png", page_png, "image/png")}
        r = self._http().post("/parse/qwen", files=files, params={"page_index": page_index})
        r.raise_for_status()
        return ParseResponse.model_validate(r.json())

    def parse_with_qwen_table(self, page_png: bytes, page_index: int) -> ParseResponse:
        """Run Qwen-VL with the SoA/complex-table-specific prompt (legacy)."""
        files = {"file": (f"page_{page_index}.png", page_png, "image/png")}
        r = self._http().post("/parse/qwen/table", files=files, params={"page_index": page_index})
        r.raise_for_status()
        return ParseResponse.model_validate(r.json())

    def parse_with_mineru_document(
        self,
        pdf_bytes: bytes,
        filename: str = "doc.pdf",
        pages: Optional[list[int]] = None,
    ) -> "MineruDocumentBlocks":
        """Send a PDF + page list to MinerU's FULL pipeline (`/v2/document`).

        Returns hierarchical output: `title`-typed headings, chrome separated
        (header/footer/page_number as their own block types), nested list
        children carry `metadata.parent_block_id`. Slower than the flat
        endpoint (`~240s` vs `~105s` on a 131-page doc) but emits real
        structure instead of flat blocks.
        """
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        params: dict = {}
        if pages is not None:
            params["pages"] = ",".join(str(p) for p in sorted(set(pages)))
        r = self._http().post("/parse/mineru/v2/document", files=files, params=params)
        r.raise_for_status()
        payload = r.json()
        pages_raw = payload.get("pages", {}) or {}
        pages_out: dict[int, list[RemoteBlock]] = {
            int(k): [RemoteBlock.model_validate(b) for b in v] for k, v in pages_raw.items()
        }
        return MineruDocumentBlocks(
            sha256=payload["sha256"],
            n_pages_requested=int(payload["n_pages_requested"]),
            pages=pages_out,
            convert_duration_ms=float(payload["convert_duration_ms"]),
        )


# ---- helpers for converting remote blocks into our pipeline model ----


def to_extracted_block(rb: RemoteBlock, parser_used: ParserRoute) -> ExtractedBlock:
    bbox = (
        BBox(x0=rb.bbox[0], y0=rb.bbox[1], x1=rb.bbox[2], y1=rb.bbox[3])
        if rb.bbox is not None
        else None
    )
    return ExtractedBlock(
        block_type=BlockType(rb.block_type),
        text=rb.text,
        page_num=rb.page_num,
        bbox=bbox,
        parser_used=parser_used,
        order_index=rb.order_index,
        metadata=rb.metadata,
    )
