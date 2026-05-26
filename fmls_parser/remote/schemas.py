"""Wire schemas for the local <-> remote heavy-parser protocol.

Keep this file dependency-light: imported by both client (local) and server
(Lightning AI), so it must NOT import torch, docling, transformers, etc.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RemoteBlock(BaseModel):
    block_type: Literal[
        "paragraph", "heading", "table", "list", "figure", "caption", "header", "footer", "footnote", "other"
    ]
    text: str
    page_num: int
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    order_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResponse(BaseModel):
    parser: Literal["docling", "qwen_vl", "mineru"]
    page_num: int
    blocks: list[RemoteBlock]
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    available_parsers: list[Literal["docling", "qwen_vl", "mineru"]]
    notes: list[str] = Field(default_factory=list)
