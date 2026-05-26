"""Fast text + layout extractor using PyMuPDF.

Reading order is preserved by PyMuPDF's block ordering. Headings are detected
heuristically by font-size relative to the page's body-text size — this is a
corpus-agnostic signal, not tied to any particular template.
"""

from __future__ import annotations

import statistics

import fitz

from ..models import BBox, BlockType, ExtractedBlock, ParserRoute
from .base import PageParser, ParseError


def _detect_body_size(page: fitz.Page) -> float | None:
    """Estimate the page's dominant body-text font size as the median of all
    text-span sizes weighted by character count. Returns None if the page has
    no text spans."""
    sizes: list[float] = []
    try:
        td = page.get_text("dict")
    except Exception:
        return None
    for block in td.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size", 0))
                text = span.get("text", "")
                if size > 0 and text.strip():
                    sizes.extend([size] * len(text.strip()))
    if not sizes:
        return None
    return statistics.median(sizes)


def _classify_block(text: str, span_sizes: list[float], body_size: float | None) -> BlockType:
    """Classify a text block. Generic rules only — no document-specific patterns."""
    stripped = text.strip()
    if not stripped:
        return BlockType.OTHER
    if not span_sizes:
        return BlockType.PARAGRAPH

    max_span = max(span_sizes)
    if body_size and max_span >= body_size * 1.15 and len(stripped) < 250:
        # Larger-than-body font + short line = likely heading.
        return BlockType.HEADING
    # Short bullet/list markers are a common cross-document signal.
    if len(stripped) < 200 and stripped[:2] in {"- ", "* ", "• "}:
        return BlockType.LIST
    return BlockType.PARAGRAPH


class PyMuPDFParser(PageParser):
    route = ParserRoute.PYMUPDF

    def parse_page(self, pdf_path: str, page_index: int) -> list[ExtractedBlock]:
        """Legacy single-page entry — opens and closes around the handle-based call."""
        with fitz.open(pdf_path) as doc:
            return self.parse_page_with_handle(doc, page_index)

    def parse_page_with_handle(self, handle, page_index: int) -> list[ExtractedBlock]:
        doc = handle
        try:
            page = doc[page_index]
        except IndexError as e:
            raise ParseError(f"page {page_index} out of range") from e

        try:
            page_height = float(page.rect.height)
            body_size = _detect_body_size(page)

            try:
                td = page.get_text("dict")
            except Exception as e:
                raise ParseError(f"text extraction failed: {e}") from e

            blocks_raw = td.get("blocks", [])
            out: list[ExtractedBlock] = []
            order = 0
            for block in blocks_raw:
                # type 0 = text, type 1 = image
                if block.get("type") == 1:
                    bbox_tuple = block.get("bbox")
                    if not bbox_tuple or len(bbox_tuple) < 4:
                        continue
                    bbox = BBox(x0=bbox_tuple[0], y0=bbox_tuple[1], x1=bbox_tuple[2], y1=bbox_tuple[3])
                    # Skip tiny images (logos in repeating headers/footers); generic
                    # threshold of 0.5% page area filters out chrome without losing
                    # real diagrams.
                    page_area = max(page.rect.width * page.rect.height, 1.0)
                    if bbox.width * bbox.height < page_area * 0.005:
                        continue
                    out.append(
                        ExtractedBlock(
                            block_type=BlockType.FIGURE,
                            text="[figure]",
                            page_num=page_index,
                            bbox=bbox,
                            parser_used=self.route,
                            order_index=order,
                            metadata={
                                "is_visual": True,
                                "ext": block.get("ext"),
                                "image_width": block.get("width"),
                                "image_height": block.get("height"),
                            },
                        )
                    )
                    order += 1
                    continue
                if block.get("type") != 0:
                    continue
                bbox_tuple = block.get("bbox")
                lines = block.get("lines", [])
                if not lines:
                    continue
                text_parts: list[str] = []
                span_sizes: list[float] = []
                fonts: set[str] = set()
                for line in lines:
                    line_parts = []
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t:
                            line_parts.append(t)
                            sz = float(span.get("size", 0))
                            if sz > 0:
                                span_sizes.append(sz)
                            f = span.get("font")
                            if f:
                                fonts.add(f)
                    if line_parts:
                        text_parts.append("".join(line_parts))
                text = "\n".join(text_parts).strip()
                if not text:
                    continue

                # Drop very-top / very-bottom one-liners as page headers/footers
                # if they're short and far from body content. Generic rule:
                # within top 5% or bottom 5% of page AND single-line AND short.
                btype = _classify_block(text, span_sizes, body_size)
                if bbox_tuple and len(text) < 120 and "\n" not in text:
                    y_top = bbox_tuple[1] / page_height
                    if y_top < 0.05:
                        btype = BlockType.HEADER
                    elif y_top > 0.95:
                        btype = BlockType.FOOTER
                # Footnote heuristic: smaller-than-body font + bottom 30% of page +
                # not already classified as something more specific. Generic rule,
                # corpus-agnostic.
                if (
                    btype == BlockType.PARAGRAPH
                    and bbox_tuple
                    and body_size
                    and span_sizes
                ):
                    max_span = max(span_sizes)
                    y_top = bbox_tuple[1] / page_height
                    if max_span <= body_size * 0.92 and y_top >= 0.7:
                        btype = BlockType.FOOTNOTE

                bbox = (
                    BBox(x0=bbox_tuple[0], y0=bbox_tuple[1], x1=bbox_tuple[2], y1=bbox_tuple[3])
                    if bbox_tuple and len(bbox_tuple) >= 4
                    else None
                )
                out.append(
                    ExtractedBlock(
                        block_type=btype,
                        text=text,
                        page_num=page_index,
                        bbox=bbox,
                        parser_used=self.route,
                        order_index=order,
                        metadata={
                            "max_font_size": max(span_sizes) if span_sizes else None,
                            "body_font_size": body_size,
                            "fonts": sorted(fonts),
                        },
                    )
                )
                order += 1
            return out
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"pymupdf parse failed on page {page_index}: {e}") from e
