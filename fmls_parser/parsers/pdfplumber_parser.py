"""Local table-aware parser using pdfplumber.

Used as the local fallback for pages that triage flagged as having tables or
multi-column layout when the remote Docling service is not available. Will
also be used for verification side-by-side in the UI.
"""

from __future__ import annotations

import pdfplumber

from ..models import BBox, BlockType, ExtractedBlock, ParserRoute
from .base import PageParser, ParseError


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows of cells) as GitHub-flavored markdown."""
    if not rows:
        return ""
    cleaned = [
        [(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in row]
        for row in rows
    ]
    n_cols = max((len(r) for r in cleaned), default=0)
    if n_cols == 0:
        return ""
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    header = cleaned[0]
    sep = ["---"] * n_cols
    body = cleaned[1:] if len(cleaned) > 1 else []
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class PdfPlumberParser(PageParser):
    route = ParserRoute.PDFPLUMBER

    def parse_page(self, pdf_path: str, page_index: int) -> list[ExtractedBlock]:
        """Legacy entry — opens and closes around the handle-based call."""
        with pdfplumber.open(pdf_path) as pdf:
            return self.parse_page_with_handle(pdf, page_index)

    def parse_page_with_handle(self, handle, page_index: int) -> list[ExtractedBlock]:
        pdf = handle
        try:
            page = pdf.pages[page_index]
        except IndexError as e:
            raise ParseError(f"page {page_index} out of range") from e

        try:
            out: list[ExtractedBlock] = []
            order = 0

            # Tables first — they're the reason this parser exists.
            try:
                table_objs = page.find_tables()
            except Exception:
                table_objs = []
            table_bboxes = []
            for tbl in table_objs:
                rows = tbl.extract() or []
                md = _table_to_markdown(rows)
                if not md.strip():
                    continue
                bx = tbl.bbox  # (x0, top, x1, bottom)
                bbox = BBox(x0=float(bx[0]), y0=float(bx[1]), x1=float(bx[2]), y1=float(bx[3]))
                table_bboxes.append(bbox)
                out.append(
                    ExtractedBlock(
                        block_type=BlockType.TABLE,
                        text=md,
                        page_num=page_index,
                        bbox=bbox,
                        parser_used=self.route,
                        order_index=order,
                        metadata={
                            "n_rows": len(rows),
                            "n_cols": max((len(r) for r in rows), default=0),
                            "cells": rows,
                        },
                    )
                )
                order += 1

            # Non-table prose: extract words and group into lines/paragraphs,
            # skipping anything that falls inside a detected table bbox.
            words = []
            try:
                words = page.extract_words(use_text_flow=True) or []
            except Exception:
                words = []

            def _in_any_table(x0: float, y0: float, x1: float, y1: float) -> bool:
                for tb in table_bboxes:
                    if x0 >= tb.x0 and y0 >= tb.y0 and x1 <= tb.x1 and y1 <= tb.y1:
                        return True
                return False

            # Group words into lines by y-coordinate clusters, then lines into
            # paragraphs by vertical gap. Generic — no template-specific rules.
            lines: list[dict] = []
            current_line: dict | None = None
            for w in words:
                x0, top, x1, bottom = float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])
                if _in_any_table(x0, top, x1, bottom):
                    continue
                if current_line is None or abs(top - current_line["top"]) > 3:
                    if current_line is not None:
                        lines.append(current_line)
                    current_line = {"top": top, "bottom": bottom, "x0": x0, "x1": x1, "text": w["text"]}
                else:
                    current_line["text"] += " " + w["text"]
                    current_line["x1"] = max(current_line["x1"], x1)
                    current_line["bottom"] = max(current_line["bottom"], bottom)
            if current_line is not None:
                lines.append(current_line)

            # Paragraph grouping: a vertical gap larger than ~1.5x the typical
            # line gap starts a new paragraph.
            if lines:
                gaps = [lines[i + 1]["top"] - lines[i]["bottom"] for i in range(len(lines) - 1)]
                pos_gaps = [g for g in gaps if g > 0]
                # Use median as the typical line gap; fall back to a constant.
                if pos_gaps:
                    pos_gaps.sort()
                    typical = pos_gaps[len(pos_gaps) // 2]
                else:
                    typical = 4.0
                para_threshold = max(typical * 1.8, 6.0)

                buf_lines: list[dict] = [lines[0]]
                for prev, cur in zip(lines, lines[1:]):
                    gap = cur["top"] - prev["bottom"]
                    if gap > para_threshold:
                        out.append(_lines_to_block(buf_lines, page_index, order, self.route))
                        order += 1
                        buf_lines = [cur]
                    else:
                        buf_lines.append(cur)
                out.append(_lines_to_block(buf_lines, page_index, order, self.route))
                order += 1
            return out
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"pdfplumber parse failed on page {page_index}: {e}") from e


def _lines_to_block(line_dicts: list[dict], page_num: int, order: int, route: ParserRoute) -> ExtractedBlock:
    text = "\n".join(ld["text"] for ld in line_dicts).strip()
    x0 = min(ld["x0"] for ld in line_dicts)
    x1 = max(ld["x1"] for ld in line_dicts)
    y0 = min(ld["top"] for ld in line_dicts)
    y1 = max(ld["bottom"] for ld in line_dicts)
    return ExtractedBlock(
        block_type=BlockType.PARAGRAPH,
        text=text,
        page_num=page_num,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        parser_used=route,
        order_index=order,
        metadata={"n_lines": len(line_dicts)},
    )
