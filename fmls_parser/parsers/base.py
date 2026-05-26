"""Common interface every page-level parser implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import ExtractedBlock, ParserRoute


class ParseError(RuntimeError):
    """Raised when a parser cannot produce a usable result for a page."""


class PageParser(ABC):
    """A page-level parser.

    Parsers accept *open document handles* in their hot path so the orchestrator
    can open each PDF exactly once per document. The legacy
    `parse_page(pdf_path, page_index)` form is preserved for callers that don't
    have a handle, but it just opens and closes around the handle-based call.
    """

    route: ParserRoute  # set by subclass

    @abstractmethod
    def parse_page_with_handle(self, handle: Any, page_index: int) -> list[ExtractedBlock]:
        """Return ordered blocks for one page using an already-open document handle.

        `handle` type is parser-specific (e.g. `fitz.Document`, `pdfplumber.PDF`).
        """

    def parse_page(self, pdf_path: str, page_index: int) -> list[ExtractedBlock]:
        """Legacy convenience: open the PDF, run on one page, close.

        Hot paths should use `parse_page_with_handle` to avoid re-opening.
        """
        raise NotImplementedError
