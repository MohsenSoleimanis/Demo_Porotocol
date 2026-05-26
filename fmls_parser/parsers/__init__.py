"""Parser implementations behind a common interface (see base.PageParser).

Local parsers run in-process. Remote parsers (Docling, Qwen-VL) call a
FastAPI service over an SSH-forwarded port — see fmls_parser.remote.client.
"""

from .base import PageParser, ParseError
from .pdfplumber_parser import PdfPlumberParser
from .pymupdf_parser import PyMuPDFParser

__all__ = [
    "PageParser",
    "ParseError",
    "PyMuPDFParser",
    "PdfPlumberParser",
]
