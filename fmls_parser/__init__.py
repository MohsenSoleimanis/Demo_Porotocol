"""FMLS clinical-protocol parser pipeline.

Public surface kept intentionally narrow — orchestrate everything through
`pipeline.parse_document` and the data models in `models`.
"""

from ._logging import setup_logging as _setup_logging
_setup_logging()  # idempotent; library users can override

from .models import (
    BBox,
    BlockType,
    DocumentResult,
    ExtractedBlock,
    PageFeatures,
    PageResult,
    ParserRoute,
    ParseStatus,
    TriageDecision,
)
from .pipeline import parse_document

__all__ = [
    "BBox",
    "BlockType",
    "DocumentResult",
    "ExtractedBlock",
    "PageFeatures",
    "PageResult",
    "ParserRoute",
    "ParseStatus",
    "TriageDecision",
    "parse_document",
]
