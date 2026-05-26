"""v3 production pipeline.

Architecture (downstream of MinerU + chunker):

  chunks.json
       │
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  enrich      : per-chunk MDKeyChunker-style enrichment +         │
  │                semantic_role (chunk-level: criterion/header/...)│
  ├──────────────────────────────────────────────────────────────────┤
  │  classify    : multi-grain (document / section / chunk-role)    │
  │                — section-classifier reuses chunker's m11; chunk  │
  │                role comes from enrich layer                      │
  ├──────────────────────────────────────────────────────────────────┤
  │  retrieve    : element-specific chunks per USDM target class     │
  │                with chunk reunion (split chunks re-merged)       │
  ├──────────────────────────────────────────────────────────────────┤
  │  extract     : schema-constrained per USDM class via instructor  │
  │                + usdm_model classes as response_model directly   │
  ├──────────────────────────────────────────────────────────────────┤
  │  validate    : Pydantic + CDISC Rules Engine + cross-record      │
  ├──────────────────────────────────────────────────────────────────┤
  │  assemble    : build Wrapper(study=Study(...)) — canonical USDM  │
  └──────────────────────────────────────────────────────────────────┘
       │
       ▼
  dataset/{stem}/usdm.json  +  validation_report.json
"""

from fmls_parser.pipeline_v3.models import (
    ChunkSemanticRole,
    SemanticRoleAnnotation,
    PipelineRunMetadata,
)

__all__ = [
    "ChunkSemanticRole",
    "SemanticRoleAnnotation",
    "PipelineRunMetadata",
]
