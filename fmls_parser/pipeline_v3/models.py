"""Pipeline-specific Pydantic models.

The key addition is `ChunkSemanticRole` — a closed enum that names what
a chunk IS in the document's logical structure (independent of MinerU's
layout-level block_type and independent of section-inherited role_hints).

This is the field we identified as the architectural gap: every chunk in
§5.1 currently inherits role_hints=["inclusion_criterion"] from the
section's m11 mapping, but no field tells you whether the SPECIFIC chunk
is the criterion itself, the category label, the intro paragraph, or a
sub-explanation. ChunkSemanticRole fills that gap.

It's universal across verticals — clinical, legal, industrial, financial
all share these chunk-role concepts. Domain-specific specialization comes
from combining `role` with `schema_class_hints` from the domain layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# === The closed universe of chunk roles ============================

ChunkSemanticRole = Literal[
    # primary content
    "primary_item",      # the main content unit: a criterion, an endpoint, a procedure step
    # parts that elaborate the primary item
    "sub_explanation",   # paragraph that elaborates the preceding primary_item
    "sub_clause",        # labeled clause like (a), (b)
    "sub_bullet",        # bullet point under primary_item or sub_clause
    "exception",         # explicit exception ("except for...")
    # structural / navigational
    "section_header",    # the section's own heading
    "section_intro",     # intro paragraph immediately after section_header
    "category_header",   # mid-section category label like "Age", "Medical Conditions"
    # tabular / visual
    "table",             # structured table content
    "table_caption",     # caption attached to a table
    "figure_caption",    # caption attached to a figure
    # noise / chrome
    "footnote_marker",   # superscript marker pointing into a footnote
    "footnote_body",     # the footnote text itself
    "header_footer",     # running page header/footer
    "page_number",       # standalone page number
    "boilerplate",       # cross-document templated text (regulatory boilerplate)
    "note",              # editorial note ("Note: ...")
    "skip",              # something to ignore
    "unknown",           # we couldn't decide
]


class SemanticRoleAnnotation(BaseModel):
    """Per-chunk semantic role with provenance.

    Populated by an enrichment pass (rules / LLM / hybrid). Persisted in
    the document's enriched.json. Every downstream stage reads this
    instead of re-deriving role from raw signals.
    """

    model_config = ConfigDict(extra="forbid")

    role: ChunkSemanticRole
    marker: Optional[str] = Field(
        default=None,
        description="Leading marker if present: '1', '(a)', 'i.', etc.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["rule", "llm", "hybrid"]
    # For sub_* roles: the chunk_id of the primary_item this attaches to
    parent_role_chunk_id: Optional[str] = None
    # For category_header: the label text
    category_label: Optional[str] = None
    # Free-form notes from the classifier
    notes: Optional[str] = None
    classified_at: datetime
    classifier_version: str


# === MDKeyChunker-style enrichment fields ==========================
# Inspired by arxiv 2603.23533: one LLM call per chunk produces a small
# bundle of fields that boost downstream retrieval and extraction quality.

class ChunkKeyEnrichment(BaseModel):
    """Single-call LLM enrichment of a chunk's semantic key fields.

    Per MDKeyChunker (arxiv 2603.23533), this enrichment alone boosts QA
    accuracy 50-60% → 72-75% without changing retrieval architecture.

    All fields are optional except `semantic_role` — the LLM may legitimately
    decide a chunk has no extractable entities or hypothetical questions.
    """

    model_config = ConfigDict(extra="forbid")

    semantic_role: ChunkSemanticRole
    marker: Optional[str] = None
    parent_role_chunk_id: Optional[str] = None
    category_label: Optional[str] = None

    # Semantic key fields (MDKeyChunker contribution)
    title: Optional[str] = Field(default=None, description="A short title summarizing the chunk (5-12 words).")
    summary: Optional[str] = Field(default=None, description="1-2 sentence summary of the chunk's content.")
    keywords: list[str] = Field(default_factory=list, description="Top 3-8 keywords from the chunk.")
    typed_entities: list[dict] = Field(
        default_factory=list,
        description="Domain-relevant entities the chunk mentions, as {type, surface} dicts.",
    )
    hypothetical_questions: list[str] = Field(
        default_factory=list,
        description="2-4 questions this chunk would answer (for HyDE-style retrieval).",
    )
    semantic_key: Optional[str] = Field(
        default=None,
        description="A short canonical key/phrase capturing the chunk's main idea.",
    )

    # Quality flags
    is_boilerplate: bool = False
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


# === Pipeline run metadata ========================================


class PipelineRunMetadata(BaseModel):
    """Audit trail for a full pipeline run on one document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    pipeline_version: str = "fmls-pipeline-v3"
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Cache stats per stage
    stage_results: dict[str, dict] = Field(default_factory=dict)

    # Aggregate stats
    n_chunks_total: int = 0
    n_chunks_enriched: int = 0
    n_chunks_with_primary_role: int = 0
    n_records_extracted: dict[str, int] = Field(default_factory=dict)
    n_records_validated: dict[str, int] = Field(default_factory=dict)
    cdisc_rules_passed: Optional[bool] = None
    cdisc_rules_violations: list[str] = Field(default_factory=list)

    # LLM cost
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


__all__ = [
    "ChunkSemanticRole",
    "SemanticRoleAnnotation",
    "ChunkKeyEnrichment",
    "PipelineRunMetadata",
]
