"""Enrichment schema for the unstructured-AI pipeline.

This is the **production** schema, not a test draft. It is deliberately
domain-agnostic and binds to no target output schema (USDM, LegalRuleML,
AAS, XBRL, etc.). Domain-specific target-schema binding happens at the
extraction stage (Stage 8), not here.

Five orthogonal layers:
  - Structural  : where the chunk sits (section path, bbox, internal refs)
  - Linguistic  : language-level facts (negation, certainty, temporal,
                  acronyms, coref) — universal across domains
  - Semantic    : entities, relations, topics, citations using an abstract
                  type taxonomy (PERSON, ORGANIZATION, QUANTITY, ...) not
                  domain-specific types
  - Quality     : OCR/layout/completeness confidence, boilerplate, PII
  - Domain      : vertical-specific HINTS (not bindings) — section
                  taxonomy mapping, role hints, schema-class hints

Optional sub-enrichments populate only when chunk_type matches:
  - TableEnrichment   : when chunk_type == "table"
  - FigureEnrichment  : when chunk_type == "figure"
  - CodeEnrichment    : when chunk_type == "code"

Every annotation carries provenance (producer + version + confidence +
timestamp + optional prompt_hash / ruleset_version). Every annotation is
span-anchored to a SpanRef(chunk_id, start, end). No annotation is
attached to a chunk by attribute alone.

The envelope supports partial re-enrichment: bump only the layer-version
that changed; consumers can detect stale layers per chunk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "fmls-enrichment-1.0"


# ===========================================================================
# Provenance + span references
# ===========================================================================


class AnnotationProvenance(BaseModel):
    """Carried by every annotation. Audit-grade."""

    producer: str
    """Logical producer name, e.g. 'gliner-bio', 'negex', 'claude-sonnet-4-5',
    'regex-citation-v2', 'human-reviewer'."""

    producer_version: str
    """Version string of the producer (model version, ruleset version,
    library version, or git SHA for hand-written rules)."""

    prompt_hash: Optional[str] = None
    """SHA-256 of the prompt template, if the producer was an LLM. None
    for deterministic producers."""

    ruleset_version: Optional[str] = None
    """Version of the ruleset for rule-based producers (e.g. NegEx ruleset
    v1.2). None for learned models."""

    created_at: datetime

    confidence: float = Field(ge=0.0, le=1.0)
    """[0.0, 1.0] confidence in this specific annotation."""

    notes: Optional[str] = None
    """Free-text producer note, if any."""


class SpanRef(BaseModel):
    """Reference to a character span within a chunk.

    `start` and `end` are 0-indexed character offsets in the chunk's
    canonical text. End is exclusive (Python slice semantics).
    """

    chunk_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


# ===========================================================================
# Layer 1 — Structural enrichment
# ===========================================================================


ChunkType = Literal[
    "heading",
    "paragraph",
    "table",
    "table_cell",
    "list_item",
    "footnote",
    "caption",
    "figure",
    "code",
    "equation",
    "definition",
    "quote",
    "other",
]


class InternalReference(BaseModel):
    """A cross-reference INSIDE this document.

    Examples: 'see Section 3.2', 'as defined in Annex B', 'per Table 7',
    'page 14'.
    """

    surface_text: str
    span: SpanRef
    target_section_path: Optional[list[str]] = None
    target_chunk_id: Optional[str] = None
    """Resolved chunk_id if cross-reference resolver succeeded; None otherwise."""
    reference_kind: Literal[
        "section",
        "table",
        "figure",
        "appendix",
        "page",
        "footnote",
        "equation",
        "list_item",
    ]
    provenance: AnnotationProvenance


class StructuralEnrichment(BaseModel):
    """Where the chunk sits in document structure."""

    chunk_type: ChunkType
    section_path: list[str]
    """Hierarchical path of section headings leading to this chunk."""

    parent_heading: Optional[str] = None
    depth: int = Field(ge=0)
    page: int = Field(ge=0)
    bbox: Optional[tuple[float, float, float, float]] = None
    reading_order_index: int = Field(ge=0)

    internal_references: list[InternalReference] = Field(default_factory=list)
    """All in-document cross-references found in this chunk."""

    co_occurring_chunks: list[str] = Field(default_factory=list)
    """chunk_ids that this chunk references OR is referenced by."""

    version_anchor: Optional[str] = None
    """For amended documents, which amendment/version this chunk belongs to."""

    supersedes_chunk_id: Optional[str] = None
    """If this chunk supersedes another (amendment), the prior chunk_id."""

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Layer 2 — Linguistic enrichment
# ===========================================================================


class AcronymUse(BaseModel):
    """An acronym mention in this chunk."""

    surface: str
    span: SpanRef
    expansion: Optional[str] = None
    """Canonical expansion, e.g. 'objective response rate'."""

    expansion_chunk_id: Optional[str] = None
    """The chunk where this acronym was defined, if found."""

    provenance: AnnotationProvenance


class DefinedTermUse(BaseModel):
    """A use of a doc-locally-defined term in this chunk.

    Examples: 'Investigational Product', 'Lessee', 'Permitted Encumbrance',
    'the Notes'. These are terms whose definition lives elsewhere in the
    same document and binds throughout.
    """

    surface: str
    span: SpanRef
    definition_chunk_id: Optional[str] = None
    provenance: AnnotationProvenance


NegationKind = Literal["definite", "possible", "history", "family", "conditional"]


class NegationAnnotation(BaseModel):
    """A negated span. NegEx / ConText style.

    Examples:
      'no evidence of hepatic impairment'   -> definite
      'rule out diabetes'                   -> definite
      'history of myocardial infarction'    -> history
      'family history of cancer'            -> family
      'possible pneumonia'                  -> possible
      'should not exceed 100 mg'            -> conditional (legal/industrial)
    """

    target_span: SpanRef
    """The span being negated (e.g. 'hepatic impairment')."""

    cue_text: str
    cue_span: SpanRef
    """The negation cue itself (e.g. 'no evidence of')."""

    negation_type: NegationKind
    provenance: AnnotationProvenance


CertaintyLevel = Literal["high", "medium", "low", "uncertain"]
Modality = Literal["epistemic", "deontic", "dynamic", "alethic", "other"]


class CertaintyAnnotation(BaseModel):
    """Hedging / modality annotation.

    Examples:
      'may cause drowsiness'     -> epistemic, medium
      'shall comply with'        -> deontic, high
      'must not exceed'          -> deontic, high
      'possible diagnosis'       -> epistemic, low
      'is required to report'    -> deontic, high
    """

    target_span: SpanRef
    cue_text: str
    cue_span: SpanRef
    certainty_level: CertaintyLevel
    modality: Modality
    provenance: AnnotationProvenance


TemporalKind = Literal[
    "absolute_date",
    "relative",
    "duration",
    "frequency",
    "marker",
    "interval",
]


class TemporalAnnotation(BaseModel):
    """A temporal expression in the chunk.

    Examples:
      'baseline'                 -> marker
      'week 4'                   -> relative
      '2024-01-15'               -> absolute_date
      'twice daily'              -> frequency
      'for 12 weeks'             -> duration
      'between Day 1 and Day 7'  -> interval
    """

    target_span: SpanRef
    surface: str
    temporal_value: Optional[str] = None
    """Normalized ISO 8601 or domain-specific normalized form if parseable."""

    temporal_type: TemporalKind
    provenance: AnnotationProvenance


class CoreferenceCluster(BaseModel):
    """A cluster of coreferent mentions within or across chunks.

    Examples:
      ['the patient', 'she', 'the subject']
      ['the Lessee', 'such party', 'it']
      ['this part', 'the assembly', 'the part above']
    """

    cluster_id: str
    mentions: list[SpanRef]
    canonical_text: Optional[str] = None
    """The representative mention of the cluster."""

    provenance: AnnotationProvenance


class LinguisticEnrichment(BaseModel):
    """Language-level facts. Universal across domains."""

    language: str
    """ISO 639-1 code, e.g. 'en', 'ja', 'zh'."""

    script: Optional[str] = None
    """ISO 15924 code, e.g. 'Latn', 'Hant', 'Arab'."""

    sentence_spans: list[SpanRef] = Field(default_factory=list)
    acronym_uses: list[AcronymUse] = Field(default_factory=list)
    defined_term_uses: list[DefinedTermUse] = Field(default_factory=list)
    negation_annotations: list[NegationAnnotation] = Field(default_factory=list)
    certainty_annotations: list[CertaintyAnnotation] = Field(default_factory=list)
    temporal_annotations: list[TemporalAnnotation] = Field(default_factory=list)
    coreference_clusters: list[CoreferenceCluster] = Field(default_factory=list)

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Layer 3 — Semantic enrichment
# ===========================================================================


# Abstract entity taxonomy. Domain-agnostic. Domain-specific subtyping
# happens via configurable adapter (clinical maps PRODUCT->DRUG, etc.).
AbstractEntityType = Literal[
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "QUANTITY",
    "MEASUREMENT",
    "PERCENT",
    "MONEY",
    "PRODUCT",
    "CONDITION",
    "PROCEDURE",
    "EVENT",
    "REFERENCE",
    "TERM",
    "IDENTIFIER",
    "ROLE",
    "OBJECT",
    "OTHER",
]


class EntityAnnotation(BaseModel):
    """An entity mention with optional canonical link."""

    surface: str
    span: SpanRef
    entity_type: AbstractEntityType
    """Abstract type from a domain-agnostic taxonomy."""

    subtype: Optional[str] = None
    """Optional domain-specific subtype, e.g. 'DRUG', 'CLAUSE', 'PART_NUMBER',
    'INSTRUMENT'. Free-form to support arbitrary verticals."""

    canonical_id: Optional[str] = None
    """ID in controlled vocabulary, e.g. UMLS CUI, LEI, ECLASS IRDI, FIGI."""

    canonical_vocab: Optional[str] = None
    """Name of the controlled vocabulary, e.g. 'UMLS', 'SNOMED-CT', 'RxNorm',
    'LEI', 'ECLASS', 'FIGI', 'CIK'."""

    canonical_uri: Optional[str] = None
    """Resolvable URI when available."""

    provenance: AnnotationProvenance


# Abstract relation taxonomy. Domain-agnostic.
AbstractRelationType = Literal[
    "REFERENCES",
    "DEFINES",
    "PART_OF",
    "CONTAINS",
    "EQUIVALENT_TO",
    "SUPERSEDES",
    "TEMPORALLY_BEFORE",
    "TEMPORALLY_AFTER",
    "TEMPORALLY_DURING",
    "CAUSES",
    "REQUIRES",
    "EXCLUDES",
    "COMPLIES_WITH",
    "GOVERNS",
    "MEASURES",
    "ASSESSES",
    "OTHER",
]


class RelationAnnotation(BaseModel):
    """A relation between two spans (typically two entities)."""

    relation_type: AbstractRelationType
    head_span: SpanRef
    tail_span: SpanRef
    subtype: Optional[str] = None
    """Domain-specific subtype, e.g. 'EXCLUDES_FOR_CONDITION', 'COMPLIES_WITH_ISO'."""

    provenance: AnnotationProvenance


class TopicLabel(BaseModel):
    """A topic/theme label attached to the chunk."""

    label: str
    score: float = Field(ge=0.0, le=1.0)
    provenance: AnnotationProvenance


CitationKind = Literal[
    "internal_section",
    "external_standard",
    "literature",
    "regulation",
    "patent",
    "trademark",
    "url",
    "case_law",
    "statute",
    "treaty",
    "rfc",
    "iso_standard",
    "iec_standard",
    "other",
]


class CitationAnnotation(BaseModel):
    """A citation to an external source or internal section.

    Examples:
      'Section 3.2'              -> internal_section
      'ISO 9001:2015'            -> iso_standard
      '21 CFR 312.62'            -> regulation
      '[Smith 2024]'             -> literature
      'doi:10.1234/abc'          -> literature
      'Brown v. Board of Education, 347 U.S. 483 (1954)' -> case_law
      'RFC 7231'                 -> rfc
    """

    surface: str
    span: SpanRef
    citation_kind: CitationKind
    target_identifier: Optional[str] = None
    """Normalized identifier (e.g. 'ISO 9001:2015', '21 CFR 312.62')."""

    target_resolved: Optional[str] = None
    """Resolved URL/DOI/chunk_id if the citation could be resolved."""

    provenance: AnnotationProvenance


class SemanticEnrichment(BaseModel):
    """Entities, relations, topics, citations."""

    entities: list[EntityAnnotation] = Field(default_factory=list)
    relations: list[RelationAnnotation] = Field(default_factory=list)
    topic_labels: list[TopicLabel] = Field(default_factory=list)
    citations: list[CitationAnnotation] = Field(default_factory=list)

    embedding_id: Optional[str] = None
    """Reference to embedding stored externally (vector store / file)."""

    nearest_neighbor_chunk_ids: list[str] = Field(default_factory=list)
    """Top-K nearest-neighbor chunks within this document or corpus.
    Filled after embedding+indexing if available."""

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Layer 4 — Quality enrichment
# ===========================================================================


class PIIAnnotation(BaseModel):
    """A PII / sensitive-data mention.

    PII types intentionally union clinical (PHI) + legal + financial.
    Add domain-specific subtypes in `pii_type` as needed.
    """

    surface: str
    span: SpanRef
    pii_type: str
    """e.g. 'PERSON_NAME', 'DOB', 'ADDRESS', 'PHONE', 'EMAIL', 'MRN',
    'SSN', 'EIN', 'PASSPORT', 'IBAN', 'CREDIT_CARD', 'LEI', 'IP_ADDRESS'."""

    redact_suggested: bool = True
    provenance: AnnotationProvenance


class QualityEnrichment(BaseModel):
    """Per-chunk quality + privacy."""

    ocr_confidence: Optional[float] = None
    layout_confidence: Optional[float] = None
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    """How complete the signal is. 1.0 = no truncation / missing parts."""

    boilerplate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    """0.0 = unique content. 1.0 = identical/near-identical to many other chunks."""

    near_duplicate_chunk_ids: list[str] = Field(default_factory=list)
    """Other chunk_ids with high content similarity (SimHash/MinHash/embedding)."""

    language_purity: float = Field(default=1.0, ge=0.0, le=1.0)
    """1.0 = fully in declared primary language. <1.0 = mixed-language."""

    pii_detected: bool = False
    pii_annotations: list[PIIAnnotation] = Field(default_factory=list)

    quality_flags: list[str] = Field(default_factory=list)
    """Free-form flags, e.g. 'truncated', 'ocr_low_confidence',
    'language_mixed', 'figure_unrendered', 'table_merge_uncertain'."""

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Layer 5 — Domain enrichment (HINTS, not bindings)
# ===========================================================================


Domain = Literal[
    "clinical",
    "legal",
    "industrial",
    "financial",
    "scientific",
    "regulatory",
    "general",
    "unknown",
]


class DomainEnrichment(BaseModel):
    """Vertical-specific HINTS that downstream extraction may use.

    These DO NOT bind to a target schema. Target-schema binding (USDM
    EligibilityCriterion, LegalRuleML Obligation, AAS Submodel, XBRL fact)
    happens at the extraction stage. This layer only PRIMES the extractor.
    """

    domain: Domain = "unknown"

    section_taxonomy_mapping: dict[str, str] = Field(default_factory=dict)
    """Maps from taxonomy name to section identifier in that taxonomy.

    Examples:
      clinical:  {'ich_m11': '5.1', 'cdisc_sdtm_domain': 'IE'}
      legal:     {'contract_section': 'representations_warranties'}
      industrial:{'isa_95_level': 'L3', 'aas_submodel': 'TechnicalData'}
      financial: {'sec_10k_item': 'Item 1A. Risk Factors'}
    """

    role_hints: list[str] = Field(default_factory=list)
    """Free-form hints about the chunk's role in the document.

    Examples:
      clinical:  ['inclusion_criterion', 'primary_endpoint', 'soa_row']
      legal:     ['indemnification_clause', 'governing_law_clause']
      industrial:['safety_requirement', 'functional_requirement']
      financial: ['risk_factor', 'segment_revenue']
    """

    schema_class_hints: list[str] = Field(default_factory=list)
    """Likely target-schema classes the extractor MAY produce from this chunk.

    Examples:
      clinical:  ['EligibilityCriterion', 'Objective', 'Endpoint']
      legal:     ['Obligation', 'Right', 'Definition']
      industrial:['Requirement', 'Specification', 'Component']
      financial: ['XBRL:Revenues', 'XBRL:CashAndCashEquivalents']
    """

    custom: dict[str, Any] = Field(default_factory=dict)
    """Escape hatch for vertical-specific fields not yet promoted to first-class."""

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Optional chunk-type-specific sub-enrichments
# ===========================================================================


class CellAnnotation(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    role: Literal["header", "row_header", "data", "computed", "merged_origin", "blank"]
    surface: str
    canonical_value: Optional[str] = None
    """Normalized value if numeric/date/categorical."""

    canonical_unit: Optional[str] = None
    provenance: AnnotationProvenance


class TableEnrichment(BaseModel):
    """Populated only when chunk_type == 'table'."""

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    header_row_indices: list[int] = Field(default_factory=list)
    header_col_indices: list[int] = Field(default_factory=list)
    merged_cells: list[tuple[int, int, int, int]] = Field(default_factory=list)
    """(row_start, col_start, row_end, col_end) for each merged region."""

    caption_chunk_id: Optional[str] = None
    table_role: Optional[
        Literal[
            "data",
            "schedule",
            "specification",
            "comparison",
            "summary",
            "matrix",
            "key_value",
        ]
    ] = None

    cell_annotations: list[CellAnnotation] = Field(default_factory=list)
    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


class FigureEnrichment(BaseModel):
    """Populated only when chunk_type == 'figure'."""

    figure_kind: Optional[
        Literal[
            "chart",
            "diagram",
            "schematic",
            "photo",
            "screenshot",
            "flowchart",
            "map",
            "equation_image",
            "other",
        ]
    ] = None
    caption_chunk_id: Optional[str] = None
    extracted_text: Optional[str] = None
    """OCR'd text from inside the figure, if any."""

    extracted_data: Optional[dict[str, Any]] = None
    """Structured data extracted from chart figures (e.g. {x: [...], y: [...]})."""

    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


class CodeEnrichment(BaseModel):
    """Populated only when chunk_type == 'code'."""

    language: Optional[str] = None
    runnable: bool = False
    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


class EquationEnrichment(BaseModel):
    """Populated only when chunk_type == 'equation'."""

    latex: Optional[str] = None
    mathml: Optional[str] = None
    symbols_used: list[str] = Field(default_factory=list)
    layer_version: str = SCHEMA_VERSION
    provenance: AnnotationProvenance


# ===========================================================================
# Per-chunk envelope
# ===========================================================================


class ChunkEnrichment(BaseModel):
    """The complete enrichment of a single chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    schema_version: str = SCHEMA_VERSION
    enriched_at: datetime

    structural: StructuralEnrichment
    linguistic: LinguisticEnrichment
    semantic: SemanticEnrichment
    quality: QualityEnrichment
    domain: DomainEnrichment

    # Optional chunk-type-specific layers; populated only when applicable.
    table: Optional[TableEnrichment] = None
    figure: Optional[FigureEnrichment] = None
    code: Optional[CodeEnrichment] = None
    equation: Optional[EquationEnrichment] = None


# ===========================================================================
# Document-level envelope
# ===========================================================================


class GlossaryEntry(BaseModel):
    """A doc-local acronym or defined-term entry."""

    surface: str
    expansion: str
    defined_in_chunk_id: Optional[str] = None
    kind: Literal["acronym", "defined_term"]
    provenance: AnnotationProvenance


class CrossReferenceIndexEntry(BaseModel):
    """An entry in the document-internal cross-reference index."""

    surface: str
    """e.g. 'Section 3.2', 'Table 7', 'Annex B'."""

    target_chunk_id: Optional[str] = None
    target_section_path: Optional[list[str]] = None
    reference_kind: Literal[
        "section",
        "table",
        "figure",
        "appendix",
        "page",
        "footnote",
        "equation",
        "list_item",
    ]
    provenance: AnnotationProvenance


class VersionEvent(BaseModel):
    """An amendment/revision event in the document's history."""

    version_anchor: str
    """e.g. 'Amendment 3', 'Revision B', 'v2.1'."""

    effective_date: Optional[datetime] = None
    description: Optional[str] = None
    affected_chunk_ids: list[str] = Field(default_factory=list)


class DocumentEnrichment(BaseModel):
    """The complete enrichment of a document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    schema_version: str = SCHEMA_VERSION
    enriched_at: datetime

    # Document-level enrichment
    primary_language: str
    additional_languages: list[str] = Field(default_factory=list)
    domain: Domain = "unknown"

    glossary: list[GlossaryEntry] = Field(default_factory=list)
    cross_reference_index: list[CrossReferenceIndexEntry] = Field(default_factory=list)
    version_history: list[VersionEvent] = Field(default_factory=list)
    document_topic_labels: list[TopicLabel] = Field(default_factory=list)

    # Per-chunk enrichment, keyed by chunk_id
    chunks: dict[str, ChunkEnrichment] = Field(default_factory=dict)

    # Run metadata
    producers_used: list[str] = Field(default_factory=list)
    """All producers that contributed annotations to this document."""

    total_annotations: int = 0
    """Sum of annotations across all layers and chunks (for telemetry)."""


__all__ = [
    "SCHEMA_VERSION",
    "AnnotationProvenance",
    "SpanRef",
    "ChunkType",
    "InternalReference",
    "StructuralEnrichment",
    "AcronymUse",
    "DefinedTermUse",
    "NegationKind",
    "NegationAnnotation",
    "CertaintyLevel",
    "Modality",
    "CertaintyAnnotation",
    "TemporalKind",
    "TemporalAnnotation",
    "CoreferenceCluster",
    "LinguisticEnrichment",
    "AbstractEntityType",
    "EntityAnnotation",
    "AbstractRelationType",
    "RelationAnnotation",
    "TopicLabel",
    "CitationKind",
    "CitationAnnotation",
    "SemanticEnrichment",
    "PIIAnnotation",
    "QualityEnrichment",
    "Domain",
    "DomainEnrichment",
    "CellAnnotation",
    "TableEnrichment",
    "FigureEnrichment",
    "CodeEnrichment",
    "EquationEnrichment",
    "ChunkEnrichment",
    "GlossaryEntry",
    "CrossReferenceIndexEntry",
    "VersionEvent",
    "DocumentEnrichment",
]
