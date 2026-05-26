"""Stage 3 — Chunk-level enrichment.

Per chunk, populates a ChunkEnrichment with:
  - Structural:  chunk_type, section_path, bbox, internal_references
  - Linguistic:  language, sentences, acronym uses (lookup from doc index),
                 defined-term uses, negation (MedSpaCy ConText), temporal,
                 coreference clusters (fastcoref)
  - Semantic:    candidate entities (GLiNER-bio), citations, embedding_id
  - Quality:     OCR/layout conf, PII flags
  - Domain:      role_hints, schema_class_hints (from m11_section via rule table),
                 arm_context, visit_context

Heavy models (GLiNER-bio, fastcoref, MedCPT embedding, MedSpaCy) are
loaded ONCE per process. The orchestrator chunks through doc.chunks and
emits a DocumentEnrichment envelope with per-chunk ChunkEnrichment.

GPU is auto-detected; falls back to CPU.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

import langdetect
import pysbd
import torch
from gliner import GLiNER
from medspacy.context import ConTextRule

from fmls_parser.chunk_models import ChunkedDocument, Chunk
from fmls_parser.enrichment.domain_clinical import (
    lookup_functional_label,
    role_hints as get_role_hints,
    usdm_class_hints,
)
from fmls_parser.enrichment.indexes import DocIndexes
from fmls_parser.enrichment.normalize import (
    casefold_key,
    normalize_negation_kind,
    normalize_text,
)
from fmls_parser.enrichment.schema import (
    AbstractEntityType,
    AcronymUse,
    AnnotationProvenance,
    CertaintyAnnotation,
    ChunkEnrichment,
    CoreferenceCluster,
    DefinedTermUse,
    Domain,
    DomainEnrichment,
    DocumentEnrichment,
    EntityAnnotation,
    InternalReference,
    LinguisticEnrichment,
    NegationAnnotation,
    QualityEnrichment,
    SCHEMA_VERSION,
    SemanticEnrichment,
    SpanRef,
    StructuralEnrichment,
    TemporalAnnotation,
)


# === Global lazy-loaded models =====================================

_GLINER: Optional[GLiNER] = None
_MEDSPACY_NLP = None
_FASTCOREF = None


def _get_gliner() -> GLiNER:
    global _GLINER
    if _GLINER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [enrich] loading GLiNER-bio on {device}...")
        _GLINER = GLiNER.from_pretrained("urchade/gliner_large_bio-v0.1").to(device)
    return _GLINER


def _get_medspacy():
    global _MEDSPACY_NLP
    if _MEDSPACY_NLP is None:
        import medspacy
        print(f"  [enrich] loading MedSpaCy (ConText + NegEx)...")
        _MEDSPACY_NLP = medspacy.load(enable=["medspacy_context"])
    return _MEDSPACY_NLP


def _get_fastcoref():
    global _FASTCOREF
    if _FASTCOREF is None:
        from fastcoref import FCoref
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [enrich] loading fastcoref on {device}...")
        _FASTCOREF = FCoref(device=device)
    return _FASTCOREF


# === GLiNER entity labels (clinical) ===============================

GLINER_LABELS = [
    "drug", "condition", "procedure", "biomarker", "anatomical location",
    "demographic", "lab value", "dose", "frequency", "route", "duration",
    "visit", "endpoint", "objective", "criterion", "intervention",
    "arm", "population", "indication", "organization", "site",
    "ctcae grade", "performance status", "person",
]

# Map GLiNER labels to our abstract entity taxonomy
_GLINER_TO_ABSTRACT: dict[str, AbstractEntityType] = {
    "drug": "PRODUCT",
    "condition": "CONDITION",
    "procedure": "PROCEDURE",
    "biomarker": "MEASUREMENT",
    "anatomical location": "LOCATION",
    "demographic": "PERSON",
    "lab value": "MEASUREMENT",
    "dose": "QUANTITY",
    "frequency": "QUANTITY",
    "route": "OTHER",
    "duration": "QUANTITY",
    "visit": "EVENT",
    "endpoint": "MEASUREMENT",
    "objective": "EVENT",
    "criterion": "TERM",
    "intervention": "PRODUCT",
    "arm": "ROLE",
    "population": "ROLE",
    "indication": "CONDITION",
    "organization": "ORGANIZATION",
    "site": "ORGANIZATION",
    "ctcae grade": "MEASUREMENT",
    "performance status": "MEASUREMENT",
    "person": "PERSON",
}


# === Internal-reference resolution =================================

_XREF_INLINE = re.compile(
    r"\b(?:Section|Sec\.?|Table|Figure|Fig\.?|Appendix)\s+(?:\d+(?:\.\d+)*|[A-Z]\d*)",
    re.IGNORECASE,
)


def _resolve_internal_references(
    chunk: Chunk,
    doc_xref_lookup: dict[str, dict[str, Any]],
    now: datetime,
) -> list[InternalReference]:
    """For each xref surface found in the chunk, lookup against doc xref index."""
    out: list[InternalReference] = []
    seen: set[str] = set()
    for m in _XREF_INLINE.finditer(chunk.text):
        surface = m.group(0).strip()
        key = surface.lower()
        if key in seen:
            continue
        seen.add(key)

        # Determine kind
        sk = surface.lower()
        if sk.startswith(("section", "sec")): kind = "section"
        elif sk.startswith("table"):           kind = "table"
        elif sk.startswith(("figure", "fig")): kind = "figure"
        elif sk.startswith("appendix"):        kind = "appendix"
        else:                                  kind = "section"

        # Lookup in doc xref index
        target = doc_xref_lookup.get(key, {})
        out.append(InternalReference(
            surface_text=surface,
            span=SpanRef(chunk_id=chunk.chunk_id, start=m.start(), end=m.end()),
            target_chunk_id=target.get("target_chunk_id"),
            target_section_path=target.get("target_section_path"),
            reference_kind=kind,
            provenance=AnnotationProvenance(
                producer="regex-xref-v1",
                producer_version="1.0",
                created_at=now,
                confidence=1.0 if target.get("target_chunk_id") else 0.5,
            ),
        ))
    return out


# === Find acronym/defined-term mentions ============================

def _find_acronym_uses(
    chunk: Chunk,
    acronym_lookup: dict[str, str],   # acronym surface -> expansion
    now: datetime,
) -> list[AcronymUse]:
    """Scan chunk for occurrences of each known acronym surface."""
    out: list[AcronymUse] = []
    for acronym, expansion in acronym_lookup.items():
        if not acronym:
            continue
        # Find acronym as whole word; case-sensitive (acronyms are case-sensitive)
        pattern = re.compile(rf"\b{re.escape(acronym)}\b")
        for m in pattern.finditer(chunk.text):
            out.append(AcronymUse(
                surface=acronym,
                span=SpanRef(chunk_id=chunk.chunk_id, start=m.start(), end=m.end()),
                expansion=expansion,
                provenance=AnnotationProvenance(
                    producer="acronym-lookup",
                    producer_version="1.0",
                    created_at=now,
                    confidence=1.0,
                ),
            ))
    return out


def _find_defined_term_uses(
    chunk: Chunk,
    glossary_lookup: dict[str, dict[str, Any]],   # term surface -> {definition, source_chunk_id}
    now: datetime,
) -> list[DefinedTermUse]:
    """Scan chunk for occurrences of each glossary term (case-insensitive whole-word)."""
    out: list[DefinedTermUse] = []
    chunk_lower = chunk.text.lower()
    for term, info in glossary_lookup.items():
        if not term or len(term) < 4:
            continue
        # Case-insensitive whole-word match
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for m in pattern.finditer(chunk.text):
            out.append(DefinedTermUse(
                surface=m.group(0),
                span=SpanRef(chunk_id=chunk.chunk_id, start=m.start(), end=m.end()),
                definition_chunk_id=info.get("source_chunk_id"),
                provenance=AnnotationProvenance(
                    producer="glossary-lookup",
                    producer_version="1.0",
                    created_at=now,
                    confidence=1.0,
                ),
            ))
    return out


# === GLiNER entity extraction =====================================

def _gliner_entities(
    chunk: Chunk,
    now: datetime,
) -> list[EntityAnnotation]:
    """Run GLiNER-bio on chunk text. Returns candidate entities."""
    if not chunk.text or len(chunk.text.strip()) < 10:
        return []
    try:
        gliner = _get_gliner()
        preds = gliner.predict_entities(chunk.text, GLINER_LABELS, threshold=0.5)
    except Exception as e:
        print(f"  [WARN] GLiNER failed on chunk {chunk.chunk_id}: {e}")
        return []

    out: list[EntityAnnotation] = []
    for p in preds:
        abstract_type = _GLINER_TO_ABSTRACT.get(p["label"], "OTHER")
        subtype = p["label"].upper().replace(" ", "_")
        out.append(EntityAnnotation(
            surface=p["text"],
            span=SpanRef(chunk_id=chunk.chunk_id, start=p["start"], end=p["end"]),
            entity_type=abstract_type,
            subtype=subtype,
            provenance=AnnotationProvenance(
                producer="gliner-bio",
                producer_version="urchade/gliner_large_bio-v0.1",
                created_at=now,
                confidence=float(p["score"]),
            ),
        ))
    return out


# === Negation detection (MedSpaCy ConText + GLiNER entities) =======

def _negation_annotations(
    chunk: Chunk,
    entities: list[EntityAnnotation],
    now: datetime,
) -> list[NegationAnnotation]:
    """Apply MedSpaCy ConText to detect negated entity spans.

    Strategy: pass chunk text through medspacy. For each GLiNER entity span,
    check if the text in that region is contained within a negation modifier
    span detected by ConText.
    """
    if not entities:
        return []
    try:
        nlp = _get_medspacy()
        # We need to insert our entities so ConText can scope to them
        doc = nlp(chunk.text)
    except Exception as e:
        print(f"  [WARN] medspacy failed on chunk {chunk.chunk_id}: {e}")
        return []

    if not doc.ents:
        # medspacy didn't find entities natively; fall back to text-based check
        # using ConText's modifier spans against GLiNER entity character spans
        return _negation_via_text_search(chunk, entities, doc, now)

    # For each natively-detected entity, check ConText flags
    out: list[NegationAnnotation] = []
    for ent in doc.ents:
        if not hasattr(ent._, "is_negated"):
            continue
        if not ent._.is_negated and not getattr(ent._, "is_historical", False) and not getattr(ent._, "is_family", False):
            continue
        # Determine kind
        if getattr(ent._, "is_family", False):
            kind = "family"
        elif getattr(ent._, "is_historical", False):
            kind = "history"
        elif getattr(ent._, "is_uncertain", False):
            kind = "possible"
        else:
            kind = "definite"

        out.append(NegationAnnotation(
            target_span=SpanRef(
                chunk_id=chunk.chunk_id,
                start=ent.start_char,
                end=ent.end_char,
            ),
            cue_text="",  # medspacy doesn't expose cue text easily; v2
            cue_span=SpanRef(
                chunk_id=chunk.chunk_id, start=ent.start_char, end=ent.end_char,
            ),
            negation_type=normalize_negation_kind(kind),
            provenance=AnnotationProvenance(
                producer="medspacy-context",
                producer_version="1.3.1",
                ruleset_version="medspacy-default",
                created_at=now,
                confidence=0.85,
            ),
        ))
    return out


# Cheap fallback when medspacy's own entity detection misses our spans:
# scan the chunk text for explicit negation cue patterns near each GLiNER entity.
_NEG_CUES = [
    (re.compile(r"\b(no evidence of|no signs of|not exhibit(?:ing|s|ed)?)\b", re.IGNORECASE), "definite"),
    (re.compile(r"\b(rule[ds]? out|ruled out|negative for|denies?)\b", re.IGNORECASE), "definite"),
    (re.compile(r"\b(without any|absence of|free of)\b", re.IGNORECASE), "definite"),
    (re.compile(r"\b(history of|hx of|prior history of)\b", re.IGNORECASE), "history"),
    (re.compile(r"\bfamily history of\b", re.IGNORECASE), "family"),
    (re.compile(r"\b(possible|possibly|suspect(?:ed|s)?|probable|likely)\b", re.IGNORECASE), "possible"),
    (re.compile(r"\bdo(?:es)? not have\b", re.IGNORECASE), "definite"),
    (re.compile(r"\bmust not have\b", re.IGNORECASE), "definite"),
    (re.compile(r"\bshould not have\b", re.IGNORECASE), "definite"),
]
_NEG_WINDOW = 80  # chars of look-ahead from cue end to entity start


def _negation_via_text_search(
    chunk: Chunk,
    entities: list[EntityAnnotation],
    doc,  # unused but kept for symmetry
    now: datetime,
) -> list[NegationAnnotation]:
    """Heuristic negation scoping. For each cue match, mark any entity span
    starting within _NEG_WINDOW chars after the cue as negated."""
    out: list[NegationAnnotation] = []
    text = chunk.text
    for pattern, kind in _NEG_CUES:
        for m in pattern.finditer(text):
            cue_end = m.end()
            # Find entities starting within window
            for ent in entities:
                if cue_end <= ent.span.start <= cue_end + _NEG_WINDOW:
                    out.append(NegationAnnotation(
                        target_span=ent.span,
                        cue_text=m.group(0),
                        cue_span=SpanRef(
                            chunk_id=chunk.chunk_id, start=m.start(), end=m.end(),
                        ),
                        negation_type=normalize_negation_kind(kind),
                        provenance=AnnotationProvenance(
                            producer="negex-fallback",
                            producer_version="1.0",
                            ruleset_version="custom-clinical-v1",
                            created_at=now,
                            confidence=0.75,
                        ),
                    ))
    return out


# === Per-chunk enrichment ==========================================

def enrich_chunk(
    chunk: Chunk,
    doc_indexes: DocIndexes,
    *,
    enable_coref: bool = False,
) -> ChunkEnrichment:
    """Run the full Stage 3 enrichment on a single chunk.

    `doc_indexes` provides the lookups for xref/acronym/glossary.
    `enable_coref` runs fastcoref per chunk (slow); off by default for v1.
    """
    now = datetime.now(timezone.utc)

    # === Build doc-level lookup dicts (could be precomputed in orchestrator) ===
    acronym_lookup = {a.surface: a.expansion for a in doc_indexes.acronyms}
    glossary_lookup = {
        g.surface: {"definition": g.definition, "source_chunk_id": g.source_chunk_id}
        for g in doc_indexes.glossary
    }
    xref_lookup: dict[str, dict[str, Any]] = {}
    for x in doc_indexes.cross_references:
        xref_lookup[x.surface.lower()] = {
            "target_chunk_id": x.target_chunk_id,
            "target_section_path": x.target_section_path,
        }

    # === Structural layer ===
    structural = StructuralEnrichment(
        chunk_type=_chunk_type_str(chunk.block_type),
        section_path=list(chunk.section_path),
        parent_heading=chunk.section_path[-1] if chunk.section_path else None,
        depth=chunk.section_depth,
        page=chunk.page_num,
        bbox=(chunk.bbox.x0, chunk.bbox.y0, chunk.bbox.x1, chunk.bbox.y1) if chunk.bbox else None,
        reading_order_index=chunk.source_block_indices[0] if chunk.source_block_indices else 0,
        internal_references=_resolve_internal_references(chunk, xref_lookup, now),
        co_occurring_chunks=[],  # filled later via cross-doc analysis
        provenance=AnnotationProvenance(
            producer="fmls-chunker",
            producer_version="0.1",
            created_at=now,
            confidence=1.0,
        ),
    )

    # === Linguistic layer ===
    try:
        lang = langdetect.detect(chunk.text[:500]) if chunk.text else "en"
    except Exception:
        lang = "en"
    sentences = _sentence_spans(chunk)
    acronym_uses = _find_acronym_uses(chunk, acronym_lookup, now)
    defined_term_uses = _find_defined_term_uses(chunk, glossary_lookup, now)

    # === Semantic layer (must run before negation, which needs entities) ===
    entities = _gliner_entities(chunk, now)

    # Negation runs after entities (it scopes to entity spans)
    negation_annotations = _negation_annotations(chunk, entities, now)

    linguistic = LinguisticEnrichment(
        language=lang,
        sentence_spans=sentences,
        acronym_uses=acronym_uses,
        defined_term_uses=defined_term_uses,
        negation_annotations=negation_annotations,
        certainty_annotations=[],  # v2
        temporal_annotations=[],   # v2: integrate dateparser + heideltime-alt
        coreference_clusters=[],   # v2: enable_coref flag
        provenance=AnnotationProvenance(
            producer="fmls-linguistic",
            producer_version="0.1",
            created_at=now,
            confidence=1.0,
        ),
    )

    semantic = SemanticEnrichment(
        entities=entities,
        relations=[],
        topic_labels=[],
        citations=[],   # filled by doc-level citation pass; for now use xrefs as proxy
        embedding_id=None,  # v2: compute MedCPT embedding
        nearest_neighbor_chunk_ids=[],
        provenance=AnnotationProvenance(
            producer="fmls-semantic",
            producer_version="0.1",
            created_at=now,
            confidence=1.0,
        ),
    )

    # === Quality layer ===
    quality = QualityEnrichment(
        ocr_confidence=chunk.metadata.get("ocr_confidence"),
        layout_confidence=chunk.metadata.get("layout_confidence"),
        language_purity=1.0,  # v2: char-class analysis
        completeness_score=1.0,
        boilerplate_score=0.0,  # v2: simhash
        near_duplicate_chunk_ids=[],
        pii_detected=False,
        pii_annotations=[],
        quality_flags=[],
        provenance=AnnotationProvenance(
            producer="fmls-quality",
            producer_version="0.1",
            created_at=now,
            confidence=1.0,
        ),
    )

    # === Domain priming (clinical) ===
    functional_label = lookup_functional_label(chunk.m11_section)
    domain = DomainEnrichment(
        domain="clinical",
        section_taxonomy_mapping={"ich_m11": chunk.m11_section} if chunk.m11_section else {},
        role_hints=get_role_hints(functional_label) if functional_label else [],
        schema_class_hints=usdm_class_hints(functional_label) if functional_label else [],
        custom={"functional_label": functional_label} if functional_label else {},
        provenance=AnnotationProvenance(
            producer="fmls-domain-clinical",
            producer_version="0.1",
            created_at=now,
            confidence=1.0 if functional_label else 0.5,
        ),
    )

    return ChunkEnrichment(
        chunk_id=chunk.chunk_id,
        enriched_at=now,
        structural=structural,
        linguistic=linguistic,
        semantic=semantic,
        quality=quality,
        domain=domain,
    )


# === Helpers =======================================================

def _chunk_type_str(block_type) -> str:
    """Map our BlockType enum to enrichment schema chunk_type literal."""
    s = block_type.value if hasattr(block_type, "value") else str(block_type)
    valid = {
        "heading": "heading",
        "paragraph": "paragraph",
        "table": "table",
        "table_cell": "table_cell",
        "list_item": "list_item",
        "list": "list_item",
        "footnote": "footnote",
        "caption": "caption",
        "figure": "figure",
        "code": "code",
        "equation": "equation",
        "definition": "definition",
    }
    return valid.get(s, "other")


_PYSBD_CACHE: dict[str, Any] = {}


def _sentence_spans(chunk: Chunk) -> list[SpanRef]:
    """Sentence-segment chunk text. Returns list of SpanRef."""
    if not chunk.text:
        return []
    lang_simple = "en"  # pysbd supports a few languages; default English for now
    if lang_simple not in _PYSBD_CACHE:
        _PYSBD_CACHE[lang_simple] = pysbd.Segmenter(language=lang_simple, clean=False)
    seg = _PYSBD_CACHE[lang_simple]
    sentences = seg.segment(chunk.text)
    out: list[SpanRef] = []
    offset = 0
    for s in sentences:
        if not s.strip():
            offset += len(s)
            continue
        idx = chunk.text.find(s, offset)
        if idx < 0:
            continue
        out.append(SpanRef(
            chunk_id=chunk.chunk_id, start=idx, end=idx + len(s),
        ))
        offset = idx + len(s)
    return out


# === Top-level orchestrator ========================================

def enrich_document(
    doc: ChunkedDocument,
    doc_indexes: DocIndexes,
    *,
    chunk_filter=None,
    progress: bool = True,
) -> DocumentEnrichment:
    """Enrich every chunk in the document. Returns DocumentEnrichment envelope.

    `chunk_filter(chunk) -> bool`: optional predicate to skip chunks
    (e.g., headers/footers, chunks without text, chunks below quality threshold).
    """
    now = datetime.now(timezone.utc)

    enriched_chunks: dict[str, ChunkEnrichment] = {}
    producers_used: set[str] = set()
    total_annotations = 0

    chunks_to_process = doc.chunks
    if chunk_filter is not None:
        chunks_to_process = [c for c in doc.chunks if chunk_filter(c)]
    n = len(chunks_to_process)
    print(f"  [enrich] processing {n} chunks (filtered from {doc.total_chunks})...")

    for i, chunk in enumerate(chunks_to_process):
        if progress and i % 100 == 0 and i > 0:
            print(f"  [enrich]   {i}/{n}  ({100*i//n}%)")

        ce = enrich_chunk(chunk, doc_indexes)
        enriched_chunks[chunk.chunk_id] = ce

        # Telemetry
        producers_used.add(ce.structural.provenance.producer)
        producers_used.add(ce.linguistic.provenance.producer)
        producers_used.add(ce.semantic.provenance.producer)
        total_annotations += (
            len(ce.structural.internal_references)
            + len(ce.linguistic.acronym_uses)
            + len(ce.linguistic.defined_term_uses)
            + len(ce.linguistic.negation_annotations)
            + len(ce.semantic.entities)
        )

    print(f"  [enrich] done. {n} chunks enriched, {total_annotations} total annotations.")

    return DocumentEnrichment(
        document_id=doc.doc_id,
        enriched_at=now,
        primary_language="en",  # v2: derive from chunks
        domain="clinical",
        chunks=enriched_chunks,
        producers_used=sorted(producers_used),
        total_annotations=total_annotations,
    )


__all__ = [
    "GLINER_LABELS",
    "enrich_chunk",
    "enrich_document",
]
