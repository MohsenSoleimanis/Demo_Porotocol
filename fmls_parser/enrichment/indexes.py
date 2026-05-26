"""Stage 1 — Document-level indexes.

Built ONCE per document. Output is consumed by every chunk's enrichment
(Stage 3) and by every extraction call (Stage 4). Caching this per-doc
means hot loops don't re-run LLM calls or regex sweeps.

Indexes built:
  - cross-reference: "Section 3.2" / "Table 7" / etc. → resolved chunk_id
  - acronyms:        "ORR" → "objective response rate" + source chunk_id
  - glossary:        "Investigational Product" → definition + source chunk_id
  - doc metadata:    protocol_number, sponsor, indication, phase, NCT ID

Costs:
  - xref index:     pure regex, free
  - acronyms:       regex + 1 LLM cleanup call (deduplicates, fixes obvious errors)
  - glossary:       1 LLM call on the definitions section (skipped if no such section)
  - metadata:       1 LLM call on §1 chunks

Total LLM cost per protocol: ~3 calls / ~$0.05 on gpt-4o-mini.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from fmls_parser.chunk_models import ChunkedDocument, Chunk
from fmls_parser.enrichment.domain_clinical import lookup_functional_label


# === Models ============================================================


class CrossRefEntry(BaseModel):
    surface: str                # "Section 3.2"
    kind: str                   # "section" | "table" | "figure" | "appendix" | "page"
    target_chunk_id: Optional[str] = None
    target_section_path: Optional[list[str]] = None


class AcronymEntry(BaseModel):
    surface: str                # "ORR"
    expansion: str              # "objective response rate"
    source_chunk_id: Optional[str] = None
    inferred_by: str = "regex"  # "regex" | "llm-cleanup"


class GlossaryEntry(BaseModel):
    surface: str                # "Investigational Product"
    definition: str
    source_chunk_id: Optional[str] = None


class DocMetadata(BaseModel):
    nct_id: Optional[str] = None
    sponsor_protocol_number: Optional[str] = None
    other_identifiers: dict[str, str] = Field(default_factory=dict)
    sponsor: Optional[str] = None
    indication: Optional[str] = None
    phase: Optional[str] = None
    full_title: Optional[str] = None
    short_title: Optional[str] = None


class DocIndexes(BaseModel):
    doc_id: str
    schema_version: str = "fmls-doc-indexes-1.0"
    built_at: datetime

    cross_references: list[CrossRefEntry] = Field(default_factory=list)
    acronyms: list[AcronymEntry] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    metadata: DocMetadata = Field(default_factory=DocMetadata)


# === Cross-reference index (regex, deterministic) ====================


_XREF_PATTERNS = [
    ("section",  re.compile(r"\bSection\s+(\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("table",    re.compile(r"\bTable\s+(\d+(?:[-.]\d+)?)", re.IGNORECASE)),
    ("figure",   re.compile(r"\bFigure\s+(\d+(?:[-.]\d+)?)", re.IGNORECASE)),
    ("appendix", re.compile(r"\bAppendix\s+([A-Z]\d*)", re.IGNORECASE)),
]


def build_xref_index(doc: ChunkedDocument) -> list[CrossRefEntry]:
    """Scan all chunk text for references, resolve against section_path index.

    Returns a flat list of CrossRefEntry. Same surface may appear multiple
    times (referenced from multiple chunks) but each entry is canonical.
    """
    # Section number -> first matching chunk_id + section_path
    section_index: dict[str, tuple[str, list[str]]] = {}
    for c in doc.chunks:
        if c.m11_section and c.m11_section not in section_index:
            section_index[c.m11_section] = (c.chunk_id, list(c.section_path))

    seen: set[tuple[str, str]] = set()
    out: list[CrossRefEntry] = []

    for c in doc.chunks:
        text = c.text or ""
        for kind, pattern in _XREF_PATTERNS:
            for m in pattern.finditer(text):
                surface = m.group(0).strip()
                key = (kind, m.group(1).strip())
                if key in seen:
                    continue
                seen.add(key)

                target_chunk_id = None
                target_section_path = None
                if kind == "section":
                    sec_num = m.group(1).strip()
                    if sec_num in section_index:
                        target_chunk_id, target_section_path = section_index[sec_num]
                    else:
                        # Try parent prefix (e.g., "3.2.1" -> "3.2" -> "3")
                        parts = sec_num.split(".")
                        while parts:
                            parts.pop()
                            prefix = ".".join(parts)
                            if prefix and prefix in section_index:
                                target_chunk_id, target_section_path = section_index[prefix]
                                break

                out.append(CrossRefEntry(
                    surface=surface,
                    kind=kind,
                    target_chunk_id=target_chunk_id,
                    target_section_path=target_section_path,
                ))
    return out


# === Acronym index (regex + optional LLM cleanup) ====================


# Pattern A: "Expansion (ACRONYM)" — common form
# Matches: "Serious Adverse Event (SAE)"
_ACRONYM_DEFINITION = re.compile(
    r"\b((?:[A-Z][A-Za-z\-]+\s+){1,6}[A-Za-z\-]+)\s*\(([A-Z]{2,10}s?)\)"
)

# Pattern B: "ACRONYM (Expansion)" — less common but happens
_ACRONYM_DEFINITION_REVERSE = re.compile(
    r"\b([A-Z]{2,10})\s*\(((?:[A-Za-z][A-Za-z\-]*\s+){1,6}[A-Za-z][A-Za-z\-]*)\)"
)


def build_acronym_index(doc: ChunkedDocument) -> list[AcronymEntry]:
    """Extract acronym definitions via pattern matching.

    Pattern A: 'Serious Adverse Event (SAE)' — expansion before, acronym in parens
    Pattern B: 'SAE (Serious Adverse Event)' — acronym before, expansion in parens

    Dedupes by acronym surface; first occurrence wins. For ambiguous cases
    (different expansions in different chunks), the first chunk wins and
    inconsistencies should be logged at run-time (caller responsibility).
    """
    seen: dict[str, AcronymEntry] = {}

    for c in doc.chunks:
        text = c.text or ""

        for m in _ACRONYM_DEFINITION.finditer(text):
            expansion = m.group(1).strip()
            acronym = m.group(2).strip()
            # Heuristic: first letters of expansion should roughly match acronym
            initials = "".join(w[0].upper() for w in expansion.split() if w)
            acronym_letters = acronym.rstrip("s").upper()  # tolerate trailing 's'
            if not _initials_match(initials, acronym_letters):
                continue
            if acronym not in seen:
                seen[acronym] = AcronymEntry(
                    surface=acronym,
                    expansion=expansion,
                    source_chunk_id=c.chunk_id,
                    inferred_by="regex",
                )

        for m in _ACRONYM_DEFINITION_REVERSE.finditer(text):
            acronym = m.group(1).strip()
            expansion = m.group(2).strip()
            initials = "".join(w[0].upper() for w in expansion.split() if w)
            acronym_letters = acronym.rstrip("s").upper()
            if not _initials_match(initials, acronym_letters):
                continue
            if acronym not in seen:
                seen[acronym] = AcronymEntry(
                    surface=acronym,
                    expansion=expansion,
                    source_chunk_id=c.chunk_id,
                    inferred_by="regex",
                )

    return list(seen.values())


def _initials_match(initials: str, acronym: str) -> bool:
    """Tolerant initials match. ORR matches 'Objective Response Rate' (initials=ORR).
    Allows missing letters for short words (of/and/in/the/...).
    """
    if not initials or not acronym:
        return False
    # Strict equality
    if initials == acronym:
        return True
    # Allow acronym to be a substring of initials (e.g., expansion has filler words)
    if acronym in initials:
        return True
    # Tolerant: at least 60% of acronym letters appear in initials in order
    i, j = 0, 0
    matches = 0
    while i < len(acronym) and j < len(initials):
        if acronym[i] == initials[j]:
            matches += 1
            i += 1
        j += 1
    return matches / max(1, len(acronym)) >= 0.6


# === Glossary index (1 LLM call on definitions section) =============


def find_definition_sections(doc: ChunkedDocument) -> list[Chunk]:
    """Find chunks whose m11_section maps to a definitions/glossary role."""
    out: list[Chunk] = []
    for c in doc.chunks:
        # Common patterns: ICH M11 doesn't have a dedicated "Definitions"
        # section; sponsors usually put them in §10 (appendix) or §0 prelim.
        # Section path heuristic: a heading containing "Definition" or
        # "Glossary" or "Abbreviation" upstream.
        path_lower = " > ".join(c.section_path).lower()
        if any(k in path_lower for k in ("definition", "glossary", "abbreviation")):
            out.append(c)
    return out


GLOSSARY_PROMPT = """You are extracting term definitions from a clinical-trial protocol's glossary/definitions section.

Below are chunks from the document's definitions sections. Extract each
distinctly-defined term as (term, definition). Use ONLY what the text says.

Rules:
- `term` is the headword being defined, exactly as it appears (verbatim).
- `definition` is the explanatory clause, lightly trimmed of leading dashes/punctuation.
- Skip acronym-only entries (e.g., "SAE - Serious Adverse Event"); they go to the acronym index.
- Skip prose paragraphs that aren't definitions.
- If no definitions found, return an empty list.

Source text:
\"\"\"
{text}
\"\"\"
"""


class GlossaryItem(BaseModel):
    term: str
    definition: str


class GlossaryExtraction(BaseModel):
    items: list[GlossaryItem] = Field(default_factory=list)


def build_glossary_index(
    doc: ChunkedDocument,
    *,
    llm_client=None,
    model: str = "gpt-4o-mini",
) -> list[GlossaryEntry]:
    """Build the glossary index via 1 LLM call over the definitions section.

    If no definitions section is found, returns empty list (no LLM cost).
    """
    if llm_client is None:
        return []

    def_chunks = find_definition_sections(doc)
    if not def_chunks:
        return []

    # Concatenate text from def chunks (cap at ~30K chars to stay in budget)
    parts: list[tuple[str, str]] = []  # (chunk_id, text)
    char_budget = 30_000
    for c in def_chunks:
        if char_budget <= 0:
            break
        text = c.text[:char_budget]
        parts.append((c.chunk_id, text))
        char_budget -= len(text)

    combined = "\n\n".join(t for _, t in parts)
    prompt = GLOSSARY_PROMPT.format(text=combined)

    try:
        result: GlossaryExtraction = llm_client.chat.completions.create(
            model=model,
            response_model=GlossaryExtraction,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4000,
        )
    except Exception as e:
        print(f"  [WARN] glossary LLM call failed: {e}")
        return []

    # Assign each definition to the most likely source chunk: pick the first
    # chunk whose text contains the term (case-insensitive).
    out: list[GlossaryEntry] = []
    for item in result.items:
        source = None
        term_low = item.term.lower()
        for chunk_id, text in parts:
            if term_low in text.lower():
                source = chunk_id
                break
        out.append(GlossaryEntry(
            surface=item.term,
            definition=item.definition,
            source_chunk_id=source,
        ))
    return out


# === Document-level metadata extraction (1 LLM call on §1) ===========


_NCT_RX = re.compile(r"\b(NCT\d{8})\b")
_EUDRACT_RX = re.compile(r"\b(\d{4}-\d{6}-\d{2})\b")


METADATA_PROMPT = """You are extracting study-level metadata from a clinical-trial protocol's first pages.

Below is the synopsis section. Extract identifiers, sponsor, indication, phase, and titles.

Rules:
- Only return what the text states. Use null for missing fields.
- `nct_id` matches NCT followed by 8 digits.
- `sponsor_protocol_number` is the sponsor's internal study ID (often near "Protocol Number" / "Study ID").
- `phase` is one of "1", "1/2", "2", "2/3", "3", "4", or null. Map textual phases ("Phase III" → "3").
- `indication` is the primary disease/condition under study (short noun phrase).
- `full_title` is the official long title; `short_title` is a shorter version if present.

Source text:
\"\"\"
{text}
\"\"\"
"""


def extract_doc_metadata(
    doc: ChunkedDocument,
    *,
    llm_client=None,
    model: str = "gpt-4o-mini",
) -> DocMetadata:
    """Extract doc-level metadata from §1 synopsis chunks via 1 LLM call.

    Falls back to a deterministic regex pass if no LLM client is provided.
    """
    # Find §1 chunks
    synopsis_chunks = [
        c for c in doc.chunks
        if c.m11_section and (
            c.m11_section == "1" or c.m11_section.startswith("1.")
        )
    ]
    # Fallback: if no §1 chunks tagged, use the first 30 chunks (title page region)
    if not synopsis_chunks:
        synopsis_chunks = doc.chunks[:30]

    combined = "\n\n".join(c.text for c in synopsis_chunks[:50])  # cap

    # Deterministic regex sweep first
    nct = _NCT_RX.search(combined)
    eudract = _EUDRACT_RX.search(combined)

    md = DocMetadata(
        nct_id=nct.group(1) if nct else None,
        other_identifiers={"eudract": eudract.group(1)} if eudract else {},
    )

    # Doc ID is usually the NCT
    if not md.nct_id and doc.doc_id.startswith("NCT"):
        md.nct_id = doc.doc_id

    if llm_client is None:
        return md

    # LLM pass to fill the rest
    capped = combined[:25_000]
    prompt = METADATA_PROMPT.format(text=capped)
    try:
        llm_md: DocMetadata = llm_client.chat.completions.create(
            model=model,
            response_model=DocMetadata,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=800,
        )
    except Exception as e:
        print(f"  [WARN] doc metadata LLM call failed: {e}")
        return md

    # Merge — prefer regex-extracted IDs (they're authoritative), use LLM for everything else
    return DocMetadata(
        nct_id=md.nct_id or llm_md.nct_id,
        sponsor_protocol_number=llm_md.sponsor_protocol_number,
        other_identifiers={**md.other_identifiers, **(llm_md.other_identifiers or {})},
        sponsor=llm_md.sponsor,
        indication=llm_md.indication,
        phase=llm_md.phase,
        full_title=llm_md.full_title,
        short_title=llm_md.short_title,
    )


# === Top-level orchestrator ==========================================


def build_doc_indexes(
    doc: ChunkedDocument,
    *,
    llm_client=None,
    model: str = "gpt-4o-mini",
) -> DocIndexes:
    """Build all doc-level indexes. Reuses LLM client across the 2 LLM calls."""
    print(f"  [indexes] building xref index...")
    xrefs = build_xref_index(doc)
    print(f"  [indexes]   {len(xrefs)} cross-references")

    print(f"  [indexes] building acronym index...")
    acronyms = build_acronym_index(doc)
    print(f"  [indexes]   {len(acronyms)} acronyms")

    print(f"  [indexes] building glossary index (1 LLM call if def section found)...")
    glossary = build_glossary_index(doc, llm_client=llm_client, model=model)
    print(f"  [indexes]   {len(glossary)} glossary entries")

    print(f"  [indexes] extracting doc metadata (1 LLM call)...")
    metadata = extract_doc_metadata(doc, llm_client=llm_client, model=model)
    print(f"  [indexes]   nct={metadata.nct_id}, sponsor={metadata.sponsor}, phase={metadata.phase}, indication={metadata.indication}")

    return DocIndexes(
        doc_id=doc.doc_id,
        built_at=datetime.now(timezone.utc),
        cross_references=xrefs,
        acronyms=acronyms,
        glossary=glossary,
        metadata=metadata,
    )


__all__ = [
    "CrossRefEntry",
    "AcronymEntry",
    "GlossaryEntry",
    "DocMetadata",
    "DocIndexes",
    "build_xref_index",
    "build_acronym_index",
    "build_glossary_index",
    "extract_doc_metadata",
    "build_doc_indexes",
]
