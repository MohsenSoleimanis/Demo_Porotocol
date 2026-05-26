"""v3 chunk enrichment with semantic_role + MDKeyChunker-style single-call enrichment.

For each chunk, we make ONE LLM call (or one rule-based call) that returns:
  - semantic_role (criterion / sub_explanation / category_header / boilerplate / ...)
  - marker, parent_role_chunk_id, category_label
  - title, summary, keywords, typed_entities, hypothetical_questions, semantic_key
  - is_boilerplate flag

This is the architectural pivot from v1/v2:
  - v1/v2 ran multiple shallow passes (GLiNER for entities, MedSpaCy for negation,
    LLM at extraction time for role) — duplicated work, scattered output
  - v3 runs ONE rich enrichment pass per chunk, persisted, reused by all downstream
    USDM extractors

Rule-first, LLM-fallback:
  - For chunks where MinerU's block_type + bbox + content gives an unambiguous answer
    (page numbers, footers, isolated headings with section numbering), we classify via
    rules (free, deterministic)
  - For everything else, an LLM call produces the full ChunkKeyEnrichment
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import instructor
from openai import OpenAI

from fmls_parser.chunk_models import Chunk, ChunkedDocument
from fmls_parser.pipeline_v3.models import (
    ChunkKeyEnrichment,
    ChunkSemanticRole,
    SemanticRoleAnnotation,
)


# === Rule-based fast-path classification =========================

_NUMERIC_SECTION_RX = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")
_PAGE_NUMBER_RX = re.compile(r"^\s*\d{1,4}\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)
_FOOTER_HEADER_HINTS = re.compile(
    r"(?i)(confidential|proprietary|amendment\s+\d|page\s+\d+\s+of\s+\d+|version\s+\d)"
)
_MARKER_RX = re.compile(r"^\s*(\d+|\([a-z]\)|\([A-Z]\)|[A-Z]\d?|\([ivx]+\))[\.\s]")


def extract_marker(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MARKER_RX.match(text)
    return m.group(1) if m else None


def rule_classify(chunk: Chunk) -> Optional[SemanticRoleAnnotation]:
    """Return a SemanticRoleAnnotation iff a rule unambiguously matches.

    Returns None when no rule matches — the LLM fallback handles those cases.
    """
    text = (chunk.text or "").strip()
    if not text:
        return _make_annotation("skip", confidence=1.0, method="rule")

    bt = chunk.block_type.value if hasattr(chunk.block_type, "value") else str(chunk.block_type)

    # Page numbers
    if _PAGE_NUMBER_RX.match(text):
        return _make_annotation("page_number", confidence=1.0, method="rule")

    # Header/footer text
    if bt in ("header", "footer") or _FOOTER_HEADER_HINTS.search(text):
        return _make_annotation("header_footer", confidence=0.9, method="rule")

    # Footnote bodies (MinerU labels these)
    if bt == "footnote":
        return _make_annotation("footnote_body", confidence=1.0, method="rule")

    # Tables (preserve as-is)
    if bt == "table":
        return _make_annotation("table", confidence=1.0, method="rule")

    if bt == "caption":
        # Could be table_caption or figure_caption; without proximity info default to figure
        return _make_annotation("figure_caption", confidence=0.6, method="rule")

    # Section headers (chunker tagged m11_section means this IS the section
    # heading itself if block_type == heading AND text matches numeric prefix)
    if bt == "heading" and _NUMERIC_SECTION_RX.match(text):
        return _make_annotation("section_header", confidence=0.95, method="rule")

    # Otherwise let the LLM decide
    return None


def _make_annotation(
    role: ChunkSemanticRole,
    *,
    confidence: float,
    method: str,
    marker: Optional[str] = None,
    parent_role_chunk_id: Optional[str] = None,
    category_label: Optional[str] = None,
    notes: Optional[str] = None,
) -> SemanticRoleAnnotation:
    return SemanticRoleAnnotation(
        role=role,
        marker=marker,
        parent_role_chunk_id=parent_role_chunk_id,
        category_label=category_label,
        confidence=confidence,
        method=method,
        notes=notes,
        classified_at=datetime.now(timezone.utc),
        classifier_version="fmls-pipeline-v3",
    )


# === LLM-based MDKeyChunker enrichment =============================

ENRICH_SYSTEM_PROMPT = """You are enriching a single chunk of a clinical-trial
protocol with structured metadata. You do NOT regenerate or extract any of the
chunk's text. You annotate.

Your output: ONE ChunkKeyEnrichment record describing this chunk.

Required fields:
  - semantic_role: what IS this chunk in the document's logical structure?
    Choose from the closed set:
      • primary_item       — the main content unit (a criterion, an endpoint, a procedure step)
      • sub_explanation    — paragraph elaborating the preceding primary_item
      • sub_clause         — labeled clause like (a), (b)
      • sub_bullet         — bullet point under primary_item or sub_clause
      • exception          — "except for..." clause
      • section_header     — section heading
      • section_intro      — intro paragraph immediately after a section_header
      • category_header    — mid-section label like "Age", "Reproduction", "Medical Conditions"
      • table / table_caption / figure_caption
      • footnote_marker / footnote_body
      • header_footer / page_number / boilerplate / note
      • skip / unknown

  - marker: leading marker "1" / "(a)" / "i." if present, else null
  - category_label: if role==category_header, the label text
  - title: a short 5-12 word title for the chunk
  - summary: 1-2 sentence summary of what this chunk says
  - keywords: top 3-8 keywords from the chunk
  - typed_entities: list of {type, surface} dicts for domain entities mentioned
    (use abstract types: PERSON, ORGANIZATION, CONDITION, PROCEDURE, PRODUCT,
    MEASUREMENT, QUANTITY, DATE, TIME, EVENT, REFERENCE, IDENTIFIER, OTHER)
  - hypothetical_questions: 2-4 questions this chunk would answer
  - semantic_key: a short canonical phrase capturing the main idea
  - is_boilerplate: true if this looks like cross-document templated text

Be decisive. Avoid "unknown" unless the content is truly opaque.
"""


def build_enrich_prompt(chunk: Chunk, neighbors: dict) -> str:
    """Build the enrichment prompt for one chunk with neighbor context."""
    bt = chunk.block_type.value if hasattr(chunk.block_type, "value") else str(chunk.block_type)
    indent = int(round(chunk.bbox.x0 / 12) * 12) if chunk.bbox else 72

    prev = neighbors.get("prev")
    next_ = neighbors.get("next")

    return f"""Chunk to classify and enrich:

  text:        \"\"\"{(chunk.text or '').strip()[:1200]}\"\"\"
  block_type:  {bt}
  indent_col:  {indent}
  section:     {" > ".join(chunk.section_path) if chunk.section_path else "(unknown)"}
  m11_section: {chunk.m11_section or "(none)"}
  page:        {chunk.page_num + 1}
  parent_id:   {chunk.parent_chunk_id or "(none)"}

Preceding chunk (for context only; do not annotate it):
  {f'block_type={prev["block_type"]}, role_hint=(see your output), text="{prev["text"][:150]}"' if prev else "(none — first chunk in section)"}

Following chunk (for context only):
  {f'block_type={next_["block_type"]}, text="{next_["text"][:150]}"' if next_ else "(none — last chunk in section)"}

Output one ChunkKeyEnrichment record describing the chunk above.
"""


def enrich_chunks_v3(
    chunks: list[Chunk],
    *,
    llm_client=None,
    model: str = "gpt-4o-mini",
    use_llm_for_clear_cases: bool = False,
) -> dict[str, ChunkKeyEnrichment]:
    """Run v3 enrichment on a list of chunks.

    Strategy:
      1. Try rule_classify() per chunk — fast, free.
      2. For chunks where rules return None (or rule confidence is low),
         call the LLM to produce the full ChunkKeyEnrichment.

    Returns dict {chunk_id: ChunkKeyEnrichment}.
    """
    out: dict[str, ChunkKeyEnrichment] = {}
    chunks_needing_llm: list[Chunk] = []

    # Pass 1: rule-based classification
    for chunk in chunks:
        rule_ann = rule_classify(chunk) if not use_llm_for_clear_cases else None
        if rule_ann is not None and rule_ann.confidence >= 0.9:
            out[chunk.chunk_id] = ChunkKeyEnrichment(
                semantic_role=rule_ann.role,
                marker=rule_ann.marker,
                parent_role_chunk_id=rule_ann.parent_role_chunk_id,
                category_label=rule_ann.category_label,
                title=None,
                summary=None,
                keywords=[],
                typed_entities=[],
                hypothetical_questions=[],
                semantic_key=None,
                is_boilerplate=rule_ann.role in ("header_footer", "page_number", "boilerplate"),
                confidence=rule_ann.confidence,
            )
        else:
            chunks_needing_llm.append(chunk)

    # Pass 2: LLM enrichment for the rest
    if not chunks_needing_llm or llm_client is None:
        # Fill remaining with low-confidence "unknown" so every chunk has an entry
        for chunk in chunks_needing_llm:
            out[chunk.chunk_id] = ChunkKeyEnrichment(
                semantic_role="unknown",
                confidence=0.0,
            )
        return out

    # Build neighbor map for context
    by_idx = {i: c for i, c in enumerate(chunks)}
    chunk_index = {c.chunk_id: i for i, c in enumerate(chunks)}

    print(f"  [enrich_v3] {len(chunks)} chunks total; {len(out)} resolved by rules; "
          f"{len(chunks_needing_llm)} need LLM ({100*len(chunks_needing_llm)//max(1,len(chunks))}%)")

    for n, chunk in enumerate(chunks_needing_llm):
        if (n + 1) % 50 == 0:
            print(f"  [enrich_v3]   LLM call {n + 1}/{len(chunks_needing_llm)}")
        idx = chunk_index[chunk.chunk_id]
        prev_chunk = by_idx.get(idx - 1)
        next_chunk = by_idx.get(idx + 1)
        neighbors = {
            "prev": _neighbor_summary(prev_chunk) if prev_chunk else None,
            "next": _neighbor_summary(next_chunk) if next_chunk else None,
        }
        prompt = build_enrich_prompt(chunk, neighbors)
        try:
            result: ChunkKeyEnrichment = llm_client.chat.completions.create(
                model=model,
                response_model=ChunkKeyEnrichment,
                messages=[
                    {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=800,
            )
            out[chunk.chunk_id] = result
        except Exception as e:
            print(f"  [enrich_v3]   FAIL on {chunk.chunk_id}: {e}")
            out[chunk.chunk_id] = ChunkKeyEnrichment(
                semantic_role="unknown",
                confidence=0.0,
            )

    return out


def _neighbor_summary(chunk: Chunk) -> dict:
    bt = chunk.block_type.value if hasattr(chunk.block_type, "value") else str(chunk.block_type)
    return {"block_type": bt, "text": (chunk.text or "")[:200]}


# === Persistence helper ============================================


def save_chunk_key_enrichments(enrichments: dict[str, ChunkKeyEnrichment], path: Path) -> None:
    """Write the enrichment dict to JSON."""
    payload = {
        "schema_version": "fmls-pipeline-v3",
        "n_chunks": len(enrichments),
        "enrichments": {
            cid: e.model_dump(mode="json") for cid, e in enrichments.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_chunk_key_enrichments(path: Path) -> dict[str, ChunkKeyEnrichment]:
    """Read back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        cid: ChunkKeyEnrichment.model_validate(e)
        for cid, e in payload["enrichments"].items()
    }


__all__ = [
    "extract_marker",
    "rule_classify",
    "enrich_chunks_v3",
    "save_chunk_key_enrichments",
    "load_chunk_key_enrichments",
]
