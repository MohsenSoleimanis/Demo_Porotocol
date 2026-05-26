"""v3 element-specific extraction.

Per USDM target class:
  1. Filter chunks by semantic_role + domain hints
  2. Retrieve element-relevant chunks (with chunk reunion for split criteria)
  3. Call LLM with the chunk-reunion text + USDM Pydantic class as response_model
  4. Output records reference chunks by ID (no text regeneration)

Compared to v2: v2 processed ALL eligibility chunks in one prompt with role
classification. v3 processes ALL USDM target classes the same way:
EligibilityCriterion, Objective, Endpoint, StudyArm, etc. — each with its own
retrieval filter and Pydantic schema.

For the v1-eligibility-equivalent use case, this produces the same result as v2
because the role-classification step happened upstream (in enrich_v3), and the
extractor here just walks chunks where semantic_role == "primary_item" within
an eligibility section.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from fmls_parser.chunk_models import Chunk
from fmls_parser.pipeline_v3.models import ChunkKeyEnrichment
from fmls_parser.extraction import (
    Provenance,
    Extracted,
    EligibilityCriterion,
    EligibilityCriterionItem,
    Code,
)


# === Element-specific retrieval ===================================


def filter_for_eligibility(
    chunks: list[Chunk],
    enrichments: dict[str, ChunkKeyEnrichment],
) -> list[tuple[Chunk, ChunkKeyEnrichment]]:
    """Return (chunk, enrichment) pairs that look like primary eligibility criteria.

    Filter rules:
      - chunk is in §5.1, §5.2, or §5.3 (Withdrawal) by m11_section
      - semantic_role is "primary_item"
      - NOT in §5.3 if we treat 5.3 as Lifestyle (we'll let role decide)
    """
    out: list[tuple[Chunk, ChunkKeyEnrichment]] = []
    for c in chunks:
        e = enrichments.get(c.chunk_id)
        if e is None:
            continue
        if e.semantic_role != "primary_item":
            continue
        m11 = c.m11_section or ""
        if not (m11 == "5.1" or m11 == "5.2" or m11 == "5.3" or
                m11.startswith("5.1.") or m11.startswith("5.2.") or m11.startswith("5.3.")):
            continue
        # Skip §5.3 Lifestyle Considerations
        section_lower = " > ".join(c.section_path).lower()
        if "lifestyle" in section_lower:
            continue
        # Skip if chunk is in §5.4 (Screen Failures — not eligibility)
        if "screen failure" in section_lower:
            continue
        out.append((c, e))
    return out


def reunite_with_sub_items(
    primary_chunks: list[tuple[Chunk, ChunkKeyEnrichment]],
    all_chunks: list[Chunk],
    enrichments: dict[str, ChunkKeyEnrichment],
) -> dict[str, dict]:
    """For each primary_item chunk, gather the sub_explanation / sub_bullet /
    exception chunks that follow it in reading order (until the next
    primary_item or category_header).

    Returns {primary_chunk_id: {"primary": Chunk, "subs": [Chunk, ...]}}.
    """
    by_id = {c.chunk_id: c for c in all_chunks}
    chunks_in_order = sorted(
        all_chunks,
        key=lambda c: (c.page_num, c.source_block_indices[0] if c.source_block_indices else 0),
    )
    idx_of = {c.chunk_id: i for i, c in enumerate(chunks_in_order)}

    primary_ids = {c.chunk_id for c, _ in primary_chunks}
    result: dict[str, dict] = {}

    for primary, e in primary_chunks:
        start_idx = idx_of[primary.chunk_id]
        subs: list[Chunk] = []
        for i in range(start_idx + 1, len(chunks_in_order)):
            cand = chunks_in_order[i]
            cand_e = enrichments.get(cand.chunk_id)
            if cand_e is None:
                continue
            # Stop boundary: next primary_item, category_header, or section_header
            if cand_e.semantic_role in ("primary_item", "category_header", "section_header"):
                break
            # Capture sub-items
            if cand_e.semantic_role in ("sub_explanation", "sub_clause", "sub_bullet", "exception"):
                subs.append(cand)
        result[primary.chunk_id] = {"primary": primary, "subs": subs}

    return result


# === Extraction (still LLM-based for category + identifier inference) ====
# but uses chunk.text VERBATIM (no regeneration)


def assemble_eligibility_records(
    reunited: dict[str, dict],
    *,
    category_inference_method: str = "section_path",
    extractor_label: str = "fmls-pipeline-v3",
) -> list[tuple[Extracted[EligibilityCriterionItem], Extracted[EligibilityCriterion]]]:
    """Build USDM records from reunited primary+sub chunks.

    Category is inferred from section_path (no LLM call needed):
      - "Inclusion Criteria" in path → inclusion
      - "Exclusion Criteria" in path → exclusion
      - "Withdrawal" in path → withdrawal

    Identifier extracted from the chunk's leading marker (regex from enrich_v3).

    Returns list of (Extracted[Item], Extracted[Criterion]) pairs.
    """
    from fmls_parser.pipeline_v3.enrich_v3 import extract_marker

    pairs: list = []

    for primary_id, payload in reunited.items():
        primary: Chunk = payload["primary"]
        subs: list[Chunk] = payload["subs"]

        # Category from section_path
        section_lower = " > ".join(primary.section_path).lower()
        if "inclusion" in section_lower:
            category = "inclusion"
        elif "exclusion" in section_lower:
            category = "exclusion"
        elif "withdrawal" in section_lower:
            category = "withdrawal"
        else:
            category = "inclusion"

        # Identifier from marker
        identifier = extract_marker(primary.text or "")

        # Text body (strip leading marker so it doesn't appear twice in the record)
        text = (primary.text or "").strip()
        if identifier:
            # Remove the marker prefix (e.g., "1 ", "(a) ")
            text = re.sub(r"^\s*\S+\s+", "", text, count=1)

        # Sub-items go into description, preserved verbatim
        description_parts = []
        for sub in subs:
            description_parts.append(f"• {(sub.text or '').strip()}")
        description = "\n".join(description_parts) if description_parts else None

        # Provenance
        bbox = (
            (primary.bbox.x0, primary.bbox.y0, primary.bbox.x1, primary.bbox.y1)
            if primary.bbox else None
        )
        provenance = Provenance(
            evidence_chunk_id=primary.chunk_id,
            evidence_quote=text,
            evidence_page=primary.page_num,
            evidence_bbox=bbox,
            extractor=extractor_label,
            extractor_version=extractor_label,
            prompt_hash="-",
            extraction_confidence=0.95,
            extracted_at=datetime.now(timezone.utc),
        )

        # USDM records
        item_id = f"crit_item_{uuid.uuid4().hex[:8]}"
        ref_id = f"crit_ref_{uuid.uuid4().hex[:8]}"

        item = EligibilityCriterionItem(
            id=item_id,
            name="EligibilityCriterionItem",
            label=identifier,
            description=description,
            text=text,
            dictionaryId=None,
            notes=[],
            extensionAttributes=[],
            instanceType="EligibilityCriterionItem",
        )
        category_code = Code(
            id=str(uuid.uuid4()),
            code=category.upper(),
            codeSystem="FMLS-Category",
            codeSystemVersion="0.1",
            decode=category,
            instanceType="Code",
        )
        ref = EligibilityCriterion(
            id=ref_id,
            name=f"EligibilityCriterion {identifier or ref_id}",
            label=identifier,
            description=None,
            category=category_code,
            identifier=identifier or "",
            criterionItemId=item_id,
            nextId=None,
            previousId=None,
            notes=[],
            extensionAttributes=[],
            instanceType="EligibilityCriterion",
        )
        pairs.append((
            Extracted[EligibilityCriterionItem](value=item, provenance=provenance),
            Extracted[EligibilityCriterion](value=ref, provenance=provenance),
        ))

    return pairs


__all__ = [
    "filter_for_eligibility",
    "reunite_with_sub_items",
    "assemble_eligibility_records",
]
