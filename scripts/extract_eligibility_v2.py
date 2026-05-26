"""Stage 4 v2 — Eligibility extraction via per-chunk role classification.

Replaces v1's flat-text generation approach. Architecturally:

  v1: LLM regenerates verbatim text → fuzzy substring match → drop on drift
  v2: LLM classifies role per chunk → assemble USDM records from chunk.text
       verbatim (no regeneration) → bytes are identical to source by construction

What changes:
  - Prompt now shows chunks with (block_type, indent, marker, text)
  - LLM output is a list of role labels, one per chunk index
  - No `text: str` field in LLM response — text comes from chunks directly
  - Sub-explanations / sub-bullets attached to parent criterion via reading order
  - LaTeX / Unicode rendering issues eliminated by construction
  - Sub-bullet false positives eliminated by explicit role labels

Usage:
    python scripts/extract_eligibility_v2.py AZ_demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from fmls_parser.chunk_models import ChunkedDocument, Chunk
from fmls_parser.enrichment.schema import DocumentEnrichment, ChunkEnrichment
from fmls_parser.enrichment.indexes import DocIndexes
from fmls_parser.extraction import (
    Provenance,
    Extracted,
    EligibilityCriterion,
    EligibilityCriterionItem,
    Code,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# === Per-chunk role classification model =========================

ChunkRole = Literal[
    "criterion",          # top-level eligibility criterion
    "sub_explanation",    # elaboration / definition of preceding criterion
    "sub_bullet",         # sub-bullet attached to preceding criterion / sub-clause
    "category_header",    # category label like "Age", "Reproduction"
    "section_intro",      # section intro paragraph, not a criterion
    "skip",               # footer, page number, table, irrelevant
]


class ChunkRoleClassification(BaseModel):
    """LLM output: one classification per chunk shown.

    NO text field. The chunk's text comes from the chunks list directly
    (zero-regeneration guarantee).
    """
    i: int = Field(description="1-based index of the chunk (matches the prompt list)")
    role: ChunkRole
    identifier: Optional[str] = Field(
        default=None,
        description="For role=criterion: the criterion's number/label as it appears in the source (e.g. '1', '6', '16'). Null otherwise.",
    )
    category: Optional[Literal["inclusion", "exclusion", "withdrawal"]] = Field(
        default=None,
        description="For role=criterion: inclusion|exclusion|withdrawal based on section context. Null otherwise.",
    )


class RoleClassificationResponse(BaseModel):
    classifications: list[ChunkRoleClassification]
    notes: Optional[str] = Field(default=None, description="Optional brief audit notes.")


# === Chunk rendering ==============================================

_MARKER_RX = re.compile(r"^\s*(\d+|\([a-z]\)|\([A-Z]\)|[A-Z]\d*|\([ivx]+\))\s+", re.IGNORECASE)


def extract_marker_and_text(text: str) -> tuple[Optional[str], str]:
    """Pull leading marker like '1 ', '(a) ', 'I.' off the front of chunk text."""
    if not text:
        return None, text
    m = _MARKER_RX.match(text)
    if not m:
        return None, text.strip()
    marker = m.group(1).strip()
    rest = text[m.end():].strip()
    return marker, rest


def quantize_indent(x0: float | None) -> int:
    """Round bbox x0 to the nearest 12 pixels to make indent levels comparable.

    PDFs typically indent in increments of ~24pt; quantizing to 12 makes the
    grouping robust to small inconsistencies.
    """
    if x0 is None:
        return 72
    return int(round(x0 / 12) * 12)


def render_chunks_for_llm(chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    """Render chunks as a numbered metadata table for the LLM.

    Returns (rendered_text, chunks_in_order). The chunks_in_order list has
    1-based indexing aligned with the prompt — chunks_in_order[i-1] is the
    chunk for classification index i.
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        bt = c.block_type.value if hasattr(c.block_type, "value") else str(c.block_type)
        indent = quantize_indent(c.bbox.x0 if c.bbox else None)
        marker, body = extract_marker_and_text(c.text or "")
        marker_str = f' marker="{marker}"' if marker else ""
        # Truncate very long text for the prompt; LLM only needs to classify role
        body_disp = body if len(body) <= 400 else body[:400] + " ..."
        # Remove embedded newlines for table readability
        body_disp = re.sub(r"\s+", " ", body_disp)
        lines.append(f'  i={i:3d}  block={bt:11s}  indent=col_{indent:3d}{marker_str:20s}  "{body_disp}"')
    return "\n".join(lines), chunks


# === Prompt =======================================================

SYSTEM_PROMPT = """You are a clinical-protocol structural analyst.

Your task: classify each chunk's role in the document. You DO NOT extract or
regenerate text. The text already exists in the source. You only output a
role label per chunk index.

Roles:
  - "criterion"        : a top-level eligibility criterion (numbered list item at outer indent)
  - "sub_explanation"  : elaboration/definition of the preceding criterion (paragraph deeper-indented after a criterion)
  - "sub_bullet"       : sub-bullet attached to preceding criterion or sub-clause (list_item at deeper indent)
  - "category_header"  : category label like "Age", "Reproduction", "Medical Conditions" (short heading without section numbering, immediately preceding criteria)
  - "section_intro"    : section intro paragraph, not a criterion (typically immediately after a section heading)
  - "skip"             : footer, page number, table caption, "Note:", irrelevant content

For role=criterion only, also provide:
  - identifier: the criterion's number/label as shown in the source (e.g. "1", "5", "16")
  - category: inclusion | exclusion | withdrawal based on section context

Decision heuristics:
  - block=list_item at outer indent (typically col 72-90) WITH a numeric marker → "criterion"
  - block=paragraph at deeper indent (col 100+) immediately after a criterion → "sub_explanation"
  - block=list_item at deeper indent → "sub_bullet"
  - block=heading with SHORT text (<6 words) NOT matching section numbering (no "5.1", "5.2") between criteria → "category_header"
  - block=heading with section numbering ("5.1", "5.2") → "skip" (it's the section header we already know)
  - block=paragraph between section heading and first criterion → "section_intro"
  - Page numbers, "Note:", table references → "skip"
"""


def build_prompt(
    section_label: str,
    section_path: list[str],
    chunks: list[Chunk],
) -> tuple[str, list[Chunk]]:
    rendered, chunks_ordered = render_chunks_for_llm(chunks)
    section_path_str = " > ".join(section_path) if section_path else section_label

    # Determine default category from section label
    sl_lower = section_label.lower()
    if "inclusion" in sl_lower:
        default_cat = "inclusion"
    elif "exclusion" in sl_lower:
        default_cat = "exclusion"
    elif "withdrawal" in sl_lower or "discontinuation" in sl_lower:
        default_cat = "withdrawal"
    else:
        default_cat = "inclusion"

    prompt = f"""Section: {section_label}
Section path: {section_path_str}
Default category for criteria in this section: {default_cat}

Chunks (in document order, 1-indexed):

{rendered}

Output one ChunkRoleClassification per chunk above, in order (i=1, i=2, ..., i={len(chunks_ordered)}).
"""
    return prompt, chunks_ordered


# === Assembly: classifications → USDM records ====================

def assemble_criteria(
    classifications: list[ChunkRoleClassification],
    chunks: list[Chunk],
    extractor_provenance: dict,
) -> list[tuple[Extracted[EligibilityCriterionItem], Extracted[EligibilityCriterion]]]:
    """Walk classifications in order; assemble USDM (Item, Criterion) pairs.

    Strategy:
      - For each chunk classified "criterion", start a new group.
      - Append following "sub_explanation" and "sub_bullet" chunks to the
        current criterion's description until the next "criterion" or
        "category_header" or "skip" / "section_intro" boundary.
      - "category_header" → remember as label for the next criterion.
    """
    by_i = {c.i: c for c in classifications}
    pairs: list[tuple] = []

    current_category_header: Optional[str] = None
    current_criterion: Optional[dict] = None  # buffer for the criterion being built

    def flush_current():
        nonlocal current_criterion
        if current_criterion is None:
            return
        pairs.append(_build_usdm_pair(current_criterion, extractor_provenance))
        current_criterion = None

    for i in range(1, len(chunks) + 1):
        cls = by_i.get(i)
        if cls is None:
            continue   # missing classification — treat as skip
        chunk = chunks[i - 1]

        if cls.role == "criterion":
            flush_current()
            # Strip the marker prefix from the criterion text so the
            # text field doesn't contain the number twice
            _, body = extract_marker_and_text(chunk.text or "")
            current_criterion = {
                "parent_chunk": chunk,
                "category": cls.category or "inclusion",
                "identifier": cls.identifier or "",
                "text": body,
                "category_header": current_category_header,
                "sub_chunks": [],
            }
        elif cls.role in ("sub_explanation", "sub_bullet"):
            if current_criterion is not None:
                current_criterion["sub_chunks"].append({"chunk": chunk, "role": cls.role})
            # else: orphan sub-item, ignore (LLM mis-classified)
        elif cls.role == "category_header":
            flush_current()
            current_category_header = (chunk.text or "").strip()
        elif cls.role in ("section_intro", "skip"):
            flush_current()
            # don't clear current_category_header — a new criterion may follow
        # else: unknown role — skip

    flush_current()
    return pairs


def _build_usdm_pair(buf: dict, extractor_provenance: dict) -> tuple:
    """Build (Extracted[Item], Extracted[Criterion]) from an assembly buffer."""
    parent_chunk = buf["parent_chunk"]
    text = buf["text"]

    # Collect sub-item text into description
    description_parts = []
    if buf["category_header"]:
        description_parts.append(f"Category: {buf['category_header']}")
    for sub in buf["sub_chunks"]:
        c = sub["chunk"]
        marker = "  ↳" if sub["role"] == "sub_bullet" else "  ※"
        description_parts.append(f"{marker} {c.text.strip()}")
    description = "\n".join(description_parts) if description_parts else None

    item_id = f"crit_item_{uuid.uuid4().hex[:8]}"
    ref_id = f"crit_ref_{uuid.uuid4().hex[:8]}"

    now = datetime.now(timezone.utc)
    bbox = (
        (parent_chunk.bbox.x0, parent_chunk.bbox.y0, parent_chunk.bbox.x1, parent_chunk.bbox.y1)
        if parent_chunk.bbox else None
    )
    provenance = Provenance(
        evidence_chunk_id=parent_chunk.chunk_id,
        evidence_quote=text,
        evidence_page=parent_chunk.page_num,
        evidence_bbox=bbox,
        extractor=extractor_provenance["model"],
        extractor_version=extractor_provenance["model"],
        prompt_hash=extractor_provenance["prompt_hash"],
        extraction_confidence=0.95,
        extracted_at=now,
    )

    item = EligibilityCriterionItem(
        id=item_id,
        name="EligibilityCriterionItem",
        label=buf["identifier"] or None,
        description=description,
        text=text,
        dictionaryId=None,
        notes=[],
        extensionAttributes=[],
        instanceType="EligibilityCriterionItem",
    )

    category_code = Code(
        id=str(uuid.uuid4()),
        code=buf["category"].upper(),
        codeSystem="FMLS-Category",
        codeSystemVersion="0.1",
        decode=buf["category"],
        instanceType="Code",
    )

    ref = EligibilityCriterion(
        id=ref_id,
        name=f"EligibilityCriterion {buf['identifier'] or ref_id}",
        label=buf["identifier"] or None,
        description=None,
        category=category_code,
        identifier=buf["identifier"] or "",
        criterionItemId=item_id,
        nextId=None,
        previousId=None,
        notes=[],
        extensionAttributes=[],
        instanceType="EligibilityCriterion",
    )

    return (
        Extracted[EligibilityCriterionItem](value=item, provenance=provenance),
        Extracted[EligibilityCriterion](value=ref, provenance=provenance),
    )


# === Main =========================================================

def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-completion-tokens", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc_dir = DATASET / args.stem
    chunks_path = doc_dir / "chunks.json"
    enriched_path = doc_dir / "enriched.json"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found", file=sys.stderr)
        return 2

    print(f"=== Loading {args.stem} ===")
    doc = ChunkedDocument.model_validate(json.loads(chunks_path.read_text(encoding="utf-8")))
    enriched = None
    if enriched_path.exists():
        enriched = DocumentEnrichment.model_validate(
            json.loads(enriched_path.read_text(encoding="utf-8"))
        )
    print(f"  chunks={doc.total_chunks}, enriched={(len(enriched.chunks) if enriched else 0)}")

    # Filter to chunks where the schema hint targets EligibilityCriterion
    chunks_by_id = {c.chunk_id: c for c in doc.chunks}
    if enriched:
        elig_chunk_ids = [
            cid for cid, ce in enriched.chunks.items()
            if "EligibilityCriterion" in ce.domain.schema_class_hints
        ]
    else:
        elig_chunk_ids = [
            c.chunk_id for c in doc.chunks
            if c.m11_section and c.m11_section.startswith("5")
        ]
    elig_chunks = [chunks_by_id[cid] for cid in elig_chunk_ids if cid in chunks_by_id]
    elig_chunks.sort(key=lambda c: (c.page_num, c.source_block_indices[0] if c.source_block_indices else 0))
    print(f"  eligibility candidate chunks: {len(elig_chunks)}")

    # Group by section heading (deepest "Inclusion"/"Exclusion"/"Withdrawal" or m11_section)
    groups: dict[str, list[Chunk]] = {}
    for c in elig_chunks:
        leaf = next(
            (h for h in reversed(c.section_path)
             if any(k in h.lower() for k in ("inclusion", "exclusion", "withdrawal"))),
            c.section_path[-1] if c.section_path else "unknown",
        )
        groups.setdefault(leaf, []).append(c)

    # Filter out groups not actually about criteria (Lifestyle/Screen failures don't belong)
    KEEP_GROUPS = lambda label: any(
        k in label.lower() for k in ("inclusion", "exclusion", "withdrawal")
    )
    groups = {k: v for k, v in groups.items() if KEEP_GROUPS(k)}

    print(f"  groups (after filter): {list(groups.keys())}")
    for k, v in groups.items():
        print(f"    {k}: {len(v)} chunks")

    client = instructor.from_openai(OpenAI()) if not args.dry_run else None

    all_pairs: list = []
    for section_label, chunks in groups.items():
        print()
        print(f"=== Classifying {section_label} ({len(chunks)} chunks) ===")
        prompt, chunks_ordered = build_prompt(
            section_label=section_label,
            section_path=chunks[0].section_path if chunks else [],
            chunks=chunks,
        )
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        if args.dry_run:
            print(f"  [DRY] prompt length: {len(prompt)} chars / ~{len(prompt)//4} tokens")
            print(prompt[:1200])
            print("  ...")
            continue

        try:
            response: RoleClassificationResponse = client.chat.completions.create(
                model=args.model,
                response_model=RoleClassificationResponse,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=args.max_completion_tokens,
            )
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            continue

        # Summarize role distribution
        from collections import Counter
        role_counts = Counter(c.role for c in response.classifications)
        print(f"  classifications: {dict(role_counts.most_common())}")
        if response.notes:
            print(f"  notes: {response.notes}")

        # Assemble USDM records from classifications + chunks
        extractor_prov = {"model": args.model, "prompt_hash": prompt_hash}
        pairs = assemble_criteria(response.classifications, chunks_ordered, extractor_prov)
        print(f"  -> {len(pairs)} criteria assembled")
        all_pairs.extend(pairs)

    if args.dry_run:
        return 0

    # Write output
    out_dir = doc_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eligibility_v2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item, ref in all_pairs:
            f.write(item.model_dump_json() + "\n")
            f.write(ref.model_dump_json() + "\n")

    print()
    print(f"=== Done ===")
    print(f"  Total criteria extracted: {len(all_pairs)}")
    print(f"  Output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
