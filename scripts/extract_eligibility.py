"""Stage 4 — Extract USDM EligibilityCriterion + EligibilityCriterionItem from a protocol.

End-to-end pipeline:
  1. Load chunks.json + enriched.json + doc_indexes.json
  2. Filter chunks where schema_class_hints contains 'EligibilityCriterion'
  3. Group by section (§5.1 inclusion vs §5.2 exclusion)
  4. For each group: build prompt using enrichment priors → call LLM → list of extracted criteria
  5. Validate evidence_quote as substring of source text
  6. Convert each extracted criterion to USDM EligibilityCriterion + EligibilityCriterionItem
  7. Wrap with Provenance, write to dataset/{stem}/extracted/eligibility.jsonl

Usage:
    python scripts/extract_eligibility.py AZ_demo
    python scripts/extract_eligibility.py AZ_demo --model gpt-4o-mini --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    USDM_SCHEMA_VERSION,
    Provenance,
    Extracted,
    EligibilityCriterion,
    EligibilityCriterionItem,
    Code,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# === Simplified extraction view (what the LLM produces) ============

class ExtractedCriterion(BaseModel):
    """LLM-friendly view of one eligibility criterion.

    Smaller than USDM EligibilityCriterion — only the fields the LLM
    needs to extract. We post-process into USDM after validation.
    """
    text: str = Field(description=(
        "VERBATIM criterion text from the source — must be a literal substring "
        "of the chunk content. No paraphrasing, no summarization."
    ))
    category: Literal["inclusion", "exclusion", "withdrawal"] = Field(description=(
        "Category. Determined by the SECTION context: inclusion section -> inclusion. "
        "Exclusion section -> exclusion. Polarity flip: if the criterion text has a "
        "negation cue (rule out, no evidence of, must not have, denies), the effective "
        "polarity flips."
    ))
    identifier: Optional[str] = Field(default=None, description=(
        "Sponsor's identifier if present in the source (e.g. 'Inc 1', 'Exclusion 3.2', "
        "'5.1.a'). null if not present."
    ))
    has_negation: bool = Field(default=False, description=(
        "True if the criterion contains explicit negation language."
    ))
    has_conditional: bool = Field(default=False, description=(
        "True if the criterion has a conditional clause ('if X then Y', 'unless', "
        "'for patients with X', etc.)."
    ))


class CriterionExtractionResponse(BaseModel):
    """LLM response shape: list of extracted criteria + brief audit note."""
    criteria: list[ExtractedCriterion]
    notes: Optional[str] = Field(default=None, description=(
        "Brief audit notes (e.g. 'some criteria spanned multiple bullets', "
        "'identifier numbering inconsistent'). Empty/null if nothing notable."
    ))


# === Prompt builder ================================================

SYSTEM_PROMPT = """You extract structured clinical-trial eligibility criteria.

Output verbatim text only — never paraphrase or summarize. Each criterion is one
atomic enrollment requirement. Compound criteria stay as one criterion (text
includes the entire compound clause).

Rules:
1. Output `text` MUST be a verbatim substring of the chunk text.
2. Polarity is determined by section type, NOT just negation:
   - Inclusion section, plain text "must have X" -> category=inclusion
   - Inclusion section, "must NOT have X" -> category=inclusion (the criterion is "must not have X" — semantically excludes X but lives in inclusion section)
   - Exclusion section, plain text "subjects with X" -> category=exclusion
3. Set `has_negation` if the criterion uses explicit negation cues.
4. Set `has_conditional` if there's an if/when/unless/for-patients-with conditional structure.
5. Sponsor identifier (e.g. "1", "Inc 1", "3.2.a") goes in `identifier`; null if not present.
6. Don't invent criteria the text doesn't state.
7. If chunks contain headings, instructions, or non-criterion text (e.g.
   "Subjects must meet ALL of the following:"), do NOT extract these as criteria —
   they are introductory, not requirements themselves.
"""


def build_prompt(
    section_label: str,
    section_path: str,
    section_text: str,
    enrichments: list[ChunkEnrichment],
    glossary: dict[str, str],
    acronyms: dict[str, str],
) -> str:
    """Build the extraction prompt with enrichment priors."""

    # Collect negation cues from enrichment
    neg_cues = []
    for ce in enrichments:
        for n in ce.linguistic.negation_annotations:
            neg_cues.append(f'  - "{n.cue_text}" (kind: {n.negation_type})')

    # Collect entity hints
    ent_lines = []
    for ce in enrichments:
        for e in ce.semantic.entities[:5]:
            ent_lines.append(f'  - {e.surface} ({e.entity_type}/{e.subtype})')

    # Acronyms used in this section
    section_text_lower = section_text.lower()
    relevant_acronyms = [
        (a, exp) for a, exp in acronyms.items()
        if a in section_text  # case-sensitive for acronyms
    ]
    relevant_glossary = [
        (t, defn) for t, defn in glossary.items()
        if t.lower() in section_text_lower
    ]

    prompt = f"""Section: {section_label}
Section path: {section_path}

Source text (concatenated chunks under this section):
\"\"\"
{section_text}
\"\"\"

Pre-tagged enrichment hints (use these as priors — don't re-infer):

Negation cues already detected in this section:
{chr(10).join(neg_cues) if neg_cues else "  (none detected)"}

Entity candidates (top spans):
{chr(10).join(ent_lines[:20]) if ent_lines else "  (none)"}

Acronyms used in this section (expansions for your reference; don't substitute in output):
{chr(10).join(f'  - {a} = {exp}' for a, exp in relevant_acronyms) if relevant_acronyms else "  (none)"}

Defined terms used in this section:
{chr(10).join(f'  - {t}: {d[:80]}...' for t, d in relevant_glossary[:6]) if relevant_glossary else "  (none)"}

Default category for this section: {"inclusion" if "inclusion" in section_label.lower() else ("exclusion" if "exclusion" in section_label.lower() else "inclusion")}
"""
    return prompt


# === Validation =====================================================

def validate_criterion(
    crit: ExtractedCriterion,
    source_chunks: list[Chunk],
) -> tuple[bool, Optional[Chunk], str]:
    """Check that crit.text is a verbatim substring of one of the source chunks.

    Returns (is_valid, matching_chunk, reason).
    """
    crit_text = crit.text.strip()
    if not crit_text:
        return False, None, "empty text"

    # Try exact substring match
    for chunk in source_chunks:
        if crit_text in chunk.text:
            return True, chunk, "exact match"

    # Try normalized match (collapse whitespace)
    import re as _re
    crit_norm = _re.sub(r"\s+", " ", crit_text)
    for chunk in source_chunks:
        chunk_norm = _re.sub(r"\s+", " ", chunk.text)
        if crit_norm in chunk_norm:
            return True, chunk, "whitespace-normalized match"

    # Fuzzy: first 50 chars of criterion in any chunk
    if len(crit_text) >= 50:
        prefix = crit_text[:50]
        for chunk in source_chunks:
            if prefix in chunk.text:
                return True, chunk, "prefix match (lossy)"

    return False, None, "no substring match"


# === USDM conversion =================================================

def make_code(decode: str, codeSystem: str = "FMLS-Local", code_value: Optional[str] = None) -> Code:
    """Construct a USDM Code object with sensible defaults for v1.

    For production, codes should reference CT (CDISC controlled terminology),
    SNOMED, ICD, etc. For now, we use a local code system for v1 — these get
    re-mapped to canonical codes during Stage 5 linking.
    """
    return Code(
        id=str(uuid.uuid4()),
        code=code_value or decode.upper().replace(" ", "_"),
        codeSystem=codeSystem,
        codeSystemVersion="0.1",
        decode=decode,
        instanceType="Code",
    )


def to_usdm_pair(
    crit: ExtractedCriterion,
    matching_chunk: Chunk,
    evidence_quote: str,
    provenance: Provenance,
) -> tuple[Extracted[EligibilityCriterionItem], Extracted[EligibilityCriterion]]:
    """Convert one extracted criterion to (Item, Criterion) USDM pair.

    Item holds the text (lives at StudyVersion level in USDM).
    Criterion holds the reference + category + ordering (lives at StudyDesign level).
    """
    item_id = f"crit_item_{uuid.uuid4().hex[:8]}"
    ref_id = f"crit_ref_{uuid.uuid4().hex[:8]}"

    item = EligibilityCriterionItem(
        id=item_id,
        name=f"EligibilityCriterionItem text",
        label=crit.identifier if crit.identifier else None,
        description=None,
        text=crit.text,
        dictionaryId=None,
        notes=[],
        extensionAttributes=[],
        instanceType="EligibilityCriterionItem",
    )

    ref = EligibilityCriterion(
        id=ref_id,
        name=f"EligibilityCriterion {crit.identifier or ref_id}",
        label=crit.identifier if crit.identifier else None,
        description=None,
        category=make_code(crit.category, codeSystem="FMLS-Category"),
        identifier=crit.identifier or "",
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


# === Main orchestrator ===============================================

def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="dataset/{stem}/ — e.g. AZ_demo, NCT04194944")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-completion-tokens", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts and skip the LLM call")
    args = ap.parse_args()

    doc_dir = DATASET / args.stem
    chunks_path = doc_dir / "chunks.json"
    enriched_path = doc_dir / "enriched.json"
    indexes_path = doc_dir / "doc_indexes.json"

    for p in [chunks_path, enriched_path, indexes_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run Stage 1/3 first.", file=sys.stderr)
            return 2

    print(f"=== Loading {args.stem} ===")
    doc = ChunkedDocument.model_validate(json.loads(chunks_path.read_text(encoding="utf-8")))
    enriched = DocumentEnrichment.model_validate(json.loads(enriched_path.read_text(encoding="utf-8")))
    indexes = DocIndexes.model_validate(json.loads(indexes_path.read_text(encoding="utf-8")))
    print(f"  chunks={doc.total_chunks}, enriched={len(enriched.chunks)}, "
          f"acronyms={len(indexes.acronyms)}, glossary={len(indexes.glossary)}")

    # === Filter chunks: those targeting EligibilityCriterion ===
    chunks_by_id = {c.chunk_id: c for c in doc.chunks}
    eligibility_chunks: list[Chunk] = []
    for cid, ce in enriched.chunks.items():
        if "EligibilityCriterion" in ce.domain.schema_class_hints:
            if cid in chunks_by_id:
                eligibility_chunks.append(chunks_by_id[cid])
    print(f"  eligibility chunks: {len(eligibility_chunks)}")

    if not eligibility_chunks:
        print("  No eligibility chunks found. Aborting.")
        return 1

    # === Group by section_path leaf (inclusion vs exclusion) ===
    groups: dict[str, list[Chunk]] = {}
    for c in eligibility_chunks:
        # Use the deepest "Inclusion" or "Exclusion" heading as the grouping key
        leaf = next(
            (h for h in reversed(c.section_path)
             if "inclusion" in h.lower() or "exclusion" in h.lower() or "withdrawal" in h.lower()),
            c.section_path[-1] if c.section_path else "unknown",
        )
        groups.setdefault(leaf, []).append(c)

    print(f"  groups: {list(groups.keys())}")
    for k, v in groups.items():
        print(f"    {k}: {len(v)} chunks")

    # === Prepare LLM client ===
    client = instructor.from_openai(OpenAI()) if not args.dry_run else None

    # === Lookups ===
    glossary = {g.surface: g.definition for g in indexes.glossary}
    acronyms = {a.surface: a.expansion for a in indexes.acronyms}

    # === Extract per group ===
    all_extracted_items: list[Extracted[EligibilityCriterionItem]] = []
    all_extracted_refs: list[Extracted[EligibilityCriterion]] = []
    total_dropped = 0

    for section_label, chunks in groups.items():
        print()
        print(f"=== Extracting from group: {section_label} ({len(chunks)} chunks) ===")

        # Concatenate chunk texts
        chunks_sorted = sorted(chunks, key=lambda c: (c.page_num, c.source_block_indices[0] if c.source_block_indices else 0))
        section_text = "\n\n".join(c.text for c in chunks_sorted)

        # Gather enrichments for these chunks
        chunk_enrichments = [enriched.chunks[c.chunk_id] for c in chunks_sorted if c.chunk_id in enriched.chunks]

        section_path = " > ".join(chunks_sorted[0].section_path) if chunks_sorted[0].section_path else section_label

        prompt = build_prompt(
            section_label=section_label,
            section_path=section_path,
            section_text=section_text,
            enrichments=chunk_enrichments,
            glossary=glossary,
            acronyms=acronyms,
        )
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        if args.dry_run:
            print(f"  [DRY RUN] prompt length: {len(prompt)} chars / ~{len(prompt)//4} tokens")
            print(f"  prompt_hash: {prompt_hash}")
            print(f"  First 800 chars of prompt:")
            print(prompt[:800])
            print("  ...")
            continue

        # Call LLM
        try:
            response: CriterionExtractionResponse = client.chat.completions.create(
                model=args.model,
                response_model=CriterionExtractionResponse,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=args.max_completion_tokens,
            )
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            continue

        print(f"  LLM returned {len(response.criteria)} criteria")
        if response.notes:
            print(f"  notes: {response.notes}")

        # === Validate + convert ===
        now = datetime.now(timezone.utc)
        for crit in response.criteria:
            is_valid, matching_chunk, reason = validate_criterion(crit, chunks_sorted)
            if not is_valid:
                total_dropped += 1
                print(f"    [DROP] '{crit.text[:80]}...' ({reason})")
                continue

            provenance = Provenance(
                evidence_chunk_id=matching_chunk.chunk_id,
                evidence_quote=crit.text,
                evidence_page=matching_chunk.page_num,
                evidence_bbox=(
                    (matching_chunk.bbox.x0, matching_chunk.bbox.y0,
                     matching_chunk.bbox.x1, matching_chunk.bbox.y1)
                    if matching_chunk.bbox else None
                ),
                extractor=args.model,
                extractor_version=args.model,
                prompt_hash=prompt_hash,
                extraction_confidence=0.9,
                extracted_at=now,
            )

            item_wrapped, ref_wrapped = to_usdm_pair(crit, matching_chunk, crit.text, provenance)
            all_extracted_items.append(item_wrapped)
            all_extracted_refs.append(ref_wrapped)

        print(f"  -> validated: {len(all_extracted_refs)} (cumulative); dropped: {total_dropped}")

    if args.dry_run:
        return 0

    # === Write output ===
    out_dir = doc_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eligibility.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for item_w in all_extracted_items:
            f.write(item_w.model_dump_json() + "\n")
        for ref_w in all_extracted_refs:
            f.write(ref_w.model_dump_json() + "\n")

    print()
    print(f"=== Done ===")
    print(f"  EligibilityCriterionItem extracted:  {len(all_extracted_items)}")
    print(f"  EligibilityCriterion extracted:      {len(all_extracted_refs)}")
    print(f"  Dropped (failed validation):         {total_dropped}")
    print(f"  Output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
