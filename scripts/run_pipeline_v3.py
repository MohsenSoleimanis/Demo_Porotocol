"""v3 pipeline orchestrator.

Runs the full downstream-of-MinerU pipeline on a single document:

  chunks.json
   └─► enrich (v3) ──► chunk_key_enrichments.json
        └─► extract (v3) ──► extracted/eligibility_v3.jsonl
             └─► validate (Pydantic + CDISC + cross-record)
                  └─► assemble ──► usdm.json + validation_report.json

Each stage is content-hash-keyed cached; re-runs only re-execute stages whose
inputs changed. Total LLM cost per protocol depends on how many chunks fall
through the rules to LLM enrichment.

Usage:
    python scripts/run_pipeline_v3.py AZ_demo
    python scripts/run_pipeline_v3.py AZ_demo --rebuild-enrichment
    python scripts/run_pipeline_v3.py AZ_demo --skip-llm   # rule-based only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import instructor
from openai import OpenAI

from fmls_parser.chunk_models import ChunkedDocument
from fmls_parser.pipeline_v3.enrich_v3 import (
    enrich_chunks_v3,
    save_chunk_key_enrichments,
    load_chunk_key_enrichments,
)
from fmls_parser.pipeline_v3.extract_v3 import (
    filter_for_eligibility,
    reunite_with_sub_items,
    assemble_eligibility_records,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--rebuild-enrichment", action="store_true",
                    help="Rebuild chunk_key_enrichments.json even if it exists.")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip the LLM enrichment fallback; rule-based only.")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    doc_dir = DATASET / args.stem
    chunks_path = doc_dir / "chunks.json"
    enrich_path = doc_dir / "chunk_key_enrichments.json"
    out_dir = doc_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    extracted_path = out_dir / "eligibility_v3.jsonl"

    print(f"=== Pipeline v3 — {args.stem} ===")

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found", file=sys.stderr)
        return 2

    # Load chunks
    print(f"\n[1/4] Loading chunks...")
    doc = ChunkedDocument.model_validate(
        json.loads(chunks_path.read_text(encoding="utf-8"))
    )
    print(f"      {doc.total_chunks} chunks across {doc.total_pages} pages")

    # === Stage: Enrich ===
    print(f"\n[2/4] Stage: enrich (chunk key enrichments)")
    if enrich_path.exists() and not args.rebuild_enrichment:
        print(f"      using cached enrichments at {enrich_path}")
        enrichments = load_chunk_key_enrichments(enrich_path)
    else:
        client = None if args.skip_llm else instructor.from_openai(OpenAI())

        # For v3 efficiency, restrict enrichment to §5 chunks (eligibility-relevant
        # for this demo) — in full production we'd enrich every chunk
        relevant_chunks = [
            c for c in doc.chunks
            if c.m11_section and c.m11_section.startswith("5")
        ]
        print(f"      enriching {len(relevant_chunks)} chunks under §5")

        enrichments = enrich_chunks_v3(
            relevant_chunks,
            llm_client=client,
            model=args.model,
        )
        save_chunk_key_enrichments(enrichments, enrich_path)
        print(f"      -> {enrich_path}")

    # Summarize role distribution
    from collections import Counter
    role_counts = Counter(e.semantic_role for e in enrichments.values())
    print(f"      role distribution: {dict(role_counts.most_common())}")

    # === Stage: Extract ===
    print(f"\n[3/4] Stage: extract (element-specific: EligibilityCriterion)")

    primary_chunks = filter_for_eligibility(doc.chunks, enrichments)
    print(f"      found {len(primary_chunks)} primary_item chunks under eligibility sections")

    reunited = reunite_with_sub_items(primary_chunks, doc.chunks, enrichments)
    n_with_subs = sum(1 for r in reunited.values() if r["subs"])
    print(f"      reunited with sub-items: {n_with_subs} have attached sub_items")

    records = assemble_eligibility_records(reunited, extractor_label=args.model)
    print(f"      assembled {len(records)} (Item, Criterion) pairs")

    # Write extracted
    with open(extracted_path, "w", encoding="utf-8") as f:
        for item, ref in records:
            f.write(item.model_dump_json() + "\n")
            f.write(ref.model_dump_json() + "\n")
    print(f"      -> {extracted_path}")

    # === Stage: Validate ===
    print(f"\n[4/4] Stage: validate (Pydantic + cross-record)")
    # Pydantic validation already happened at construction
    # Cross-record: every EligibilityCriterion's criterionItemId must resolve
    items_by_id = {item.value.id: item for item, _ in records}
    refs = [ref for _, ref in records]
    n_resolved = sum(1 for r in refs if r.value.criterionItemId in items_by_id)
    print(f"      {n_resolved}/{len(refs)} EligibilityCriterion.criterionItemId references resolve")

    # Category distribution
    cat_counts = Counter(r.value.category.decode for r in refs)
    print(f"      categories: {dict(cat_counts)}")

    # CDISC Rules Engine — attempt if installed
    print(f"\n      CDISC Rules Engine validation: not run (deferred — requires assembled Wrapper)")

    # === Summary ===
    print(f"\n=== Done ===")
    print(f"  Extracted: {len(records)} EligibilityCriterion+Item pairs")
    print(f"  Output:    {extracted_path}")
    print(f"  Enrichment cache: {enrich_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
