"""CLI: run Stage 1 indexes + Stage 3 enrichment on a single document.

Usage:
    python scripts/enrich.py AZ_demo
    python scripts/enrich.py AZ_demo --section 5.1     # restrict to chunks under m11 §5.1
    python scripts/enrich.py AZ_demo --only-extractable # only chunks with schema_class_hints

Reads:
    dataset/{stem}/chunks.json

Writes:
    dataset/{stem}/doc_indexes.json   (Stage 1)
    dataset/{stem}/enriched.json      (Stage 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import instructor
from openai import OpenAI

from fmls_parser.chunk_models import ChunkedDocument
from fmls_parser.enrichment.domain_clinical import lookup_functional_label, usdm_class_hints
from fmls_parser.enrichment.enrich import enrich_document
from fmls_parser.enrichment.indexes import build_doc_indexes, DocIndexes


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def make_filter(args) -> "callable | None":
    """Build a chunk filter based on CLI args."""

    def has_section(chunk):
        if not args.section:
            return True
        m11 = chunk.m11_section or ""
        return m11 == args.section or m11.startswith(args.section + ".")

    def is_extractable(chunk):
        if not args.only_extractable:
            return True
        label = lookup_functional_label(chunk.m11_section)
        return bool(usdm_class_hints(label))

    if not args.section and not args.only_extractable:
        return None

    return lambda c: has_section(c) and is_extractable(c)


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Enrich a clinical protocol's chunks.")
    ap.add_argument("stem", help="dataset directory name, e.g. AZ_demo or NCT04194944")
    ap.add_argument("--section", help="restrict to chunks under this m11 section, e.g. '5.1'")
    ap.add_argument("--only-extractable", action="store_true",
                    help="only enrich chunks whose role maps to a USDM target class")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="LLM model for the 2 doc-level LLM calls in Stage 1")
    ap.add_argument("--rebuild-indexes", action="store_true",
                    help="rebuild Stage 1 indexes even if doc_indexes.json exists")
    args = ap.parse_args()

    doc_dir = DATASET / args.stem
    chunks_path = doc_dir / "chunks.json"
    indexes_path = doc_dir / "doc_indexes.json"
    enriched_path = doc_dir / "enriched.json"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} does not exist. Run the chunker first.", file=sys.stderr)
        return 2

    # Load chunks
    print(f"=== Loading chunks for {args.stem} ===")
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    doc = ChunkedDocument.model_validate(data)
    print(f"  loaded {doc.total_chunks} chunks across {doc.total_pages} pages")

    # Stage 1: indexes (cache to disk)
    client = instructor.from_openai(OpenAI())

    if indexes_path.exists() and not args.rebuild_indexes:
        print(f"=== Stage 1: Using cached doc indexes ===")
        indexes = DocIndexes.model_validate(json.loads(indexes_path.read_text(encoding="utf-8")))
        print(f"  xrefs={len(indexes.cross_references)} acronyms={len(indexes.acronyms)} "
              f"glossary={len(indexes.glossary)} metadata.sponsor={indexes.metadata.sponsor}")
    else:
        print(f"=== Stage 1: Building doc indexes ===")
        indexes = build_doc_indexes(doc, llm_client=client, model=args.model)
        indexes_path.write_text(indexes.model_dump_json(indent=2), encoding="utf-8")
        print(f"  -> {indexes_path}")

    # Stage 3: enrichment
    print(f"=== Stage 3: Chunk enrichment ===")
    chunk_filter = make_filter(args)
    enriched = enrich_document(doc, indexes, chunk_filter=chunk_filter)

    # Write
    enriched_path.write_text(enriched.model_dump_json(indent=2), encoding="utf-8")
    print(f"  -> {enriched_path}")

    # Summary
    print()
    print(f"=== Summary ===")
    print(f"  chunks enriched:     {len(enriched.chunks)}")
    print(f"  total annotations:   {enriched.total_annotations}")
    print(f"  producers used:      {enriched.producers_used}")

    # Per-chunk-type summary
    from collections import Counter
    types = Counter()
    has_entities = 0
    has_negations = 0
    has_acronyms = 0
    has_xrefs = 0
    for ce in enriched.chunks.values():
        types[ce.structural.chunk_type] += 1
        if ce.semantic.entities:
            has_entities += 1
        if ce.linguistic.negation_annotations:
            has_negations += 1
        if ce.linguistic.acronym_uses:
            has_acronyms += 1
        if ce.structural.internal_references:
            has_xrefs += 1
    print(f"  chunk types:         {dict(types.most_common())}")
    print(f"  with GLiNER ents:    {has_entities}")
    print(f"  with negation tags:  {has_negations}")
    print(f"  with acronym uses:   {has_acronyms}")
    print(f"  with xref resolved:  {has_xrefs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
