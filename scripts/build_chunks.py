"""Run the chunking layer over every parsed.json in dataset/.

Reads `dataset/{NCT}/parsed.json`, builds a `ChunkedDocument`, writes:
  - dataset/{NCT}/chunks.json        (full ChunkedDocument as one JSON)
  - dataset/{NCT}/chunks.jsonl       (one chunk per line, for streaming consumers)

Usage:
    .venv/Scripts/python.exe scripts/build_chunks.py [--stem NCT04194944]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fmls_parser.chunking import chunk_document
from fmls_parser.models import DocumentResult

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def chunk_one(parsed_path: Path) -> dict:
    payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    doc = DocumentResult.model_validate(payload)
    doc_id = parsed_path.parent.name  # NCT
    chunked = chunk_document(doc, doc_id=doc_id)
    # Full document
    (parsed_path.parent / "chunks.json").write_text(
        chunked.model_dump_json(indent=2), encoding="utf-8"
    )
    # Streaming JSONL
    with open(parsed_path.parent / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunked.chunks:
            f.write(c.model_dump_json() + "\n")

    # Summary stats
    from collections import Counter
    type_counts = Counter(c.block_type.value for c in chunked.chunks)
    sections = chunked.section_index
    m11_sections = [s.get("m11_section") for s in sections if s.get("m11_section")]
    n_with_parent = sum(1 for c in chunked.chunks if c.parent_chunk_id)
    n_with_refs = sum(1 for c in chunked.chunks if c.references)
    n_with_footnotes = sum(1 for c in chunked.chunks if c.resolved_footnotes)
    n_continuations = sum(1 for c in chunked.chunks if c.continuation_of)
    return {
        "doc_id": doc_id,
        "total_chunks": chunked.total_chunks,
        "total_pages": chunked.total_pages,
        "sections": len(sections),
        "m11_sections_recognized": sorted(set(m11_sections)),
        "type_counts": dict(type_counts),
        "n_with_parent": n_with_parent,
        "n_with_refs": n_with_refs,
        "n_with_footnotes": n_with_footnotes,
        "n_table_continuations": n_continuations,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", help="only this NCT (else all in dataset/)")
    args = ap.parse_args()
    parsed_paths = sorted(DATASET.glob("*/parsed.json"))
    if args.stem:
        parsed_paths = [p for p in parsed_paths if p.parent.name == args.stem]
    if not parsed_paths:
        print("no parsed.json found in dataset/")
        return 2
    for p in parsed_paths:
        try:
            summary = chunk_one(p)
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"FAIL {p.parent.name}: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
