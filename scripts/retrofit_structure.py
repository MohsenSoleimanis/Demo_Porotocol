"""Apply structural enrichment to saved parser results.

Reads each corpus/results/*.json and re-saves with indent_level,
parent_block_id, section_path, list_marker, references_to attached.

Operates only on what's already in the JSON — no PDF re-render, no remote.

Usage:
    .venv/Scripts/python.exe scripts/retrofit_structure.py [--stem AZ_demo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fmls_parser.models import DocumentResult
from fmls_parser.structure import enrich_blocks_structure

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "corpus" / "results"


def retrofit_one(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # Validate into model objects so the enricher gets typed BlockType etc.
    res = DocumentResult.model_validate(payload)
    n = enrich_blocks_structure(res.pages)
    # Save back as JSON.
    json_path.write_text(json.dumps(res.model_dump(mode="json")), encoding="utf-8")
    # Quick summary
    parent_count = sum(
        1 for p in res.pages for b in p.blocks
        if (b.metadata or {}).get("parent_block_id") is not None
    )
    return {"stem": json_path.stem, "blocks_enriched": n, "blocks_with_parent": parent_count}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", help="only this stem (else all)")
    args = ap.parse_args()
    paths = sorted(p for p in RESULTS_DIR.glob("*.json") if p.stem != "summary")
    if args.stem:
        paths = [p for p in paths if p.stem == args.stem]
    for p in paths:
        print(retrofit_one(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
