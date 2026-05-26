"""Remove PDFs from the corpus that look censored / redacted.

Heuristic: if more than `--threshold` fraction of pages have no extractable
text layer, the PDF is treated as redacted (every page is a black-bar image)
and removed from disk + manifest.

Usage:
    .venv/Scripts/python.exe scripts/prune_corpus.py [--threshold 0.3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz


def censored_fraction(path: Path) -> float:
    try:
        doc = fitz.open(str(path))
    except Exception:
        return 1.0
    try:
        empty = 0
        n = doc.page_count
        for page in doc:
            txt = page.get_text("text") or ""
            if len(txt.strip()) < 20:
                empty += 1
        return empty / max(n, 1)
    finally:
        doc.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="if > this fraction of pages have no text layer, delete the PDF")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus_dir = Path(args.out).resolve()
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print("no manifest")
        return 2
    manifest = json.loads(manifest_path.read_text())

    keep: list[dict] = []
    dropped: list[tuple[dict, float]] = []
    for entry in manifest:
        p = corpus_dir / entry["filename"]
        if not p.exists():
            dropped.append((entry, 1.0))  # already gone
            continue
        frac = censored_fraction(p)
        if frac > args.threshold:
            dropped.append((entry, frac))
            if not args.dry_run:
                p.unlink(missing_ok=True)
        else:
            entry["censored_fraction"] = round(frac, 3)
            keep.append(entry)

    print(f"kept   : {len(keep)}")
    print(f"dropped: {len(dropped)} (threshold {args.threshold:.0%} pages with no text)")
    for e, frac in dropped:
        print(f"  - {e['nct']} ({e['stratum']}): {frac:.1%} pages no-text -> {e['filename']}")
    if not args.dry_run:
        manifest_path.write_text(json.dumps(keep, indent=2))
        print(f"\nmanifest updated: {manifest_path}")
    else:
        print("\n(dry run — nothing deleted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
