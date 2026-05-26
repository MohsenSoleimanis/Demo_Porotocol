"""Generate a one-page profile of the downloaded corpus so we know what we have.

Reads corpus/manifest.json + the PDFs and prints:
  - per-doc: page count, raw image count, has text layer, est. # tables
  - aggregate: stratum distribution, phase distribution, page-count percentiles,
    pages-with-images count, pages-with-tables count

Use this BEFORE we decide eval-set composition.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import fitz
import pdfplumber


def profile_pdf(path: Path) -> dict:
    out = {"path": path.name, "pages": 0, "images": 0, "pages_with_imgs": 0,
           "no_text_pages": 0, "tables_est": 0, "errors": []}
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        out["errors"].append(f"open: {e}")
        return out
    out["pages"] = doc.page_count
    try:
        for i, page in enumerate(doc):
            imgs = page.get_images(full=True)
            out["images"] += len(imgs)
            if imgs:
                out["pages_with_imgs"] += 1
            txt = page.get_text("text") or ""
            if len(txt.strip()) < 20:
                out["no_text_pages"] += 1
    except Exception as e:
        out["errors"].append(f"fitz scan: {e}")
    finally:
        doc.close()
    # Tables (sample first 30 pages only — pdfplumber is slow per page)
    try:
        with pdfplumber.open(str(path)) as pp:
            sample = pp.pages[:30]
            count = 0
            for page in sample:
                try:
                    count += len(page.find_tables())
                except Exception:
                    pass
            # Extrapolate from sample to full doc
            scale = out["pages"] / max(len(sample), 1) if sample else 1.0
            out["tables_est"] = int(count * scale)
    except Exception as e:
        out["errors"].append(f"pdfplumber: {e}")
    return out


def main() -> int:
    corpus_dir = Path("corpus").resolve()
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print("no manifest — run scripts/fetch_corpus.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())

    rows: list[dict] = []
    print(f"profiling {len(manifest)} PDFs (this scans every page; can take ~1 min)...")
    for entry in manifest:
        p = corpus_dir / entry["filename"]
        if not p.exists():
            continue
        prof = profile_pdf(p)
        prof.update({
            "nct": entry["nct"],
            "stratum": entry["stratum"],
            "phase": entry["phase_actual"],
            "size_kb": entry["size_bytes"] // 1024,
        })
        rows.append(prof)

    rows.sort(key=lambda r: r["pages"])

    print()
    print(f"{'nct':<12} {'stratum':<10} {'phase':<12} {'pages':>5} {'imgs':>5} "
          f"{'with_img':>9} {'no_txt':>7} {'tables*':>8} {'size_kb':>8}")
    print("-" * 95)
    for r in rows:
        print(f"{r['nct']:<12} {r['stratum']:<10} {r['phase'][:12]:<12} {r['pages']:>5} "
              f"{r['images']:>5} {r['pages_with_imgs']:>9} {r['no_text_pages']:>7} "
              f"{r['tables_est']:>8} {r['size_kb']:>8}")

    page_counts = [r["pages"] for r in rows]
    img_total = sum(r["images"] for r in rows)
    no_text_total = sum(r["no_text_pages"] for r in rows)
    tables_total = sum(r["tables_est"] for r in rows)

    print()
    print("=== aggregate ===")
    print(f"  total docs:                {len(rows)}")
    if page_counts:
        print(f"  total pages:               {sum(page_counts)}")
        print(f"  page count min/med/max:    {min(page_counts)}/{int(statistics.median(page_counts))}/{max(page_counts)}")
    print(f"  total raw images:          {img_total}")
    print(f"  pages with images:         {sum(r['pages_with_imgs'] for r in rows)}")
    print(f"  pages with no text layer:  {no_text_total}  (these likely need VLM)")
    print(f"  total tables estimated:    {tables_total}*  (*pdfplumber's ruled-line detector, first 30 pages extrapolated)")

    strata = sorted({r["stratum"] for r in rows})
    print(f"  strata represented:        {strata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
