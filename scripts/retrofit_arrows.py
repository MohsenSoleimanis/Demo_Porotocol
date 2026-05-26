"""Re-process saved parser results to attach arrow annotations.

Reads each corpus/results/*.json, opens the source PDF, runs arrow detection,
and re-saves with arrows attached to the relevant table/figure block metadata.
This is purely local — no Lightning AI / remote calls.

Usage:
    .venv/Scripts/python.exe scripts/retrofit_arrows.py [--stem AZ_demo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

from fmls_parser.arrows import detect_arrows_on_page

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = CORPUS_DIR / "results"


def _find_pdf_for_stem(stem: str, payload: dict) -> Path | None:
    candidates = []
    src_fn = payload.get("source_filename")
    if src_fn:
        candidates.append(CORPUS_DIR / src_fn)
        candidates.append(ROOT / src_fn)
    src_path = payload.get("source_path")
    if src_path:
        candidates.append(Path(src_path))
    candidates.append(CORPUS_DIR / f"{stem}.pdf")
    candidates.append(ROOT / f"{stem}.pdf")
    for c in candidates:
        if c.exists():
            return c
    return None


def _arrow_to_dict(a) -> dict:
    return {
        "start": [round(a.start[0], 2), round(a.start[1], 2)],
        "end": [round(a.end[0], 2), round(a.end[1], 2)],
        "axis": a.axis,
        "two_headed": a.two_headed,
        "span_pt": round(
            abs(a.end[0] - a.start[0]) if a.axis == "horizontal" else abs(a.end[1] - a.start[1]),
            2,
        ),
    }


def _bbox_contains(bbox: dict, px: float, py: float) -> bool:
    return bbox is not None and bbox["x0"] <= px <= bbox["x1"] and bbox["y0"] <= py <= bbox["y1"]


def retrofit_one(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pdf_path = _find_pdf_for_stem(json_path.stem, payload)
    if pdf_path is None:
        return {"stem": json_path.stem, "skipped": True, "reason": "no source PDF"}
    doc = fitz.open(str(pdf_path))
    n_arrows_total = 0
    n_pages_with_arrows = 0
    try:
        for page in payload.get("pages") or []:
            pn = page.get("page_num")
            if pn is None or pn >= doc.page_count:
                continue
            arrows = detect_arrows_on_page(doc[pn])
            if not arrows:
                continue
            blocks = page.get("blocks") or []
            tables = [b for b in blocks if b.get("block_type") == "table" and b.get("bbox")]
            figures = [b for b in blocks if b.get("block_type") == "figure" and b.get("bbox")]
            page_n = 0
            for ar in arrows:
                mx = (ar.start[0] + ar.end[0]) / 2
                my = (ar.start[1] + ar.end[1]) / 2
                target = None
                for b in tables:
                    if _bbox_contains(b["bbox"], mx, my):
                        target = b
                        break
                if target is None:
                    for b in figures:
                        if _bbox_contains(b["bbox"], mx, my):
                            target = b
                            break
                rec = _arrow_to_dict(ar)
                if target is None:
                    if not blocks:
                        continue
                    target = blocks[0]
                    meta = target.setdefault("metadata", {}) or {}
                    target["metadata"] = meta
                    meta.setdefault("page_level_arrows", []).append(rec)
                else:
                    meta = target.setdefault("metadata", {}) or {}
                    target["metadata"] = meta
                    meta.setdefault("arrows", []).append(rec)
                    meta["has_arrows"] = True
                page_n += 1
            if page_n:
                n_pages_with_arrows += 1
                n_arrows_total += page_n
    finally:
        doc.close()
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "stem": json_path.stem,
        "arrows": n_arrows_total,
        "pages_with_arrows": n_pages_with_arrows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", help="only process this stem (otherwise all results)")
    args = ap.parse_args()
    paths = sorted(RESULTS_DIR.glob("*.json"))
    paths = [p for p in paths if p.stem != "summary"]
    if args.stem:
        paths = [p for p in paths if p.stem == args.stem]
    for p in paths:
        r = retrofit_one(p)
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
