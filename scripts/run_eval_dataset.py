"""Run the parser pipeline against every PDF in `dataset/` and save the
DocumentResult next to its source as `parsed.json`.

Reads `dataset/manifest.json`, runs pipeline sequentially (workers=1 — MinerU
serializes on GPU anyway), saves per-doc results. Skips docs that already
have a parsed.json.

Usage:
    .venv/Scripts/python.exe scripts/run_eval_dataset.py [--remote http://localhost:8000] [--force]
"""

from __future__ import annotations

import argparse
import functools
import builtins
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Unbuffered prints so progress streams immediately under nohup / pipes.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

from fmls_parser import parse_document
from fmls_parser.models import ParseStatus

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def run_one(nct: str, remote_url: str | None, force: bool = False) -> dict:
    pdf = DATASET / nct / f"{nct}.pdf"
    out = DATASET / nct / "parsed.json"
    if not pdf.exists():
        return {"nct": nct, "skipped": True, "reason": "no PDF"}
    if out.exists() and not force:
        return {"nct": nct, "skipped": True, "reason": "already parsed"}
    print(f"  START {nct} ({pdf.stat().st_size // 1024} KB)")
    t0 = time.perf_counter()
    try:
        last_phase = [""]
        def _progress(idx, total, msg):
            phase = msg.split()[0] if msg else ""
            if phase != last_phase[0]:
                print(f"    {nct}: {msg} ({idx + 1}/{total})")
                last_phase[0] = phase
        res = parse_document(pdf_path=str(pdf), remote_url=remote_url, progress=_progress)
        out.write_text(json.dumps(res.model_dump(mode="json")), encoding="utf-8")
        n_blocks = sum(len(p.blocks) for p in res.pages)
        n_err = sum(1 for p in res.pages if p.parse_status == ParseStatus.ERROR)
        n_head = sum(1 for p in res.pages for b in p.blocks if b.block_type.value == "heading")
        n_par = sum(1 for p in res.pages for b in p.blocks if (b.metadata or {}).get("parent_block_id") is not None)
        return {
            "nct": nct,
            "ok": True,
            "pages": res.total_pages,
            "blocks": n_blocks,
            "headings": n_head,
            "blocks_with_parent": n_par,
            "errors": n_err,
            "routes": res.route_distribution(),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    except Exception as e:
        return {
            "nct": nct,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "wall_s": round(time.perf_counter() - t0, 1),
            "tb": traceback.format_exc()[-500:],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=os.getenv("FMLS_REMOTE_URL", ""))
    ap.add_argument("--force", action="store_true", help="re-parse even if parsed.json exists")
    args = ap.parse_args()

    manifest_path = DATASET / "manifest.json"
    if not manifest_path.exists():
        print("no dataset/manifest.json — run scripts/build_eval_dataset.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())
    if not args.remote:
        print("WARN: FMLS_REMOTE_URL not set — pipeline will be local-only", file=sys.stderr)

    print(f"dataset: {len(manifest)} protocols, remote: {args.remote or '(local-only)'}")
    print()

    overall_t0 = time.perf_counter()
    summaries: list[dict] = []
    for i, entry in enumerate(manifest, start=1):
        nct = entry["nct_id"]
        r = run_one(nct, args.remote or None, args.force)
        summaries.append(r)
        if r.get("skipped"):
            print(f"  [{i:>2}/{len(manifest)}] {nct}: SKIP ({r['reason']})")
        elif r.get("ok"):
            print(
                f"  [{i:>2}/{len(manifest)}] {nct}: "
                f"{r['pages']}p {r['blocks']}b heads={r['headings']} par={r['blocks_with_parent']} err={r['errors']} "
                f"{r['wall_s']}s  routes={r['routes']}"
            )
        else:
            print(f"  [{i:>2}/{len(manifest)}] {nct}: FAIL {r['error']}")

    overall_wall = time.perf_counter() - overall_t0
    summary_path = DATASET / "run_summary.json"
    summary_path.write_text(json.dumps({
        "remote": args.remote or None,
        "total_wall_s": round(overall_wall, 1),
        "total_docs": len(manifest),
        "results": summaries,
    }, indent=2), encoding="utf-8")
    print()
    print(f"total wall: {overall_wall:.1f}s ({overall_wall/60:.1f} min)")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
