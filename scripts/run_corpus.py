"""Run the parser pipeline across the entire corpus, concurrently.

Concurrency strategy:
  - ThreadPoolExecutor on the CLIENT side runs N pipelines in parallel.
  - The Docling server holds the GPU, so Docling calls effectively serialize
    server-side. But while one doc is on the GPU, others are doing local
    triage + PyMuPDF parsing, so we still get pipeline overlap.
  - Max workers = 4 by default. More just adds queueing without throughput.

Outputs:
  corpus/results/{nct}.json    DocumentResult per PDF (full provenance)
  corpus/results/summary.json  aggregate timing + routing stats

Usage:
    .venv/Scripts/python.exe scripts/run_corpus.py [--workers 4] [--remote http://localhost:8000]
"""

from __future__ import annotations

import argparse
import functools
import builtins

# Flush stdout on every print so progress lines stream out immediately even
# when run in background / piped through PowerShell.
print = functools.partial(builtins.print, flush=True)  # noqa: A001
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fmls_parser import parse_document
from fmls_parser.models import ParseStatus


def run_one(pdf_path: Path, remote_url: str | None, results_dir: Path) -> dict:
    out_path = results_dir / f"{pdf_path.stem}.json"
    if out_path.exists():
        return {"pdf": pdf_path.name, "skipped": True, "reason": "already parsed"}
    print(f"  START   {pdf_path.name}")
    t0 = time.perf_counter()
    try:
        last_phase = [""]
        def _progress(idx, total, msg):
            phase = msg.split()[0] if msg else ""
            if phase != last_phase[0]:
                print(f"  {pdf_path.name} -> {msg} (page {idx+1}/{total})")
                last_phase[0] = phase
        res = parse_document(pdf_path=str(pdf_path), remote_url=remote_url, progress=_progress)
        out_path.write_text(json.dumps(res.model_dump(mode="json")))
        n_blocks = sum(len(p.blocks) for p in res.pages)
        n_err = sum(1 for p in res.pages if p.parse_status == ParseStatus.ERROR)
        return {
            "pdf": pdf_path.name,
            "ok": True,
            "pages": res.total_pages,
            "blocks": n_blocks,
            "errors": n_err,
            "routes": res.route_distribution(),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    except Exception as e:
        return {
            "pdf": pdf_path.name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "wall_s": round(time.perf_counter() - t0, 1),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="default 1 — the GPU on the remote server is the bottleneck; extra workers just queue and slow /health")
    ap.add_argument("--remote", default=os.getenv("FMLS_REMOTE_URL", ""))
    ap.add_argument("--corpus", default="corpus")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus).resolve()
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print("no manifest — run scripts/fetch_corpus.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())

    results_dir = corpus_dir / "results"
    results_dir.mkdir(exist_ok=True)

    pdfs = [corpus_dir / m["filename"] for m in manifest if (corpus_dir / m["filename"]).exists()]
    # Sort smallest first so the user sees quick wins early.
    pdfs.sort(key=lambda p: p.stat().st_size)
    print(f"corpus: {len(pdfs)} PDFs, workers: {args.workers}, remote: {args.remote or '(local-only)'}")
    print(f"order:  smallest first ({pdfs[0].stat().st_size//1024}KB) -> largest ({pdfs[-1].stat().st_size//1024}KB)")
    print()

    overall_t0 = time.perf_counter()
    summaries: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, p, args.remote or None, results_dir): p for p in pdfs}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            r = fut.result()
            summaries.append(r)
            if r.get("skipped"):
                print(f"  [{done_count:>2}/{len(pdfs)}] {r['pdf']:<40} SKIP ({r['reason']})")
            elif r.get("ok"):
                print(
                    f"  [{done_count:>2}/{len(pdfs)}] {r['pdf']:<40} "
                    f"{r['pages']:>4}p {r['blocks']:>5}b err={r['errors']:>2} "
                    f"{r['wall_s']:>5.0f}s  routes={r['routes']}"
                )
            else:
                print(f"  [{done_count:>2}/{len(pdfs)}] {r['pdf']:<40} FAIL  {r['error']}")

    overall_wall = time.perf_counter() - overall_t0
    summary_path = results_dir / "summary.json"
    summary_payload = {
        "workers": args.workers,
        "remote": args.remote or None,
        "total_wall_s": round(overall_wall, 1),
        "total_docs": len(pdfs),
        "results": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print()
    print(f"total wall time: {overall_wall:.1f}s ({overall_wall/60:.1f} min)")
    print(f"summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
