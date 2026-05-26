"""Per-parser timing benchmark.

Picks a small set of pages from a PDF (default: first PDF in project root)
and runs each parser independently on each, recording wall time and number
of extracted blocks. Prints a comparison table.

This is a TIMING measurement, not an accuracy measurement. The sample PDF is
ONE document out of a much larger corpus — these numbers describe latency
on this hardware and this document, not parser quality.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_parsers.py [pdf_path] [--pages 0,1,5,10] [--include qwen,docling]
    .venv/Scripts/python.exe scripts/benchmark_parsers.py --pages 0,1,2,3,4 --remote http://localhost:8000

The first Docling/Qwen call pays a model-load tax. We report both the first
call and the steady-state (subsequent calls) so the warmup cost is visible.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import fitz

from fmls_parser.parsers import PdfPlumberParser, PyMuPDFParser
from fmls_parser.remote.client import RemoteClient, RemoteUnavailable


def parse_pages_arg(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", help="path to PDF (default: first PDF in project root)")
    ap.add_argument("--pages", default="0,1,2", help="comma-separated 0-indexed page numbers (default: 0,1,2)")
    ap.add_argument("--remote", default=os.getenv("FMLS_REMOTE_URL", ""), help="remote server base URL (e.g. http://localhost:8000)")
    ap.add_argument("--include", default="pymupdf,pdfplumber,docling,qwen", help="comma-separated parsers to run")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.pdf:
        pdf = Path(args.pdf)
    else:
        pdfs = sorted(root.glob("*.pdf"))
        if not pdfs:
            print("no PDF found; pass a path")
            return 2
        pdf = pdfs[0]
    pdf = pdf.resolve()

    pages = parse_pages_arg(args.pages)
    parsers_requested = set(p.strip() for p in args.include.split(",") if p.strip())

    print(f"pdf:      {pdf}")
    print(f"pages:    {pages}")
    print(f"parsers:  {sorted(parsers_requested)}")
    print(f"remote:   {args.remote or '(disabled)'}")
    print()

    pdf_bytes = pdf.read_bytes()
    timings: dict[str, list[dict]] = {}

    def _record(parser_name: str, page: int, t: float, n_blocks: int, error: str | None = None):
        timings.setdefault(parser_name, []).append(
            {"page": page, "ms": t * 1000.0, "blocks": n_blocks, "error": error}
        )

    # ---- local: pymupdf ----
    if "pymupdf" in parsers_requested:
        parser = PyMuPDFParser()
        for p in pages:
            t0 = time.perf_counter()
            try:
                blocks = parser.parse_page(str(pdf), p)
                _record("pymupdf", p, time.perf_counter() - t0, len(blocks))
            except Exception as e:
                _record("pymupdf", p, time.perf_counter() - t0, 0, error=str(e))

    # ---- local: pdfplumber ----
    if "pdfplumber" in parsers_requested:
        parser = PdfPlumberParser()
        for p in pages:
            t0 = time.perf_counter()
            try:
                blocks = parser.parse_page(str(pdf), p)
                _record("pdfplumber", p, time.perf_counter() - t0, len(blocks))
            except Exception as e:
                _record("pdfplumber", p, time.perf_counter() - t0, 0, error=str(e))

    # ---- remote: docling + qwen ----
    if args.remote and ("docling" in parsers_requested or "qwen" in parsers_requested):
        client = RemoteClient(base_url=args.remote, timeout=600)
        try:
            if "docling" in parsers_requested:
                for p in pages:
                    t0 = time.perf_counter()
                    try:
                        resp = client.parse_with_docling(pdf_bytes, p, filename=pdf.name)
                        _record("docling", p, time.perf_counter() - t0, len(resp.blocks))
                    except Exception as e:
                        _record("docling", p, time.perf_counter() - t0, 0, error=str(e))
            if "qwen" in parsers_requested:
                # render pages locally first
                doc = fitz.open(str(pdf))
                page_pngs: dict[int, bytes] = {}
                for p in pages:
                    pix = doc[p].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
                    page_pngs[p] = pix.tobytes("png")
                doc.close()
                for p in pages:
                    t0 = time.perf_counter()
                    try:
                        resp = client.parse_with_qwen_vl(page_pngs[p], p)
                        _record("qwen_vl", p, time.perf_counter() - t0, len(resp.blocks))
                    except Exception as e:
                        _record("qwen_vl", p, time.perf_counter() - t0, 0, error=str(e))
        finally:
            client.close()
    elif "docling" in parsers_requested or "qwen" in parsers_requested:
        print("(skipping remote parsers — --remote not set)\n")

    # ---- report ----
    headers = ["parser", "page", "blocks", "ms"]
    print(f"{'parser':<12} {'page':>4} {'blocks':>7} {'ms':>10}  notes")
    print("-" * 60)
    for parser_name, runs in timings.items():
        for i, r in enumerate(runs):
            note = ""
            if r["error"]:
                note = f"ERROR: {r['error'][:60]}"
            elif i == 0 and parser_name in ("docling", "qwen_vl"):
                note = "(cold start - includes model load)"
            print(f"{parser_name:<12} {r['page']:>4} {r['blocks']:>7} {r['ms']:>10.1f}  {note}")

    print()
    print("steady-state averages (excluding first call):")
    print(f"{'parser':<12} {'avg_ms':>10} {'avg_blocks':>11} {'n':>4}")
    for parser_name, runs in timings.items():
        steady = runs[1:] if parser_name in ("docling", "qwen_vl") and len(runs) > 1 else runs
        ok_runs = [r for r in steady if r["error"] is None]
        if not ok_runs:
            print(f"{parser_name:<12} {'-':>10} {'-':>11} {len(steady):>4}")
            continue
        avg_ms = sum(r["ms"] for r in ok_runs) / len(ok_runs)
        avg_blocks = sum(r["blocks"] for r in ok_runs) / len(ok_runs)
        print(f"{parser_name:<12} {avg_ms:>10.1f} {avg_blocks:>11.1f} {len(ok_runs):>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
