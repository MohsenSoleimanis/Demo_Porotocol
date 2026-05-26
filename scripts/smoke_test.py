"""Smoke test — runs the pipeline against any PDF in the project dir and
prints a triage + extraction summary. Does NOT assert accuracy: the sample
PDF is one example out of a 50k+ corpus and must not be used to tune the
pipeline.

Usage:
    .venv/Scripts/python.exe scripts/smoke_test.py [path/to/pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from fmls_parser import parse_document
from fmls_parser.models import ParseStatus


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent.parent
    if len(argv) >= 2:
        pdf = Path(argv[1])
    else:
        pdfs = sorted(here.glob("*.pdf"))
        if not pdfs:
            print("no PDF found in project root; pass a path explicitly")
            return 2
        pdf = pdfs[0]
    print(f"running pipeline on: {pdf}")

    result = parse_document(pdf_path=str(pdf))

    print()
    print(f"total pages       : {result.total_pages}")
    print(f"total duration ms : {result.total_duration_ms:.0f}")
    print(f"remote configured : {result.remote_configured}")
    print(f"route counts      : {result.route_distribution()}")

    status_counts: dict[str, int] = {}
    for p in result.pages:
        status_counts[p.parse_status.value] = status_counts.get(p.parse_status.value, 0) + 1
    print(f"status counts     : {status_counts}")

    print("\nfirst 5 pages:")
    for p in result.pages[:5]:
        print(
            f"  page {p.page_num:>3} | route={p.parser_used.value:<11} "
            f"| status={p.parse_status.value:<8} | blocks={len(p.blocks):>4} "
            f"| {p.parse_duration_ms:>6.0f} ms"
        )
        print(f"           reason: {p.triage.reason}")
        if p.blocks:
            sample = p.blocks[0].text.replace("\n", " ")[:120]
            print(f"           sample: {sample!r}")

    n_errors = sum(1 for p in result.pages if p.parse_status == ParseStatus.ERROR)
    if n_errors:
        print(f"\nWARNING: {n_errors} page(s) errored. Inspect via the Streamlit UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
