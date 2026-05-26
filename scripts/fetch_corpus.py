"""Download a diverse set of real clinical-trial protocol PDFs from ClinicalTrials.gov.

Diversity strategy: stratify across therapeutic area and phase so the corpus
isn't dominated by one sponsor/template. We pull from the public ClinicalTrials.gov
v2 API, find studies with a posted Protocol document, and download via the
ProvidedDocs CDN URL.

Usage:
    .venv/Scripts/python.exe scripts/fetch_corpus.py
        [--target 25]
        [--out corpus]
        [--max-mb 30]            # skip PDFs larger than this
        [--min-pages 30]         # skip PDFs smaller than this (post-download)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable, Optional

API = "https://clinicaltrials.gov/api/v2/studies"
DOC_URL_TMPL = "https://clinicaltrials.gov/ProvidedDocs/{tail}/{nct}/{fn}"

# ClinicalTrials.gov blocks httpx (HTTP/2 + ALPN fingerprint), accepts urllib.
HTTP_HEADERS = {"User-Agent": "curl/8.0.0", "Accept": "application/json, */*"}


def _http_get(url: str, *, timeout: int = 60, binary: bool = False):
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("content-type", "")
        body = resp.read()
        return resp.status, content_type, body

# Stratification axes. Each tuple = (label_for_filename, query.cond, optional phase).
# Tuned for COMPLEXITY: prefers industry-sponsored Phase 3/4 protocols where
# templates are densest. Mix of therapy areas keeps the corpus diverse.
STRATA: list[tuple[str, str, Optional[str]]] = [
    # Industry oncology Phase 3 — typically dense SoA, biomarker tables, adaptive designs.
    ("oncology_p3",        "lung cancer",       "PHASE3"),
    ("oncology_p3",        "breast cancer",     "PHASE3"),
    ("oncology_p3",        "melanoma",          "PHASE3"),
    ("oncology_p3",        "leukemia",          "PHASE3"),
    ("oncology_p3",        "myeloma",           "PHASE3"),
    ("oncology_p2",        "lymphoma",          "PHASE2"),
    # Vaccines — multi-visit schedules, complex SoA.
    ("vaccine_p3",         "vaccine",           "PHASE3"),
    ("vaccine_p3",         "rsv vaccine",       None),
    # Rare/genetic — gene therapy and biologics have complex pharmacokinetic schedules.
    ("rare_genetic",       "gene therapy",      None),
    ("rare_genetic",       "spinal muscular atrophy", None),
    # CV outcome trials — large Phase 3 with complex endpoint definitions.
    ("cardio_p3",          "heart failure",     "PHASE3"),
    ("cardio_p3",          "myocardial infarction", "PHASE3"),
    # Endocrine — diabetes Phase 3 protocols (Novo Nordisk, Lilly, etc).
    ("endocrine_p3",       "type 2 diabetes",   "PHASE3"),
    ("endocrine_p3",       "obesity",           "PHASE3"),
    # CNS — Alzheimer and MS protocols are very long with cognitive/MRI SoA.
    ("neuro_p3",           "alzheimer disease", "PHASE3"),
    ("neuro_p3",           "multiple sclerosis","PHASE3"),
    # Adaptive / basket / platform trials.
    ("adaptive",           "platform trial",    None),
    ("adaptive",           "basket trial",      None),
    # Immunology Phase 3 — IBD/RA biologics typically have rich SoA + biomarker panels.
    ("immuno_p3",          "ulcerative colitis","PHASE3"),
    ("immuno_p3",          "rheumatoid arthritis","PHASE3"),
    ("immuno_p3",          "psoriasis",         "PHASE3"),
    # Pediatric Phase 3 — extra safety tables, age-stratified dosing.
    ("pediatric_p3",       "pediatric oncology","PHASE3"),
    # Pulmonary Phase 3.
    ("respiratory_p3",     "copd",              "PHASE3"),
    ("respiratory_p3",     "asthma",            "PHASE3"),
    # COVID-19 industry-sponsored — modern templates.
    ("covid_p3",           "covid-19",          "PHASE3"),
]


def safe_slug(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen] or "untitled"


def search_strata(condition: str, phase: Optional[str], page_size: int = 25) -> list[dict]:
    params = {
        "query.cond": condition,
        "aggFilters": "results:with",
        "pageSize": str(page_size),
        "fields": "protocolSection.identificationModule,documentSection,"
                  "protocolSection.designModule,protocolSection.sponsorCollaboratorsModule",
    }
    if phase:
        params["query.term"] = f"AREA[Phase]{phase}"
    url = API + "?" + urllib.parse.urlencode(params)
    time.sleep(0.5)
    try:
        status, _ctype, body = _http_get(url, timeout=30)
        if status != 200:
            print(f"  search failed for cond={condition!r}: status {status}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"  search failed for cond={condition!r} phase={phase!r}: {e}", file=sys.stderr)
        return []
    try:
        return json.loads(body).get("studies", []) or []
    except Exception as e:
        print(f"  failed to parse JSON for cond={condition!r}: {e}", file=sys.stderr)
        return []


def pick_protocol_doc(study: dict) -> Optional[dict]:
    docs = (
        study.get("documentSection", {})
        .get("largeDocumentModule", {})
        .get("largeDocs", [])
        or []
    )
    for d in docs:
        if d.get("hasProtocol"):
            return d
    return None


def download(nct: str, doc: dict, out_dir: Path, max_mb: int, min_kb: int = 0) -> Optional[Path]:
    size = int(doc.get("size") or 0)
    if size and size > max_mb * 1024 * 1024:
        return None
    if size and size < min_kb * 1024:
        # Skip tiny PDFs — usually summaries/synopses, not real protocols
        return None
    fn = doc.get("filename") or "protocol.pdf"
    url = DOC_URL_TMPL.format(tail=nct[-2:], nct=nct, fn=fn)
    try:
        # urllib follows 30x by default
        status, ctype, body = _http_get(url, timeout=180)
        if status != 200:
            print(f"  {nct}: download status {status}", file=sys.stderr)
            return None
        if not ctype.lower().startswith("application/pdf"):
            print(f"  {nct}: unexpected content-type {ctype}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"  {nct}: download failed: {e}", file=sys.stderr)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nct}_{Path(fn).stem}.pdf"
    out_path.write_bytes(body)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25, help="number of PDFs to download")
    ap.add_argument("--out", default="corpus", help="output directory")
    ap.add_argument("--max-mb", type=int, default=30, help="skip PDFs larger than this many MB")
    ap.add_argument("--min-kb", type=int, default=500, help="skip PDFs smaller than this many KB (usually summaries, not full protocols)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    manifest_path = out_dir / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    seen_ncts = {m["nct"] for m in manifest}
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"target: {args.target} PDFs into {out_dir}")
    print(f"already have: {len(seen_ncts)}")

    # Iterate strata in a round-robin until we hit the target.
    per_stratum_results: dict[int, list[dict]] = {}
    for i, (label, cond, phase) in enumerate(STRATA):
        per_stratum_results[i] = search_strata(cond, phase)
        print(f"  [{label:<12}] cond={cond!r} phase={phase}: {len(per_stratum_results[i])} candidates")

    idx_in_stratum = [0] * len(STRATA)
    downloaded_this_run = 0
    while len(manifest) < args.target:
        progressed = False
        for i, (label, cond, phase) in enumerate(STRATA):
            if len(manifest) >= args.target:
                break
            studies = per_stratum_results[i]
            while idx_in_stratum[i] < len(studies):
                s = studies[idx_in_stratum[i]]
                idx_in_stratum[i] += 1
                ident = s.get("protocolSection", {}).get("identificationModule", {}) or {}
                nct = ident.get("nctId")
                if not nct or nct in seen_ncts:
                    continue
                doc = pick_protocol_doc(s)
                if not doc:
                    continue
                title = (ident.get("briefTitle") or "").strip()
                design = s.get("protocolSection", {}).get("designModule", {}) or {}
                phases = ",".join(design.get("phases", []) or []) or "unknown"
                sponsor = (
                    s.get("protocolSection", {})
                    .get("sponsorCollaboratorsModule", {})
                    .get("leadSponsor", {})
                    .get("name", "unknown")
                )
                path = download(nct, doc, out_dir, args.max_mb, args.min_kb)
                if path is None:
                    continue
                seen_ncts.add(nct)
                entry = {
                    "nct": nct,
                    "stratum": label,
                    "condition": cond,
                    "phase_requested": phase,
                    "phase_actual": phases,
                    "sponsor": sponsor,
                    "title": title[:200],
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "doc_type": doc.get("typeAbbrev"),
                    "doc_date": doc.get("date"),
                }
                manifest.append(entry)
                downloaded_this_run += 1
                print(f"  [{label:<12}] {nct} ({phases:<10}) -> {path.name} ({entry['size_bytes']//1024} KB)")
                progressed = True
                break  # next stratum
        if not progressed:
            break

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"manifest: {manifest_path}")
    print(f"downloaded this run: {downloaded_this_run}")
    print(f"total in corpus:     {len(manifest)}")
    print(f"strata represented:  {sorted({m['stratum'] for m in manifest})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
