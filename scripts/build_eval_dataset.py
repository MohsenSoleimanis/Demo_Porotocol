"""Build an evaluation dataset of real clinical-protocol PDFs paired with
ClinicalTrials.gov structured metadata as ground truth.

Quality criteria (corpus-generic, no document-specific tuning):
  - Industry-sponsored Phase 3 / Phase 2-3 trials
  - Has a Protocol document (`hasProtocol=True`) AND posted results
  - PDF size in a reasonable range (1-15 MB) — too small = synopsis;
    too big = scan-heavy
  - Recent (start year >= 2020) — newer protocols are closer to M11
  - Redaction filter applied AFTER download:
      * blank-page fraction      <= 5%
      * estimated-redaction-rate <= 10% (looks at filled black rectangles
        that overlap the text region of each page)

Output layout:
  dataset/{NCT_ID}/
    {NCT_ID}.pdf
    ctgov_metadata.json     # structured ground-truth fields from CT.gov v2 API
    quality_report.json     # blank-page %, redaction %, page count, etc

  dataset/manifest.json     # top-level index

Usage:
  .venv/Scripts/python.exe scripts/build_eval_dataset.py [--target 25]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import fitz

API = "https://clinicaltrials.gov/api/v2/studies"
DOC_URL_TMPL = "https://clinicaltrials.gov/ProvidedDocs/{tail}/{nct}/{fn}"

HTTP_HEADERS = {
    "User-Agent": "curl/8.0.0",
    "Accept": "application/json, application/pdf, */*",
}

# Stratify by therapeutic area but keep filters tight on quality.
STRATA = [
    ("oncology",       "lung cancer"),
    ("oncology",       "breast cancer"),
    ("oncology",       "melanoma"),
    ("oncology",       "leukemia"),
    ("oncology",       "lymphoma"),
    ("vaccine",        "vaccine"),
    ("infectious",     "covid-19"),
    ("cardio",         "heart failure"),
    ("cardio",         "atrial fibrillation"),
    ("endocrine",      "type 2 diabetes"),
    ("endocrine",      "obesity"),
    ("neuro",          "alzheimer disease"),
    ("neuro",          "multiple sclerosis"),
    ("neuro",          "parkinson"),
    ("psych",          "depression"),
    ("psych",          "schizophrenia"),
    ("immuno",         "ulcerative colitis"),
    ("immuno",         "rheumatoid arthritis"),
    ("immuno",         "psoriasis"),
    ("respiratory",    "asthma"),
    ("respiratory",    "copd"),
    ("rare",           "spinal muscular atrophy"),
    ("rare",           "cystic fibrosis"),
]


def _http_get(url: str, timeout: int = 60) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.headers.get("content-type", ""), resp.read()


def search_phase3_industry(condition: str, page_size: int = 50) -> list[dict]:
    """Find industry-sponsored Phase 3 trials with posted protocols + results."""
    params = {
        "query.cond": condition,
        "query.term": "AREA[Phase]PHASE3 AND AREA[LeadSponsorClass]INDUSTRY",
        "aggFilters": "results:with",
        "pageSize": str(page_size),
        "fields": (
            "protocolSection.identificationModule,"
            "protocolSection.statusModule.startDateStruct,"
            "protocolSection.statusModule.lastUpdatePostDateStruct,"
            "protocolSection.statusModule.overallStatus,"
            "protocolSection.designModule,"
            "protocolSection.sponsorCollaboratorsModule,"
            "protocolSection.conditionsModule,"
            "protocolSection.eligibilityModule,"
            "protocolSection.outcomesModule,"
            "protocolSection.armsInterventionsModule,"
            "documentSection"
        ),
    }
    url = API + "?" + urllib.parse.urlencode(params)
    time.sleep(0.4)
    try:
        status, _ct, body = _http_get(url, timeout=30)
        if status != 200:
            return []
    except Exception as e:
        print(f"  search '{condition}' failed: {e}", file=sys.stderr)
        return []
    try:
        return json.loads(body).get("studies", []) or []
    except json.JSONDecodeError:
        return []


def fetch_full_study(nct: str) -> Optional[dict]:
    """Fetch the FULL study record (all sections) for metadata ground truth."""
    url = f"{API}/{nct}"
    try:
        time.sleep(0.2)
        status, _ct, body = _http_get(url, timeout=30)
        if status != 200:
            return None
        return json.loads(body)
    except Exception as e:
        print(f"  full-study fetch {nct} failed: {e}", file=sys.stderr)
        return None


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


def download_pdf(nct: str, doc: dict, max_mb: int = 15, min_kb: int = 800) -> Optional[bytes]:
    size = int(doc.get("size") or 0)
    if size and (size > max_mb * 1024 * 1024 or size < min_kb * 1024):
        return None
    fn = doc.get("filename") or "protocol.pdf"
    url = DOC_URL_TMPL.format(tail=nct[-2:], nct=nct, fn=fn)
    try:
        status, ctype, body = _http_get(url, timeout=180)
        if status != 200 or not ctype.lower().startswith("application/pdf"):
            return None
        return body
    except Exception as e:
        print(f"  download {nct} failed: {e}", file=sys.stderr)
        return None


# ---- redaction detection (corpus-generic) ----


def _is_black_fill(fill) -> bool:
    """A drawing op is a likely redaction if its fill is near-black."""
    if not fill:
        return False
    # fill is an RGB tuple in PyMuPDF (each channel 0-1)
    try:
        r, g, b = fill[:3]
        return r < 0.15 and g < 0.15 and b < 0.15
    except (TypeError, ValueError):
        return False


def assess_pdf_quality(pdf_bytes: bytes) -> dict:
    """Compute blank-page %, redaction %, and other quality signals.

    Redaction detection: look for filled black rectangles that overlap a
    page's text region. A typical redaction is a `re` op with fill near
    (0,0,0) covering a chunk of the text column. Stamps/logos won't trigger
    because they're tiny relative to the page.
    """
    out = {
        "n_pages": 0,
        "blank_pages": 0,
        "redaction_pages": 0,
        "redaction_area_fraction": 0.0,
        "page_count_with_text": 0,
    }
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return out
    try:
        n = doc.page_count
        out["n_pages"] = n
        total_redact_area = 0.0
        total_area = 0.0
        for page in doc:
            page_area = float(page.rect.width) * float(page.rect.height)
            total_area += page_area
            text = (page.get_text("text") or "").strip()
            if len(text) < 20:
                out["blank_pages"] += 1
                continue
            out["page_count_with_text"] += 1
            page_redact_area = 0.0
            for dr in page.get_drawings() or []:
                fill = dr.get("fill")
                if not _is_black_fill(fill):
                    continue
                rect = dr.get("rect")
                if rect is None:
                    continue
                try:
                    w = float(rect.width)
                    h = float(rect.height)
                except Exception:
                    continue
                area = w * h
                # Filter out trivial marks (logos, ticks)
                if area < page_area * 0.002:
                    continue
                # Filter out borders / decorative strokes
                if min(w, h) < 6:
                    continue
                page_redact_area += area
            if page_redact_area > page_area * 0.05:
                out["redaction_pages"] += 1
            total_redact_area += page_redact_area
        out["redaction_area_fraction"] = (
            total_redact_area / total_area if total_area else 0.0
        )
    finally:
        doc.close()
    return out


def is_high_quality(quality: dict, max_blank_frac: float = 0.05, max_redact_frac: float = 0.10) -> tuple[bool, str]:
    if quality["n_pages"] < 40:
        return False, "too-short"
    if quality["n_pages"] > 600:
        return False, "too-long"
    if quality["blank_pages"] / max(quality["n_pages"], 1) > max_blank_frac:
        return False, f"blank-pages-{quality['blank_pages']}"
    if quality["redaction_area_fraction"] > max_redact_frac:
        return False, f"redaction-{quality['redaction_area_fraction']:.1%}"
    return True, "ok"


# ---- structured ground truth extraction ----


def extract_ground_truth(study: dict) -> dict:
    """Pull the structured fields from CT.gov that we'll score extraction against."""
    proto = study.get("protocolSection", {}) or {}
    ident = proto.get("identificationModule", {}) or {}
    status_m = proto.get("statusModule", {}) or {}
    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
    design = proto.get("designModule", {}) or {}
    conditions = proto.get("conditionsModule", {}) or {}
    elig = proto.get("eligibilityModule", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    outcomes = proto.get("outcomesModule", {}) or {}
    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "lead_sponsor": (sponsor.get("leadSponsor") or {}).get("name"),
        "lead_sponsor_class": (sponsor.get("leadSponsor") or {}).get("class"),
        "collaborators": [c.get("name") for c in (sponsor.get("collaborators") or [])],
        "overall_status": status_m.get("overallStatus"),
        "start_date": (status_m.get("startDateStruct") or {}).get("date"),
        "completion_date": (status_m.get("completionDateStruct") or {}).get("date"),
        "phases": design.get("phases") or [],
        "study_type": design.get("studyType"),
        "allocation": (design.get("designInfo") or {}).get("allocation"),
        "intervention_model": (design.get("designInfo") or {}).get("interventionModel"),
        "masking": ((design.get("designInfo") or {}).get("maskingInfo") or {}).get("masking"),
        "primary_purpose": (design.get("designInfo") or {}).get("primaryPurpose"),
        "enrollment_count": (design.get("enrollmentInfo") or {}).get("count"),
        "enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
        "conditions": conditions.get("conditions") or [],
        "keywords": conditions.get("keywords") or [],
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        "sex": elig.get("sex"),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "arms": [
            {
                "label": a.get("label"),
                "type": a.get("type"),
                "description": a.get("description"),
                "intervention_names": a.get("interventionNames") or [],
            }
            for a in (arms.get("armGroups") or [])
        ],
        "interventions": [
            {
                "type": i.get("type"),
                "name": i.get("name"),
                "description": i.get("description"),
                "arm_group_labels": i.get("armGroupLabels") or [],
            }
            for i in (arms.get("interventions") or [])
        ],
        "primary_outcomes": [
            {"measure": o.get("measure"), "description": o.get("description"), "time_frame": o.get("timeFrame")}
            for o in (outcomes.get("primaryOutcomes") or [])
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure"), "description": o.get("description"), "time_frame": o.get("timeFrame")}
            for o in (outcomes.get("secondaryOutcomes") or [])
        ],
    }


# ---- main ----


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--min-start-year", type=int, default=2020)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    seen = {m["nct_id"] for m in manifest}

    print(f"target: {args.target} clean industry Phase-3 protocols")
    print(f"already have: {len(seen)}")

    # Search each stratum.
    per_stratum: dict[int, list[dict]] = {}
    for i, (label, cond) in enumerate(STRATA):
        per_stratum[i] = search_phase3_industry(cond)
        print(f"  [{label:<12}] {cond!r}: {len(per_stratum[i])} candidates")

    idx_in = [0] * len(STRATA)
    added_this_run = 0
    while len(manifest) < args.target:
        progressed = False
        for i, (label, cond) in enumerate(STRATA):
            if len(manifest) >= args.target:
                break
            studies = per_stratum[i]
            while idx_in[i] < len(studies):
                s = studies[idx_in[i]]
                idx_in[i] += 1
                ident = (s.get("protocolSection", {}).get("identificationModule") or {})
                nct = ident.get("nctId")
                if not nct or nct in seen:
                    continue

                status_m = (s.get("protocolSection", {}).get("statusModule") or {})
                start_date = ((status_m.get("startDateStruct") or {}).get("date") or "")[:4]
                try:
                    if int(start_date) < args.min_start_year:
                        continue
                except (TypeError, ValueError):
                    continue

                doc = pick_protocol_doc(s)
                if not doc:
                    continue

                pdf_bytes = download_pdf(nct, doc)
                if not pdf_bytes:
                    continue

                quality = assess_pdf_quality(pdf_bytes)
                ok, reason = is_high_quality(quality)
                if not ok:
                    print(f"  [{label:<12}] {nct} REJECT ({reason}): {quality}")
                    continue

                # Full study record for ground truth
                full = fetch_full_study(nct)
                if not full:
                    continue
                gt = extract_ground_truth(full)

                # Save dataset/{NCT}/
                d = out_dir / nct
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{nct}.pdf").write_bytes(pdf_bytes)
                (d / "ctgov_metadata.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
                (d / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

                manifest.append({
                    "nct_id": nct,
                    "stratum": label,
                    "condition": cond,
                    "title": (ident.get("briefTitle") or "")[:200],
                    "sponsor": gt["lead_sponsor"],
                    "phases": gt["phases"],
                    "start_year": int(start_date),
                    "n_pages": quality["n_pages"],
                    "redaction_area_fraction": round(quality["redaction_area_fraction"], 3),
                    "size_kb": len(pdf_bytes) // 1024,
                })
                seen.add(nct)
                added_this_run += 1
                progressed = True
                print(
                    f"  [{label:<12}] {nct} ({gt['lead_sponsor']}) "
                    f"OK: {quality['n_pages']}p, redact {quality['redaction_area_fraction']:.1%} "
                    f"-> {d.name}"
                )
                break
        if not progressed:
            break

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nadded this run: {added_this_run}; total: {len(manifest)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
