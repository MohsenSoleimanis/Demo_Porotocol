"""Score parser output against ClinicalTrials.gov structured ground truth.

For each NCT in `dataset/`, compare:
  - parsed.json            (our pipeline's extraction)
  - ctgov_metadata.json    (CT.gov's structured fields = ground truth)

Tasks scored:
  1. eligibility_criteria  text-overlap recall
  2. primary_outcomes      `measure` recall
  3. secondary_outcomes    `measure` recall
  4. arms                  arm-label recall
  5. interventions         intervention-name recall
  6. design metadata       (phase, allocation, masking, intervention_model, primary_purpose)
                           detection via keyword presence

Output:
  dataset/eval_report.json
  dataset/eval_report.md     (human-readable summary)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# ---- text utilities ----


_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation that varies between
    PDF extraction and the CT.gov free-text fields."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = re.sub(r"[\-‐-―−]", "-", s)  # unify dash variants
    s = re.sub(r"[‘’“”]", "'", s)  # smart quotes
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return _normalize(s).split()


def _ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def fuzzy_present(needle: str, haystack: str, *, n: int = 4, threshold: float = 0.6) -> bool:
    """Fuzzy substring check: do at least `threshold` of needle's n-grams
    appear in haystack? Robust to small wording differences (paraphrasing,
    inserted whitespace) — does NOT need an exact substring match."""
    nt = _tokens(needle)
    if not nt:
        return False
    if len(nt) < n:
        return _normalize(needle) in _normalize(haystack)
    ngrams = _ngram_set(nt, n)
    h_ngrams = _ngram_set(_tokens(haystack), n)
    if not ngrams:
        return False
    overlap = len(ngrams & h_ngrams)
    return overlap / len(ngrams) >= threshold


# ---- pulling content from parsed.json ----


def _parsed_blocks_text(parsed: dict) -> str:
    """Concatenate every block's text into one big string for fuzzy matching."""
    parts: list[str] = []
    for p in parsed.get("pages") or []:
        for b in p.get("blocks") or []:
            t = b.get("text") or ""
            if t:
                parts.append(t)
    return "\n".join(parts)


def _parsed_headings(parsed: dict) -> list[str]:
    out: list[str] = []
    for p in parsed.get("pages") or []:
        for b in p.get("blocks") or []:
            if b.get("block_type") == "heading":
                t = b.get("text") or ""
                if t:
                    out.append(t)
    return out


# ---- eligibility criteria scoring ----


def _split_eligibility(text: str) -> list[str]:
    """CT.gov stores eligibility as free text with bullets / sections.
    Split into individual criterion statements heuristically."""
    if not text:
        return []
    out: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        # Strip common bullet prefixes
        s = re.sub(r"^[-•\*·●∘°]\s*", "", s)
        s = re.sub(r"^\d{1,2}[\.\)]\s*", "", s)
        s = re.sub(r"^[A-Za-z][\.\)]\s*", "", s)
        if len(s) >= 25:  # require minimum substance
            out.append(s)
    return out


def score_eligibility(parsed: dict, gt: dict) -> dict:
    criteria = _split_eligibility(gt.get("eligibility_criteria") or "")
    body = _parsed_blocks_text(parsed)
    hits = [c for c in criteria if fuzzy_present(c, body)]
    return {
        "n_criteria_in_gt": len(criteria),
        "n_criteria_found": len(hits),
        "recall": (len(hits) / len(criteria)) if criteria else None,
        "missed_examples": [c[:200] for c in criteria if c not in hits][:5],
    }


# ---- outcomes scoring ----


def _score_outcomes(parsed: dict, outcomes: list[dict], label: str) -> dict:
    body = _parsed_blocks_text(parsed)
    hits = 0
    found: list[str] = []
    missed: list[str] = []
    for o in outcomes:
        measure = (o.get("measure") or "").strip()
        if not measure:
            continue
        if fuzzy_present(measure, body):
            hits += 1
            found.append(measure[:120])
        else:
            missed.append(measure[:120])
    return {
        f"n_{label}_in_gt": len(outcomes),
        f"n_{label}_found": hits,
        "recall": (hits / len(outcomes)) if outcomes else None,
        "missed": missed[:5],
    }


# ---- arms / interventions ----


def score_arms(parsed: dict, gt: dict) -> dict:
    body = _parsed_blocks_text(parsed)
    arms = gt.get("arms") or []
    hits = 0
    for a in arms:
        lbl = (a.get("label") or "").strip()
        if not lbl:
            continue
        if fuzzy_present(lbl, body):
            hits += 1
    return {
        "n_arms_in_gt": len(arms),
        "n_arms_found": hits,
        "recall": (hits / len(arms)) if arms else None,
    }


def score_interventions(parsed: dict, gt: dict) -> dict:
    body = _parsed_blocks_text(parsed)
    items = gt.get("interventions") or []
    hits = 0
    for it in items:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        if fuzzy_present(name, body):
            hits += 1
    return {
        "n_interventions_in_gt": len(items),
        "n_interventions_found": hits,
        "recall": (hits / len(items)) if items else None,
    }


# ---- design metadata ----


def score_design(parsed: dict, gt: dict) -> dict:
    body_norm = _normalize(_parsed_blocks_text(parsed))
    fields_present: dict[str, bool] = {}
    for key in ("phases", "allocation", "masking", "intervention_model", "primary_purpose"):
        gt_val = gt.get(key)
        if not gt_val:
            fields_present[key] = None  # type: ignore
            continue
        values = gt_val if isinstance(gt_val, list) else [gt_val]
        # Field is "present" if ANY of its allowed values is mentioned (lowercased)
        # CT.gov uses uppercased enums like "RANDOMIZED" — we look for lowercased.
        ok = any(_normalize(v) in body_norm for v in values if v)
        fields_present[key] = ok
    return fields_present


# ---- per-doc + aggregate ----


def evaluate_one(nct_dir: Path) -> dict:
    parsed_path = nct_dir / "parsed.json"
    gt_path = nct_dir / "ctgov_metadata.json"
    if not parsed_path.exists() or not gt_path.exists():
        return {"nct": nct_dir.name, "skipped": True, "reason": "missing parsed or gt"}
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    return {
        "nct": nct_dir.name,
        "sponsor": gt.get("lead_sponsor"),
        "eligibility": score_eligibility(parsed, gt),
        "primary_outcomes": _score_outcomes(parsed, gt.get("primary_outcomes") or [], "primary_outcomes"),
        "secondary_outcomes": _score_outcomes(parsed, gt.get("secondary_outcomes") or [], "secondary_outcomes"),
        "arms": score_arms(parsed, gt),
        "interventions": score_interventions(parsed, gt),
        "design": score_design(parsed, gt),
        "n_pages": parsed.get("total_pages"),
        "n_blocks": sum(len(p.get("blocks") or []) for p in parsed.get("pages") or []),
        "n_headings": sum(1 for p in parsed.get("pages") or [] for b in p.get("blocks") or [] if b.get("block_type") == "heading"),
    }


def aggregate(rows: list[dict]) -> dict:
    def _avg_recall(key: str) -> float | None:
        vals = [r[key]["recall"] for r in rows if r.get(key) and r[key].get("recall") is not None]
        return sum(vals) / len(vals) if vals else None
    design_keys = ("phases", "allocation", "masking", "intervention_model", "primary_purpose")
    design_rates = {}
    for k in design_keys:
        vals = [r["design"].get(k) for r in rows if r.get("design")]
        vals = [v for v in vals if v is not None]
        design_rates[k] = (sum(v for v in vals) / len(vals)) if vals else None
    return {
        "n_docs": len(rows),
        "avg_recall": {
            "eligibility": _avg_recall("eligibility"),
            "primary_outcomes": _avg_recall("primary_outcomes"),
            "secondary_outcomes": _avg_recall("secondary_outcomes"),
            "arms": _avg_recall("arms"),
            "interventions": _avg_recall("interventions"),
        },
        "design_detection_rate": design_rates,
    }


def _fmt_pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "—"


def write_markdown_report(report: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# FMLS extraction eval — vs ClinicalTrials.gov structured ground truth\n")
    agg = report["aggregate"]
    lines.append(f"**Docs evaluated:** {agg['n_docs']}\n")
    lines.append("## Aggregate recall (per task)\n")
    lines.append("| Task | Avg recall |")
    lines.append("|---|---|")
    for k, v in agg["avg_recall"].items():
        lines.append(f"| {k} | {_fmt_pct(v)} |")
    lines.append("\n## Design metadata detection rate (per field)\n")
    lines.append("| Field | Detection rate |")
    lines.append("|---|---|")
    for k, v in agg["design_detection_rate"].items():
        lines.append(f"| {k} | {_fmt_pct(v)} |")
    lines.append("\n## Per-protocol results\n")
    lines.append("| NCT | Sponsor | Pages | Headings | Eligibility | Primary | Secondary | Arms | Interventions |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in report["per_doc"]:
        if r.get("skipped"):
            lines.append(f"| {r['nct']} | — | — | — | SKIPPED ({r['reason']}) | | | | |")
            continue
        elig = _fmt_pct(r["eligibility"]["recall"])
        prim = _fmt_pct(r["primary_outcomes"]["recall"])
        sec = _fmt_pct(r["secondary_outcomes"]["recall"])
        arms = _fmt_pct(r["arms"]["recall"])
        ints = _fmt_pct(r["interventions"]["recall"])
        lines.append(
            f"| {r['nct']} | {(r.get('sponsor') or '')[:24]} | {r.get('n_pages')} | {r.get('n_headings')} | "
            f"{elig} | {prim} | {sec} | {arms} | {ints} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    if not (DATASET / "manifest.json").exists():
        print("no dataset/manifest.json — run scripts/build_eval_dataset.py first")
        return 2
    manifest = json.loads((DATASET / "manifest.json").read_text())
    rows: list[dict] = []
    for entry in manifest:
        rows.append(evaluate_one(DATASET / entry["nct_id"]))
    report = {"aggregate": aggregate([r for r in rows if not r.get("skipped")]), "per_doc": rows}
    out = DATASET / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out = DATASET / "eval_report.md"
    write_markdown_report(report, md_out)
    # Brief stdout summary
    print(json.dumps(report["aggregate"], indent=2))
    print(f"\nfull report: {out}")
    print(f"markdown:    {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
