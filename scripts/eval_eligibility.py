"""Eval Stage 4 extraction against a manual gold standard.

Reads:
  dataset/{stem}/gold_eligibility.json  — manually-curated ground truth from the PDF
  dataset/{stem}/extracted/eligibility.jsonl  — our extraction output

Computes:
  - Precision = (correct extracted) / (total extracted)
  - Recall    = (correct extracted) / (total in gold)
  - F1
  - Category accuracy on matched pairs
  - Per-criterion classification:
      MATCH        — correct text + correct category
      WRONG_CAT    — text matches but category is wrong (polarity error)
      FALSE_POS    — extracted but not in gold (over-extraction; e.g., sub-bullets, intros)
      FALSE_NEG    — gold criterion not extracted (under-extraction)
      PARTIAL      — partial text overlap (≥0.6 ratio)

Matching uses normalized fuzzy comparison (token-set + bigram overlap) so the
LaTeX/Unicode rendering differences don't unfairly penalize correct extractions.

Usage:
    python scripts/eval_eligibility.py AZ_demo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def normalize_for_matching(s: str) -> str:
    """Aggressive normalization for fuzzy text matching.

    Strips LaTeX math (\\geq, \\mathrm{}, \\circ etc), converts ≥/≤ to >=/<=,
    collapses whitespace, lowercases.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # Strip LaTeX commands
    s = re.sub(r"\\geq\b", ">=", s)
    s = re.sub(r"\\leq\b", "<=", s)
    s = re.sub(r"\\neq\b", "!=", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\circ\b", "°", s)
    s = re.sub(r"\\[a-zA-Z]+\b", "", s)
    s = re.sub(r"[{}~]", "", s)
    # Unicode → ASCII for comparison operators
    s = s.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=").replace("−", "-")
    # Collapse whitespace + lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def token_set(s: str) -> set:
    """Bag-of-tokens for fuzzy matching."""
    s = normalize_for_matching(s)
    return set(re.findall(r"\w+", s))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_match(
    candidate_text: str,
    gold_pool: list[dict],
    threshold: float = 0.5,
) -> tuple[dict | None, float]:
    """Find the best-matching gold criterion for a candidate extracted text.

    Returns (gold_record, score). Score is jaccard token similarity.
    """
    cand_tokens = token_set(candidate_text)
    best = None
    best_score = 0.0
    for gold in gold_pool:
        gold_tokens = token_set(gold["text"])
        score = jaccard(cand_tokens, gold_tokens)
        if score > best_score:
            best_score = score
            best = gold
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="dataset/{stem}/")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Jaccard threshold for considering a match (default 0.5)")
    ap.add_argument("--partial-threshold", type=float, default=0.3)
    ap.add_argument("--extracted", default="extracted/eligibility.jsonl",
                    help="Path to extracted JSONL relative to dataset/{stem}/")
    args = ap.parse_args()

    doc_dir = DATASET / args.stem
    gold_path = doc_dir / "gold_eligibility.json"
    extracted_path = doc_dir / args.extracted

    if not gold_path.exists():
        print(f"ERROR: {gold_path} not found.", file=sys.stderr)
        return 2
    if not extracted_path.exists():
        print(f"ERROR: {extracted_path} not found.", file=sys.stderr)
        return 2

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_crits = gold["criteria"]
    gold_by_id: dict[str, dict] = {g["id"]: g for g in gold_crits}

    # Load extracted (Extracted[T] envelope; only EligibilityCriterion records carry category)
    extracted_records = []
    with open(extracted_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            extracted_records.append(rec)

    # Pair EligibilityCriterion (ref) with EligibilityCriterionItem (text)
    items_by_id = {
        r["value"]["id"]: r["value"]["text"]
        for r in extracted_records
        if r["value"]["instanceType"] == "EligibilityCriterionItem"
    }
    extracted_crits = []
    for r in extracted_records:
        v = r["value"]
        if v["instanceType"] != "EligibilityCriterion":
            continue
        text = items_by_id.get(v["criterionItemId"], "(no text)")
        extracted_crits.append({
            "id": v["id"],
            "category": v["category"]["decode"],
            "identifier": v["identifier"] or "",
            "text": text,
        })

    print(f"=== Eligibility Eval — {args.stem} ===")
    print(f"  Gold:      {len(gold_crits)} criteria ({sum(1 for g in gold_crits if g['category']=='inclusion')} inc, "
          f"{sum(1 for g in gold_crits if g['category']=='exclusion')} exc)")
    print(f"  Extracted: {len(extracted_crits)} criteria ({sum(1 for c in extracted_crits if c['category']=='inclusion')} inc, "
          f"{sum(1 for c in extracted_crits if c['category']=='exclusion')} exc)")
    print()

    # === Score each extracted against gold ===
    used_gold: set[str] = set()
    matches: list[dict] = []
    false_positives: list[dict] = []
    partials: list[dict] = []
    wrong_cats: list[dict] = []

    for ext in extracted_crits:
        gold_match, score = best_match(ext["text"], gold_crits, threshold=args.threshold)
        if gold_match is None:
            # Try partial threshold
            partial, partial_score = best_match(ext["text"], gold_crits, threshold=args.partial_threshold)
            if partial is not None:
                partials.append({"ext": ext, "gold": partial, "score": partial_score})
            else:
                false_positives.append(ext)
            continue

        if gold_match["id"] in used_gold:
            # Already matched — this is a duplicate extraction (over-fragmentation)
            partials.append({"ext": ext, "gold": gold_match, "score": score, "duplicate": True})
            continue
        used_gold.add(gold_match["id"])

        if ext["category"] != gold_match["category"]:
            wrong_cats.append({"ext": ext, "gold": gold_match, "score": score})
        else:
            matches.append({"ext": ext, "gold": gold_match, "score": score})

    false_negatives = [g for g in gold_crits if g["id"] not in used_gold]

    # === Metrics ===
    n_extracted = len(extracted_crits)
    n_correct = len(matches)
    n_wrong_cat = len(wrong_cats)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    n_partial = len(partials)
    n_gold = len(gold_crits)

    precision = n_correct / n_extracted if n_extracted else 0
    recall = n_correct / n_gold if n_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Category accuracy among text-matched (matches + wrong_cats)
    cat_total = n_correct + n_wrong_cat
    cat_acc = n_correct / cat_total if cat_total else 0

    print(f"=== METRICS ===")
    print(f"  Precision (text + category):   {precision:.2%}  ({n_correct}/{n_extracted})")
    print(f"  Recall (text):                 {recall:.2%}  ({n_correct}/{n_gold})")
    print(f"  F1:                            {f1:.2%}")
    print(f"  Category accuracy (on text-matched):  {cat_acc:.2%}  ({n_correct}/{cat_total})")
    print(f"  Partial matches:               {n_partial}")
    print(f"  False positives (extracted not in gold):   {n_fp}")
    print(f"  False negatives (gold not extracted):      {n_fn}")
    print()

    print("=== MATCHES ===")
    for m in matches:
        print(f"  ✓ {m['gold']['id']}: {m['gold']['text'][:80]}  [score={m['score']:.2f}]")
    print()

    if wrong_cats:
        print("=== WRONG CATEGORY (text correct, polarity wrong) ===")
        for w in wrong_cats:
            print(f"  ⚠ gold {w['gold']['id']} ({w['gold']['category']}) -> extracted as {w['ext']['category']}")
            print(f"    gold:    {w['gold']['text'][:90]}")
            print(f"    extract: {w['ext']['text'][:90]}")
        print()

    if false_negatives:
        print(f"=== FALSE NEGATIVES — {n_fn} gold criteria we missed ===")
        for g in false_negatives:
            print(f"  ✗ {g['id']} ({g['category']}/{g['identifier']}): {g['text'][:100]}")
        print()

    if false_positives:
        print(f"=== FALSE POSITIVES — {n_fp} extracted that don't match gold ===")
        for fp in false_positives:
            print(f"  ⚠ ({fp['category']}): {fp['text'][:100]}")
        print()

    if partials:
        print(f"=== PARTIAL MATCHES — {n_partial} ===")
        for p in partials[:10]:
            tag = "DUP" if p.get("duplicate") else "PARTIAL"
            print(f"  ~ {tag} {p['gold']['id']}  [score={p['score']:.2f}]")
            print(f"    gold:    {p['gold']['text'][:90]}")
            print(f"    extract: {p['ext']['text'][:90]}")
        print()

    # Save the report
    report = {
        "doc_id": args.stem,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "category_accuracy_on_matched": cat_acc,
            "n_correct": n_correct,
            "n_extracted": n_extracted,
            "n_gold": n_gold,
            "n_wrong_category": n_wrong_cat,
            "n_false_positive": n_fp,
            "n_false_negative": n_fn,
            "n_partial": n_partial,
        },
        "matches": [{"gold_id": m["gold"]["id"], "score": m["score"]} for m in matches],
        "wrong_categories": [
            {"gold_id": w["gold"]["id"], "extracted_category": w["ext"]["category"], "gold_category": w["gold"]["category"]}
            for w in wrong_cats
        ],
        "false_negatives": [{"gold_id": g["id"], "text": g["text"]} for g in false_negatives],
        "false_positives": [{"text": fp["text"], "category": fp["category"]} for fp in false_positives],
        "partials": [
            {"gold_id": p["gold"]["id"], "score": p["score"], "duplicate": p.get("duplicate", False)}
            for p in partials
        ],
    }
    out_path = doc_dir / "eval_eligibility.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"=> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
