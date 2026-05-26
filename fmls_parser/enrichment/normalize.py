"""Normalization helpers — text / operators / units / frequency / route.

Rule: NEVER destructively normalize. Always store both raw and canonical.

Coverage:
  - Unicode NFKC + whitespace + smart-quote/dash fixes (universal)
  - Operator normalization (>=, <=, etc) (universal)
  - Identifier format normalization (NCT, EudraCT, DOI, LEI) (universal)
  - Number normalization (spelled-out, decimals, ranges) (universal)
  - UCUM unit mapping (clinical/scientific)
  - FHIR Timing mapping for clinical frequencies (BID, TID, QD, ...)
  - SNOMED route code mapping (PO, IV, SC, ...)
  - CTCAE / ECOG / performance-status normalization (clinical)
  - Negation kind classification (clinical)

The "clinical specific" maps live here for now because they're small and
co-located keeps things obvious. If/when we add legal/industrial, split
into normalize_universal.py + normalize_clinical.py + normalize_legal.py.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from text_to_num import alpha2digit


# === Text-level normalization (universal) ===========================

_SMART_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
})

_DASH_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-", "―": "-",
    "−": "-",  # minus sign
})

_NBSP_WS = re.compile(r"[     ​]")
_MULTI_WS = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_text(s: str) -> str:
    """Universal text normalization. Stores result alongside raw text.

    Steps:
      1. Unicode NFKC (compose combining chars, decompose compat chars)
      2. Smart quotes/dashes → ASCII equivalents
      3. NBSP and other zero-width/thin spaces → regular space
      4. Collapse runs of internal whitespace (preserves single newlines)
      5. Collapse 3+ newlines to 2 (keeps paragraph structure)
    """
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_SMART_QUOTE_MAP)
    s = s.translate(_DASH_MAP)
    s = _NBSP_WS.sub(" ", s)
    s = _MULTI_WS.sub(" ", s)
    s = _MULTI_NL.sub("\n\n", s)
    return s.strip()


def casefold_key(s: str) -> str:
    """Aggressive lower-cased key for internal matching only.

    Handles Turkish dotless i, German ß, full Unicode case mappings.
    Not for display; use only for hash/dedup/equality comparisons.
    """
    return normalize_text(s).casefold()


# === Operator normalization (universal) =============================

_OP_NORMALIZE = {
    "≥": ">=", "≤": "<=", "≠": "!=",
    "⩾": ">=", "⩽": "<=",
    "≧": ">=", "≦": "<=",
    "<=": "<=", ">=": ">=", "!=": "!=", "==": "=",
}

# English phrasal operators -> canonical form
_OP_PHRASE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bat\s+least\s+", re.IGNORECASE), ">= "),
    (re.compile(r"\bno\s+(?:more|fewer|less)\s+than\s+", re.IGNORECASE), "<= "),
    (re.compile(r"\bgreater\s+than\s+or\s+equal\s+to\s+", re.IGNORECASE), ">= "),
    (re.compile(r"\bless\s+than\s+or\s+equal\s+to\s+", re.IGNORECASE), "<= "),
    (re.compile(r"\bgreater\s+than\s+", re.IGNORECASE), "> "),
    (re.compile(r"\bless\s+than\s+", re.IGNORECASE), "< "),
    (re.compile(r"\bmore\s+than\s+", re.IGNORECASE), "> "),
    (re.compile(r"\bup\s+to\s+", re.IGNORECASE), "<= "),
    (re.compile(r"\bminimum\s+(?:of\s+)?", re.IGNORECASE), ">= "),
    (re.compile(r"\bmaximum\s+(?:of\s+)?", re.IGNORECASE), "<= "),
    (re.compile(r"\bnot\s+exceeding\s+", re.IGNORECASE), "<= "),
    (re.compile(r"\bexceeding\s+", re.IGNORECASE), "> "),
    (re.compile(r"\bbelow\s+", re.IGNORECASE), "< "),
    (re.compile(r"\babove\s+", re.IGNORECASE), "> "),
]


def normalize_operators(s: str) -> str:
    """Map ≥/≤/≠ and English phrasal operators to canonical >=, <=, !=.

    Conservative: only rewrites well-known patterns. Falls back to raw text.
    """
    if not s:
        return s
    for symbol, canonical in _OP_NORMALIZE.items():
        s = s.replace(symbol, canonical)
    for pattern, replacement in _OP_PHRASE_PATTERNS:
        s = pattern.sub(replacement, s)
    return s


# === Number normalization (universal) ==============================

def normalize_numbers(s: str, lang: str = "en") -> str:
    """Convert spelled-out numbers to digits in-place ("eighteen" -> "18").

    Uses text_to_num.alpha2digit which respects context. Idempotent on
    already-numeric strings. Returns original on failure.
    """
    try:
        return alpha2digit(s, lang)
    except Exception:
        return s


# === Identifier format normalization (universal) ===================

# NCT (ClinicalTrials.gov): NCT + 8 digits, total 11 chars
_NCT_PATTERN = re.compile(r"\bNCT[\s\-_]*0*(\d{1,8})\b", re.IGNORECASE)
# EudraCT: YYYY-NNNNNN-CC
_EUDRACT_PATTERN = re.compile(r"\b(\d{4})[\s\-]+(\d{6})[\s\-]+(\d{2})\b")
# DOI
_DOI_PATTERN = re.compile(r"\b(?:doi[:\s]+)?(10\.\d{4,9}/[^\s]+)", re.IGNORECASE)
# LEI: 20 chars
_LEI_PATTERN = re.compile(r"\b([A-Z0-9]{20})\b")


def normalize_identifier(s: str, kind: Optional[str] = None) -> str:
    """Format-normalize an identifier surface form. Best effort."""
    if not s:
        return s
    s = s.strip()
    if kind == "nct" or s.upper().startswith("NCT"):
        m = _NCT_PATTERN.search(s)
        if m:
            return f"NCT{int(m.group(1)):08d}"
    if kind == "eudract":
        m = _EUDRACT_PATTERN.search(s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if kind == "doi":
        m = _DOI_PATTERN.search(s)
        if m:
            return m.group(1)
    return s


# === UCUM unit normalization (clinical) =============================

# Static map of the ~80 most-common clinical/scientific units to UCUM.
# Keys are lowercased + whitespace-collapsed surface forms.
# Values are valid UCUM codes (https://ucum.org/ucum.html).
_UCUM_MAP: dict[str, str] = {
    # Mass
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "mcg": "ug", "microgram": "ug", "micrograms": "ug",
    "ug": "ug", "ng": "ng", "nanogram": "ng",
    "pg": "pg",
    # Volume
    "ml": "mL", "milliliter": "mL", "milliliters": "mL",
    "l": "L", "liter": "L", "liters": "L",
    "dl": "dL", "deciliter": "dL",
    "ul": "uL", "microliter": "uL",
    # Mass per volume
    "mg/dl": "mg/dL", "mg per dl": "mg/dL", "mg per deciliter": "mg/dL",
    "mg/l": "mg/L",
    "ng/ml": "ng/mL", "ng per ml": "ng/mL",
    "ug/ml": "ug/mL", "mcg/ml": "ug/mL",
    "g/dl": "g/dL", "g per dl": "g/dL",
    "g/l": "g/L",
    "iu/ml": "[IU]/mL", "iu/l": "[IU]/L",
    "mmol/l": "mmol/L", "mmol per l": "mmol/L",
    "umol/l": "umol/L", "mcmol/l": "umol/L",
    "meq/l": "meq/L",
    "nmol/l": "nmol/L",
    # Per body mass / surface
    "mg/kg": "mg/kg", "mg per kg": "mg/kg",
    "mg/m2": "mg/m2", "mg per m2": "mg/m2", "mg per square meter": "mg/m2",
    "ml/kg": "mL/kg",
    "ml/min": "mL/min",
    "ml/min/1.73m2": "mL/min/{1.73_m2}",
    # Time
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "min": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "d": "d", "day": "d", "days": "d",
    "wk": "wk", "week": "wk", "weeks": "wk",
    "mo": "mo", "month": "mo", "months": "mo",
    "yr": "a", "y": "a", "year": "a", "years": "a", "yo": "a",
    # Ratios / counts
    "%": "%", "percent": "%", "per cent": "%",
    "iu": "[IU]", "u": "U", "unit": "U", "units": "U",
    # Pressure
    "mmhg": "mm[Hg]", "mm hg": "mm[Hg]",
    # Temperature
    "celsius": "Cel", "°c": "Cel", "deg c": "Cel",
    "fahrenheit": "[degF]", "°f": "[degF]",
    # Counts per volume (cell counts)
    "cells/ml": "{cells}/mL", "cells/ul": "{cells}/uL",
    "x10^9/l": "10*9/L", "x10^6/ml": "10*6/mL",
    "/ul": "/uL", "/ml": "/mL", "/l": "/L",
    "k/ul": "10*3/uL", "k/ml": "10*3/mL",
}


def normalize_unit_to_ucum(unit_surface: str) -> Optional[str]:
    """Map a surface unit string to a UCUM code. Returns None if unknown."""
    if not unit_surface:
        return None
    key = unit_surface.strip().lower()
    key = re.sub(r"\s+", " ", key)
    return _UCUM_MAP.get(key)


# === FHIR Timing frequency normalization (clinical) ================

# https://www.hl7.org/fhir/valueset-timing-abbreviation.html
# Mapped to {frequency, period, periodUnit} triple per FHIR Timing.
_FHIR_TIMING_MAP: dict[str, dict] = {
    "qd":   {"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},
    "q.d.": {"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},
    "od":   {"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},
    "daily":{"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},
    "once daily": {"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},
    "every day":  {"frequency": 1, "period": 1, "periodUnit": "d", "label": "QD"},

    "bid":  {"frequency": 2, "period": 1, "periodUnit": "d", "label": "BID"},
    "b.i.d.":{"frequency": 2, "period": 1, "periodUnit": "d", "label": "BID"},
    "twice daily": {"frequency": 2, "period": 1, "periodUnit": "d", "label": "BID"},
    "twice a day": {"frequency": 2, "period": 1, "periodUnit": "d", "label": "BID"},
    "every 12 hours": {"frequency": 1, "period": 12, "periodUnit": "h", "label": "BID"},
    "q12h": {"frequency": 1, "period": 12, "periodUnit": "h", "label": "BID"},

    "tid":  {"frequency": 3, "period": 1, "periodUnit": "d", "label": "TID"},
    "t.i.d.":{"frequency": 3, "period": 1, "periodUnit": "d", "label": "TID"},
    "three times daily": {"frequency": 3, "period": 1, "periodUnit": "d", "label": "TID"},
    "every 8 hours": {"frequency": 1, "period": 8, "periodUnit": "h", "label": "TID"},
    "q8h":  {"frequency": 1, "period": 8, "periodUnit": "h", "label": "TID"},

    "qid":  {"frequency": 4, "period": 1, "periodUnit": "d", "label": "QID"},
    "q.i.d.":{"frequency": 4, "period": 1, "periodUnit": "d", "label": "QID"},
    "four times daily": {"frequency": 4, "period": 1, "periodUnit": "d", "label": "QID"},
    "q6h":  {"frequency": 1, "period": 6, "periodUnit": "h", "label": "QID"},

    "qhs":  {"frequency": 1, "period": 1, "periodUnit": "d", "when": ["HS"], "label": "QHS"},
    "qam":  {"frequency": 1, "period": 1, "periodUnit": "d", "when": ["MORN"], "label": "QAM"},
    "qpm":  {"frequency": 1, "period": 1, "periodUnit": "d", "when": ["EVE"], "label": "QPM"},

    "prn":  {"asNeeded": True, "label": "PRN"},
    "as needed": {"asNeeded": True, "label": "PRN"},

    "weekly": {"frequency": 1, "period": 1, "periodUnit": "wk", "label": "Q1W"},
    "q1w":  {"frequency": 1, "period": 1, "periodUnit": "wk", "label": "Q1W"},
    "qw":   {"frequency": 1, "period": 1, "periodUnit": "wk", "label": "Q1W"},
    "every week": {"frequency": 1, "period": 1, "periodUnit": "wk", "label": "Q1W"},
    "biweekly": {"frequency": 1, "period": 2, "periodUnit": "wk", "label": "Q2W"},
    "q2w":  {"frequency": 1, "period": 2, "periodUnit": "wk", "label": "Q2W"},
    "every 2 weeks": {"frequency": 1, "period": 2, "periodUnit": "wk", "label": "Q2W"},
    "q3w":  {"frequency": 1, "period": 3, "periodUnit": "wk", "label": "Q3W"},
    "every 3 weeks": {"frequency": 1, "period": 3, "periodUnit": "wk", "label": "Q3W"},
    "q4w":  {"frequency": 1, "period": 4, "periodUnit": "wk", "label": "Q4W"},
    "every 4 weeks": {"frequency": 1, "period": 4, "periodUnit": "wk", "label": "Q4W"},
    "monthly": {"frequency": 1, "period": 1, "periodUnit": "mo", "label": "QM"},
}


def normalize_frequency_to_fhir(freq_surface: str) -> Optional[dict]:
    """Map BID/TID/etc. to FHIR Timing dict. Returns None if unknown."""
    if not freq_surface:
        return None
    key = freq_surface.strip().lower()
    return _FHIR_TIMING_MAP.get(key)


# === SNOMED route normalization (clinical) =========================

# Route surface -> SNOMED CT code
# https://www.hl7.org/fhir/valueset-route-codes.html
_ROUTE_SNOMED_MAP: dict[str, dict] = {
    "po":   {"code": "26643006", "display": "Oral route", "system": "SNOMED"},
    "p.o.": {"code": "26643006", "display": "Oral route", "system": "SNOMED"},
    "oral": {"code": "26643006", "display": "Oral route", "system": "SNOMED"},
    "orally": {"code": "26643006", "display": "Oral route", "system": "SNOMED"},
    "by mouth": {"code": "26643006", "display": "Oral route", "system": "SNOMED"},

    "iv":   {"code": "47625008", "display": "Intravenous route", "system": "SNOMED"},
    "i.v.": {"code": "47625008", "display": "Intravenous route", "system": "SNOMED"},
    "intravenous": {"code": "47625008", "display": "Intravenous route", "system": "SNOMED"},
    "intravenously": {"code": "47625008", "display": "Intravenous route", "system": "SNOMED"},

    "sc":   {"code": "34206005", "display": "Subcutaneous route", "system": "SNOMED"},
    "s.c.": {"code": "34206005", "display": "Subcutaneous route", "system": "SNOMED"},
    "sq":   {"code": "34206005", "display": "Subcutaneous route", "system": "SNOMED"},
    "subcutaneous": {"code": "34206005", "display": "Subcutaneous route", "system": "SNOMED"},
    "subcutaneously": {"code": "34206005", "display": "Subcutaneous route", "system": "SNOMED"},

    "im":   {"code": "78421000", "display": "Intramuscular route", "system": "SNOMED"},
    "i.m.": {"code": "78421000", "display": "Intramuscular route", "system": "SNOMED"},
    "intramuscular": {"code": "78421000", "display": "Intramuscular route", "system": "SNOMED"},

    "it":   {"code": "72607000", "display": "Intrathecal route", "system": "SNOMED"},
    "intrathecal": {"code": "72607000", "display": "Intrathecal route", "system": "SNOMED"},

    "topical": {"code": "6064005", "display": "Topical route", "system": "SNOMED"},
    "cutaneous": {"code": "6064005", "display": "Topical route", "system": "SNOMED"},

    "inhaled": {"code": "447694001", "display": "Respiratory inhalation", "system": "SNOMED"},
    "inhalation": {"code": "447694001", "display": "Respiratory inhalation", "system": "SNOMED"},
    "nasal":   {"code": "46713006",  "display": "Nasal route", "system": "SNOMED"},
    "intranasal": {"code": "46713006", "display": "Nasal route", "system": "SNOMED"},

    "ocular":  {"code": "54485002",  "display": "Ophthalmic route", "system": "SNOMED"},
    "ophthalmic": {"code": "54485002", "display": "Ophthalmic route", "system": "SNOMED"},
    "rectal":  {"code": "37161004",  "display": "Rectal route", "system": "SNOMED"},
    "vaginal": {"code": "16857009",  "display": "Vaginal route", "system": "SNOMED"},
    "sublingual": {"code": "37839007", "display": "Sublingual route", "system": "SNOMED"},
    "buccal":  {"code": "54471007",  "display": "Buccal route", "system": "SNOMED"},
}


def normalize_route_to_snomed(route_surface: str) -> Optional[dict]:
    """Map PO/IV/SC/etc. to {code, display, system}. None if unknown."""
    if not route_surface:
        return None
    key = route_surface.strip().lower()
    return _ROUTE_SNOMED_MAP.get(key)


# === Demographic / clinical-scale normalization =====================

_SEX_MAP = {
    "m": "male", "male": "male", "men": "male", "man": "male",
    "f": "female", "female": "female", "women": "female", "woman": "female",
    "intersex": "intersex", "other": "other", "unknown": "unknown",
}


def normalize_sex(s: str) -> Optional[str]:
    if not s:
        return None
    return _SEX_MAP.get(s.strip().lower())


# === Negation kind classification (clinical) =======================

# Maps NegEx/ConText output to a small closed set.
_NEGATION_KIND_NORMALIZE = {
    "definite_negated_existence": "definite",
    "negated": "definite",
    "neg": "definite",
    "definite": "definite",
    "possible_negated_existence": "possible",
    "possible": "possible",
    "uncertain": "possible",
    "historical": "history",
    "history": "history",
    "family_history": "family",
    "family": "family",
    "conditional": "conditional",
}


def normalize_negation_kind(raw: str) -> str:
    """Map medspacy/NegEx raw kind to canonical 5-value set."""
    if not raw:
        return "definite"
    return _NEGATION_KIND_NORMALIZE.get(raw.lower(), "definite")


__all__ = [
    "normalize_text",
    "casefold_key",
    "normalize_operators",
    "normalize_numbers",
    "normalize_identifier",
    "normalize_unit_to_ucum",
    "normalize_frequency_to_fhir",
    "normalize_route_to_snomed",
    "normalize_sex",
    "normalize_negation_kind",
]
