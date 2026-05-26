"""Vector-arrow detection from PDF drawing operations.

Why this exists: table extractors (MinerU, Docling, pdfplumber, etc.) recover
cells but drop **vector annotations between cells** — duration spans, flow
indicators, "from→to" relationships. In clinical-protocol Schedule-of-Activities
tables, those arrows often carry critical semantics (e.g. "this procedure
spans Day 1 through Day 29").

Approach (corpus-generic, no document-specific tuning):
  1. Walk `page.get_drawings()` and collect every short line segment.
  2. Cluster line endpoints within `tol` points → group lines meeting at a
     common point. A pair of short lines (≈5–20pt long) meeting at a single
     point with a wide opening angle (~120°) is an arrowhead.
  3. For each arrowhead, infer its TIP and DIRECTION (the unit vector from
     the bisector of the two opening lines pointing toward the tip).
  4. Pair arrowheads facing each other along a roughly horizontal or vertical
     axis → a full arrow (span). Or a single arrowhead with its shaft inferred
     from a nearby long line → directional arrow.
  5. Return a list of normalized Arrow records.

The detector is intentionally conservative: false negatives (missed arrows)
are easier to spot than false positives (random short lines mistaken for
arrows). We can tune later from corpus evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

# Type stub friendliness
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


# ---------- data ----------


@dataclass
class Arrowhead:
    """One detected arrowhead.

    tip:        (x, y) of the point where the two lines meet (the arrow tip).
    direction:  unit vector (dx, dy) from the average opening direction toward
                the tip. So an arrowhead `>` (pointing right) has direction
                ≈ (1, 0). `<` ≈ (-1, 0). `^` ≈ (0, -1). `v` ≈ (0, 1).
    leg_length: average length of the two legs in pt.
    """

    tip: tuple[float, float]
    direction: tuple[float, float]
    leg_length: float


@dataclass
class Arrow:
    """A full arrow connecting two points.

    - For two-headed `↔` arrows: `start` and `end` are both arrowhead tips.
    - For one-headed `→` arrows: `start` is the shaft tail, `end` is the tip.
    - `axis` is "horizontal" if |dx| > |dy|, else "vertical".
    """

    start: tuple[float, float]
    end: tuple[float, float]
    axis: str
    two_headed: bool


# ---------- helpers ----------


def _length(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _norm(v) -> tuple[float, float]:
    L = math.hypot(v[0], v[1])
    if L < 1e-6:
        return (0.0, 0.0)
    return (v[0] / L, v[1] / L)


def _collect_line_segments(page) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Walk drawings, return [((x1,y1),(x2,y2)), …] for every 'l' segment."""
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for dr in page.get_drawings() or []:
        last_pt: Optional[tuple[float, float]] = None
        for it in dr.get("items") or []:
            if not it:
                continue
            op = it[0]
            if op == "l" and len(it) >= 3:
                p1 = (float(it[1].x), float(it[1].y))
                p2 = (float(it[2].x), float(it[2].y))
                out.append((p1, p2))
                last_pt = p2
            elif op == "m" and len(it) >= 2:
                last_pt = (float(it[1].x), float(it[1].y))
    return out


# ---------- public API ----------


def detect_arrowheads(
    page,
    min_leg: float = 3.0,
    max_leg: float = 20.0,
    tol: float = 1.0,
    opening_angle_min_deg: float = 20.0,
    opening_angle_max_deg: float = 170.0,
) -> list[Arrowhead]:
    """Return all arrowheads detected on the page.

    Heuristics, generic to any PDF:
      - leg must be 3–20pt (typical arrowhead size in a paper-sized doc)
      - lines must meet at a common point within `tol` pt
      - the angle between the two legs must be in [opening_angle_min, max]
        (true arrowheads typically open ~110–140°)
    """
    if fitz is None:
        return []

    segs = _collect_line_segments(page)
    # Filter to segments of arrowhead-leg length, capture non-axis-aligned ones
    short_segs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for p1, p2 in segs:
        L = _length(p1, p2)
        if L < min_leg or L > max_leg:
            continue
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        if dx < 0.5 or dy < 0.5:
            continue  # axis-aligned line, not arrowhead leg
        short_segs.append((p1, p2, L))

    # Index segment endpoints into rounded buckets so we can find shared points.
    bucket_size = max(tol, 0.5)
    endpoints: dict[tuple[int, int], list[tuple[int, tuple[float, float], tuple[float, float]]]] = {}
    for idx, (p1, p2, L) in enumerate(short_segs):
        for end_idx, ep in enumerate((p1, p2)):
            key = (int(ep[0] / bucket_size), int(ep[1] / bucket_size))
            other = p2 if end_idx == 0 else p1
            endpoints.setdefault(key, []).append((idx, ep, other))

    found: list[Arrowhead] = []
    seen_pairs: set[tuple[int, int]] = set()
    for key, group in endpoints.items():
        if len(group) < 2:
            continue
        # Try each pair sharing this approximate endpoint.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a_idx, a_tip, a_other = group[i]
                b_idx, b_tip, b_other = group[j]
                if a_idx == b_idx:
                    continue
                pair_key = (min(a_idx, b_idx), max(a_idx, b_idx))
                if pair_key in seen_pairs:
                    continue
                # Are tips actually close?
                if _length(a_tip, b_tip) > tol:
                    continue
                # Direction of each leg (from tip toward the other endpoint).
                va = _norm((a_other[0] - a_tip[0], a_other[1] - a_tip[1]))
                vb = _norm((b_other[0] - b_tip[0], b_other[1] - b_tip[1]))
                # Opening angle between legs (the inside of the arrowhead).
                dot = max(-1.0, min(1.0, va[0] * vb[0] + va[1] * vb[1]))
                opening_rad = math.acos(dot)
                opening_deg = math.degrees(opening_rad)
                if opening_deg < opening_angle_min_deg or opening_deg > opening_angle_max_deg:
                    continue
                # Direction = unit vector from average-of-legs toward tip.
                avg = ((va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0)
                # The tip lies opposite the legs' average direction.
                direction = _norm((-avg[0], -avg[1]))
                if direction == (0.0, 0.0):
                    continue
                leg_len = (short_segs[a_idx][2] + short_segs[b_idx][2]) / 2.0
                tip = ((a_tip[0] + b_tip[0]) / 2.0, (a_tip[1] + b_tip[1]) / 2.0)
                found.append(Arrowhead(tip=tip, direction=direction, leg_length=leg_len))
                seen_pairs.add(pair_key)
    return found


def pair_arrowheads_into_arrows(
    arrowheads: list[Arrowhead],
    same_axis_tol: float = 3.0,
    min_span: float = 10.0,
) -> list[Arrow]:
    """Pair arrowheads on the same horizontal or vertical line into one Arrow.

    We accept BOTH patterns:
      `→ ←`  (point toward each other)  — sometimes seen for "from-to" arrows
      `← →`  (point away from each other) — common SoA "duration / span" arrow

    What matters is that two arrowheads sit on the same line and have
    opposite axis-direction signs (one with dx<0, the other dx>0) — that
    geometrically describes the arrow's extent. Single unpaired arrowheads
    are emitted as one-headed arrows with a short inferred shaft.
    """
    arrows: list[Arrow] = []
    used: set[int] = set()
    for i, a in enumerate(arrowheads):
        if i in used:
            continue
        ax_h = abs(a.direction[0]) >= abs(a.direction[1])
        best_j: Optional[int] = None
        best_dist: float = float("inf")
        for j in range(i + 1, len(arrowheads)):
            if j in used:
                continue
            b = arrowheads[j]
            bx_h = abs(b.direction[0]) >= abs(b.direction[1])
            if ax_h != bx_h:
                continue
            if ax_h:
                if abs(a.tip[1] - b.tip[1]) > same_axis_tol:
                    continue
                if (a.direction[0] > 0) == (b.direction[0] > 0):
                    continue  # both point same way — not a span
                span = abs(a.tip[0] - b.tip[0])
                if span < min_span:
                    continue
            else:
                if abs(a.tip[0] - b.tip[0]) > same_axis_tol:
                    continue
                if (a.direction[1] > 0) == (b.direction[1] > 0):
                    continue
                span = abs(a.tip[1] - b.tip[1])
                if span < min_span:
                    continue
            # Prefer the nearest partner on the same axis (in case of multiple).
            if span < best_dist:
                best_dist = span
                best_j = j
        if best_j is not None:
            b = arrowheads[best_j]
            arrows.append(Arrow(
                start=a.tip,
                end=b.tip,
                axis="horizontal" if ax_h else "vertical",
                two_headed=True,
            ))
            used.add(i)
            used.add(best_j)

    for i, a in enumerate(arrowheads):
        if i in used:
            continue
        arrows.append(Arrow(
            start=(a.tip[0] - a.direction[0] * 20, a.tip[1] - a.direction[1] * 20),
            end=a.tip,
            axis="horizontal" if abs(a.direction[0]) >= abs(a.direction[1]) else "vertical",
            two_headed=False,
        ))
    return arrows


def detect_arrows_on_page(page) -> list[Arrow]:
    """One-shot helper: arrowheads + pairing for one page."""
    return pair_arrowheads_into_arrows(detect_arrowheads(page))
