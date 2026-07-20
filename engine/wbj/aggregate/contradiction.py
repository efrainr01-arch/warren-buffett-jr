"""Contradiction resolution — Cerebro/00_main_agent/CONTRADICTION_RESOLUTION.md.

"Contradictions are information, not errors to smooth away" -- these
functions only *label* tension between category scores; they never
mutate a category's score (Resolution rule 1).

Cerebro's table describes each combination qualitatively ("strong",
"weak") without numeric thresholds. This module uses a disclosed,
documented split: "strong" = category points >= 70% of its max, "weak"
= category points <= 40% of its max -- consistent with the raw-score
descriptive bands' 70/50 boundaries used elsewhere in SCORING_AND_GATES.md.
"""

from __future__ import annotations

STRONG_THRESHOLD_PCT = 0.70
WEAK_THRESHOLD_PCT = 0.40


def _pct(points: float, max_points: float) -> float:
    return points / max_points if max_points else 0.0


def _is_strong(points: float, max_points: float) -> bool:
    return _pct(points, max_points) >= STRONG_THRESHOLD_PCT


def _is_weak(points: float, max_points: float) -> bool:
    return _pct(points, max_points) <= WEAK_THRESHOLD_PCT


def contradictions(cats: dict[str, float], max_points: dict[str, float]) -> list[dict]:
    """The 6-row lookup table. `cats`/`max_points` are keyed by category
    name (business/financial/market/technical/risk/valuation). Returns
    only the rows whose condition is actually met."""
    business, technical = cats.get("business", 0), cats.get("technical", 0)
    valuation, risk = cats.get("valuation", 0), cats.get("risk", 0)
    market = cats.get("market", 0)
    bmax, tmax = max_points.get("business", 20), max_points.get("technical", 20)
    vmax, rmax = max_points.get("valuation", 10), max_points.get("risk", 15)
    mmax = max_points.get("market", 20)

    found: list[dict] = []

    if _is_strong(business, bmax) and _is_weak(technical, tmax):
        found.append({
            "combination": "Strong business, weak technical",
            "interpretation": "Quality may be intact; timing is unconfirmed",
            "label": "Quality watch / wait for confirmation",
        })
    if _is_weak(business, bmax) and _is_strong(technical, tmax):
        found.append({
            "combination": "Weak business, strong technical",
            "interpretation": "Price leadership without durable economics",
            "label": "Speculative momentum only",
        })
    if _is_strong(valuation, vmax) and _is_weak(technical, tmax):
        found.append({
            "combination": "Strong valuation, weak technical",
            "interpretation": "Possible value trap",
            "label": "Value watch",
        })
    if _is_weak(valuation, vmax) and _is_strong(market, mmax) and _is_strong(technical, tmax):
        found.append({
            "combination": "Expensive valuation, strong growth and technical",
            "interpretation": "Premium still validated",
            "label": "Momentum candidate if gates pass",
        })
    total_pct = sum(cats.values()) / sum(max_points.values()) if sum(max_points.values()) else 0.0
    if total_pct >= STRONG_THRESHOLD_PCT and _is_weak(risk, rmax):
        found.append({
            "combination": "Strong total, low risk score",
            "interpretation": "Aggregate hides survival risk",
            "label": "Apply risk override",
        })

    return found


def dcf_reverse_dcf_contradiction(valuation_output) -> dict | None:
    """The 6th row -- 'DCF high, reverse DCF demanding' -- needs the
    valuation specialist's own DCF value vs. its reverse-DCF implied
    growth, not just category points, so it's a separate check rather
    than a `contradictions()` row keyed on category scores."""
    base_value = valuation_output.reference_bands.get("base")
    price = (valuation_output.reverse_dcf or {}).get("current_price")
    implied_growth = (valuation_output.reverse_dcf or {}).get("implied_revenue_cagr")
    if base_value is None or price is None or implied_growth is None:
        return None
    if base_value > price and implied_growth > 0.15:  # disclosed "demanding" threshold
        return {
            "combination": "DCF high, reverse DCF demanding",
            "interpretation": "Model assumptions may be optimistic",
            "label": "Lower valuation confidence",
        }
    return None
