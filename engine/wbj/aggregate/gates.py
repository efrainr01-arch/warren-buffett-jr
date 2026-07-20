"""Profile gates — Cerebro/00_main_agent/SCORING_AND_GATES.md.

Raw-score descriptive bands, the three profile gates (Momentum
Candidate, Quality Opportunity, Value Opportunity) with their exact
thresholds, and the Speculative / Conditional-Watch / Avoid-Wait
fallback logic that folds in the Task-21 mandatory overrides.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from wbj.aggregate.overrides import Override
from wbj.core.confidence import total_confidence

RAW_BANDS: list[tuple[float, str]] = [
    (90, "Elite raw score"),
    (80, "Strong raw score"),
    (70, "Conditional raw score"),
    (60, "Mixed / wait"),
    (50, "Weak"),
]


def raw_total(category_points: list[float]) -> float:
    """MAIN-002."""
    return sum(category_points)


def raw_band(raw: float) -> str:
    for threshold, label in RAW_BANDS:
        if raw >= threshold:
            return label
    return "Avoid on raw score"


_GATE_CHECKS: dict[str, list[tuple[str, str]]] = {
    # gate name -> list of (predicate-key, failure-reason); predicates are
    # evaluated in apply_gates against `raw`/`cats`/`confidences`.
    "Momentum Candidate": [
        ("raw>=78", "raw_total<78"),
        ("technical>=17", "technical<17"),
        ("market>=16", "market<16"),
        ("business+financial>=28", "business+financial<28"),
        ("risk>=8", "risk<8"),
        ("technical_confidence>=70", "technical_confidence<70"),
    ],
    "Quality Opportunity": [
        ("raw>=80", "raw_total<80"),
        ("business>=16", "business<16"),
        ("financial>=11", "financial<11"),
        ("risk>=10", "risk<10"),
        ("valuation>=5", "valuation<5"),
        ("technical>=12", "technical<12"),
    ],
    "Value Opportunity": [
        ("raw>=75", "raw_total<75"),
        ("valuation>=8", "valuation<8"),
        ("business>=13", "business<13"),
        ("risk>=10", "risk<10"),
        ("technical>=9", "technical<9"),
    ],
}


def _predicate_holds(key: str, raw: float, cats: dict[str, float], confidences: dict[str, float]) -> bool:
    if key.startswith("raw>="):
        return raw >= float(key.split(">=")[1])
    if key == "business+financial>=28":
        return cats.get("business", 0) + cats.get("financial", 0) >= 28
    if key == "technical_confidence>=70":
        return confidences.get("technical", 0) >= 70
    cat, threshold = key.split(">=")
    return cats.get(cat, 0) >= float(threshold)


class ProfileResult(BaseModel):
    label: str
    raw_score: float
    raw_band: str
    total_confidence: float
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[dict] = Field(default_factory=list)
    override_ids: list[str] = Field(default_factory=list)


def apply_gates(raw: float, cats: dict[str, float], confidences: dict[str, float], overrides: list[Override]) -> ProfileResult:
    """SCORING_AND_GATES.md's three profile gates plus the Speculative /
    Conditional-Watch / Avoid-Wait fallback ladder."""
    total_conf = total_confidence(confidences)
    override_by_id = {o.id: o for o in overrides}
    coverage_blocked = bool(override_by_id.get("coverage_override") and override_by_id["coverage_override"].condition_met)
    value_creation_blocked = bool(override_by_id.get("value_creation") and override_by_id["value_creation"].condition_met)

    passed: list[str] = []
    failed: list[dict] = []
    for gate_name, checks in _GATE_CHECKS.items():
        reasons = [reason for key, reason in checks if not _predicate_holds(key, raw, cats, confidences)]
        if coverage_blocked:
            reasons.append("coverage_override: a category is below 70% coverage")
        if gate_name == "Quality Opportunity" and value_creation_blocked:
            reasons.append("value_creation_override: ROIC<WACC blocks Quality/Elite")
        if reasons:
            failed.append({"gate": gate_name, "reasons": reasons})
        else:
            passed.append(gate_name)

    avoid_override = any(
        override_by_id[i].condition_met for i in ("capital_dependence", "premium_breakdown") if i in override_by_id
    )
    risk_override_active = bool(override_by_id.get("risk_override") and override_by_id["risk_override"].condition_met)

    if avoid_override or raw < 50:
        label = "Avoid/Wait"
    elif risk_override_active or total_conf < 60:
        label = "Speculative"
    elif passed:
        label = " / ".join(passed)
    elif raw >= 60:
        label = "Conditional/Watch"
    else:
        label = "Avoid/Wait"

    return ProfileResult(
        label=label,
        raw_score=raw,
        raw_band=raw_band(raw),
        total_confidence=total_conf,
        passed_gates=passed,
        failed_gates=failed,
        override_ids=[o.id for o in overrides if o.condition_met],
    )
