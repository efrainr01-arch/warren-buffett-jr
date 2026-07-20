"""Handoff validation and mandatory overrides — Cerebro/shared/
HANDOFF_CONTRACT.md and Cerebro/00_main_agent/SCORING_AND_GATES.md
"Mandatory overrides" (7 rules).
"""

from __future__ import annotations

from pydantic import BaseModel

from wbj.core.nullstates import NullState
from wbj.specialists.common import SpecialistOutput


def validate_handoff(output: SpecialistOutput) -> list[str]:
    """HANDOFF_CONTRACT.md's rejection conditions. Returns a list of
    rejection reasons; empty means the handoff is accepted."""
    reasons: list[str] = []

    dim_points = [
        d["max_points"] * (d["score_10"] / 10.0) for d in output.dimensions if d.get("score_10") is not None
    ]
    if dim_points and output.category.awarded_points is not None:
        if abs(sum(dim_points) - output.category.awarded_points) > 1e-6:
            reasons.append("category points do not reproduce from dimension scores")

    for m in output.metrics:
        if not m.formula:
            reasons.append(f"metric {m.metric_id!r} lacks a formula ID")

    if not output.knowledge_timestamp:
        reasons.append("knowledge timestamp is absent")
    if output.category.confidence is None:
        reasons.append("confidence is absent")
    if output.coverage is None:
        reasons.append("coverage is absent")

    return reasons


class Override(BaseModel):
    id: str
    name: str
    condition_met: bool
    action: str


def apply_overrides(outputs: dict[str, SpecialistOutput], packet=None) -> list[Override]:
    """The 7 mandatory overrides. `outputs` is keyed by agent_id (only
    the agents actually run need be present -- missing agents simply
    can't trigger their associated override)."""
    financial = outputs.get("financial_analysis")
    business = outputs.get("business_analysis")
    risk = outputs.get("risk_analysis")
    valuation = outputs.get("valuation_analysis")
    technical = outputs.get("technical_momentum")

    overrides: list[Override] = []

    capital_dependence = bool(financial and any("OVERRIDE_1_CANDIDATE" in f for f in financial.mandatory_overrides))
    overrides.append(
        Override(id="capital_dependence", name="Capital dependence override", condition_met=capital_dependence,
                  action="net loss + negative FCF + external-capital dependence caps the profile at Avoid/Speculative")
    )

    value_creation = bool(business and "VALUE_DESTRUCTION" in business.mandatory_flags)
    overrides.append(
        Override(id="value_creation", name="Value-creation override", condition_met=value_creation,
                  action="ROIC below WACC prevents Elite/Quality Opportunity/Excellent business")
    )

    solvency = bool(
        (financial and "SOLVENCY_WARNING" in financial.mandatory_flags)
        or (risk and "SOLVENCY_WARNING" in risk.mandatory_flags)
    )
    overrides.append(
        Override(id="solvency_warning", name="Solvency warning", condition_met=solvency,
                  action="interest coverage below 1.5x always displayed prominently")
    )

    risk_override = bool(risk and risk.category.awarded_points is not None and risk.category.awarded_points <= 4.0)
    overrides.append(Override(id="risk_override", name="Risk override", condition_met=risk_override, action="caps the profile at Speculative"))

    premium_breakdown = bool(
        valuation and technical
        and valuation.category.awarded_points is not None and valuation.category.awarded_points <= 4.0
        and technical.category.awarded_points is not None and technical.category.awarded_points <= 8.0
    )
    overrides.append(
        Override(id="premium_breakdown", name="Premium breakdown override", condition_met=premium_breakdown, action="becomes Wait/Avoid")
    )

    low_coverage = any(o.coverage is not None and o.coverage < 0.70 for o in outputs.values())
    overrides.append(
        Override(id="coverage_override", name="Coverage override", condition_met=low_coverage, action="cannot pass a profile gate")
    )

    conflicted_or_missing = False
    if packet is not None:
        critical = ("diluted_shares", "total_debt", "cash", "price")
        for field_name in critical:
            v = packet.facts_table.get(field_name)
            if v is not None and (v.is_null and v.state in (NullState.CONFLICTED, NullState.MISSING)):
                conflicted_or_missing = True
                break
    overrides.append(
        Override(id="data_conflict", name="Data-conflict override", condition_met=conflicted_or_missing,
                  action="unresolved/missing material share-count, debt, cash, or price prevents per-share valuation publication")
    )

    return overrides
