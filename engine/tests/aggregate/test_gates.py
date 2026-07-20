"""Tests for wbj.aggregate.gates and wbj.aggregate.overrides, per
Cerebro/00_main_agent/VALIDATION_TESTS.md (MAIN-002/003/007/009) and the
plan's momentum-gate exact-thresholds test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wbj.aggregate.gates import apply_gates, raw_total
from wbj.aggregate.overrides import Override, apply_overrides


def _fake_output(agent_id, points, max_points, coverage=0.90, mandatory_flags=None, mandatory_overrides=None):
    return SimpleNamespace(
        agent_id=agent_id,
        category=SimpleNamespace(awarded_points=points, max_points=max_points),
        coverage=coverage,
        mandatory_flags=mandatory_flags or [],
        mandatory_overrides=mandatory_overrides or [],
    )


_ALL_CONFIDENT = {"business": 80, "financial": 80, "market": 80, "technical": 80, "risk": 80, "valuation": 80}


# --- MAIN-002: raw total -------------------------------------------------


def test_MAIN_002_raw_total():
    assert raw_total([16, 10.5, 18, 16, 9, 7]) == 76.5


# --- MAIN-003: risk<=4/15 caps Speculative regardless of total ----------


def test_MAIN_003_risk_cap():
    outputs = {"risk_analysis": _fake_output("risk_analysis", 4.0, 15.0)}
    overrides = apply_overrides(outputs)
    result = apply_gates(raw=90.0, cats={"business": 20, "financial": 15, "market": 20, "technical": 20, "risk": 4, "valuation": 10}, confidences=_ALL_CONFIDENT, overrides=overrides)
    assert result.label == "Speculative"
    assert "risk_override" in result.override_ids


# --- MAIN-007: any category coverage <0.70 blocks every gate -----------


def test_MAIN_007_low_coverage_blocks_gates():
    outputs = {
        "financial_analysis": _fake_output("financial_analysis", 14.0, 15.0, coverage=0.65),
        "business_analysis": _fake_output("business_analysis", 18.0, 20.0, coverage=0.95),
    }
    overrides = apply_overrides(outputs)
    # Otherwise gate-passing numbers, but coverage_override must veto all of them.
    result = apply_gates(
        raw=85.0,
        cats={"business": 18, "financial": 14, "market": 18, "technical": 18, "risk": 12, "valuation": 8},
        confidences=_ALL_CONFIDENT,
        overrides=overrides,
    )
    assert result.passed_gates == []
    assert all(any("coverage_override" in r for r in f["reasons"]) for f in result.failed_gates)


# --- MAIN-009: missing share count suppresses per-share valuation ------


def test_MAIN_009_conflict_suppresses_per_share():
    from wbj.core.nullstates import NullState, Value

    class FakePacket:
        facts_table = {"diluted_shares": Value.null(NullState.MISSING, unit="shares")}

    overrides = apply_overrides({}, packet=FakePacket())
    data_conflict = next(o for o in overrides if o.id == "data_conflict")
    assert data_conflict.condition_met is True
    assert "per-share" in data_conflict.action


# --- momentum gate: exact thresholds ------------------------------------


def _momentum_cats():
    return {"business": 18, "financial": 10, "market": 16, "technical": 17, "risk": 8, "valuation": 9}


def test_momentum_gate_passes_at_exact_thresholds():
    result = apply_gates(raw=78.0, cats=_momentum_cats(), confidences=_ALL_CONFIDENT, overrides=[])
    assert "Momentum Candidate" in result.passed_gates


def test_momentum_gate_fails_just_below_raw_threshold():
    result = apply_gates(raw=77.9, cats=_momentum_cats(), confidences=_ALL_CONFIDENT, overrides=[])
    assert "Momentum Candidate" not in result.passed_gates
    momentum_failure = next(f for f in result.failed_gates if f["gate"] == "Momentum Candidate")
    assert "raw_total<78" in momentum_failure["reasons"]


def test_momentum_gate_requires_technical_confidence_70():
    confidences = dict(_ALL_CONFIDENT, technical=69.9)
    result = apply_gates(raw=78.0, cats=_momentum_cats(), confidences=confidences, overrides=[])
    momentum_failure = next(f for f in result.failed_gates if f["gate"] == "Momentum Candidate")
    assert "technical_confidence<70" in momentum_failure["reasons"]


# --- Avoid/Wait and value-creation override interactions --------------


def test_capital_dependence_override_forces_avoid_wait():
    outputs = {"financial_analysis": _fake_output("financial_analysis", 5.0, 15.0, mandatory_overrides=["OVERRIDE_1_CANDIDATE: ..."])}
    overrides = apply_overrides(outputs)
    result = apply_gates(raw=90.0, cats={"business": 20, "financial": 5, "market": 20, "technical": 20, "risk": 15, "valuation": 10}, confidences=_ALL_CONFIDENT, overrides=overrides)
    assert result.label == "Avoid/Wait"


def test_value_creation_override_blocks_quality_label():
    outputs = {"business_analysis": _fake_output("business_analysis", 18.0, 20.0, mandatory_flags=["VALUE_DESTRUCTION"])}
    overrides = apply_overrides(outputs)
    cats = {"business": 18, "financial": 12, "market": 18, "technical": 14, "risk": 12, "valuation": 6}
    result = apply_gates(raw=80.0, cats=cats, confidences=_ALL_CONFIDENT, overrides=overrides)
    quality_failure = next(f for f in result.failed_gates if f["gate"] == "Quality Opportunity")
    assert any("value_creation_override" in r for r in quality_failure["reasons"])
