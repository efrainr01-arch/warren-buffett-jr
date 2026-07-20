"""Tests for wbj.specialists.market, per Cerebro/03_market_analysis/
VALIDATION_TESTS.md (MKT-T001..T008).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.schemas.packet import Packet
from wbj.specialists.market import (
    SOURCE_TIER_SCORE_CAP,
    catalyst_impact_index,
    earnings_surprise,
    forecast_consistency_gate,
    penetration,
    revision_breadth,
    run,
    share_delta,
    tam_cagr,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def test_MKT_T001_tam_cagr():
    v = tam_cagr(tam_end=1210.0, tam_begin=1000.0, years=2)
    assert v.value == pytest.approx(0.10)


def test_MKT_T002_penetration():
    v = penetration(company_revenue=50.0, tam=1000.0)
    assert v.value == pytest.approx(0.05)


def test_MKT_T003_share_delta():
    assert share_delta(0.057, 0.050) == pytest.approx(0.007)


def test_MKT_T004_revision_breadth():
    v = revision_breadth(upward_count=8, total_count=10)
    assert v.value == pytest.approx(0.80)


def test_MKT_T004b_revision_breadth_requires_5_estimates():
    v = revision_breadth(upward_count=2, total_count=3)
    assert v.is_null


def test_MKT_T005_catalyst_impact_index():
    assert catalyst_impact_index(probability=0.6, impact=100.0, evidence_quality=0.8, time_decay_factor=0.5) == pytest.approx(24.0)


def test_MKT_T006_forecast_exceeds_tam_fails_gate():
    assert forecast_consistency_gate(forecast_revenue=1200.0, tam=1000.0) is False
    assert forecast_consistency_gate(forecast_revenue=800.0, tam=1000.0) is True


def test_MKT_T007_issuer_tam_source_tier_4_caps_score_at_6():
    assert SOURCE_TIER_SCORE_CAP[4] == 6.0

    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet, overlay={"tam": 500_000_000_000, "tam_source_tier": 4})
    tam_dim = next(d for d in out.dimensions if d["name"] == "tam_and_industry_tailwind")
    assert tam_dim["score_10"] <= 6.0


def test_MKT_T008_surprise_rejected_when_snapshot_after_release():
    v = earnings_surprise(actual=1.0, pre_release_consensus=0.9, snapshot_before_earnings=False)
    assert v.is_null
    assert any("REJECTED" in w for w in v.warnings)


def test_earnings_surprise_computed_when_frozen_before_release():
    v = earnings_surprise(actual=0.81, pre_release_consensus=0.75, snapshot_before_earnings=True)
    assert v.value == pytest.approx((0.81 - 0.75) / 0.75)


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet)

    assert out.agent_id == "market_analysis"
    assert out.category.max_points == 20
    assert len(out.dimensions) == 5
    assert 0.0 <= out.coverage <= 1.0
    # TAM figure/tier weren't supplied -> must ask.
    assert any(j.metric_id == "tam_and_source_tier" for j in out.judgment_requests)
    # No catalysts supplied -> must ask, and the dimension is narrative-capped at 3.
    assert any(j.metric_id == "catalysts" for j in out.judgment_requests)
    catalysts_dim = next(d for d in out.dimensions if d["name"] == "product_and_business_catalysts")
    assert catalysts_dim["score_10"] == pytest.approx(3.0)

    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6
