"""Tests for wbj.specialists.business, per Cerebro/01_business_analysis/
VALIDATION_TESTS.md (BUS-T001..T008).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.engines.valuation_engine import eva, nopat, roic, spread
from wbj.schemas.packet import Packet
from wbj.specialists.business import (
    _cumulative_fcf_conversion,
    _diluted_share_cagr,
    _margin_series,
    _revenue_cagr,
    run,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def _packet(annual_rows: list[dict]) -> Packet:
    return Packet.model_validate(
        {
            "security": {"ticker": "TST", "exchange": "NASDAQ", "security_type": "operating_company",
                         "reporting_currency": "USD", "valuation_currency": "USD"},
            "analysis": {"knowledge_timestamp": "2026-01-01T00:00:00Z", "market_timestamp": "2026-01-01",
                         "industry_adapter": "default_nonfinancial"},
            "fundamentals": {"annual": annual_rows, "quarterly": []},
            "market_data": {"daily": [], "benchmark": [], "sector": [], "adjusted": True},
            "estimates": {},
            "capital_structure": {},
            "insiders": [],
            "institutional_holders": [],
            "facts_table": {},
            "staleness": {},
            "packet_hash": "",
        }
    )


# --- BUS-T001/T002: NOPAT / ROIC / spread / EVA (reused from Task 13) -----


def test_BUS_T001_nopat_and_roic():
    n = nopat(norm_ebit=100.0, tax_rate=0.25)
    assert n == pytest.approx(75.0)
    r = roic(n, avg_invested_capital=500.0)
    assert r.value == pytest.approx(0.15)


def test_BUS_T002_spread_and_economic_value():
    assert spread(0.15, 0.10) == pytest.approx(0.05)
    assert eva(0.15, 0.10, 500.0) == pytest.approx(25.0)


# --- BUS-T003: margin range/stability -----------------------------------


def test_BUS_T003_margin_range_is_3_points_and_stable():
    annual = [
        {"date": f"202{6-i}-01-01", "revenue": 1000.0, "ebit": m * 1000.0}
        for i, m in enumerate([0.20, 0.21, 0.19, 0.22, 0.20])
    ]
    margins = _margin_series(annual, years=5)
    valid = [m for m in margins if m is not None]
    margin_range = max(valid) - min(valid)
    assert margin_range == pytest.approx(0.03)
    assert margin_range <= 0.05  # passes the wide-moat gate's 5pp threshold


# --- BUS-T004: customer concentration red flag -----------------------


def test_BUS_T004_concentration_red_flag():
    packet = _packet([{"date": "2026-01-01", "revenue": 1000.0, "ebit": 200.0, "net_income": 150.0,
                        "total_debt": 100.0, "total_equity": 500.0, "cash": 50.0}])
    out = run(packet, wacc=0.10, overlay={"largest_customer_concentration": 0.35})
    assert "CONCENTRATION_RED_FLAG" in out.mandatory_flags


# --- BUS-T005: cumulative FCF conversion --------------------------------


def test_BUS_T005_cumulative_fcf_conversion():
    # Five years summing to FCF=500, net income=450.
    annual = [
        {"date": f"202{6-i}-01-01", "operating_cash_flow": 120.0, "capex": -20.0, "net_income": 90.0}
        for i in range(5)
    ]
    ratio = _cumulative_fcf_conversion(annual, years=5)
    assert ratio == pytest.approx(500.0 / 450.0)


# --- BUS-T006: revenue CAGR not meaningful for non-positive base -------


def test_BUS_T006_revenue_cagr_nonpositive_base_not_meaningful():
    annual = [{"date": "2026-01-01", "revenue": 100.0}, {"date": "2025-01-01", "revenue": -10.0}]
    v = _revenue_cagr(annual, years=1)
    assert v.is_null


def test_diluted_share_cagr_nonpositive_base_returns_none():
    annual = [{"date": "2026-01-01", "diluted_shares": 100.0}, {"date": "2025-01-01", "diluted_shares": 0.0}]
    assert _diluted_share_cagr(annual, years=1) is None


# --- BUS-T007: ROIC<WACC blocks Excellent/wide-moat --------------------


def test_BUS_T007_roic_below_wacc_blocks_wide_moat_and_flags_value_destruction():
    annual = [
        {"date": "2026-01-01", "ebit": 40.0, "pretax_income": 35.0, "income_tax_expense": 8.75,
         "total_debt": 200.0, "total_equity": 800.0, "cash": 50.0, "revenue": 1000.0, "net_income": 30.0},
        {"date": "2025-01-01", "ebit": 38.0, "pretax_income": 33.0, "income_tax_expense": 8.25,
         "total_debt": 190.0, "total_equity": 750.0, "cash": 45.0, "revenue": 950.0, "net_income": 28.0},
    ]
    packet = _packet(annual)
    out = run(packet, wacc=0.20)  # deliberately high WACC vs. a modest-ROIC business
    assert "VALUE_DESTRUCTION" in out.mandatory_flags
    assert out.moat["classification"] != "Wide"


# --- BUS-T008: missing NRR for non-subscription industrial -> no penalty --


def test_BUS_T008_missing_customer_economics_not_scorable_without_penalty():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet, wacc=0.10)
    customer_dim = next(d for d in out.dimensions if d["name"] == "customer_economics")
    assert customer_dim["score_10"] is None  # NOT_SCORABLE, not silently scored BAD
    # Category math must still reconcile without a fabricated penalty term.
    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet, wacc=0.10)
    assert out.agent_id == "business_analysis"
    assert out.category.max_points == 20
    assert len(out.dimensions) == 5
    assert 0.0 <= out.coverage <= 1.0
    assert out.moat["classification"] in {"Wide", "Narrow", "None", "NotScorable"}
    # Moat effects count wasn't supplied via overlay -> must ask for it.
    assert any(j.metric_id == "moat_quantitative_effects_count" for j in out.judgment_requests)
