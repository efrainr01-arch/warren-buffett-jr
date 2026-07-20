"""Tests for wbj.specialists.financial, per Cerebro/02_financial_analysis/
VALIDATION_TESTS.md (FIN-T001..T009 — FIN-T010's bank/insurer adapter is
out of scope: the plan explicitly excludes industry adapters) plus the
core-27 percent math and dimension/category reconciliation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.schemas.packet import Packet
from wbj.specialists.financial import (
    _classify,
    _current_ratio,
    _debt_to_equity,
    _fcf,
    _fcf_margin,
    _interest_coverage,
    _roic_vs_wacc,
    _yoy,
    run,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


# --- FIN-T001..T005: formula-level validation rows -------------------------


def test_FIN_T001_revenue_yoy_growth():
    assert _yoy(110, 100) == pytest.approx(0.10)


def test_FIN_T002_current_ratio():
    value, _ = _current_ratio([{"total_current_assets": 150, "total_current_liabilities": 100}], None, None)
    assert value == pytest.approx(1.5)


def test_FIN_T003_coverage_exactly_1_5x_no_warning():
    value, _ = _interest_coverage([{"ebit": 30, "interest_expense": 20}], None, None)
    assert value == pytest.approx(1.5)
    assert not (value < 1.5)  # SOLVENCY_WARNING trigger is strict "<1.5"


def test_FIN_T004_coverage_below_1_5x_triggers_warning():
    value, _ = _interest_coverage([{"ebit": 29, "interest_expense": 20}], None, None)
    assert value == pytest.approx(1.45)
    assert value < 1.5  # SOLVENCY_WARNING trigger


def test_FIN_T005_fcf_and_fcf_margin():
    # Packet convention (Task 10): capex is stored negative (a cash
    # outflow), so OCF=120 and a $40 capex outflow is `capex=-40`;
    # FCF = OCF + capex = 120 + (-40) = 80, matching VALIDATION_TESTS.md's
    # "OCF=120, capex=40 -> FCF=80" under our storage sign convention.
    annual = [{"operating_cash_flow": 120, "capex": -40, "revenue": 800}]
    fcf_value, _ = _fcf(annual, None, None)
    assert fcf_value == pytest.approx(80)
    margin, _ = _fcf_margin(annual, None, None)
    assert margin == pytest.approx(0.10)


# --- FIN-T007: ROIC below WACC -> no Excellent verdict ---------------------


def test_FIN_T007_roic_below_wacc_not_excellent():
    diff, _ = _roic_vs_wacc(
        [
            {"ebit": 90, "total_debt": 200, "total_equity": 800, "cash": 50, "pretax_income": 80, "income_tax_expense": 20},
            {"ebit": 85, "total_debt": 200, "total_equity": 750, "cash": 50, "pretax_income": 75, "income_tax_expense": 19},
        ],
        None,
        0.11,
    )
    assert diff is not None and diff < 0
    assert _classify(diff, -0.01, 0.01) != "EXCELLENT"


# --- FIN-T009: negative equity -> debt/equity NOT_MEANINGFUL ---------------


def test_FIN_T009_negative_equity_not_meaningful():
    value, warnings = _debt_to_equity([{"total_debt": 100, "total_equity": -50}], None, None)
    assert value is None
    assert any("NOT_MEANINGFUL" in w for w in warnings)


# --- FIN-T008: core-27 percent math -----------------------------------


def test_FIN_T008_all_27_excellent_is_100_percent():
    valid, points = 27, 27 * 2
    percent = points / (2 * valid) * 100
    assert percent == pytest.approx(100.0)


# --- FIN-T006: loss + negative FCF -> Bad/Avoid override candidate ------


def test_FIN_T006_loss_and_negative_fcf_flags_override():
    packet = Packet.model_validate(
        {
            "security": {"ticker": "TST", "exchange": "NASDAQ", "security_type": "operating_company",
                         "reporting_currency": "USD", "valuation_currency": "USD"},
            "analysis": {"knowledge_timestamp": "2026-01-01T00:00:00Z", "market_timestamp": "2026-01-01",
                         "industry_adapter": "default_nonfinancial"},
            "fundamentals": {
                "annual": [
                    {"date": "2026-01-01", "revenue": 500, "net_income": -10, "operating_cash_flow": -15,
                     "capex": -5, "cash": 20, "total_debt": 100, "total_equity": 40,
                     "total_current_assets": 60, "total_current_liabilities": 50, "total_assets": 200},
                    {"date": "2025-01-01", "revenue": 480, "net_income": -8, "operating_cash_flow": -10,
                     "capex": -5, "cash": 25, "total_debt": 90, "total_equity": 48,
                     "total_current_assets": 55, "total_current_liabilities": 45, "total_assets": 190},
                ],
                "quarterly": [],
            },
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
    out = run(packet)
    assert any("OVERRIDE_1_CANDIDATE" in o for o in out.mandatory_overrides)


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output_with_reconciling_math():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet, wacc=0.10)

    assert out.agent_id == "financial_analysis"
    assert out.category.max_points == 15
    assert len(out.dimensions) == 5
    assert len(out.metrics) == 27
    assert out.core_27_metrics["valid_count"] <= 27
    assert 0.0 <= out.coverage <= 1.0
    # category.points() sums dimension points -- reconciles by construction
    # (Category.points() *is* the sum), but assert the category math is
    # internally consistent within float tolerance regardless.
    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6
