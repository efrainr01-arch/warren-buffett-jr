"""Tests for wbj.specialists.risk, per Cerebro/05_risk_analysis/
VALIDATION_TESTS.md (RSK-T001..T009).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.schemas.packet import Packet
from wbj.specialists.risk import (
    SOLVENCY_WARNING_TEXT,
    BENEISH_M_SCREEN_THRESHOLD,
    concentration_hhi,
    interest_coverage,
    is_financial_sector,
    max_drawdown,
    net_debt_to_ebitda,
    run,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def _packet(annual_rows: list[dict], security_type: str = "operating_company") -> Packet:
    return Packet.model_validate(
        {
            "security": {"ticker": "TST", "exchange": "NASDAQ", "security_type": security_type,
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


# --- RSK-T001/T002: interest coverage and solvency warning -----------


def test_RSK_T001_coverage_exactly_1_5x():
    v = interest_coverage(ebit=15.0, interest=10.0)
    assert v.value == pytest.approx(1.5)


def test_RSK_T002_coverage_1_49_triggers_mandatory_warning():
    packet = _packet([{"date": "2026-01-01", "ebit": 14.9, "interest_expense": 10.0}])
    out = run(packet)
    assert SOLVENCY_WARNING_TEXT in out.mandatory_warnings
    assert "SOLVENCY_WARNING" in out.mandatory_flags


# --- RSK-T003: cash runway ----------------------------------------------


def test_RSK_T003_cash_runway_12_months():
    from wbj.specialists.risk import cash_runway

    v = cash_runway(cash=120.0, committed_liquidity=0.0, monthly_burn=10.0)
    assert v.value == pytest.approx(12.0)


# --- RSK-T004: max drawdown ---------------------------------------------


def test_RSK_T004_max_drawdown_60_percent():
    prices = [100.0, 90.0, 70.0, 40.0, 50.0, 60.0]
    out = max_drawdown(prices)
    assert out["mdd"] == pytest.approx(-0.60)


# --- RSK-T005: customer HHI ----------------------------------------------


def test_RSK_T005_two_customers_50pct_each_hhi():
    v = concentration_hhi([0.5, 0.5])
    assert v.value == pytest.approx(0.50)


# --- RSK-T006: negative EBITDA -> net debt/EBITDA not meaningful -------


def test_RSK_T006_negative_ebitda_not_meaningful():
    v = net_debt_to_ebitda(net_debt=100.0, ebitda=-10.0)
    assert v.is_null


# --- RSK-T007: bank -> Altman/Beneish not automatic ---------------------


def test_RSK_T007_bank_excludes_forensic_scoring():
    assert is_financial_sector("bank") is True
    assert is_financial_sector("insurer") is True
    assert is_financial_sector("operating_company") is False


# --- RSK-T008: risk category <=4/15 flags Speculative cap candidate ----


def test_RSK_T008_low_risk_score_flags_override_candidate():
    # No fundamentals at all -> every dimension NOT_SCORABLE -> awarded=0 <= 4.
    packet = _packet([])
    out = run(packet)
    assert out.category.awarded_points <= 4.0
    assert any("RISK_OVERRIDE_CANDIDATE" in f for f in out.mandatory_flags)


# --- RSK-T009: forensic flag is a screen, not an accusation ------------


def test_RSK_T009_beneish_flag_is_screening_language_only():
    packet = _packet(
        [
            {"date": "2026-01-01", "net_income": 100.0, "operating_cash_flow": 50.0, "total_assets": 1000.0},
            {"date": "2025-01-01", "net_income": 90.0, "operating_cash_flow": 60.0, "total_assets": 900.0},
        ]
    )
    out = run(packet, overlay={"beneish_m_score": BENEISH_M_SCREEN_THRESHOLD + 0.5})
    flag = out.earnings_quality_and_forensics["beneish_flag"]
    assert flag is not None
    assert "screening flag" in flag
    assert "manipulation" not in flag.split("not")[0]  # no bare accusation before the disclaimer


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet)

    assert out.agent_id == "risk_analysis"
    assert out.category.max_points == 15
    assert len(out.dimensions) == 6
    assert sum(d["max_points"] for d in out.dimensions) == 15
    assert 0.0 <= out.coverage <= 1.0
    # No benchmark in this packet -> beta-dependent volatility signal is absent,
    # but drawdown/vol from price alone should still be computable.
    assert out.market_risk["annualized_volatility"] is not None

    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6
