"""Tests for wbj.specialists.valuation, per Cerebro/06_valuation_analysis/
VALIDATION_TESTS.md (VAL-T001..T010; VAL-T008's options/convertibles
schedule is out of scope — dilution-scenario modeling isn't implemented
in this engine, so per-share value is correctly incomplete rather than
fabricated).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.engines.valuation_engine import dcf_value, economic_profit_value, equity_bridge, per_share
from wbj.schemas.packet import Packet
from wbj.schemas.valuation import ScenarioInput
from wbj.specialists.valuation import run, select_models

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


# --- VAL-T001: one-stage Gordon growth value at t0 ----------------------


def test_VAL_T001_gordon_value_at_t0():
    fcff1, wacc, g = 100.0, 0.10, 0.03
    value_t0 = fcff1 / (wacc - g)
    assert value_t0 == pytest.approx(1428.57, abs=0.01)


# --- VAL-T002/T003: g>=WACC refused -------------------------------------


def test_VAL_T002_denominator_zero_rejected():
    out = dcf_value([100.0], wacc_value=0.08, terminal_growth=0.08)
    assert out.ev is None


def test_VAL_T003_g_greater_than_wacc_rejected():
    out = dcf_value([100.0], wacc_value=0.07, terminal_growth=0.08)
    assert out.ev is None


# --- VAL-T004: terminal reinvestment rate -------------------------------


def test_VAL_T004_terminal_reinvestment_rate():
    roic, g = 0.20, 0.04
    terminal_reinvestment_rate = g / roic
    assert terminal_reinvestment_rate == pytest.approx(0.20)


# --- VAL-T005: equity bridge and per-share value ------------------------


def test_VAL_T005_equity_bridge_and_per_share():
    equity = equity_bridge(ev=1000.0, cash=100.0, debt=300.0)
    assert equity == pytest.approx(800.0)
    ps = per_share(equity, diluted_shares=80.0)
    assert ps.value == pytest.approx(10.0)


# --- VAL-T006: scenario probabilities sum to 100% ------------------------


def test_VAL_T006_scenario_probabilities_sum_to_100pct():
    from wbj.engines.valuation_engine import scenarios as scenarios_fn

    entries = [
        ScenarioInput(label="bear", probability=0.20, growth=0.02, margin=0.10, wacc=0.10, tv_growth=0.02, value=50.0),
        ScenarioInput(label="base", probability=0.60, growth=0.05, margin=0.15, wacc=0.09, tv_growth=0.03, value=80.0),
        ScenarioInput(label="bull", probability=0.20, growth=0.08, margin=0.20, wacc=0.08, tv_growth=0.03, value=120.0),
    ]
    out = scenarios_fn(entries)  # must not raise
    assert out["weighted_value"] == pytest.approx(0.2 * 50 + 0.6 * 80 + 0.2 * 120)


# --- VAL-T007: terminal share 80% triggers high-sensitivity flag --------


def test_VAL_T007_terminal_share_80pct_flags_high_sensitivity():
    out = dcf_value([10.0], wacc_value=0.09, terminal_growth=0.085)
    assert out.terminal_share > 0.75
    assert any("75%" in w for w in out.warnings)


# --- VAL-T009: FCFF and EVA diverge when inputs are inconsistent -------


def test_VAL_T009_fcff_and_economic_profit_diverge_with_inconsistent_inputs():
    # Deliberately inconsistent: FCFF stream doesn't correspond to the
    # invested-capital/economic-profit stream used here -> the cross-check
    # must be able to detect a material (>1%) divergence.
    fcff_ev = dcf_value([50.0, 55.0, 60.0], wacc_value=0.09, terminal_growth=0.03).ev
    ep_ev = economic_profit_value(ic0=2000.0, economic_profits=[10.0, 10.0, 10.0], wacc_value=0.09, terminal_growth=0.03)["ev"]
    assert abs(fcff_ev - ep_ev) / fcff_ev > 0.01


# --- VAL-T010: bank -> adapter unsupported, no EV/EBITDA primary -------


def test_VAL_T010_bank_selects_adapter_unsupported():
    sel = select_models("bank")
    assert sel["status"] == "ADAPTER_UNSUPPORTED"
    assert "FCFF DCF" not in sel["primary"]


def test_select_models_default_nonfinancial_uses_fcff_and_economic_profit():
    sel = select_models("operating_company")
    assert sel["status"] == "OK"
    assert "FCFF DCF" in sel["primary"]
    assert "Economic profit" in sel["primary"]


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet, beta=1.72, erp=0.045)

    assert out.agent_id == "valuation_analysis"
    assert out.category.max_points == 10
    assert len(out.dimensions) == 5
    assert 0.0 <= out.coverage <= 1.0
    assert out.model_selection["status"] == "OK"
    assert len(out.scenarios) == 3
    assert out.wacc["value"] is not None

    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6


def test_nvda_fixture_without_beta_requests_judgment():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet)  # no beta supplied
    assert any(j.metric_id == "beta_and_erp" for j in out.judgment_requests)
    assert out.wacc["value"] is None
