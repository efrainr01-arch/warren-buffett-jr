"""Tests for wbj.engines.valuation_engine, per Cerebro/special_sauces/
INSTITUTIONAL_VALUATION_ENGINE.md and Cerebro/06_valuation_analysis/
FORMULAS.md.
"""

from __future__ import annotations

import pytest

from wbj.core.nullstates import NullState
from wbj.engines.valuation_engine import (
    ReverseDCFInputs,
    dcf_value,
    economic_profit_value,
    ensemble,
    equity_bridge,
    fcff,
    fundamental_growth,
    hist_zscore,
    justified_pe,
    margin_of_safety,
    monte_carlo,
    nopat,
    per_share,
    relever_beta,
    reverse_dcf_implied_growth,
    roic,
    scenarios,
    spread,
    synthetic_kd,
    unlever_beta,
    wacc,
)
from wbj.schemas.valuation import ScenarioInput


# --- Gordon terminal value / DCF -----------------------------------------


def test_gordon_terminal_math():
    # single FCFF=100 growing 2%, wacc 10% -> TV = 100*1.02/0.08 = 1275
    terminal_fcff = 100 * 1.02
    tv = terminal_fcff / (0.10 - 0.02)
    assert tv == pytest.approx(1275.0)

    out = dcf_value([100.0], wacc_value=0.10, terminal_growth=0.02)
    assert out.pv_terminal == pytest.approx(1275.0 / 1.10)


def test_g_greater_than_wacc_refused():
    out = dcf_value([100.0, 110.0], wacc_value=0.08, terminal_growth=0.08)
    assert out.ev is None
    assert out.state == str(NullState.NOT_MEANINGFUL)


def test_terminal_share_warning_above_75pct():
    # Long, nearly-flat explicit stream + rich terminal growth -> most of
    # the value sits in the terminal component.
    out = dcf_value([10.0], wacc_value=0.09, terminal_growth=0.085)
    assert out.terminal_share > 0.75
    assert any("75%" in w for w in out.warnings)


def test_terminal_share_no_warning_when_below_threshold():
    out = dcf_value([100.0, 100.0, 100.0, 100.0, 100.0], wacc_value=0.10, terminal_growth=0.02)
    assert out.terminal_share < 0.75
    assert out.warnings == []


# --- WACC ------------------------------------------------------------------


def test_wacc():
    # E=800, D=200, Ke=10%, Kd=5%, tax=25% -> 0.8*.10 + 0.2*.05*.75 = 8.75%
    assert wacc(e=800, d=200, ke=0.10, kd=0.05, tax_rate=0.25) == pytest.approx(0.0875)


def test_unlever_relever_beta_round_trip():
    levered = 1.4
    unlevered = unlever_beta(levered, tax_rate=0.25, de_ratio=0.5)
    relevered = relever_beta(unlevered, tax_rate=0.25, target_de=0.5)
    assert relevered == pytest.approx(levered)


def test_synthetic_kd_uses_spread_table():
    # coverage 6.0 falls in [5.5, 6.5) -> spread 0.0100
    assert synthetic_kd(risk_free=0.04, interest_coverage=6.0) == pytest.approx(0.05)
    # coverage 0.3, below the lowest bucket -> the worst-case default spread
    assert synthetic_kd(risk_free=0.04, interest_coverage=0.3) == pytest.approx(0.04 + 0.12)


# --- equity bridge / per share ---------------------------------------------


def test_equity_bridge_and_per_share():
    equity = equity_bridge(ev=1000.0, cash=50.0, nonop=10.0, debt=300.0, lease_debt_value=20.0, preferred=5.0)
    assert equity == pytest.approx(1000 + 50 + 10 - 300 - 20 - 5)
    ps = per_share(equity, diluted_shares=50.0)
    assert ps.value == pytest.approx(equity / 50.0)


def test_per_share_zero_shares_not_meaningful():
    assert per_share(100.0, diluted_shares=0.0).is_null


# --- ROIC block --------------------------------------------------------


def test_nopat_roic_spread_fundamental_growth():
    n = nopat(norm_ebit=200.0, tax_rate=0.25)
    assert n == pytest.approx(150.0)
    r = roic(n, avg_invested_capital=1000.0)
    assert r.value == pytest.approx(0.15)
    assert spread(r.value, wacc_value=0.09) == pytest.approx(0.06)
    assert fundamental_growth(reinvestment_rate=0.25, roic_value=0.12) == pytest.approx(0.03)


# --- justified multiples / robust z-score -----------------------------


def test_justified_pe_requires_positive_roe_and_g_below_ke():
    val = justified_pe(g=0.03, roe=0.15, ke=0.09)
    assert val.value == pytest.approx((1 - 0.03 / 0.15) / (0.09 - 0.03))
    assert justified_pe(g=0.10, roe=0.15, ke=0.09).is_null  # g >= ke


def test_hist_zscore_robust():
    history = [10, 11, 12, 13, 14, 15, 16]
    z = hist_zscore(current=13, history=history)
    assert z.value == pytest.approx(0.0)  # 13 is the median


# --- reverse DCF ------------------------------------------------------


def test_reverse_dcf_recovers_known_growth():
    inputs = ReverseDCFInputs(
        revenue0=1000.0,
        years=5,
        tax_rate=0.25,
        wacc=0.09,
        terminal_growth=0.03,
        dna_pct_revenue=0.05,
        capex_pct_revenue=0.06,
        dnwc_pct_revenue=0.01,
        net_debt_and_claims=300.0,
        diluted_shares=100.0,
    )
    known_growth = 0.12
    margin = 0.20

    from wbj.engines.valuation_engine import _implied_equity

    equity = _implied_equity(known_growth, margin, inputs)
    price = equity / inputs.diluted_shares

    recovered = reverse_dcf_implied_growth(price, margin, inputs)
    assert recovered == pytest.approx(known_growth, abs=1e-4)


# --- scenarios / monte carlo / ensemble ---------------------------------


def test_scenario_probabilities_must_sum_to_1():
    entries = [
        ScenarioInput(label="bear", probability=0.3, growth=0.02, margin=0.10, wacc=0.10, tv_growth=0.02, value=50.0),
        ScenarioInput(label="base", probability=0.3, growth=0.05, margin=0.15, wacc=0.09, tv_growth=0.03, value=80.0),
        ScenarioInput(label="bull", probability=0.3, growth=0.08, margin=0.20, wacc=0.08, tv_growth=0.03, value=120.0),
    ]
    with pytest.raises(ValueError, match="sum to 1.0"):
        scenarios(entries)


def test_scenario_weighted_value():
    entries = [
        ScenarioInput(label="bear", probability=0.25, growth=0.02, margin=0.10, wacc=0.10, tv_growth=0.02, value=50.0),
        ScenarioInput(label="base", probability=0.50, growth=0.05, margin=0.15, wacc=0.09, tv_growth=0.03, value=80.0),
        ScenarioInput(label="bull", probability=0.25, growth=0.08, margin=0.20, wacc=0.08, tv_growth=0.03, value=120.0),
    ]
    out = scenarios(entries)
    assert out["weighted_value"] == pytest.approx(0.25 * 50 + 0.50 * 80 + 0.25 * 120)


def test_monte_carlo_deterministic_given_seed():
    kwargs = dict(
        revenue0=1000.0, years=5, tax_rate=0.25, dna_pct_revenue=0.05, capex_pct_revenue=0.06,
        dnwc_pct_revenue=0.01, net_debt_and_claims=300.0, diluted_shares=100.0,
        growth_range=(0.02, 0.06, 0.12), margin_range=(0.10, 0.15, 0.22), wacc_range=(0.07, 0.09, 0.11),
        terminal_growth=0.03, n=500, seed=42,
    )
    out1 = monte_carlo(**kwargs)
    out2 = monte_carlo(**kwargs)
    assert out1 == out2
    assert out1["p10"] <= out1["p25"] <= out1["median"] <= out1["p75"] <= out1["p90"]


def test_ensemble_weighted_value_and_dispersion():
    out = ensemble([(100.0, 0.5), (120.0, 0.3), (90.0, 0.2)])
    expected = (100 * 0.5 + 120 * 0.3 + 90 * 0.2) / 1.0
    assert out["value"] == pytest.approx(expected)
    assert out["dispersion"] >= 0


def test_margin_of_safety():
    mos = margin_of_safety(value=100.0, price=85.0)
    assert mos.value == pytest.approx(0.15)


# --- economic profit reconciles with FCFF -----------------------------


def test_economic_profit_reconciles_with_fcff():
    # Constant-growth identity: g = reinvestment_rate * ROIC holds NOPAT,
    # invested capital, and FCFF all growing at exactly g -> the FCFF DCF
    # and economic-profit DCF must reconcile exactly (see derivation in
    # commit message / plan notes), well inside the 1% tolerance.
    ic0 = 1000.0
    target_roic = 0.12
    g = 0.03
    wacc_value = 0.09
    reinvestment_rate = g / target_roic  # 0.25

    ic = ic0
    fcffs = []
    eps = []
    for _ in range(5):
        nopat_t = ic * target_roic
        reinvestment_t = reinvestment_rate * nopat_t
        ep_t = nopat_t - wacc_value * ic
        fcff_t = nopat_t - reinvestment_t
        eps.append(ep_t)
        fcffs.append(fcff_t)
        ic = ic + reinvestment_t

    fcff_ev = dcf_value(fcffs, wacc_value, terminal_growth=g).ev
    ep_ev = economic_profit_value(ic0, eps, wacc_value, terminal_growth=g)["ev"]

    assert abs(fcff_ev - ep_ev) / fcff_ev < 0.01
