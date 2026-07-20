"""Institutional valuation engine — Cerebro/special_sauces/
INSTITUTIONAL_VALUATION_ENGINE.md and Cerebro/06_valuation_analysis/
FORMULAS.md VAL-001..044.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from wbj.core.nullstates import NullState, Value
from wbj.schemas.valuation import DCFResult, ScenarioInput

_TERMINAL_SHARE_WARNING_THRESHOLD = 0.75
_RECONCILE_TOLERANCE = 0.05

# --- 3: normalization --------------------------------------------------


def normalized_ebit(
    reported: float,
    unusual_gains: float = 0.0,
    nonrecurring_charges: float = 0.0,
    misclassified_recurring: float = 0.0,
) -> float:
    """3.1: Normalized EBIT = reported - unusual gains + non-recurring
    charges added back + recurring costs mislabeled non-recurring."""
    return reported - unusual_gains + nonrecurring_charges + misclassified_recurring


def rd_capitalize(rd_history: list[float], life: int) -> dict:
    """3.2: capitalize R&D over a useful life of `life` years.
    `rd_history` is newest-first and must cover at least `life` years."""
    if len(rd_history) < life:
        raise ValueError("rd_history must cover at least `life` years")
    asset = sum(rd_history[j] * (1 - j / life) for j in range(life))
    amortization = sum(rd_history[:life]) / life
    return {"asset": asset, "amortization": amortization, "ebit_addback": rd_history[0] - amortization}


def lease_debt(commitments: list[float], pretax_kd: float) -> float:
    """3.3: PV of future lease commitments at the pre-tax cost of debt.
    `commitments[i]` is the payment due in year `i+1`."""
    return sum(c / (1 + pretax_kd) ** (i + 1) for i, c in enumerate(commitments))


# --- 4: ROIC, WACC, and economic value creation -----------------------


def nopat(norm_ebit: float, tax_rate: float) -> float:
    """4.1."""
    return norm_ebit * (1 - tax_rate)


def invested_capital(
    debt: float,
    equity: float,
    excess_cash: float,
    debt_like_claims: float = 0.0,
    operating_assets: float | None = None,
    operating_liabilities: float | None = None,
) -> dict:
    """4.2: financing-view invested capital, reconciled to the operating
    view when supplied (warn if the two views differ by more than 5%)."""
    financing_view = debt + equity - excess_cash + debt_like_claims
    result: dict = {"financing_view": financing_view}
    if operating_assets is not None and operating_liabilities is not None:
        operating_view = operating_assets - operating_liabilities
        result["operating_view"] = operating_view
        diff = abs(financing_view - operating_view) / abs(financing_view) if financing_view else 0.0
        result["reconciled"] = diff <= _RECONCILE_TOLERANCE
        if diff > _RECONCILE_TOLERANCE:
            result["warning"] = f"financing/operating invested-capital views differ {diff:.1%} (>5%)"
    return result


def roic(nopat_value: float, avg_invested_capital: float) -> Value:
    """4.3."""
    if avg_invested_capital == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(nopat_value / avg_invested_capital, unit="ratio")


def spread(roic_value: float, wacc_value: float) -> float:
    """4.4."""
    return roic_value - wacc_value


def eva(roic_value: float, wacc_value: float, avg_invested_capital: float) -> float:
    """4.4: economic value added."""
    return (roic_value - wacc_value) * avg_invested_capital


def incremental_roic(delta_nopat: float, delta_ic: float) -> Value:
    """4.5."""
    if delta_ic == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(delta_nopat / delta_ic, unit="ratio")


def fundamental_growth(reinvestment_rate: float, roic_value: float) -> float:
    """4.6: fundamental growth ~= reinvestment rate * ROIC."""
    return reinvestment_rate * roic_value


# --- 5: discount rate -----------------------------------------------------


def unlever_beta(levered_beta: float, tax_rate: float, de_ratio: float) -> float:
    """5.3."""
    return levered_beta / (1 + (1 - tax_rate) * de_ratio)


def relever_beta(unlevered_beta: float, tax_rate: float, target_de: float) -> float:
    """5.3."""
    return unlevered_beta * (1 + (1 - tax_rate) * target_de)


def cost_of_equity(risk_free: float, beta: float, erp: float, country_risk_premium: float = 0.0) -> float:
    """5.2."""
    return risk_free + beta * erp + country_risk_premium


# Synthetic default-spread table (interest coverage -> spread over the
# risk-free rate), a disclosed Damodaran-style mapping for non-rated /
# thinly-traded debt. Dated 2024Q4 vintage; recalibrate periodically.
SYNTHETIC_SPREAD_TABLE: list[tuple[float, float]] = [
    (8.50, 0.0060),
    (6.50, 0.0080),
    (5.50, 0.0100),
    (4.25, 0.0125),
    (3.00, 0.0150),
    (2.50, 0.0200),
    (2.00, 0.0250),
    (1.50, 0.0350),
    (1.25, 0.0450),
    (0.80, 0.0600),
    (0.50, 0.0800),
]
_DEFAULT_SPREAD_BELOW_MIN = 0.1200


def synthetic_kd(risk_free: float, interest_coverage: float) -> float:
    """5.4: pre-tax cost of debt via interest-coverage -> default-spread
    mapping. Coverage below 1.5x carries a mandatory solvency warning
    elsewhere in the system (Cerebro/05_risk_analysis)."""
    for threshold, spread_bp in SYNTHETIC_SPREAD_TABLE:
        if interest_coverage >= threshold:
            return risk_free + spread_bp
    return risk_free + _DEFAULT_SPREAD_BELOW_MIN


def wacc(e: float, d: float, ke: float, kd: float, tax_rate: float) -> float:
    """5.1."""
    total = e + d
    if total == 0:
        return ke
    return (e / total) * ke + (d / total) * kd * (1 - tax_rate)


def wacc_sensitivity(w: float, bp: int = 100) -> dict:
    """5.5: +/- `bp` basis points around base WACC."""
    delta = bp / 10000
    return {"low": w - delta, "base": w, "high": w + delta}


# --- 6: FCFF discounted cash flow --------------------------------------


def fcff(ebit: float, tax_rate: float, dna: float, capex: float, dnwc: float) -> float:
    """6.1."""
    return ebit * (1 - tax_rate) + dna - capex - dnwc


def dcf_value(fcffs: list[float], wacc_value: float, terminal_growth: float) -> DCFResult:
    """6.3/6.4/6.6: enterprise value via explicit FCFF + Gordon terminal
    value. Refuses (NOT_MEANINGFUL) when terminal growth >= WACC; warns
    when the terminal-value share exceeds 75%."""
    if terminal_growth >= wacc_value:
        return DCFResult(
            ev=None,
            state=str(NullState.NOT_MEANINGFUL),
            warnings=["terminal growth >= WACC: Gordon terminal value is undefined"],
        )
    pv_explicit = sum(cf / (1 + wacc_value) ** t for t, cf in enumerate(fcffs, start=1))
    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcff / (wacc_value - terminal_growth)
    n = len(fcffs)
    pv_terminal = terminal_value / (1 + wacc_value) ** n
    ev = pv_explicit + pv_terminal
    terminal_share = pv_terminal / ev if ev else None
    warnings = []
    if terminal_share is not None and terminal_share > _TERMINAL_SHARE_WARNING_THRESHOLD:
        warnings.append(f"terminal-value share {terminal_share:.1%} exceeds 75% high-sensitivity threshold")
    return DCFResult(ev=ev, pv_explicit=pv_explicit, pv_terminal=pv_terminal, terminal_share=terminal_share, warnings=warnings)


def equity_bridge(
    ev: float,
    cash: float,
    nonop: float = 0.0,
    debt: float = 0.0,
    lease_debt_value: float = 0.0,
    preferred: float = 0.0,
    minority: float = 0.0,
    pension: float = 0.0,
) -> float:
    """6.7."""
    return ev + cash + nonop - debt - lease_debt_value - preferred - minority - pension


def per_share(equity_value: float, diluted_shares: float) -> Value:
    """6.8."""
    if diluted_shares <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="usd_per_share")
    return Value.of(equity_value / diluted_shares, unit="usd_per_share")


# --- 7/9/10: cross-checks -----------------------------------------------


def fcfe_value(fcfes: list[float], ke: float, terminal_growth: float) -> dict:
    """Section 7: FCFE valuation at the cost of equity."""
    if terminal_growth >= ke:
        return {"equity_value": None, "state": str(NullState.NOT_MEANINGFUL)}
    pv_explicit = sum(cf / (1 + ke) ** t for t, cf in enumerate(fcfes, start=1))
    terminal_fcfe = fcfes[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcfe / (ke - terminal_growth)
    n = len(fcfes)
    pv_terminal = terminal_value / (1 + ke) ** n
    return {"equity_value": pv_explicit + pv_terminal, "pv_explicit": pv_explicit, "pv_terminal": pv_terminal}


def economic_profit_value(
    ic0: float, economic_profits: list[float], wacc_value: float, terminal_growth: float
) -> dict:
    """Section 9: EV = IC0 + PV(future economic profits). Should reconcile
    with FCFF DCF under consistent assumptions."""
    if terminal_growth >= wacc_value:
        return {"ev": None, "state": str(NullState.NOT_MEANINGFUL)}
    pv_explicit = sum(ep / (1 + wacc_value) ** t for t, ep in enumerate(economic_profits, start=1))
    terminal_ep = economic_profits[-1] * (1 + terminal_growth)
    terminal_value = terminal_ep / (wacc_value - terminal_growth)
    n = len(economic_profits)
    pv_terminal = terminal_value / (1 + wacc_value) ** n
    return {"ev": ic0 + pv_explicit + pv_terminal, "pv_explicit": pv_explicit, "pv_terminal": pv_terminal}


def residual_income_value(
    book_equity0: float, residual_incomes: list[float], ke: float, terminal_growth: float
) -> dict:
    """Section 10: equity value = book equity + PV(future residual income)."""
    if terminal_growth >= ke:
        return {"equity_value": None, "state": str(NullState.NOT_MEANINGFUL)}
    pv_explicit = sum(ri / (1 + ke) ** t for t, ri in enumerate(residual_incomes, start=1))
    terminal_ri = residual_incomes[-1] * (1 + terminal_growth)
    terminal_value = terminal_ri / (ke - terminal_growth)
    n = len(residual_incomes)
    pv_terminal = terminal_value / (1 + ke) ** n
    return {"equity_value": book_equity0 + pv_explicit + pv_terminal}


def justified_pe(g: float, roe: float, ke: float) -> Value:
    """14.1: requires positive sustainable ROE and g < cost of equity."""
    if roe <= 0 or ke <= g:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of((1 - g / roe) / (ke - g), unit="ratio")


def justified_ev_sales(margin: float, tax_rate: float, g: float, roic_value: float, wacc_value: float) -> Value:
    """14.2: after-tax operating margin * (1 - g/ROIC) / (WACC - g)."""
    if roic_value <= 0 or wacc_value <= g:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    after_tax_margin = margin * (1 - tax_rate)
    return Value.of(after_tax_margin * (1 - g / roic_value) / (wacc_value - g), unit="ratio")


def hist_zscore(current: float, history: list[float]) -> Value:
    """Section 15: robust z-score = (x - median) / (1.4826 * MAD)."""
    median = statistics.median(history)
    mad = statistics.median([abs(h - median) for h in history])
    if mad == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="zscore")
    return Value.of((current - median) / (1.4826 * mad), unit="zscore")


# --- 13: reverse DCF -----------------------------------------------------


@dataclass
class ReverseDCFInputs:
    """Shared base assumptions for a reverse-DCF solve."""

    revenue0: float
    years: int
    tax_rate: float
    wacc: float
    terminal_growth: float
    dna_pct_revenue: float
    capex_pct_revenue: float
    dnwc_pct_revenue: float
    net_debt_and_claims: float
    diluted_shares: float


def _fcff_stream_from_growth_margin(
    revenue0: float, growth: float, margin: float, years: int, tax_rate: float, dna_pct: float, capex_pct: float, dnwc_pct: float
) -> list[float]:
    fcffs = []
    revenue = revenue0
    for _ in range(years):
        revenue *= 1 + growth
        ebit = revenue * margin
        fcffs.append(fcff(ebit, tax_rate, revenue * dna_pct, revenue * capex_pct, revenue * dnwc_pct))
    return fcffs


def _implied_equity(growth: float, margin: float, inputs: ReverseDCFInputs) -> float | None:
    fcffs = _fcff_stream_from_growth_margin(
        inputs.revenue0, growth, margin, inputs.years, inputs.tax_rate,
        inputs.dna_pct_revenue, inputs.capex_pct_revenue, inputs.dnwc_pct_revenue,
    )
    dcf = dcf_value(fcffs, inputs.wacc, inputs.terminal_growth)
    if dcf.ev is None:
        return None
    return dcf.ev - inputs.net_debt_and_claims


def reverse_dcf_implied_growth(price: float, margin: float, inputs: ReverseDCFInputs) -> float:
    """13: implied revenue CAGR that reconciles modeled equity value to
    `price * diluted_shares`, holding margin fixed."""
    target_equity = price * inputs.diluted_shares

    def objective(growth: float) -> float:
        equity = _implied_equity(growth, margin, inputs)
        return (equity if equity is not None else float("-inf")) - target_equity

    return brentq(objective, -0.5, 3.0, xtol=1e-8)


def reverse_dcf_implied_margin(price: float, growth: float, inputs: ReverseDCFInputs) -> float:
    """13: implied terminal operating margin at a given (e.g. consensus)
    revenue CAGR."""
    target_equity = price * inputs.diluted_shares

    def objective(margin: float) -> float:
        equity = _implied_equity(growth, margin, inputs)
        return (equity if equity is not None else float("-inf")) - target_equity

    return brentq(objective, -0.5, 0.90, xtol=1e-8)


# --- 16: scenarios and Monte Carlo --------------------------------------


def scenarios(entries: list[ScenarioInput]) -> dict:
    """16.1: probability-weighted scenario value. `entries[i].value` is
    the scenario's already-computed per-share (or equity) value, e.g. from
    `dcf_value` + `equity_bridge` + `per_share` run at that scenario's
    growth/margin/WACC/terminal-growth. Probabilities must sum to 1.0."""
    total_p = sum(e.probability for e in entries)
    if abs(total_p - 1.0) > 1e-6:
        raise ValueError(f"scenario probabilities must sum to 1.0, got {total_p}")
    weighted = sum(e.probability * e.value for e in entries)
    return {"weighted_value": weighted, "scenarios": entries}


def monte_carlo(
    revenue0: float,
    years: int,
    tax_rate: float,
    dna_pct_revenue: float,
    capex_pct_revenue: float,
    dnwc_pct_revenue: float,
    net_debt_and_claims: float,
    diluted_shares: float,
    growth_range: tuple[float, float, float],
    margin_range: tuple[float, float, float],
    wacc_range: tuple[float, float, float],
    terminal_growth: float,
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """16.2: Monte Carlo over triangular (low, mode, high) draws of
    growth/margin/WACC. Seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    growths = rng.triangular(*growth_range, n)
    margins = rng.triangular(*margin_range, n)
    waccs = rng.triangular(*wacc_range, n)

    values = np.full(n, np.nan)
    for i in range(n):
        g, m, w = float(growths[i]), float(margins[i]), float(waccs[i])
        if terminal_growth >= w:
            continue
        fcffs = _fcff_stream_from_growth_margin(revenue0, g, m, years, tax_rate, dna_pct_revenue, capex_pct_revenue, dnwc_pct_revenue)
        dcf = dcf_value(fcffs, w, terminal_growth)
        if dcf.ev is None:
            continue
        values[i] = (dcf.ev - net_debt_and_claims) / diluted_shares

    valid = values[~np.isnan(values)]
    p10, p25, median, p75, p90 = np.percentile(valid, [10, 25, 50, 75, 90])
    return {
        "p10": float(p10),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p90": float(p90),
        "seed": seed,
        "trials": n,
    }


# --- 19: ensemble ----------------------------------------------------------


def ensemble(model_values: list[tuple[float, float]]) -> dict:
    """Section 19: reliability-weighted model ensemble; reports dispersion
    rather than forcing false precision."""
    total_w = sum(w for _, w in model_values)
    if total_w == 0:
        return {"value": None, "dispersion": None, "models": model_values}
    weighted = sum(v * w for v, w in model_values) / total_w
    values = [v for v, _ in model_values]
    dispersion = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"value": weighted, "dispersion": dispersion, "models": model_values}


def margin_of_safety(value: float, price: float) -> Value:
    """1.2/16.1 margin-of-safety reference: (value - price) / value."""
    if value == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of((value - price) / value, unit="ratio")
