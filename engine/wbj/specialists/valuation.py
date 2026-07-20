"""Valuation specialist (10 pts) — Cerebro/06_valuation_analysis/.

Orchestrates the Task-13 institutional valuation engine against the
packet's facts table and annual fundamentals into Bear/Base/Bull
scenarios, a reverse DCF, a seeded Monte Carlo, a reliability-weighted
ensemble, and the five weighted dimensions (SCORING.md).

The packet built in Task 10 does not carry a peer multiple panel or
beta/benchmark data (Tasks 17-18 note the same gap): "Historical and
peer comparison" is correctly NOT_SCORABLE rather than forced through an
unreliable single-price historical-multiple proxy, and beta/ERP are
optional parameters to `run()` (mirroring how Financial/Business accept
`wacc`) rather than fabricated. Interest coverage for the synthetic cost
of debt falls back to a disclosed, clearly-flagged assumption when
interest expense is absent from the packet (evidence class A, not a
silent estimate).
"""

from __future__ import annotations

import statistics
from typing import Literal

from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension, anchor_score
from wbj.engines.valuation_engine import (
    ReverseDCFInputs,
    cost_of_equity,
    dcf_value,
    economic_profit_value,
    ensemble,
    equity_bridge,
    fcff,
    hist_zscore,
    justified_pe,
    margin_of_safety,
    monte_carlo,
    nopat,
    per_share,
    reverse_dcf_implied_growth,
    roic as _roic_fn,
    scenarios as _scenarios_fn,
    synthetic_kd,
    wacc as _wacc_fn,
)
from wbj.schemas.valuation import ScenarioInput
from wbj.specialists.common import CategorySummary, JudgmentRequest, MetricRow, SpecialistOutput

AGENT_ID = "valuation_analysis"
MAX_POINTS = float(CATEGORY_WEIGHTS["valuation"])  # 10
FORECAST_YEARS = 5
DEFAULT_ERP = 0.045  # long-run US equity risk premium, disclosed assumption
DEFAULT_INTEREST_COVERAGE_ASSUMPTION = 10.0  # used only when interest expense is absent from the packet


def select_models(security_type: str | None) -> dict:
    """DECISION_RULES.md model-selection matrix. Only the "mature
    non-financial" row (FCFF DCF + economic profit) is implemented;
    other adapters are out of scope for this engine (per the plan)."""
    if (security_type or "").lower() in {"bank", "insurer", "insurance", "financial"}:
        return {"primary": [], "rejected": ["FCFF DCF", "EV/EBITDA"], "status": "ADAPTER_UNSUPPORTED"}
    return {"primary": ["FCFF DCF", "Economic profit"], "rejected": ["DDM (payout not stable-mature)"], "status": "OK"}


def _annual_field(annual: list[dict], field_name: str, i: int = 0) -> float | None:
    if i >= len(annual):
        return None
    return annual[i].get(field_name)


def _historical_avg_op_margin(annual: list[dict], years: int = 5) -> float | None:
    margins = []
    for i in range(min(years, len(annual))):
        rev, ebit = _annual_field(annual, "revenue", i), _annual_field(annual, "ebit", i)
        if rev:
            margins.append(ebit / rev)
    return statistics.mean(margins) if margins else None


def _capex_dna_dnwc_pct(annual: list[dict]) -> tuple[float, float, float]:
    """Disclosed proxy: latest-year capex/D&A/NWC-change as a percent of
    revenue, held flat across the forecast (documented assumption, since
    the packet has no explicit multi-driver forecast model)."""
    row = annual[0] if annual else {}
    rev = row.get("revenue") or 1.0
    capex_pct = abs(row.get("capex") or 0.0) / rev
    dna_pct = capex_pct * 0.8  # D&A not separately in the canonical mapping; proxy near capex, documented
    dnwc_pct = 0.01
    return dna_pct, capex_pct, dnwc_pct


def _forecast_fcffs(revenue0: float, growth: float, margin: float, years: int, tax_rate: float, dna_pct: float, capex_pct: float, dnwc_pct: float) -> list[float]:
    fcffs = []
    revenue = revenue0
    for _ in range(years):
        revenue *= 1 + growth
        ebit = revenue * margin
        fcffs.append(fcff(ebit, tax_rate, revenue * dna_pct, revenue * capex_pct, revenue * dnwc_pct))
    return fcffs


class ValuationOutput(SpecialistOutput):
    model_selection: dict = {}
    normalization_reconciliation: list = []
    wacc: dict = {}
    scenarios: list = []
    reverse_dcf: dict = {}
    model_cross_checks: dict = {}
    fair_value_distribution: dict = {}
    reference_bands: dict = {}


def run(packet, beta: float | None = None, erp: float = DEFAULT_ERP, overlay: dict | None = None) -> ValuationOutput:
    overlay = overlay or {}
    beta_erp_answer = overlay.get("beta_and_erp")
    if beta is None and isinstance(beta_erp_answer, dict):
        beta = beta_erp_answer.get("beta")
        erp = beta_erp_answer.get("erp", erp)
    annual = packet.fundamentals.get("annual", [])
    facts = packet.facts_table

    model_selection = select_models(packet.security.security_type)
    judgment_requests: list[JudgmentRequest] = []

    revenue = facts.get("revenue")
    diluted_shares = facts.get("diluted_shares")
    cash = facts.get("cash")
    total_debt = facts.get("total_debt")
    price = facts.get("price")

    have_core_facts = all(v is not None and v.is_valid for v in (revenue, diluted_shares, cash, total_debt, price))

    wacc_value = None
    wacc_components: dict = {}
    if have_core_facts:
        pretax, tax_expense = _annual_field(annual, "pretax_income"), _annual_field(annual, "income_tax_expense")
        tax_rate = tax_expense / pretax if pretax else 0.21
        risk_free = (packet.estimates or {}).get("risk_free_rate")
        ebit, interest = _annual_field(annual, "ebit"), _annual_field(annual, "interest_expense")

        assumptions: list[str] = []
        if interest is None or interest == 0:
            interest_coverage = DEFAULT_INTEREST_COVERAGE_ASSUMPTION
            assumptions.append(
                f"ASSUMPTION (evidence class A): interest expense not in packet; assumed interest "
                f"coverage {DEFAULT_INTEREST_COVERAGE_ASSUMPTION}x for synthetic cost of debt"
            )
        else:
            interest_coverage = ebit / interest if ebit else DEFAULT_INTEREST_COVERAGE_ASSUMPTION

        if risk_free is not None and beta is not None:
            ke = cost_of_equity(risk_free, beta, erp)
            kd = synthetic_kd(risk_free, interest_coverage)
            market_cap = price.value * diluted_shares.value
            wacc_value = _wacc_fn(e=market_cap, d=total_debt.value, ke=ke, kd=kd, tax_rate=tax_rate)
            wacc_components = {"risk_free": risk_free, "beta": beta, "erp": erp, "ke": ke, "kd": kd, "tax_rate": tax_rate}
        else:
            judgment_requests.append(
                JudgmentRequest(
                    request_id="valuation.beta",
                    agent_id=AGENT_ID,
                    metric_id="beta_and_erp",
                    question="What bottom-up beta and equity risk premium should be used (no benchmark series in this packet to compute beta directly)?",
                    schema_hint="{beta: number, erp: number}",
                )
            )
    else:
        assumptions = ["core valuation facts (revenue, diluted shares, cash, debt, price) are incomplete/conflicted in the facts table"]

    # --- scenarios --------------------------------------------------
    scenario_results: list[ScenarioInput] = []
    base_value_per_share = None
    fcff_ev = None
    ep_ev = None
    revenue0 = revenue.value if revenue and revenue.is_valid else None
    hist_cagr_v = None

    if have_core_facts and wacc_value is not None and revenue0 is not None:
        n_years = min(5, len(annual) - 1)
        if n_years >= 1 and annual[0].get("revenue") and annual[n_years].get("revenue"):
            hist_cagr_v = _cagr(annual[0]["revenue"], annual[n_years]["revenue"], n_years)
        base_growth = hist_cagr_v.value if hist_cagr_v and hist_cagr_v.is_valid else 0.05
        base_margin = _historical_avg_op_margin(annual) or 0.15
        dna_pct, capex_pct, dnwc_pct = _capex_dna_dnwc_pct(annual)
        tv_growth = min(base_growth, 0.03)

        scenario_defs = [
            ("Bear", 0.25, base_growth - 0.05, base_margin - 0.02, wacc_value + 0.01, min(tv_growth, 0.02)),
            ("Base", 0.50, base_growth, base_margin, wacc_value, tv_growth),
            ("Bull", 0.25, base_growth + 0.05, base_margin + 0.02, wacc_value - 0.01, tv_growth),
        ]
        for name, prob, growth, margin, scen_wacc, scen_tv_g in scenario_defs:
            if scen_tv_g >= scen_wacc:
                scen_tv_g = scen_wacc - 0.01
            fcffs = _forecast_fcffs(revenue0, growth, margin, FORECAST_YEARS, 0.21, dna_pct, capex_pct, dnwc_pct)
            dcf = dcf_value(fcffs, scen_wacc, scen_tv_g)
            if dcf.ev is None:
                continue
            equity = equity_bridge(dcf.ev, cash=cash.value, debt=total_debt.value)
            ps = per_share(equity, diluted_shares.value)
            value = ps.value if ps.is_valid else 0.0
            scenario_results.append(ScenarioInput(label=name, probability=prob, growth=growth, margin=margin, wacc=scen_wacc, tv_growth=scen_tv_g, value=value))
            if name == "Base":
                base_value_per_share = value
                fcff_ev = dcf.ev

        # economic-profit cross-check on the base case
        if base_value_per_share is not None:
            nopat0 = nopat(revenue0 * base_margin, 0.21)
            eps_list = []
            ic0 = total_debt.value + (price.value * diluted_shares.value) - cash.value
            ic = ic0
            for _ in range(FORECAST_YEARS):
                roic_v = _roic_fn(nopat0, ic).value if ic else None
                ep = (roic_v - wacc_value) * ic if roic_v is not None else 0.0
                eps_list.append(ep)
            ep_result = economic_profit_value(ic0, eps_list, wacc_value, tv_growth)
            ep_ev = ep_result.get("ev")

    weighted_scenario_value = None
    if len(scenario_results) == 3:
        weighted = _scenarios_fn(scenario_results)
        weighted_scenario_value = weighted["weighted_value"]

    # --- reverse DCF -----------------------------------------------
    reverse_dcf_out: dict = {}
    if have_core_facts and wacc_value is not None and revenue0 is not None and price.value:
        dna_pct, capex_pct, dnwc_pct = _capex_dna_dnwc_pct(annual)
        base_margin = _historical_avg_op_margin(annual) or 0.15
        inputs = ReverseDCFInputs(
            revenue0=revenue0, years=FORECAST_YEARS, tax_rate=0.21, wacc=wacc_value,
            terminal_growth=min(0.03, wacc_value - 0.01), dna_pct_revenue=dna_pct,
            capex_pct_revenue=capex_pct, dnwc_pct_revenue=dnwc_pct,
            net_debt_and_claims=total_debt.value - cash.value, diluted_shares=diluted_shares.value,
        )
        try:
            implied_growth = reverse_dcf_implied_growth(price.value, base_margin, inputs)
            reverse_dcf_out = {"current_price": price.value, "implied_revenue_cagr": implied_growth, "implied_margin": base_margin}
        except ValueError:
            reverse_dcf_out = {"current_price": price.value, "warning": "reverse DCF did not converge within bounds"}

    # --- Monte Carlo ---------------------------------------------------
    mc_result = None
    if len(scenario_results) == 3 and revenue0 is not None:
        bear, base, bull = scenario_results
        dna_pct, capex_pct, dnwc_pct = _capex_dna_dnwc_pct(annual)
        mc_result = monte_carlo(
            revenue0=revenue0, years=FORECAST_YEARS, tax_rate=0.21, dna_pct_revenue=dna_pct,
            capex_pct_revenue=capex_pct, dnwc_pct_revenue=dnwc_pct,
            net_debt_and_claims=total_debt.value - cash.value, diluted_shares=diluted_shares.value,
            growth_range=(bear.growth, base.growth, bull.growth), margin_range=(bear.margin, base.margin, bull.margin),
            wacc_range=(bull.wacc, base.wacc, bear.wacc), terminal_growth=base.tv_growth, n=2000, seed=42,
        )

    # --- ensemble / margin of safety ------------------------------------
    ensemble_out = None
    mos_v = None
    if base_value_per_share is not None and price and price.is_valid:
        model_values = [(base_value_per_share, 0.6)]
        if ep_ev is not None and diluted_shares.is_valid:
            ep_equity = equity_bridge(ep_ev, cash=cash.value, debt=total_debt.value)
            ep_ps = per_share(ep_equity, diluted_shares.value)
            if ep_ps.is_valid:
                model_values.append((ep_ps.value, 0.4))
        ensemble_out = ensemble(model_values)
        mos_v = margin_of_safety(ensemble_out["value"], price.value)

    # --- dimensions -------------------------------------------------
    growth_adjusted_score = None
    ni, eps_diluted = _annual_field(annual, "net_income"), _annual_field(annual, "eps_diluted")
    if price and price.is_valid and eps_diluted and hist_cagr_v and hist_cagr_v.is_valid:
        equity0 = (price.value * diluted_shares.value) if diluted_shares and diluted_shares.is_valid else None
        roe = ni / (total_debt.value * 0 + (equity0 or 1)) if ni and equity0 else None  # market-cap-based proxy ROE, documented
        if roe and wacc_value:
            jpe = justified_pe(hist_cagr_v.value, roe, wacc_value + erp * 0)  # use cost of equity proxy via wacc when ke unavailable
            actual_pe = price.value / eps_diluted if eps_diluted else None
            if jpe.is_valid and actual_pe and jpe.value:
                ratio = actual_pe / jpe.value
                growth_adjusted_score = anchor_score(ratio, [(0.7, 10), (1.0, 6), (1.2, 3), (2.0, 0)])

    peer_score = None  # no peer multiple panel in this packet -> NOT_SCORABLE, not proxied

    cf_yield_score = None
    if price and price.is_valid and eps_diluted:
        earnings_yield = eps_diluted / price.value
        cf_yield_score = anchor_score(earnings_yield, [(-0.05, 0), (0.0, 3), (0.03, 6), (0.06, 10)])

    fair_value_score = None
    if price and price.is_valid and weighted_scenario_value is not None:
        rel = (weighted_scenario_value - price.value) / price.value
        fair_value_score = anchor_score(rel, [(-0.30, 0), (0.0, 5), (0.20, 8), (0.50, 10)])

    mos_score = None
    if mos_v is not None and mos_v.is_valid:
        mos_score = anchor_score(mos_v.value, [(-0.10, 2), (0.0, 4), (0.15, 7), (0.30, 10)])

    dims_spec = [
        ("growth_adjusted_multiples", 3.0, growth_adjusted_score),
        ("historical_and_peer_comparison", 2.0, peer_score),
        ("cash_flow_and_earnings_yield", 2.0, cf_yield_score),
        ("fair_value_by_scenarios", 2.0, fair_value_score),
        ("margin_of_safety", 1.0, mos_score),
    ]
    dims = []
    for name, max_pts, score in dims_spec:
        if score is None:
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[]))
        else:
            v = Value.of(score, unit="score", evidence_class=EvidenceClass.C)
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[(1.0, v)]))

    category = Category(name="valuation", max_points=MAX_POINTS, dimensions=dims)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    mandatory_flags: list[str] = []

    metrics = [
        MetricRow(metric_id="wacc", value=wacc_value, formula="VAL-WACC-007", unit="ratio",
                  score=None, evidence_class=str(EvidenceClass.C), source="derived"),
        MetricRow(metric_id="base_case_per_share_value", value=base_value_per_share, formula="VAL-PS-016",
                  unit="usd_per_share", score=None, evidence_class=str(EvidenceClass.C), source="derived"),
        MetricRow(metric_id="margin_of_safety", value=mos_v.value if mos_v and mos_v.is_valid else None,
                  formula="VAL-MOS-040", unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="derived"),
    ]

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return ValuationOutput(
        agent_id=AGENT_ID,
        status=status,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange, "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategorySummary(max_points=MAX_POINTS, awarded_points=awarded, score_10=score_10, confidence=coverage * 100),
        coverage=coverage,
        dimensions=[{"name": d.name, "max_points": d.max_points, "score_10": _dim_score10_or_none(d)} for d in dims],
        metrics=metrics,
        mandatory_flags=mandatory_flags,
        assumptions=assumptions if have_core_facts else ["core valuation facts incomplete/conflicted"],
        judgment_requests=judgment_requests,
        source_lineage=["packet.facts_table", "packet.fundamentals.annual"],
        model_selection=model_selection,
        wacc={"value": wacc_value, "components": wacc_components, "sensitivity": []},
        scenarios=[s.model_dump() for s in scenario_results],
        reverse_dcf=reverse_dcf_out,
        model_cross_checks={"fcff": fcff_ev, "economic_profit": ep_ev},
        fair_value_distribution=mc_result or {},
        reference_bands={
            "bear": scenario_results[0].value if scenario_results else None,
            "base": base_value_per_share,
            "bull": scenario_results[2].value if len(scenario_results) == 3 else None,
            "margin_of_safety_15pct": (base_value_per_share * 0.85) if base_value_per_share else None,
            "margin_of_safety_25pct": (base_value_per_share * 0.75) if base_value_per_share else None,
        },
    )


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value
