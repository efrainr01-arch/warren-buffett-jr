"""Risk & Resilience specialist (15 pts, higher = safer) —
Cerebro/05_risk_analysis/.

Implements RSK-001..035 (FORMULAS.md), the six weighted dimensions
(SCORING.md), the resilience anchors, mandatory solvency warning, and
forensic-screen rules (DECISION_RULES.md). Per DECISION_RULES.md's
"Direction rule": a high category score means resilience — raw risk
measures are inverted before scoring.

Market beta and downside beta need aligned benchmark returns, which the
Task-10 packet builder does not populate (`market_data.benchmark` is
empty) — beta is on no imputation list but is nonetheless never proxied
here; it is correctly MISSING. Customer/product/geographic concentration
(customer_concentration is a prohibited-imputation metric) and the
regulatory/legal event registry and valuation-agent handoff aren't in
this packet either, so those dimensions are NOT_SCORABLE or judgment-
gated rather than fabricated.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal

import numpy as np

from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension, anchor_score
from wbj.specialists.common import CategorySummary, JudgmentRequest, MetricRow, SpecialistOutput

AGENT_ID = "risk_analysis"
MAX_POINTS = float(CATEGORY_WEIGHTS["risk"])  # 15
SOLVENCY_WARNING_TEXT = "SOLVENCY_WARNING: Operating earnings do not provide a comfortable interest buffer."
BENEISH_M_SCREEN_THRESHOLD = -1.78
RISK_OVERRIDE_POINTS_THRESHOLD = 4.0


# --- market risk ----------------------------------------------------


def annualized_vol(returns: list[float]) -> Value:
    """RSK-VOL-001."""
    if len(returns) < 2:
        return Value.null(NullState.MISSING, unit="ratio")
    return Value.of(statistics.pstdev(returns) * math.sqrt(252), unit="ratio", evidence_class=EvidenceClass.C)


def downside_deviation(returns: list[float], mar: float = 0.0) -> Value:
    """RSK-DOWN-002."""
    if not returns:
        return Value.null(NullState.MISSING, unit="ratio")
    sq = [min(r - mar, 0.0) ** 2 for r in returns]
    return Value.of(math.sqrt(statistics.mean(sq)) * math.sqrt(252), unit="ratio", evidence_class=EvidenceClass.C)


def beta(stock_returns: list[float], bench_returns: list[float]) -> Value:
    """RSK-BETA-003. Never proxied when a benchmark series is unavailable."""
    if len(stock_returns) != len(bench_returns) or len(stock_returns) < 2:
        return Value.null(NullState.MISSING, unit="ratio")
    cov = np.cov(stock_returns, bench_returns)[0, 1]
    var = np.var(bench_returns, ddof=1)
    if var == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(float(cov / var), unit="ratio", evidence_class=EvidenceClass.C)


def downside_beta(stock_returns: list[float], bench_returns: list[float], min_obs: int = 30) -> Value:
    """RSK-DBETA-004: minimum 30 down-market observations."""
    pairs = [(s, b) for s, b in zip(stock_returns, bench_returns) if b < 0]
    if len(pairs) < min_obs:
        return Value.null(NullState.NOT_SCORABLE, unit="ratio", warnings=[f"fewer than {min_obs} down-market observations ({len(pairs)})"])
    s_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    return beta(s_vals, b_vals)


def max_drawdown(prices: list[float]) -> dict:
    """RSK-MDD-006: returns {mdd, peak_index, trough_index}."""
    if not prices:
        return {"mdd": None, "peak_index": None, "trough_index": None}
    peak = prices[0]
    peak_idx = 0
    mdd = 0.0
    mdd_peak_idx = 0
    mdd_trough_idx = 0
    for i, p in enumerate(prices):
        if p > peak:
            peak = p
            peak_idx = i
        dd = p / peak - 1
        if dd < mdd:
            mdd = dd
            mdd_peak_idx = peak_idx
            mdd_trough_idx = i
    return {"mdd": mdd, "peak_index": mdd_peak_idx, "trough_index": mdd_trough_idx}


def historical_var(returns: list[float], confidence: float = 0.95) -> Value:
    """RSK-VAR-008."""
    if not returns:
        return Value.null(NullState.MISSING, unit="ratio")
    q = float(np.percentile(returns, (1 - confidence) * 100))
    return Value.of(-q, unit="ratio", evidence_class=EvidenceClass.C)


def cvar(returns: list[float], confidence: float = 0.95) -> Value:
    """RSK-CVAR-009: minimum 500 observations preferred for tail stability."""
    if not returns:
        return Value.null(NullState.MISSING, unit="ratio")
    var_v = historical_var(returns, confidence)
    tail = [r for r in returns if r <= -var_v.value]
    warnings = [] if len(returns) >= 500 else [f"fewer than 500 observations ({len(returns)}); tail estimate less stable"]
    if not tail:
        return Value.of(var_v.value, unit="ratio", evidence_class=EvidenceClass.C, warnings=warnings)
    return Value.of(-statistics.mean(tail), unit="ratio", evidence_class=EvidenceClass.C, warnings=warnings)


# --- liquidity and solvency -------------------------------------------


def interest_coverage(ebit: float, interest: float) -> Value:
    """RSK-ICOV-011."""
    if interest == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(ebit / interest, unit="ratio", evidence_class=EvidenceClass.C)


def net_debt_to_ebitda(net_debt: float, ebitda: float) -> Value:
    """RSK-ND-013: not meaningful for negative EBITDA."""
    if ebitda <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(net_debt / ebitda, unit="ratio", evidence_class=EvidenceClass.C)


def cash_runway(cash: float, committed_liquidity: float, monthly_burn: float) -> Value:
    """RSK-RUN-015: only meaningful for negative FCF/burn."""
    if monthly_burn <= 0:
        return Value.null(NullState.NOT_APPLICABLE, unit="months", warnings=["not a cash burner"])
    return Value.of((cash + committed_liquidity) / monthly_burn, unit="months", evidence_class=EvidenceClass.C)


def maturity_wall_coverage(cash: float, expected_fcf_before_maturity: float, committed_liquidity: float, debt_due: float) -> Value:
    """RSK-MAT-016: values <1 imply refinancing need."""
    if debt_due == 0:
        return Value.null(NullState.NOT_APPLICABLE, unit="ratio", warnings=["no debt due in this horizon"])
    return Value.of((cash + expected_fcf_before_maturity + committed_liquidity) / debt_due, unit="ratio", evidence_class=EvidenceClass.C)


# --- concentration -----------------------------------------------------


def concentration_hhi(shares: list[float]) -> Value:
    """RSK-CUST-017..019."""
    if not shares:
        return Value.null(NullState.MISSING, unit="ratio")
    return Value.of(sum(s ** 2 for s in shares), unit="ratio", evidence_class=EvidenceClass.C)


# --- execution / earnings quality ------------------------------------


def accrual_ratio(net_income: float, ocf: float, avg_total_assets: float) -> Value:
    """RSK-ACCR-020."""
    if avg_total_assets == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of((net_income - ocf) / avg_total_assets, unit="ratio", evidence_class=EvidenceClass.C)


def beneish_m_score(dsri: float, gmi: float, aqi: float, sgi: float, depi: float, sgai: float, tata: float, lvgi: float) -> float:
    """RSK-MSCR-029."""
    return -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi


def altman_z_double_prime(working_capital: float, total_assets: float, retained_earnings: float, ebit: float, book_equity: float, total_liabilities: float) -> Value:
    """RSK-ALT-030."""
    if total_assets == 0 or total_liabilities == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="score")
    z = (
        6.56 * (working_capital / total_assets)
        + 3.26 * (retained_earnings / total_assets)
        + 6.72 * (ebit / total_assets)
        + 1.05 * (book_equity / total_liabilities)
    )
    return Value.of(z, unit="score", evidence_class=EvidenceClass.C)


def piotroski_f_score(signals: dict[str, bool]) -> int:
    """RSK-PIO-031: sum of up to 9 binary signals (0-9)."""
    return sum(1 for v in signals.values() if v)


def sbc_to_fcf(sbc: float, fcf: float, materiality_floor: float = 1.0) -> Value:
    """RSK-SBC-033."""
    denom = max(abs(fcf), materiality_floor)
    return Value.of(sbc / denom, unit="ratio", evidence_class=EvidenceClass.C, warnings=([] if fcf > 0 else ["denominator uses |FCF|, FCF is not positive"]))


def thesis_killer_priority(probability: float, impact: float, detectability: float, time_urgency: float) -> float:
    """RSK-THESIS-035."""
    return probability * impact * (1 - detectability) * time_urgency


def is_financial_sector(security_type: str | None) -> bool:
    """DECISION_RULES.md forensic-screen rule: exclude financial companies
    and other inapplicable industries from Altman/Beneish scoring."""
    return (security_type or "").lower() in {"bank", "insurer", "insurance", "financial"}


class RiskOutput(SpecialistOutput):
    market_risk: dict = {}
    liquidity_and_solvency: dict = {}
    concentrations: dict = {}
    earnings_quality_and_forensics: dict = {}
    regulatory_legal_macro: list = []
    valuation_compression: dict = {}
    thesis_killers: list = []
    mandatory_warnings: list = []


def _annual_field(annual: list[dict], field_name: str, i: int = 0) -> float | None:
    if i >= len(annual):
        return None
    return annual[i].get(field_name)


def _daily_returns(daily_rows) -> list[float]:
    """Packet OHLCV is newest-first; returns oldest-first log returns."""
    closes = [r.adj_close for r in reversed(daily_rows)]
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]


def run(packet, overlay: dict | None = None) -> RiskOutput:
    overlay = overlay or {}
    annual = packet.fundamentals.get("annual", [])
    daily = packet.market_data.daily

    judgment_requests: list[JudgmentRequest] = []
    mandatory_flags: list[str] = []
    mandatory_warnings: list[str] = []

    # --- financing / balance-sheet ---------------------------------
    ebit, interest = _annual_field(annual, "ebit"), _annual_field(annual, "interest_expense")
    coverage_v = interest_coverage(ebit, interest) if ebit is not None and interest else Value.null(NullState.MISSING)
    if coverage_v.is_valid and coverage_v.value < 1.5:
        mandatory_flags.append("SOLVENCY_WARNING")
        mandatory_warnings.append(SOLVENCY_WARNING_TEXT)

    financing_score = None
    if coverage_v.is_valid:
        financing_score = anchor_score(coverage_v.value, [(0.5, 0), (1.5, 3), (3.0, 6.5), (5.0, 10)])

    # --- concentration: not in this packet (prohibited imputation) ----
    concentration_score = None
    judgment_requests.append(
        JudgmentRequest(
            request_id="risk.concentration",
            agent_id=AGENT_ID,
            metric_id="customer_product_geo_concentration",
            question="What are the largest customer/product/geographic revenue concentrations (as shares)?",
            schema_hint="{largest_customer_share: number, largest_product_share: number, largest_geo_share: number}",
        )
    )

    # --- execution & earnings quality: accrual ratio + M-score ---------
    ni, ocf = _annual_field(annual, "net_income"), _annual_field(annual, "operating_cash_flow")
    ta0, ta1 = _annual_field(annual, "total_assets", 0), _annual_field(annual, "total_assets", 1)
    accrual_v = Value.null(NullState.MISSING)
    if ni is not None and ocf is not None and ta0 is not None and ta1 is not None:
        accrual_v = accrual_ratio(ni, ocf, (ta0 + ta1) / 2)

    beneish_flag = None
    m_score = None
    financial_sector = is_financial_sector(packet.security.security_type)
    if not financial_sector and len(annual) >= 2:
        m_score = overlay.get("beneish_m_score")  # full 8-component calc needs many extra fields; overlay-suppliable
        if m_score is not None and m_score > BENEISH_M_SCREEN_THRESHOLD:
            beneish_flag = "FORENSIC_SCREEN: Beneish M-score above the -1.78 screening threshold (a screening flag, not an accusation of manipulation)"

    execution_score = None
    if accrual_v.is_valid:
        execution_score = anchor_score(accrual_v.value, [(0.10, 0), (0.02, 6), (-0.02, 9), (-0.10, 10)])
        if beneish_flag:
            execution_score = min(execution_score, 5.0)

    # --- regulatory/legal/macro: no event registry in this packet -------
    regulatory_score = None
    judgment_requests.append(
        JudgmentRequest(
            request_id="risk.regulatory_legal_macro",
            agent_id=AGENT_ID,
            metric_id="regulatory_legal_macro_events",
            question="List material unresolved regulatory/legal/macro threats with estimated probability and impact.",
            schema_hint="[{event, probability_assumption, impact, time_horizon}]",
        )
    )

    # --- valuation compression: cross-agent handoff -----------------
    valuation_compression_score = None
    margin_of_safety = overlay.get("margin_of_safety")
    if margin_of_safety is not None:
        valuation_compression_score = anchor_score(margin_of_safety, [(-0.30, 0), (0.0, 5), (0.15, 8), (0.30, 10)])

    # --- volatility & drawdown -----------------------------------------
    returns = _daily_returns(daily)
    vol_v = annualized_vol(returns) if returns else Value.null(NullState.MISSING)
    prices_3y = [r.adj_close for r in reversed(daily)][-756:]
    mdd = max_drawdown(prices_3y) if prices_3y else {"mdd": None}

    volatility_score = None
    if mdd.get("mdd") is not None:
        volatility_score = anchor_score(mdd["mdd"], [(-0.80, 0), (-0.60, 3), (-0.30, 6.5), (-0.10, 10)])

    dims_spec = [
        ("financing_and_balance_sheet_risk", 3.0, financing_score),
        ("competition_and_concentration_risk", 3.0, concentration_score),
        ("execution_and_earnings_quality_risk", 3.0, execution_score),
        ("regulatory_legal_and_macro_risk", 2.0, regulatory_score),
        ("valuation_compression_risk", 2.0, valuation_compression_score),
        ("volatility_and_drawdown_risk", 2.0, volatility_score),
    ]
    dims = []
    for name, max_pts, score in dims_spec:
        if score is None:
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[]))
        else:
            v = Value.of(score, unit="score", evidence_class=EvidenceClass.C)
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[(1.0, v)]))

    category = Category(name="risk", max_points=MAX_POINTS, dimensions=dims)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    if awarded <= RISK_OVERRIDE_POINTS_THRESHOLD:
        mandatory_flags.append(f"RISK_OVERRIDE_CANDIDATE: category points {awarded:.2f} <= {RISK_OVERRIDE_POINTS_THRESHOLD}/15 caps the main profile at Speculative")

    metrics = [
        MetricRow(metric_id="interest_coverage", value=coverage_v.value, formula="RSK-ICOV-011",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="accrual_ratio", value=accrual_v.value, formula="RSK-ACCR-020",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="annualized_volatility", value=vol_v.value, formula="RSK-VOL-001",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="max_drawdown_3y", value=mdd.get("mdd"), formula="RSK-MDD-006",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
    ]

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return RiskOutput(
        agent_id=AGENT_ID,
        status=status,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange, "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategorySummary(max_points=MAX_POINTS, awarded_points=awarded, score_10=score_10, confidence=coverage * 100),
        coverage=coverage,
        dimensions=[{"name": d.name, "max_points": d.max_points, "score_10": _dim_score10_or_none(d)} for d in dims],
        metrics=metrics,
        mandatory_flags=mandatory_flags,
        assumptions=[
            "Beta/downside beta need a benchmark series not populated by the Task-10 packet builder yet "
            "-- correctly MISSING, never proxied. Concentration and the regulatory event registry are "
            "judgment requests; customer_concentration is a prohibited-imputation metric.",
        ],
        judgment_requests=judgment_requests,
        source_lineage=["packet.fundamentals.annual", "packet.market_data.daily"],
        market_risk={"annualized_volatility": vol_v.value, "max_drawdown_3y": mdd.get("mdd")},
        liquidity_and_solvency={"interest_coverage": coverage_v.value},
        concentrations={},
        earnings_quality_and_forensics={"accrual_ratio": accrual_v.value, "beneish_m_score": m_score, "beneish_flag": beneish_flag},
        regulatory_legal_macro=[],
        valuation_compression={"margin_of_safety": margin_of_safety},
        thesis_killers=[],
        mandatory_warnings=mandatory_warnings,
    )


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value
