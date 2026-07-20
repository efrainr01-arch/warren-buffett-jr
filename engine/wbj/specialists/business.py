"""Business specialist (20 pts) — Cerebro/01_business_analysis/.

Implements BUS-001..030 (FORMULAS.md), the five weighted dimensions
(SCORING.md), the wide-moat gate and mandatory flags (DECISION_RULES.md),
and the extension fields (OUTPUT_SCHEMA.md). ROIC/spread/EVA/incremental-
ROIC reuse `wbj.engines.valuation_engine`'s Task-13 functions.

The packet built in Task 10 does not carry segment revenue, customer
concentration, recurring-revenue %, market share, peer panels, guidance,
or subscription-cohort data (NRR/GRR/churn/LTV/CAC) — several of those
(customer_concentration, market_share, organic_growth) are on Cerebro's
MISSING_DATA_POLICY.md prohibited-imputation list, so they are correctly
MISSING/NOT_SCORABLE here rather than estimated. Moat classification and
the "≥2 quantitative moat effects" wide-moat condition are judgment
requests per the plan (qualitative evidence, not mechanically derivable).
"""

from __future__ import annotations

import statistics
from typing import Literal

from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension, anchor_score
from wbj.engines.valuation_engine import eva as _eva, incremental_roic as _incremental_roic, nopat as _nopat, roic as _roic_fn, spread as _spread
from wbj.specialists.common import CategorySummary, JudgmentRequest, MetricRow, SpecialistOutput

AGENT_ID = "business_analysis"
MAX_POINTS = float(CATEGORY_WEIGHTS["business"])  # 20
WIDE_MOAT_SPREAD_THRESHOLD = 0.05
WIDE_MOAT_YEARS_REQUIRED = 4
WIDE_MOAT_MARGIN_RANGE_MAX = 0.05
CONCENTRATION_THRESHOLD = 0.30
DILUTION_CAGR_THRESHOLD = 0.05

MoatClass = Literal["Wide", "Narrow", "None", "NotScorable"]


def _row(annual: list[dict], i: int) -> dict:
    return annual[i] if i < len(annual) else {}


def _nopat_ic_for_row(row: dict) -> tuple[float | None, float | None]:
    ebit = row.get("ebit")
    debt, equity, cash = row.get("total_debt"), row.get("total_equity"), row.get("cash")
    pretax, tax_expense = row.get("pretax_income"), row.get("income_tax_expense")
    if ebit is None or debt is None or equity is None or cash is None:
        return None, None
    tax_rate = tax_expense / pretax if pretax else 0.21
    return _nopat(ebit, tax_rate), debt + equity - cash


def _roic_history(annual: list[dict], years: int = 5) -> list[float | None]:
    """ROIC per fiscal year, newest-first, using average(begin, end) invested capital."""
    out = []
    for i in range(min(years, max(len(annual) - 1, 0))):
        nopat_t, ic_t = _nopat_ic_for_row(_row(annual, i))
        _, ic_prev = _nopat_ic_for_row(_row(annual, i + 1))
        if nopat_t is None or ic_t is None or ic_prev is None:
            out.append(None)
            continue
        avg_ic = (ic_t + ic_prev) / 2
        v = _roic_fn(nopat_t, avg_ic)
        out.append(v.value if v.is_valid else None)
    return out


def _margin_series(annual: list[dict], years: int = 5) -> list[float | None]:
    out = []
    for i in range(min(years, len(annual))):
        row = _row(annual, i)
        rev, ebit = row.get("revenue"), row.get("ebit")
        out.append(ebit / rev if rev else None)
    return out


def _cumulative_fcf_conversion(annual: list[dict], years: int = 5) -> float | None:
    fcfs, nis = [], []
    for i in range(min(years, len(annual))):
        row = _row(annual, i)
        ocf, capex, ni = row.get("operating_cash_flow"), row.get("capex"), row.get("net_income")
        if ocf is None or capex is None or ni is None:
            continue
        fcfs.append(ocf + capex)
        nis.append(ni)
    total_ni = sum(nis)
    if not nis or total_ni == 0:
        return None
    return sum(fcfs) / total_ni


def _diluted_share_cagr(annual: list[dict], years: int = 5) -> float | None:
    """BUS-DIL-028."""
    n = min(years, len(annual) - 1)
    if n < 1:
        return None
    d_end, d_begin = _row(annual, 0).get("diluted_shares"), _row(annual, n).get("diluted_shares")
    if d_end is None or d_begin is None:
        return None
    v = _cagr(d_end, d_begin, n)
    return v.value if v.is_valid else None


def _revenue_cagr(annual: list[dict], years: int = 5) -> Value:
    """BUS-CAGR-006: revenue CAGR, NOT_MEANINGFUL across a non-positive
    beginning value or a sign change (delegates to `wbj.core.formulas.cagr`)."""
    n = min(years, len(annual) - 1)
    if n < 1:
        return Value.null(NullState.MISSING, unit="pct")
    rev_end, rev_begin = _row(annual, 0).get("revenue"), _row(annual, n).get("revenue")
    if rev_end is None or rev_begin is None:
        return Value.null(NullState.MISSING, unit="pct")
    return _cagr(rev_end, rev_begin, n)


def _incremental_roic_3y(annual: list[dict]) -> float | None:
    if len(annual) < 4:
        return None
    nopat_t, ic_t = _nopat_ic_for_row(_row(annual, 0))
    nopat_t3, ic_t3 = _nopat_ic_for_row(_row(annual, 3))
    if None in (nopat_t, ic_t, nopat_t3, ic_t3):
        return None
    v = _incremental_roic(nopat_t - nopat_t3, ic_t - ic_t3)
    return v.value if v.is_valid else None


def _sbc_burden(annual: list[dict]) -> float | None:
    row = _row(annual, 0)
    sbc, rev = row.get("stock_based_comp"), row.get("revenue")
    if sbc is None or rev is None or rev == 0:
        return None
    return sbc / rev


class BusinessOutput(SpecialistOutput):
    business_in_one_sentence: str | None = None
    moat: dict = {}
    roic_history: list = []
    roic_wacc_spread_history: list = []
    margin_stability: dict = {}
    customer_economics: dict = {}
    capital_allocation: dict = {}
    competitive_position: dict = {}
    three_thesis_killers: list = []


def _classify_moat(
    spread_years_pass: bool | None,
    margin_range_pass: bool | None,
    moat_effects_count: int | None,
    concentration_flag: bool,
) -> MoatClass:
    if spread_years_pass is False:
        return "None"
    if moat_effects_count is None or spread_years_pass is None or margin_range_pass is None:
        return "NotScorable"
    if spread_years_pass and margin_range_pass and moat_effects_count >= 2 and not concentration_flag:
        return "Wide"
    if spread_years_pass or moat_effects_count >= 1:
        return "Narrow"
    return "None"


def run(packet, wacc: float | None = None, overlay: dict | None = None) -> BusinessOutput:
    overlay = overlay or {}
    annual = packet.fundamentals.get("annual", [])

    roic_hist = _roic_history(annual)
    spread_hist = [(r - wacc) if r is not None and wacc is not None else None for r in roic_hist]
    valid_spreads = [s for s in spread_hist if s is not None]
    spread_avg = statistics.mean(valid_spreads) if valid_spreads else None

    years_with_5pt = sum(1 for s in spread_hist if s is not None and s >= WIDE_MOAT_SPREAD_THRESHOLD)
    spread_years_pass = (years_with_5pt >= WIDE_MOAT_YEARS_REQUIRED) if valid_spreads else None

    margins = _margin_series(annual)
    valid_margins = [m for m in margins if m is not None]
    margin_range = (max(valid_margins) - min(valid_margins)) if len(valid_margins) >= 2 else None
    margin_stability = statistics.pstdev(valid_margins) if len(valid_margins) >= 2 else None
    margin_range_pass = (margin_range <= WIDE_MOAT_MARGIN_RANGE_MAX) if margin_range is not None else None

    fcf_conversion = _cumulative_fcf_conversion(annual)
    diluted_cagr = _diluted_share_cagr(annual)
    incremental_roic_v = _incremental_roic_3y(annual)
    capital_allocation_spread = (incremental_roic_v - wacc) if incremental_roic_v is not None and wacc is not None else None
    sbc_burden = _sbc_burden(annual)

    judgment_requests: list[JudgmentRequest] = []
    moat_effects_count = overlay.get("moat_quantitative_effects_count")
    if moat_effects_count is None:
        judgment_requests.append(
            JudgmentRequest(
                request_id="business.moat_effects_count",
                agent_id=AGENT_ID,
                metric_id="moat_quantitative_effects_count",
                question=(
                    "How many independent, quantitatively-visible moat effects does this business show "
                    "(retention/switching costs, cost advantage, network scale, regulated/intangible "
                    "protection, efficient scale)? Provide an integer count and cite the evidence."
                ),
                schema_hint="integer >= 0",
            )
        )

    concentration_flag = bool(overlay.get("largest_customer_concentration", 0) > CONCENTRATION_THRESHOLD)
    moat_classification = _classify_moat(spread_years_pass, margin_range_pass, moat_effects_count, concentration_flag)

    # --- dimension scores ---------------------------------------------

    moat_score = 3.0
    if spread_avg is not None:
        moat_score = anchor_score(spread_avg, [(-0.05, 0), (0.0, 3), (0.03, 5), (0.05, 7.5), (0.10, 10)])
        if spread_avg <= 0:
            moat_score = min(moat_score, 6.0)
    else:
        moat_score = None
    if moat_score is not None and not (spread_years_pass and margin_range_pass):
        moat_score = min(moat_score, 6.9)  # reserve 7-10 for the fuller wide-moat evidence bar

    competitive_score = None  # market share / peer economics not in this packet (prohibited imputation)

    management_score = None
    if capital_allocation_spread is not None:
        management_score = anchor_score(capital_allocation_spread, [(-0.05, 0), (0.0, 3), (0.02, 6), (0.05, 9)])
    if diluted_cagr is not None and diluted_cagr > DILUTION_CAGR_THRESHOLD and management_score is not None:
        management_score = min(management_score, 5.0)

    durability_score = None
    if roic_hist and any(v is not None for v in roic_hist):
        stable_positive_roic = sum(1 for v in roic_hist if v is not None and v > 0)
        durability_score = anchor_score(stable_positive_roic, [(0, 2), (2, 5), (4, 7), (5, 9)])
        if margin_stability is not None:
            durability_score = anchor_score(margin_stability, [(0.15, 3), (0.05, 6), (0.0, 9)]) if margin_stability is not None else durability_score
    if concentration_flag and durability_score is not None:
        durability_score = min(durability_score, 6.0)

    customer_economics_score = None  # no subscription-cohort data (NRR/GRR/churn/LTV/CAC) in this packet

    dims = []
    for name, max_pts, score in [
        ("moat_and_pricing_power", 5.0, moat_score),
        ("competitive_position", 4.0, competitive_score),
        ("management_and_capital_allocation", 4.0, management_score),
        ("business_durability", 4.0, durability_score),
        ("customer_economics", 3.0, customer_economics_score),
    ]:
        if score is None:
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[]))
        else:
            v = Value.of(score, unit="score", evidence_class=EvidenceClass.C)
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[(1.0, v)]))

    category = Category(name="business", max_points=MAX_POINTS, dimensions=dims)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    mandatory_flags: list[str] = []
    if roic_hist and roic_hist[0] is not None and wacc is not None and roic_hist[0] < wacc:
        mandatory_flags.append("VALUE_DESTRUCTION")
    if concentration_flag:
        mandatory_flags.append("CONCENTRATION_RED_FLAG")
    if diluted_cagr is not None and diluted_cagr > DILUTION_CAGR_THRESHOLD:
        mandatory_flags.append("DILUTION_RED_FLAG")

    revenue_cagr_v = _revenue_cagr(annual)

    metrics = [
        MetricRow(metric_id="revenue_cagr_5y", value=revenue_cagr_v.value, formula="BUS-CAGR-006",
                  state=str(revenue_cagr_v.state) if revenue_cagr_v.is_null else None,
                  unit="pct", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals",
                  warnings=list(revenue_cagr_v.warnings)),
        MetricRow(metric_id="roic_latest", value=roic_hist[0] if roic_hist else None, formula="BUS-ROIC-013",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="margin_range_5y", value=margin_range, formula="BUS-RANGE-010",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="margin_stability_5y", value=margin_stability, formula="BUS-STAB-009",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="cumulative_fcf_conversion_5y", value=fcf_conversion, formula="BUS-FCFC-017",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="incremental_roic_3y", value=incremental_roic_v, formula="BUS-IROIC-016",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="diluted_share_cagr", value=diluted_cagr, formula="BUS-DIL-028",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="sbc_burden", value=sbc_burden, formula="BUS-SBC-030",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
    ]

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return BusinessOutput(
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
            "competitive_position and customer_economics dimensions require peer/market-share and "
            "subscription-cohort data not present in this packet; correctly NOT_SCORABLE rather than imputed",
        ],
        judgment_requests=judgment_requests,
        source_lineage=["packet.fundamentals.annual"],
        business_in_one_sentence=None,
        moat={
            "classification": moat_classification,
            "quantitative_evidence": [],
            "spread_years_pass": spread_years_pass,
            "margin_range_pass": margin_range_pass,
        },
        roic_history=roic_hist,
        roic_wacc_spread_history=spread_hist,
        margin_stability={"stdev_5y": margin_stability, "range_5y": margin_range},
        customer_economics={},
        capital_allocation={
            "incremental_roic_3y": incremental_roic_v,
            "capital_allocation_spread": capital_allocation_spread,
            "diluted_share_cagr": diluted_cagr,
        },
        competitive_position={},
        three_thesis_killers=[],
    )


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value
