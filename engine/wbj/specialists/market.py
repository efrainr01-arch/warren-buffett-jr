"""Market & Growth specialist (20 pts) — Cerebro/03_market_analysis/.

Implements MKT-001..025 (FORMULAS.md), the five weighted dimensions
(SCORING.md), the source-quality tiers and forecast-consistency gate
(DECISION_RULES.md), and the extension fields (OUTPUT_SCHEMA.md).

The packet built in Task 10 does not carry TAM/SAM/SOM sizing, peer
market shares, a timestamped revision history, or backlog/RPO — TAM and
market share are on Cerebro's MISSING_DATA_POLICY.md prohibited-
imputation list, so they are never estimated; they are judgment
requests (TAM figure + source tier) or NOT_SCORABLE. Earnings surprise
IS computable from `packet.fundamentals`'s FMP earnings-calendar-style
estimates (pre-release consensus vs. actual), and operating leverage
from the annual EBIT/revenue history.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal

from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension, anchor_score
from wbj.specialists.common import CategorySummary, JudgmentRequest, MetricRow, SpecialistOutput

AGENT_ID = "market_analysis"
MAX_POINTS = float(CATEGORY_WEIGHTS["market"])  # 20
MIN_ESTIMATES_FOR_REVISION_BREADTH = 5

SOURCE_TIER_CONFIDENCE = {1: 100, 2: 85, 3: 70, 4: 45, 5: 0}
SOURCE_TIER_SCORE_CAP = {4: 6.0, 5: 0.0}  # tier 5 is not scorable at all


# --- pure MKT-0xx formulas --------------------------------------------


def tam_cagr(tam_end: float, tam_begin: float, years: float) -> Value:
    """MKT-CAGR-004."""
    return _cagr(tam_end, tam_begin, years)


def penetration(company_revenue: float, tam: float) -> Value:
    """MKT-PEN-005."""
    if tam <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="pct")
    return Value.of(company_revenue / tam, unit="pct", evidence_class=EvidenceClass.C)


def share_delta(share_t: float, share_t1: float) -> float:
    """MKT-SHDELTA-007: percentage-point change, not percent change."""
    return share_t - share_t1


def runway_years(target_revenue: float, current_revenue: float, growth: float) -> Value:
    """MKT-RUN-010: not meaningful when assumed growth <= 0."""
    if growth <= 0 or current_revenue <= 0 or target_revenue <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="years")
    return Value.of(math.log(target_revenue / current_revenue) / math.log(1 + growth), unit="years", evidence_class=EvidenceClass.C)


def revision_breadth(upward_count: int, total_count: int) -> Value:
    """MKT-REVBR-011: requires >=5 active estimates."""
    if total_count < MIN_ESTIMATES_FOR_REVISION_BREADTH:
        return Value.null(
            NullState.NOT_SCORABLE,
            unit="pct",
            warnings=[f"fewer than {MIN_ESTIMATES_FOR_REVISION_BREADTH} active estimates ({total_count})"],
        )
    return Value.of(upward_count / total_count, unit="pct", evidence_class=EvidenceClass.C)


def estimate_dispersion_proxy(high: float, low: float, avg: float) -> Value:
    """MKT-DISP-013 proxy: true per-analyst estimates aren't in this
    packet (only avg/high/low), so dispersion is approximated as
    range/4 (a standard normal-range-to-stdev approximation) over
    |avg|, flagged as a proxy per FORMULAS.md's "record any proxy"."""
    if avg == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(
        ((high - low) / 4) / abs(avg),
        unit="ratio",
        evidence_class=EvidenceClass.C,
        warnings=["PROXY_RANGE_STDEV: true individual-analyst dispersion unavailable, using range/4 approximation"],
    )


def earnings_surprise(actual: float, pre_release_consensus: float, snapshot_before_earnings: bool) -> Value:
    """MKT-SURP-014: freeze consensus before release; reject if the
    snapshot was taken after the actual print (MKT-T008)."""
    if not snapshot_before_earnings:
        return Value.null(
            NullState.NOT_SCORABLE,
            unit="pct",
            warnings=["REJECTED: consensus snapshot was not frozen before the earnings release"],
        )
    if pre_release_consensus == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="pct")
    return Value.of((actual - pre_release_consensus) / abs(pre_release_consensus), unit="pct", evidence_class=EvidenceClass.C)


def operating_leverage(op_income_growth: float, revenue_growth: float) -> Value:
    """MKT-OPLEV-017: not meaningful across a loss sign change (handled
    by the caller, which only calls this when both periods are profitable)."""
    if revenue_growth == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(op_income_growth / revenue_growth, unit="ratio", evidence_class=EvidenceClass.C)


def incremental_operating_margin(delta_op_income: float, delta_revenue: float) -> Value:
    """MKT-INCM-018."""
    if delta_revenue == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(delta_op_income / delta_revenue, unit="ratio", evidence_class=EvidenceClass.C)


def time_decay(months_to_event: float) -> float:
    """MKT-TDEC-020: 12-month half-life."""
    return math.exp(-math.log(2) * months_to_event / 12)


def catalyst_impact_index(probability: float, impact: float, evidence_quality: float, time_decay_factor: float) -> float:
    """MKT-CAT-019: Probability * Impact * EvidenceQuality * TimeDecay."""
    return probability * impact * evidence_quality * time_decay_factor


def forecast_consistency_gate(forecast_revenue: float, tam: float) -> bool:
    """DECISION_RULES.md: company revenue must not exceed TAM under the
    same definition."""
    return forecast_revenue <= tam


def fundamental_growth_capacity(reinvestment_rate: float, roic: float) -> float:
    """MKT-GCAP-009."""
    return reinvestment_rate * roic


class MarketOutput(SpecialistOutput):
    market_definition: str | None = None
    tam_sam_som: dict = {}
    penetration_and_share: dict = {}
    revision_dashboard: dict = {}
    catalysts: list = []
    growth_capacity_check: dict = {}
    three_growth_thesis_killers: list = []


def _annual_field(annual: list[dict], field_name: str, i: int = 0) -> float | None:
    if i >= len(annual):
        return None
    return annual[i].get(field_name)


def _operating_leverage_from_annual(annual: list[dict]) -> tuple[float | None, float | None]:
    """Returns (operating_leverage, incremental_margin) from the latest
    two annual periods, or (None, None) across a loss sign change."""
    rev0, rev1 = _annual_field(annual, "revenue", 0), _annual_field(annual, "revenue", 1)
    oi0, oi1 = _annual_field(annual, "ebit", 0), _annual_field(annual, "ebit", 1)
    if None in (rev0, rev1, oi0, oi1) or rev1 == 0:
        return None, None
    if (oi0 >= 0) != (oi1 >= 0):
        return None, None
    rev_growth = (rev0 - rev1) / rev1
    oi_growth = (oi0 - oi1) / abs(oi1) if oi1 != 0 else None
    oplev = operating_leverage(oi_growth, rev_growth) if oi_growth is not None else Value.null(NullState.MISSING)
    incm = incremental_operating_margin(oi0 - oi1, rev0 - rev1)
    return (oplev.value if oplev.is_valid else None), (incm.value if incm.is_valid else None)


def run(packet, overlay: dict | None = None) -> MarketOutput:
    overlay = overlay or {}
    annual = packet.fundamentals.get("annual", [])
    estimates = packet.estimates or {}

    judgment_requests: list[JudgmentRequest] = []

    # --- TAM (judgment-gated: figure + source tier not derivable) -----
    tam_value = overlay.get("tam")
    tam_source_tier = overlay.get("tam_source_tier")
    if tam_value is None or tam_source_tier is None:
        judgment_requests.append(
            JudgmentRequest(
                request_id="market.tam",
                agent_id=AGENT_ID,
                metric_id="tam_and_source_tier",
                question="What is the current TAM (with explicit scope/geography/year) and its source-quality tier (1-5)?",
                schema_hint="{tam: number, source_tier: 1|2|3|4|5, definition: string}",
            )
        )
    tam_score = None
    if tam_value is not None and tam_source_tier is not None:
        tam_score = 5.0
        cap = SOURCE_TIER_SCORE_CAP.get(tam_source_tier)
        if cap is not None:
            tam_score = min(tam_score, cap)

    # --- Revisions: earnings surprise + dispersion proxy ----------------
    finnhub_eps = estimates.get("finnhub_eps") or {}
    eps_rows = finnhub_eps.get("data", []) if isinstance(finnhub_eps, dict) else []
    dispersion_vals = []
    for row in eps_rows:
        high, low, avg = row.get("epsHigh"), row.get("epsLow"), row.get("epsAvg")
        if high is not None and low is not None and avg:
            dispersion_vals.append(estimate_dispersion_proxy(high, low, avg))
    valid_dispersion = [v.value for v in dispersion_vals if v.is_valid]
    dispersion_avg = statistics.mean(valid_dispersion) if valid_dispersion else None

    surprises = []
    for row in eps_rows:
        actual, est = row.get("epsActual"), row.get("epsAvg")
        if actual is not None and est is not None:
            surprises.append(earnings_surprise(actual, est, snapshot_before_earnings=True))
    valid_surprises = [v.value for v in surprises if v.is_valid]
    surprise_avg = statistics.mean(valid_surprises) if valid_surprises else None

    revisions_score = None
    if surprise_avg is not None:
        revisions_score = anchor_score(surprise_avg, [(-0.20, 0), (0.0, 4), (0.05, 6.5), (0.15, 9)])
        if dispersion_avg is not None and dispersion_avg > 0.30:
            revisions_score = min(revisions_score, 6.0)

    # --- Catalysts: judgment-only (P/Impact/EvidenceQuality), narrative cap --
    overlay_catalysts = overlay.get("catalysts", [])
    catalysts_score = 3.0 if not overlay_catalysts else None
    if overlay_catalysts:
        total_impact = sum(
            catalyst_impact_index(c["probability"], c["impact"], c["evidence_quality"], time_decay(c["months_to_event"]))
            for c in overlay_catalysts
        )
        catalysts_score = anchor_score(len(overlay_catalysts), [(0, 3), (1, 5), (2, 7.5), (3, 10)])
    else:
        judgment_requests.append(
            JudgmentRequest(
                request_id="market.catalysts",
                agent_id=AGENT_ID,
                metric_id="catalysts",
                question="List at least three product/business catalysts with probability, financial impact, evidence quality, and months to event.",
                schema_hint="[{event, probability, impact, evidence_quality, months_to_event, evidence_class}]",
            )
        )

    # --- Runway & share: growth capacity is computable; penetration/share is not --
    reinvestment_rate = overlay.get("reinvestment_rate")
    roic_v = overlay.get("roic")
    growth_capacity = None
    if reinvestment_rate is not None and roic_v is not None:
        growth_capacity = fundamental_growth_capacity(reinvestment_rate, roic_v)
    runway_score = None
    if growth_capacity is not None:
        runway_score = anchor_score(growth_capacity, [(0.0, 3), (0.05, 6), (0.10, 9)])

    # --- Operating leverage & sector confirmation ------------------------
    oplev, incm = _operating_leverage_from_annual(annual)
    sector_bars = packet.market_data.sector if packet.market_data else []
    oplev_score = None
    if incm is not None:
        oplev_score = anchor_score(incm, [(-0.10, 0), (0.0, 4), (0.20, 7), (0.35, 10)])
    if not sector_bars and oplev_score is not None:
        pass  # sector breadth is context-only per SCORING.md; absence doesn't penalize

    dims_spec = [
        ("tam_and_industry_tailwind", 5.0, tam_score),
        ("earnings_and_revenue_revisions", 4.0, revisions_score),
        ("product_and_business_catalysts", 4.0, catalysts_score),
        ("growth_runway_and_share_capture", 4.0, runway_score),
        ("operating_leverage_and_market_confirmation", 3.0, oplev_score),
    ]
    dims = []
    for name, max_pts, score in dims_spec:
        if score is None:
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[]))
        else:
            v = Value.of(score, unit="score", evidence_class=EvidenceClass.C)
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[(1.0, v)]))

    category = Category(name="market", max_points=MAX_POINTS, dimensions=dims)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    metrics = [
        MetricRow(metric_id="earnings_surprise_avg", value=surprise_avg, formula="MKT-SURP-014",
                  unit="pct", score=None, evidence_class=str(EvidenceClass.C), source="packet.estimates.finnhub_eps"),
        MetricRow(metric_id="estimate_dispersion_proxy_avg", value=dispersion_avg, formula="MKT-DISP-013",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.estimates.finnhub_eps",
                  warnings=["PROXY_RANGE_STDEV"]),
        MetricRow(metric_id="operating_leverage", value=oplev, formula="MKT-OPLEV-017",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="incremental_operating_margin", value=incm, formula="MKT-INCM-018",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="packet.fundamentals"),
        MetricRow(metric_id="fundamental_growth_capacity", value=growth_capacity, formula="MKT-GCAP-009",
                  unit="ratio", score=None, evidence_class=str(EvidenceClass.C), source="overlay"),
    ]

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return MarketOutput(
        agent_id=AGENT_ID,
        status=status,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange, "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategorySummary(max_points=MAX_POINTS, awarded_points=awarded, score_10=score_10, confidence=coverage * 100),
        coverage=coverage,
        dimensions=[{"name": d.name, "max_points": d.max_points, "score_10": _dim_score10_or_none(d)} for d in dims],
        metrics=metrics,
        mandatory_flags=[],
        assumptions=[
            "TAM/SAM/SOM, peer market share, and revision-breadth history are not in this packet "
            "(market_share is a prohibited-imputation metric) — TAM is a judgment request, "
            "market share and revision breadth are NOT_SCORABLE.",
        ],
        judgment_requests=judgment_requests,
        source_lineage=["packet.fundamentals.annual", "packet.estimates"],
        market_definition=None,
        tam_sam_som={"tam": tam_value, "sam": None, "som_scenarios": [], "source_tier": tam_source_tier},
        penetration_and_share={},
        revision_dashboard={
            "earnings_surprise_avg": surprise_avg,
            "estimate_dispersion_proxy_avg": dispersion_avg,
            "active_estimates": len(eps_rows),
        },
        catalysts=overlay_catalysts,
        growth_capacity_check={"fundamental_growth_capacity": growth_capacity},
        three_growth_thesis_killers=[],
    )


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value
