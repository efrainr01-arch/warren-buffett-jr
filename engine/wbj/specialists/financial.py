"""Financial specialist (15 pts) — Cerebro/02_financial_analysis/.

Implements the core-27 diagnostic (DECISION_RULES.md) and the five
weighted dimensions (SCORING.md) over FIN-GR-001..EF-027, plus the
DX-028..033 diagnostics as unscored audit rows. Bands come verbatim from
FORMULAS.md; where FORMULAS.md describes a metric only qualitatively
(GR-004 organic growth, CF-016 capital dependence, GR-003/005 peer and
market-share trend, and the three-state formulas PR-006/010/011,
CF-013/015, BS-021/022, EF-027 that give no explicit numeric band) a
documented, disclosed threshold is used — noted inline and in the commit
message, per the plan's "Cerebro is law, discrepancies noted" rule.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension
from wbj.specialists.common import CategorySummary, MetricRow, SpecialistOutput

AGENT_ID = "financial_analysis"
MAX_POINTS = float(CATEGORY_WEIGHTS["financial"])  # 15
RECONCILE_TOLERANCE_POINTS = 1.5

Band = Literal["BAD", "GOOD", "EXCELLENT"]
_CLASS_SCORE = {"BAD": 2.0, "GOOD": 5.5, "EXCELLENT": 9.0}


def _classify(x: float, bad_edge: float, excellent_edge: float, higher_is_better: bool = True) -> Band:
    if higher_is_better:
        if x < bad_edge:
            return "BAD"
        return "GOOD" if x <= excellent_edge else "EXCELLENT"
    if x > bad_edge:
        return "BAD"
    return "GOOD" if x >= excellent_edge else "EXCELLENT"


def _ols_slope(y: list[float]) -> float | None:
    valid = [v for v in y if v is not None]
    if len(valid) < 2:
        return None
    x = list(range(len(valid)))
    slope, _ = np.polyfit(x, valid, 1)
    return float(slope)


class FinancialOutput(SpecialistOutput):
    core_27_metrics: dict = field(default_factory=dict)
    profitability_and_cash: dict = field(default_factory=dict)
    balance_and_maturities: dict = field(default_factory=dict)
    return_on_capital: dict = field(default_factory=dict)
    dilution_and_sbc: dict = field(default_factory=dict)
    mandatory_overrides: list[str] = field(default_factory=list)
    strongest_metric: str | None = None
    weakest_metric: str | None = None


@dataclass
class _MetricDef:
    metric_id: str
    formula: str
    dimensions: list[str]  # a metric may feed more than one dimension (e.g. FIN-CF-015)
    compute: object  # Callable[[list[dict], list[dict], float | None], tuple[float | None, list[str]]]
    bad_edge: float | None
    excellent_edge: float | None
    higher_is_better: bool = True
    scorable: bool = True  # False for formulas with no registered numeric band
    classify_override: object = None  # Callable[[list[dict]], Band | None], bypasses bad/excellent edges


def _annual_field(annual: list[dict], field_name: str, i: int = 0) -> float | None:
    if i >= len(annual):
        return None
    return annual[i].get(field_name)


def _yoy(cur: float | None, prior: float | None) -> float | None:
    if cur is None or prior is None or prior == 0:
        return None
    return (cur - prior) / prior


def _revenue_yoy(a: list[dict], q: list[dict], wacc: float | None) -> tuple[float | None, list[str]]:
    return _yoy(_annual_field(a, "revenue", 0), _annual_field(a, "revenue", 1)), []


def _revenue_growth_trend(a: list[dict], q, wacc) -> tuple[float | None, list[str]]:
    growths = [_yoy(_annual_field(a, "revenue", i), _annual_field(a, "revenue", i + 1)) for i in range(3)]
    return _ols_slope(list(reversed(growths))), []


def _growth_vs_peers(a, q, wacc) -> tuple[float | None, list[str]]:
    return None, ["FIN-GR-003 requires a peer growth panel not present in this packet"]


def _organic_growth_quality(a, q, wacc) -> tuple[float | None, list[str]]:
    return None, ["FIN-GR-004 requires an organic/total revenue-growth bridge not present in this packet"]


def _market_share_trend(a, q, wacc) -> tuple[float | None, list[str]]:
    return None, ["FIN-GR-005 requires market-size/share history not present in this packet"]


def _net_profit_status(a, q, wacc) -> tuple[float | None, list[str]]:
    """Returns the latest net income (a real, meaningful value); band
    classification is a special case in `run()` (loss/volatile/consistent
    is categorical, not a single numeric band) — see `_classify_net_profit_status`."""
    ni_latest = _annual_field(a, "net_income", 0)
    return ni_latest, []


def _classify_net_profit_status(a: list[dict]) -> Band | None:
    ni = [v for v in (_annual_field(a, "net_income", i) for i in range(3)) if v is not None]
    if not ni:
        return None
    if ni[0] <= 0:
        return "BAD"
    if len(ni) >= 2 and statistics.mean(ni) != 0:
        cv = statistics.pstdev(ni) / abs(statistics.mean(ni))
        return "EXCELLENT" if cv < 0.30 else "GOOD"
    return "GOOD"


def _gross_margin(a, q, wacc) -> tuple[float | None, list[str]]:
    rev, cogs = _annual_field(a, "revenue"), _annual_field(a, "cogs")
    if rev is None or cogs is None or rev == 0:
        return None, []
    return (rev - cogs) / rev, []


def _operating_margin(a, q, wacc) -> tuple[float | None, list[str]]:
    rev, ebit = _annual_field(a, "revenue"), _annual_field(a, "ebit")
    if rev is None or ebit is None or rev == 0:
        return None, []
    return ebit / rev, []


def _net_margin(a, q, wacc) -> tuple[float | None, list[str]]:
    rev, ni = _annual_field(a, "revenue"), _annual_field(a, "net_income")
    if rev is None or ni is None or rev == 0:
        return None, []
    return ni / rev, []


def _margin_trend(a, q, wacc) -> tuple[float | None, list[str]]:
    margins = []
    for i in range(3):
        rev, ni = _annual_field(a, "revenue", i), _annual_field(a, "net_income", i)
        margins.append(ni / rev if rev else None)
    return _ols_slope(list(reversed(margins))), []


def _profit_vs_revenue_growth(a, q, wacc) -> tuple[float | None, list[str]]:
    rev_g = _yoy(_annual_field(a, "revenue", 0), _annual_field(a, "revenue", 1))
    ni0, ni1 = _annual_field(a, "net_income", 0), _annual_field(a, "net_income", 1)
    if ni0 is None or ni1 is None or rev_g is None:
        return None, []
    if (ni0 <= 0) != (ni1 <= 0):
        return None, ["FIN-PR-011 not meaningful across a loss-to-profit sign change"]
    ni_g = _yoy(ni0, ni1)
    if ni_g is None:
        return None, []
    return ni_g - rev_g, []


def _fcf(a, q, wacc) -> tuple[float | None, list[str]]:
    ocf, capex = _annual_field(a, "operating_cash_flow"), _annual_field(a, "capex")
    if ocf is None or capex is None:
        return None, []
    return ocf + capex, []  # capex is stored negative (cash outflow), per Task-10 canonical mapping


def _fcf_growth(a, q, wacc) -> tuple[float | None, list[str]]:
    def _fcf_at(i):
        ocf, capex = _annual_field(a, "operating_cash_flow", i), _annual_field(a, "capex", i)
        return ocf + capex if ocf is not None and capex is not None else None

    cur, prior = _fcf_at(0), _fcf_at(1)
    if cur is None or prior is None or prior == 0:
        return None, []
    if (cur >= 0) != (prior >= 0):
        return None, ["FIN-CF-013 sign change: reporting transition, not a percentage"]
    return (cur - prior) / abs(prior), []


def _fcf_margin(a, q, wacc) -> tuple[float | None, list[str]]:
    rev = _annual_field(a, "revenue")
    ocf, capex = _annual_field(a, "operating_cash_flow"), _annual_field(a, "capex")
    if rev is None or rev == 0 or ocf is None or capex is None:
        return None, []
    return (ocf + capex) / rev, []


def _cash_vs_earnings(a, q, wacc) -> tuple[float | None, list[str]]:
    ocf, ni = _annual_field(a, "operating_cash_flow"), _annual_field(a, "net_income")
    if ocf is None or ni is None or ni == 0:
        return None, []
    return ocf / ni, []


def _capital_dependence(a, q, wacc) -> tuple[float | None, list[str]]:
    return None, ["FIN-CF-016 requires a financing-need bridge (debt/equity issuance) not modeled in this packet"]


def _current_ratio(a, q, wacc) -> tuple[float | None, list[str]]:
    ca, cl = _annual_field(q or a, "total_current_assets"), _annual_field(q or a, "total_current_liabilities")
    if ca is None or cl is None or cl == 0:
        return None, []
    warnings = ["idle-capital note: current ratio exceeds 3.0"] if ca / cl > 3.0 else []
    return ca / cl, warnings


def _quick_ratio(a, q, wacc) -> tuple[float | None, list[str]]:
    src = q or a
    ca, inv, cl = _annual_field(src, "total_current_assets"), _annual_field(src, "inventory"), _annual_field(src, "total_current_liabilities")
    if ca is None or inv is None or cl is None or cl == 0:
        return None, []
    return (ca - inv) / cl, []


def _debt_to_equity(a, q, wacc) -> tuple[float | None, list[str]]:
    src = q or a
    debt, equity = _annual_field(src, "total_debt"), _annual_field(src, "total_equity")
    if debt is None or equity is None:
        return None, []
    if equity <= 0:
        return None, ["FIN-BS-019 NOT_MEANINGFUL: negative equity"]
    return debt / equity, []


def _interest_coverage(a, q, wacc) -> tuple[float | None, list[str]]:
    ebit, interest = _annual_field(a, "ebit"), _annual_field(a, "interest_expense")
    if ebit is None or interest is None or interest == 0:
        return None, []
    return ebit / interest, []


def _debt_vs_revenue_trend(a, q, wacc) -> tuple[float | None, list[str]]:
    debt_g = _yoy(_annual_field(a, "total_debt", 0), _annual_field(a, "total_debt", 1))
    rev_g = _yoy(_annual_field(a, "revenue", 0), _annual_field(a, "revenue", 1))
    if debt_g is None or rev_g is None:
        return None, []
    return debt_g - rev_g, []


def _liquidity_trend(a, q, wacc) -> tuple[float | None, list[str]]:
    ratios = []
    for i in range(3):
        ca, cl = _annual_field(a, "total_current_assets", i), _annual_field(a, "total_current_liabilities", i)
        ratios.append(ca / cl if ca is not None and cl else None)
    return _ols_slope(list(reversed(ratios))), []


def _roe(a, q, wacc) -> tuple[float | None, list[str]]:
    ni = _annual_field(a, "net_income")
    e0, e1 = _annual_field(a, "total_equity", 0), _annual_field(a, "total_equity", 1)
    if ni is None or e0 is None or e1 is None or (e0 + e1) == 0:
        return None, []
    return ni / ((e0 + e1) / 2), []


def _roic(a, q, wacc) -> tuple[float | None, list[str]]:
    ebit = _annual_field(a, "ebit")
    debt0, eq0, cash0 = _annual_field(a, "total_debt", 0), _annual_field(a, "total_equity", 0), _annual_field(a, "cash", 0)
    debt1, eq1, cash1 = _annual_field(a, "total_debt", 1), _annual_field(a, "total_equity", 1), _annual_field(a, "cash", 1)
    pretax, tax_expense = _annual_field(a, "pretax_income"), _annual_field(a, "income_tax_expense")
    if ebit is None or None in (debt0, eq0, cash0, debt1, eq1, cash1):
        return None, []
    tax_rate = tax_expense / pretax if pretax else 0.21
    nopat_v = ebit * (1 - tax_rate)
    ic0, ic1 = debt0 + eq0 - cash0, debt1 + eq1 - cash1
    avg_ic = (ic0 + ic1) / 2
    if avg_ic == 0:
        return None, []
    return nopat_v / avg_ic, []


def _roa(a, q, wacc) -> tuple[float | None, list[str]]:
    ni = _annual_field(a, "net_income")
    ta0, ta1 = _annual_field(a, "total_assets", 0), _annual_field(a, "total_assets", 1)
    if ni is None or ta0 is None or ta1 is None or (ta0 + ta1) == 0:
        return None, []
    return ni / ((ta0 + ta1) / 2), []


def _roic_vs_wacc(a, q, wacc) -> tuple[float | None, list[str]]:
    roic_v, _ = _roic(a, q, wacc)
    if roic_v is None or wacc is None:
        return None, ["FIN-EF-026 requires WACC (supplied by the valuation specialist or overlay)"]
    return roic_v - wacc, []


def _return_trend(a, q, wacc) -> tuple[float | None, list[str]]:
    values = []
    for i in range(3):
        v, _ = _roic(a[i:], q, wacc)
        values.append(v)
    return _ols_slope(list(reversed(values))), []


# metric_id -> (_MetricDef)
_METRICS: list[_MetricDef] = [
    _MetricDef("revenue_yoy_growth", "FIN-GR-001", ["revenue_quality_and_growth"], _revenue_yoy, 0.0, 0.10),
    _MetricDef("revenue_growth_trend", "FIN-GR-002", ["revenue_quality_and_growth"], _revenue_growth_trend, -0.01, 0.01),
    _MetricDef("growth_vs_peers", "FIN-GR-003", ["revenue_quality_and_growth"], _growth_vs_peers, 0.0, 0.0, scorable=False),
    _MetricDef("organic_growth_quality", "FIN-GR-004", ["revenue_quality_and_growth"], _organic_growth_quality, 0.0, 0.0, scorable=False),
    _MetricDef("market_share_trend", "FIN-GR-005", ["revenue_quality_and_growth"], _market_share_trend, 0.0, 0.0, scorable=False),
    _MetricDef(
        "net_profit_status", "FIN-PR-006", ["eps_and_fcf"], _net_profit_status, None, None,
        classify_override=_classify_net_profit_status,
    ),
    _MetricDef("gross_margin", "FIN-PR-007", ["margins"], _gross_margin, 0.20, 0.40),
    _MetricDef("operating_margin", "FIN-PR-008", ["margins"], _operating_margin, 0.10, 0.20),
    _MetricDef("net_margin", "FIN-PR-009", ["margins"], _net_margin, 0.05, 0.10),
    _MetricDef("margin_trend", "FIN-PR-010", ["margins"], _margin_trend, -0.005, 0.005),
    _MetricDef("profit_vs_revenue_growth", "FIN-PR-011", ["eps_and_fcf"], _profit_vs_revenue_growth, -0.02, 0.02),
    _MetricDef("fcf", "FIN-CF-012", ["eps_and_fcf"], _fcf, 0.0, 0.0, scorable=False),
    _MetricDef("fcf_growth", "FIN-CF-013", ["eps_and_fcf"], _fcf_growth, 0.0, 0.10),
    _MetricDef("fcf_margin", "FIN-CF-014", ["eps_and_fcf"], _fcf_margin, 0.0, 0.10),
    _MetricDef("cash_vs_earnings", "FIN-CF-015", ["eps_and_fcf", "cash_conversion_and_capital_efficiency"], _cash_vs_earnings, 0.9, 1.1),
    _MetricDef("capital_dependence", "FIN-CF-016", ["eps_and_fcf"], _capital_dependence, 0.0, 0.0, scorable=False),
    _MetricDef("current_ratio", "FIN-BS-017", ["balance_and_liquidity"], _current_ratio, 1.0, 1.5),
    _MetricDef("quick_ratio", "FIN-BS-018", ["balance_and_liquidity"], _quick_ratio, 0.7, 1.0),
    _MetricDef("debt_to_equity", "FIN-BS-019", ["balance_and_liquidity"], _debt_to_equity, 2.0, 1.0, higher_is_better=False),
    _MetricDef("interest_coverage", "FIN-BS-020", ["balance_and_liquidity"], _interest_coverage, 1.5, 3.0),
    _MetricDef("debt_vs_revenue_trend", "FIN-BS-021", ["balance_and_liquidity"], _debt_vs_revenue_trend, 0.02, -0.02, higher_is_better=False),
    _MetricDef("liquidity_trend", "FIN-BS-022", ["balance_and_liquidity"], _liquidity_trend, -0.05, 0.05),
    _MetricDef("roe", "FIN-EF-023", ["cash_conversion_and_capital_efficiency"], _roe, 0.08, 0.15),
    _MetricDef("roic", "FIN-EF-024", ["cash_conversion_and_capital_efficiency"], _roic, 0.08, 0.15),
    _MetricDef("roa", "FIN-EF-025", ["cash_conversion_and_capital_efficiency"], _roa, 0.03, 0.08),
    _MetricDef("roic_vs_wacc", "FIN-EF-026", ["cash_conversion_and_capital_efficiency"], _roic_vs_wacc, -0.01, 0.01),
    _MetricDef("return_trend", "FIN-EF-027", ["cash_conversion_and_capital_efficiency"], _return_trend, -0.01, 0.01),
]

_DIMENSION_NAMES = [
    "revenue_quality_and_growth",
    "eps_and_fcf",
    "margins",
    "balance_and_liquidity",
    "cash_conversion_and_capital_efficiency",
]


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value


def run(packet, wacc: float | None = None, overlay: dict | None = None) -> FinancialOutput:
    annual = packet.fundamentals.get("annual", [])
    quarterly = packet.fundamentals.get("quarterly", [])

    metric_values: dict[str, float | None] = {}
    metric_warnings: dict[str, list[str]] = {}
    metric_rows: list[MetricRow] = []
    classification_scores: dict[str, float | None] = {}

    core_points = 0.0
    core_valid = 0

    for m in _METRICS:
        raw, warnings = m.compute(annual, quarterly, wacc)
        metric_values[m.metric_id] = raw
        metric_warnings[m.metric_id] = warnings

        score: float | str | None = "NOT_SCORABLE"
        cls: Band | None = None
        if m.classify_override is not None:
            cls = m.classify_override(annual)
        elif m.scorable and raw is not None:
            cls = _classify(raw, m.bad_edge, m.excellent_edge, m.higher_is_better)

        if cls is not None:
            classification_scores[m.metric_id] = _CLASS_SCORE[cls]
            score = _CLASS_SCORE[cls]
            core_valid += 1
            core_points += {"BAD": 0, "GOOD": 1, "EXCELLENT": 2}[cls]
        else:
            classification_scores[m.metric_id] = None

        metric_rows.append(
            MetricRow(
                metric_id=m.metric_id,
                value=raw,
                state=None if raw is not None else str(NullState.MISSING),
                unit="ratio",
                period=annual[0]["date"] if annual else None,
                formula=m.formula,
                score=score,
                evidence_class=str(EvidenceClass.C),
                source="packet.fundamentals",
                confidence=None,
                warnings=warnings,
            )
        )

    core_27_percent = (core_points / (2 * core_valid) * 100) if core_valid else None
    core_27_score10 = core_27_percent / 10 if core_27_percent is not None else None

    dimensions: list[Dimension] = []
    for name in _DIMENSION_NAMES:
        members = [m for m in _METRICS if name in m.dimensions]
        metric_scores = []
        for m in members:
            s = classification_scores.get(m.metric_id)
            v = Value.of(s, unit="score", evidence_class=EvidenceClass.C) if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")
            metric_scores.append((1.0 / len(members), v))
        dimensions.append(Dimension(name=name, max_points=3.0, metric_scores=metric_scores))

    category = Category(name="financial", max_points=MAX_POINTS, dimensions=dimensions)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    reconciliation_warning = None
    if core_27_score10 is not None and abs(score_10 - core_27_score10) > RECONCILE_TOLERANCE_POINTS:
        reconciliation_warning = (
            f"MAIN_RECONCILE_WARNING: weighted score {score_10:.2f} vs core-27 score "
            f"{core_27_score10:.2f} differ by more than {RECONCILE_TOLERANCE_POINTS}"
        )

    mandatory_flags: list[str] = []
    mandatory_overrides: list[str] = []
    ic = metric_values.get("interest_coverage")
    if ic is not None and ic < 1.5:
        mandatory_flags.append("SOLVENCY_WARNING")

    ni_latest = annual[0].get("net_income") if annual else None
    fcf_latest = metric_values.get("fcf")
    if ni_latest is not None and fcf_latest is not None and ni_latest < 0 and fcf_latest < 0:
        mandatory_overrides.append(
            "OVERRIDE_1_CANDIDATE: normalized net income and FCF are both negative — "
            "verdict capped at Bad/Avoid if external financing dependence is confirmed"
        )

    roic_v = metric_values.get("roic")
    if roic_v is not None and wacc is not None and roic_v < wacc:
        mandatory_overrides.append("OVERRIDE_2: ROIC < WACC — category verdict cannot be Excellent")

    scored = [(m.metric_id, classification_scores[m.metric_id]) for m in _METRICS if classification_scores[m.metric_id] is not None]
    strongest = max(scored, key=lambda t: t[1])[0] if scored else None
    weakest = min(scored, key=lambda t: t[1])[0] if scored else None

    diluted_cagr = None
    d0, d1 = (annual[0].get("diluted_shares") if annual else None), (annual[-1].get("diluted_shares") if annual else None)
    if d0 and d1 and d0 > 0 and d1 > 0 and len(annual) > 1:
        diluted_cagr = (d0 / d1) ** (1 / (len(annual) - 1)) - 1

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return FinancialOutput(
        agent_id=AGENT_ID,
        status=status,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange, "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategorySummary(max_points=MAX_POINTS, awarded_points=awarded, score_10=score_10, confidence=coverage * 100),
        coverage=coverage,
        dimensions=[
            {"name": d.name, "max_points": d.max_points, "score_10": _dim_score10_or_none(d)}
            for d in dimensions
        ],
        metrics=metric_rows,
        mandatory_flags=mandatory_flags,
        assumptions=[
            "net_profit_status, profit_vs_revenue_growth, fcf_growth, cash_vs_earnings, debt_vs_revenue_trend, "
            "and liquidity_trend use disclosed bands not given numerically in FORMULAS.md (see module docstring)",
        ]
        + ([reconciliation_warning] if reconciliation_warning else []),
        source_lineage=["packet.fundamentals.annual", "packet.fundamentals.quarterly"],
        core_27_metrics={
            "valid_count": core_valid,
            "points": core_points,
            "maximum_valid_points": 2 * core_valid,
            "percent": core_27_percent,
            "rows": [{"metric_id": m.metric_id, "value": metric_values[m.metric_id]} for m in _METRICS],
        },
        profitability_and_cash={
            "gross_margin": metric_values.get("gross_margin"),
            "operating_margin": metric_values.get("operating_margin"),
            "net_margin": metric_values.get("net_margin"),
            "fcf": metric_values.get("fcf"),
            "fcf_margin": metric_values.get("fcf_margin"),
        },
        balance_and_maturities={
            "current_ratio": metric_values.get("current_ratio"),
            "quick_ratio": metric_values.get("quick_ratio"),
            "debt_to_equity": metric_values.get("debt_to_equity"),
            "interest_coverage": metric_values.get("interest_coverage"),
        },
        return_on_capital={
            "roe": metric_values.get("roe"),
            "roic": metric_values.get("roic"),
            "roa": metric_values.get("roa"),
            "roic_vs_wacc": metric_values.get("roic_vs_wacc"),
        },
        dilution_and_sbc={"diluted_share_cagr": diluted_cagr},
        mandatory_overrides=mandatory_overrides,
        strongest_metric=strongest,
        weakest_metric=weakest,
    )
