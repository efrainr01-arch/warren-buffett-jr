"""Staged pipeline — Task 24. Wires the packet builder (Task 10), the six
specialists (Tasks 14-19), the judgment overlay (Task 20), aggregation
(Task 21), charts (Task 22), and the renderer (Task 23) into one
end-to-end `wbj analyze <TICKER>` run, plus the individual
`fetch|packet|compute|aggregate|report` CLI stages.

Each `stage_*` function is independently callable; `run_all` chains them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wbj.aggregate.contradiction import contradictions
from wbj.aggregate.gates import apply_gates, raw_total
from wbj.aggregate.overrides import apply_overrides, validate_handoff
from wbj.aggregate.synthesis import synthesize_levels
from wbj.config import Settings
from wbj.core.nullstates import NullState
from wbj.engines import indicators as ind
from wbj.overlay.merge import collect_requests, merge_overlay
from wbj.packet.builder import Providers, build_packet
from wbj.providers.cache import Cache
from wbj.providers.edgar import EdgarProvider
from wbj.providers.finnhub import FinnhubProvider
from wbj.providers.fmp import FMPProvider
from wbj.providers.fred import FredProvider
from wbj.report.charts import football_field_chart, price_levels_chart, scenario_fan_chart, scorecard_chart
from wbj.report.render import render
from wbj.schemas.final_report import CategoryScore, ExecutiveThesis, FinalReport, ReportProfile, ReportSecurity
from wbj.schemas.overlay import Judgment
from wbj.schemas.packet import Packet
from wbj.specialists import business, financial, market, risk, technical, valuation
from wbj.specialists.common import SpecialistOutput

DEFAULT_ERP = 0.045

_AGENT_TO_CATEGORY = {
    "business_analysis": "business",
    "financial_analysis": "financial",
    "market_analysis": "market",
    "technical_momentum": "technical",
    "risk_analysis": "risk",
    "valuation_analysis": "valuation",
}


def _providers(settings: Settings, offline: bool = False) -> Providers:
    cache = Cache(settings.cache_dir)
    return Providers(
        fmp=FMPProvider(settings, cache, offline=offline),
        edgar=EdgarProvider(settings, cache, offline=offline),
        finnhub=FinnhubProvider(settings, cache, offline=offline),
        fred=FredProvider(settings, cache, offline=offline),
    )


def stage_fetch(ticker: str, settings: Settings) -> None:
    """Warms the provider cache for `ticker` by touching every endpoint
    the packet builder needs. Not meaningful with `offline=True` (there
    is nothing to warm without network access)."""
    ticker = ticker.upper()
    providers = _providers(settings, offline=False)
    providers.fmp.profile(ticker)
    providers.fmp.ohlcv_daily(ticker)
    providers.fmp.income_annual(ticker)
    providers.fmp.income_quarterly(ticker)
    providers.fmp.balance_annual(ticker)
    providers.fmp.balance_quarterly(ticker)
    providers.fmp.cashflow_annual(ticker)
    providers.fmp.cashflow_quarterly(ticker)
    providers.fmp.earnings_calendar(ticker)
    providers.fmp.institutional_holders(ticker)
    providers.fmp.insider_trades(ticker)
    providers.fmp.analyst_estimates(ticker)
    cik = providers.edgar.cik_for(ticker)
    if cik is not None:
        providers.edgar.companyfacts(cik)
    providers.finnhub.quote(ticker)
    providers.finnhub.estimates(ticker)
    providers.finnhub.revenue_estimates(ticker)
    providers.fred.risk_free_rate()


def stage_packet(ticker: str, settings: Settings, offline: bool = False, now: datetime | None = None) -> Packet:
    """Builds and caches the analysis packet for `ticker`."""
    now = now or datetime.now(timezone.utc)
    providers = _providers(settings, offline=offline)
    packet = build_packet(ticker, providers, now)
    out_path = settings.cache_dir / packet.security.ticker / "artifacts" / "packet.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(packet.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return packet


def stage_compute(packet: Packet, settings: Settings, beta: float | None = None, erp: float = DEFAULT_ERP) -> dict[str, SpecialistOutput]:
    """Runs the six specialists and writes their outputs (plus any
    outstanding judgment requests) as artifacts under
    `<cache_dir>/<TICKER>/artifacts/`.

    Valuation runs first so its WACC feeds Financial's and Business's
    ROIC-vs-WACC checks -- Valuation itself never depends on another
    specialist's *score* (Cerebro's independence rule), only on the
    packet's raw facts.
    """
    valuation_out = valuation.run(packet, beta=beta, erp=erp)
    wacc_value = valuation_out.wacc.get("value")

    outputs: dict[str, SpecialistOutput] = {
        "valuation_analysis": valuation_out,
        "financial_analysis": financial.run(packet, wacc=wacc_value),
        "business_analysis": business.run(packet, wacc=wacc_value),
        "market_analysis": market.run(packet),
        "technical_momentum": technical.run(packet),
        "risk_analysis": risk.run(packet),
    }

    artifacts_dir = settings.cache_dir / packet.security.ticker / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, out in outputs.items():
        (artifacts_dir / f"{agent_id}.json").write_text(out.model_dump_json(indent=2), encoding="utf-8")

    requests = collect_requests(list(outputs.values()))
    (artifacts_dir / "judgment_requests.json").write_text(
        json.dumps([r.model_dump() for r in requests], indent=2), encoding="utf-8"
    )
    return outputs


def _default_monitoring_triggers(profile_label: str) -> list[dict]:
    if "avoid" in profile_label.lower():
        return [
            {
                "revisit_event": "next quarterly earnings release, or 90 days, whichever comes first",
                "reason": "gate/override-driven avoid classification -- default policy revisit window",
            }
        ]
    return []


def _executive_thesis(packet: Packet, outputs: dict[str, SpecialistOutput], wacc_value: float | None) -> ExecutiveThesis:
    ticker = packet.security.ticker
    business_out, financial_out, risk_out = outputs["business_analysis"], outputs["financial_analysis"], outputs["risk_analysis"]
    valuation_out, market_out = outputs["valuation_analysis"], outputs["market_analysis"]

    roic_latest = business_out.roic_history[0] if business_out.roic_history else None
    if roic_latest is not None and wacc_value is not None:
        spread = roic_latest - wacc_value
        durability = (
            f"ROIC is {roic_latest:.1%} against an estimated WACC of {wacc_value:.1%} "
            f"(spread {spread:+.1%}), {'supporting' if spread > 0 else 'not clearly supporting'} durable value creation."
        )
    else:
        durability = "The ROIC-vs-WACC spread could not be computed from the data available in this packet."

    fcf = financial_out.profitability_and_cash.get("fcf")
    growth_funding = (
        "Growth appears self-funded from operating cash flow." if fcf is not None and fcf > 0
        else "Growth funding could not be confirmed as self-sustaining from the available cash-flow data."
    )

    surprise_avg = market_out.revision_dashboard.get("earnings_surprise_avg")
    market_validation = (
        f"Recent earnings surprises average {surprise_avg:+.1%}, a {'supportive' if surprise_avg and surprise_avg > 0 else 'mixed or unfavorable'} market signal."
        if surprise_avg is not None else "Market validation could not be assessed from the available estimate history."
    )

    implied_growth = (valuation_out.reverse_dcf or {}).get("implied_revenue_cagr")
    price_implied = (
        f"The current price appears to require roughly {implied_growth:.1%} revenue CAGR." if implied_growth is not None
        else "Reverse-DCF implied growth could not be solved from the available inputs."
    )

    bands = valuation_out.reference_bands or {}
    if bands.get("base") is not None:
        nearest_levels = f"Base intrinsic-value reference is {bands['base']:.2f}; see the important-levels table for nearby technical zones."
    else:
        nearest_levels = "Intrinsic-value reference bands could not be computed."

    if risk_out.mandatory_warnings:
        primary_risk = risk_out.mandatory_warnings[0]
    elif business_out.mandatory_flags:
        primary_risk = f"Flagged: {business_out.mandatory_flags[0]}"
    else:
        primary_risk = "No single dominant deterministic risk flag was raised; see thesis killers for judgment-based risks."

    return ExecutiveThesis(
        what_the_company_does=f"{ticker} is analyzed as a {packet.security.security_type.replace('_', ' ')} under the {packet.analysis.industry_adapter} adapter.",
        value_creation_durability=durability,
        growth_funding=growth_funding,
        market_validation=market_validation,
        price_implied_assumptions=price_implied,
        nearest_levels=nearest_levels,
        primary_invalidation_risk=primary_risk,
    )


def stage_aggregate(packet: Packet, outputs: dict[str, SpecialistOutput], overlay_path: Path | None = None) -> FinalReport:
    """Optionally merges judgment answers, validates each handoff,
    applies overrides/gates/contradictions, synthesizes price levels, and
    assembles the `FinalReport`."""
    if overlay_path is not None and Path(overlay_path).exists():
        raw_judgments = json.loads(Path(overlay_path).read_text(encoding="utf-8"))
        judgments = [Judgment.model_validate(j) for j in raw_judgments]
        merged = merge_overlay(packet, list(outputs.values()), judgments)
        outputs = {o.agent_id: o for o in merged}

    for out in outputs.values():
        reasons = validate_handoff(out)
        if reasons:
            raise ValueError(f"handoff rejected for {out.agent_id}: {reasons}")

    overrides = apply_overrides(outputs, packet=packet)

    cats: dict[str, float] = {}
    cats_max: dict[str, float] = {}
    confidences: dict[str, float] = {}
    for agent_id, category in _AGENT_TO_CATEGORY.items():
        out = outputs.get(agent_id)
        if out is None:
            continue
        cats[category] = out.category.awarded_points or 0.0
        cats_max[category] = out.category.max_points
        confidences[category] = out.category.confidence or 0.0

    raw = raw_total(list(cats.values()))
    profile_result = apply_gates(raw, cats, confidences, overrides)
    contradiction_rows = contradictions(cats, cats_max)

    valuation_out, technical_out = outputs["valuation_analysis"], outputs["technical_momentum"]
    price = None
    atr = technical_out.indicators.get("atr14")
    if packet.facts_table.get("price") and packet.facts_table["price"].is_valid:
        price = packet.facts_table["price"].value
    important_levels = synthesize_levels(technical_out, valuation_out, price, atr) if price is not None and atr else []

    missing_or_conflicted: list[str] = []
    for name, v in packet.facts_table.items():
        if v.is_null and v.state == NullState.CONFLICTED:
            missing_or_conflicted.append(f"facts_table.{name}: CONFLICTED")
    for out in outputs.values():
        missing_or_conflicted.extend(out.assumptions)

    formula_versions = sorted({m.formula for out in outputs.values() for m in out.metrics if m.formula})

    monitoring_triggers = _default_monitoring_triggers(profile_result.label)
    for row in contradiction_rows:
        monitoring_triggers.append({"contradiction": row["combination"], "label": row["label"]})

    thesis_killers: list[dict] = []
    for out in outputs.values():
        thesis_killers.extend(getattr(out, "thesis_killers", []) or [])
        thesis_killers.extend(getattr(out, "three_thesis_killers", []) or [])

    return FinalReport(
        security=ReportSecurity(
            ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(), knowledge_timestamp=packet.analysis.knowledge_timestamp,
        ),
        profile=ReportProfile(
            label=profile_result.label, raw_score=profile_result.raw_score, total_confidence=profile_result.total_confidence,
            passed_gates=profile_result.passed_gates, failed_gates=profile_result.failed_gates,
            overrides=[o.id for o in overrides if o.condition_met],
        ),
        category_scorecard={
            category: CategoryScore(points=cats.get(category), max=cats_max.get(category, 0), confidence=confidences.get(category))
            for category in _AGENT_TO_CATEGORY.values()
        },
        executive_thesis=_executive_thesis(packet, outputs, valuation_out.wacc.get("value")),
        important_levels=important_levels,
        valuation_scenarios=valuation_out.scenarios,
        reverse_dcf=valuation_out.reverse_dcf,
        notable_holders=list(packet.institutional_holders or []),
        management_track_record=[],
        insider_trades=list(packet.insiders or []),
        thesis_killers=thesis_killers,
        monitoring_triggers=monitoring_triggers,
        missing_or_conflicted_data=missing_or_conflicted,
        audit={
            "packet_hashes": {packet.security.ticker: packet.packet_hash},
            "formula_versions": formula_versions,
            "validation_summary": {"passed": len(outputs), "failed": 0},
        },
    )


def _reference_bands_for_chart(valuation_out) -> dict[str, tuple[float, float]]:
    mc = valuation_out.fair_value_distribution or {}
    if all(k in mc for k in ("p10", "p25", "p75", "p90")):
        return {"Bear": (mc["p10"], mc["p25"]), "Base": (mc["p25"], mc["p75"]), "Bull": (mc["p75"], mc["p90"])}
    bands = valuation_out.reference_bands or {}
    out = {}
    for label in ("bear", "base", "bull"):
        v = bands.get(label)
        if v is not None:
            out[label.title()] = (v * 0.98, v * 1.02)
    return out


def stage_report(packet: Packet, outputs: dict[str, SpecialistOutput], final: FinalReport, out_dir: Path) -> Path:
    """Builds the chart set and renders `report.md`/`report.json` into `out_dir`."""
    out_dir = Path(out_dir)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    valuation_out, technical_out = outputs["valuation_analysis"], outputs["technical_momentum"]
    daily = list(reversed(packet.market_data.daily))
    closes = [r.adj_close for r in daily]

    charts: dict[str, Path] = {}

    smas = {}
    for n in (50, 200):
        series = ind.sma(pd.Series(closes), n).dropna().tolist()
        if series:
            smas[f"sma{n}"] = series
    chart_levels = [
        {"type": lvl["type"], "lower": lvl.get("lower", lvl["value"]), "center": lvl["value"], "upper": lvl.get("upper", lvl["value"]), "status": lvl.get("status")}
        for lvl in final.important_levels if lvl.get("type") in ("support", "resistance")
    ]
    charts["price_levels"] = price_levels_chart(closes, chart_levels, smas, charts_dir / "price_levels.png")

    band_map = _reference_bands_for_chart(valuation_out)
    fan_scenarios = []
    for s in valuation_out.scenarios:
        name = s["label"]
        low, high = band_map.get(name.title(), (s["value"] * 0.95, s["value"] * 1.05))
        fan_scenarios.append({"name": name, "growth": s["growth"], "margin": s["margin"], "low": low, "high": high, "years": 5})
    if fan_scenarios:
        charts["scenario_fan"] = scenario_fan_chart(closes, fan_scenarios, charts_dir / "scenario_fan.png")

    points = {cat: cs.points or 0 for cat, cs in final.category_scorecard.items()}
    maxes = {cat: cs.max for cat, cs in final.category_scorecard.items()}
    charts["scorecard"] = scorecard_chart(points, maxes, charts_dir / "scorecard.png")

    if band_map and packet.facts_table.get("price") and packet.facts_table["price"].is_valid:
        charts["football_field"] = football_field_chart(band_map, packet.facts_table["price"].value, charts_dir / "football_field.png")

    return render(final, charts, out_dir)


def run_all(
    ticker: str,
    settings: Settings,
    offline: bool = False,
    beta: float | None = None,
    erp: float = DEFAULT_ERP,
    overlay_path: Path | None = None,
    now: datetime | None = None,
) -> FinalReport:
    """Runs the full pipeline for `ticker` and writes the report under
    `<settings.reports_dir>/<TICKER>/<YYYY-MM-DD>/`. Returns the `FinalReport`."""
    now = now or datetime.now(timezone.utc)
    packet = stage_packet(ticker, settings, offline=offline, now=now)
    outputs = stage_compute(packet, settings, beta=beta, erp=erp)
    final = stage_aggregate(packet, outputs, overlay_path=overlay_path)

    out_dir = settings.reports_dir / packet.security.ticker / now.date().isoformat()
    stage_report(packet, outputs, final, out_dir)
    return final
