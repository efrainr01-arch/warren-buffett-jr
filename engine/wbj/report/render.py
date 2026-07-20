"""Final report renderer — Cerebro/00_main_agent/FINAL_REPORT_SCHEMA.md,
Cerebro/examples/FINAL_REPORT_EXAMPLE.md, and root CLAUDE.md's
"Contenido obligatorio del reporte final".

Writes `report.json` (a schema-validated dump of the `FinalReport`) and
`report.md` (the English narrative report) into `out_dir`.
"""

from __future__ import annotations

from pathlib import Path

from wbj.schemas.final_report import FinalReport

INSUFFICIENT_DATA_SENTENCE = "Insufficient data to reach an investment conclusion"
INSIDER_SIGNIFICANCE_THRESHOLD = 1_000_000
FORBIDDEN_PHRASES = {"guaranteed target", "must hold", "certain floor"}

# Perfil Inversionista/Victor Gonzalez.md -- fixed for this system's single
# investor profile (root CLAUDE.md "Perfil del inversionista").
PROFILE_FIT_TEXT = (
    "Capital: $25,000 USD. Max position size: 30-60% of capital. Horizon: 3-5 years. "
    "Universe: United States only, no forex. Style: aggressive/speculative "
    "(stocks, ETFs, options). Priority: probability of success and entry/exit timing."
)


class RenderError(ValueError):
    """Raised when a `FinalReport` cannot be rendered as specified."""


def significant_insider_trades(trades: list[dict]) -> list[dict]:
    """root CLAUDE.md #5: only trades whose total value exceeds
    $1,000,000 are flagged 'significant' (Forms 4, SEC EDGAR)."""
    out = []
    for t in trades:
        total = abs(t.get("shares", 0) * t.get("price", 0))
        if total > INSIDER_SIGNIFICANCE_THRESHOLD:
            out.append({**t, "total_value": total, "significant": True})
    return out


def _require_revisit_if_avoid(final: FinalReport) -> None:
    """root CLAUDE.md #2: an 'avoid' classification requires a concrete
    revisit date/event in `monitoring_triggers`."""
    if "avoid" not in (final.profile.label or "").lower():
        return
    has_revisit = any(t.get("revisit_date") or t.get("revisit_event") for t in final.monitoring_triggers)
    if not has_revisit:
        raise RenderError(
            "profile label is an 'avoid' classification but no revisit_date/revisit_event "
            "was supplied in monitoring_triggers"
        )


def _check_forbidden_language(text: str) -> None:
    lowered = text.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in lowered]
    if hits:
        raise RenderError(f"forbidden phrase(s) present in rendered report: {hits}")


def _executive_summary_md(final: FinalReport) -> list[str]:
    t = final.executive_thesis
    sentences = [
        t.what_the_company_does,
        t.value_creation_durability,
        t.growth_funding,
        t.market_validation,
        t.price_implied_assumptions,
        t.nearest_levels,
        t.primary_invalidation_risk,
    ]
    lines = ["## Executive Summary", ""]
    if any(not s or not s.strip() for s in sentences):
        lines.append(f"*{INSUFFICIENT_DATA_SENTENCE}.*")
        lines.append("")
    for i, s in enumerate(sentences, start=1):
        lines.append(f"{i}. {s or INSUFFICIENT_DATA_SENTENCE}")
    lines.append("")
    return lines


def _classification_md(final: FinalReport) -> list[str]:
    label = final.profile.label or ""
    is_avoid = "avoid" in label.lower()
    classification = "Avoid" if is_avoid else "Favorable to invest (subject to profile fit and confirmation conditions)"
    lines = ["## Research Classification", ""]
    lines.append(f"**Classification:** {classification} — profile gate label: `{label}`.")
    lines.append(
        "This is a research classification with disclosed evidence, never an automatic buy/sell order."
    )
    if is_avoid:
        revisit = next(
            (t for t in final.monitoring_triggers if t.get("revisit_date") or t.get("revisit_event")), {}
        )
        lines.append(
            f"**Revisit:** {revisit.get('revisit_date') or revisit.get('revisit_event')}"
        )
    lines.append("")
    return lines


def _scorecard_md(final: FinalReport) -> list[str]:
    lines = ["## Category Scorecard", "", "| Category | Points | Max | Confidence |", "|---|---:|---:|---:|"]
    for name, cs in final.category_scorecard.items():
        lines.append(f"| {name} | {cs.points if cs.points is not None else 'N/S'} | {cs.max} | {cs.confidence if cs.confidence is not None else 'N/S'} |")
    lines.append("")
    lines.append(
        f"Raw score: {final.profile.raw_score:.1f}/100. Total confidence: {final.profile.total_confidence:.1f}/100."
    )
    lines.append("")
    return lines


def _scenarios_md(final: FinalReport) -> list[str]:
    lines = ["## Price Scenario Ranges", ""]
    if not final.valuation_scenarios:
        lines.append(f"*{INSUFFICIENT_DATA_SENTENCE} for valuation scenarios.*")
        lines.append("")
        return lines
    lines.append("| Scenario | Growth | Margin | Value |")
    lines.append("|---|---:|---:|---:|")
    for s in final.valuation_scenarios:
        lines.append(f"| {s.get('label', s.get('name', '?'))} | {s.get('growth', 0):.1%} | {s.get('margin', 0):.1%} | {s.get('value', 'N/S')} |")
    lines.append("")
    lines.append("Ranges with declared assumptions — never a single price.")
    lines.append("")
    return lines


def _levels_md(final: FinalReport) -> list[str]:
    lines = [
        "## Important Levels",
        "",
        "| Rank | Type | Lower | Center | Upper | Distance % | Distance ATR | Strength | Status | Confirmation | Invalidation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    ranked = sorted(
        (lvl for lvl in final.important_levels if lvl.get("type") != "current_price"),
        key=lambda lvl: abs(lvl.get("distance_percent") or 0),
    )
    for i, lvl in enumerate(ranked, start=1):
        center = lvl.get("value")
        lower = lvl.get("lower", center)
        upper = lvl.get("upper", center)
        lines.append(
            f"| {i} | {lvl.get('type')} | {lower:.2f} | {center:.2f} | {upper:.2f} | "
            f"{lvl.get('distance_percent', 0):.1%} | {lvl.get('distance_atr', 0):.2f} | "
            f"{lvl.get('strength_0_100', 'N/S')} | {lvl.get('status', 'N/S')} | "
            f"{lvl.get('confirmation_rule', '')} | {lvl.get('invalidation_rule', '')} |"
        )
    lines.append("")
    return lines


def _holders_and_insiders_md(final: FinalReport) -> list[str]:
    lines = ["## Notable Holders & Management Track Record", ""]
    if final.notable_holders:
        for h in final.notable_holders:
            lines.append(f"- {h.get('holder')}: {h.get('shares')} shares")
    else:
        lines.append(f"*{INSUFFICIENT_DATA_SENTENCE} for institutional ownership.*")
    for t in final.management_track_record:
        lines.append(f"- {t}")
    lines.append("")

    lines.append("## Insider Activity")
    lines.append("")
    sig = significant_insider_trades(final.insider_trades)
    if sig:
        lines.append("| Insider | Type | Shares | Price | Total value |")
        lines.append("|---|---|---:|---:|---:|")
        for t in sig:
            lines.append(
                f"| {t.get('reportingName', '?')} | {t.get('transactionType', '?')} | {t.get('shares')} | "
                f"{t.get('price')} | ${t.get('total_value', 0):,.0f} (significant) |"
            )
    else:
        lines.append("No insider trades exceeding $1,000,000 total were found.")
    lines.append("")
    return lines


def _thesis_killers_md(final: FinalReport) -> list[str]:
    lines = ["## Thesis Killers & Monitoring Triggers", ""]
    for k in final.thesis_killers:
        lines.append(f"- **{k.get('risk', '?')}** (impact: {k.get('impact', '?')}) — early warning: {k.get('early_warning_metric', '?')}")
    if not final.thesis_killers:
        lines.append(f"*{INSUFFICIENT_DATA_SENTENCE} for thesis-killer risks.*")
    lines.append("")
    for m in final.monitoring_triggers:
        lines.append(f"- Monitor: {m}")
    lines.append("")
    return lines


def _profile_fit_md() -> list[str]:
    return ["## Profile Fit", "", PROFILE_FIT_TEXT, ""]


def _missing_data_md(final: FinalReport) -> list[str]:
    lines = ["## Missing / Conflicted Data", ""]
    if final.missing_or_conflicted_data:
        for item in final.missing_or_conflicted_data:
            lines.append(f"- {item}")
    else:
        lines.append("None disclosed.")
    lines.append("")
    return lines


def _audit_md(final: FinalReport) -> list[str]:
    lines = ["## Audit Appendix", "", "**Packet hashes:**"]
    for k, v in (final.audit.get("packet_hashes") or {}).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("**Formula versions:** " + ", ".join(final.audit.get("formula_versions") or []) or "N/S")
    vs = final.audit.get("validation_summary") or {}
    lines.append(f"**Validation summary:** {vs}")
    lines.append("")
    return lines


def render(final: FinalReport, charts: dict[str, Path], out_dir: Path) -> Path:
    """Writes `report.json` and `report.md` into `out_dir`, returning
    `out_dir`. Raises `RenderError` if the report can't be rendered
    (avoid classification without a revisit date, or forbidden language)."""
    _require_revisit_if_avoid(final)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# {final.security.ticker} — Research Report", ""]
    lines.append(
        f"*Report version {final.report_version} · Analysis timestamp {final.security.analysis_timestamp} · "
        f"Knowledge timestamp {final.security.knowledge_timestamp}*"
    )
    lines.append("")
    lines += _executive_summary_md(final)
    lines += _classification_md(final)
    lines += _scorecard_md(final)
    lines += _scenarios_md(final)
    lines += _levels_md(final)
    lines += _holders_and_insiders_md(final)
    lines += _thesis_killers_md(final)
    lines += _profile_fit_md()
    lines += _missing_data_md(final)
    lines += _audit_md(final)

    if charts:
        lines.append("## Charts")
        lines.append("")
        for name, path in charts.items():
            lines.append(f"- {name}: `{path}`")
        lines.append("")

    md_text = "\n".join(lines)
    _check_forbidden_language(md_text)

    (out_dir / "report.md").write_text(md_text, encoding="utf-8")
    (out_dir / "report.json").write_text(final.model_dump_json(indent=2), encoding="utf-8")

    return out_dir
