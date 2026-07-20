"""Tests for wbj.report.render, per Cerebro/00_main_agent/
FINAL_REPORT_SCHEMA.md and root CLAUDE.md's "Contenido obligatorio del
reporte final".
"""

from __future__ import annotations

import json

import pytest

from wbj.report.render import RenderError, render, significant_insider_trades
from wbj.schemas.final_report import (
    CategoryScore,
    ExecutiveThesis,
    FinalReport,
    ReportProfile,
    ReportSecurity,
)


def _minimal_report(label: str = "Conditional/Watch", monitoring_triggers: list | None = None) -> FinalReport:
    return FinalReport(
        security=ReportSecurity(
            ticker="EXMPL", exchange="NASDAQ", currency="USD",
            analysis_timestamp="2026-07-20T00:00:00Z", knowledge_timestamp="2026-07-20T00:00:00Z",
        ),
        profile=ReportProfile(label=label, raw_score=65.0, total_confidence=72.0, passed_gates=[], failed_gates=[], overrides=[]),
        category_scorecard={
            "business": CategoryScore(points=14.0, max=20, confidence=80),
            "financial": CategoryScore(points=10.0, max=15, confidence=85),
            "market": CategoryScore(points=12.0, max=20, confidence=70),
            "technical": CategoryScore(points=10.0, max=20, confidence=75),
            "risk": CategoryScore(points=9.0, max=15, confidence=80),
            "valuation": CategoryScore(points=6.0, max=10, confidence=65),
        },
        executive_thesis=ExecutiveThesis(
            what_the_company_does="The company sells widgets to industrial customers.",
            value_creation_durability="ROIC has exceeded WACC in 4 of the last 5 years.",
            growth_funding="Growth is self-funded from operating cash flow.",
            market_validation="Consensus estimates have been revised upward this quarter.",
            price_implied_assumptions="The current price implies roughly 8% revenue CAGR.",
            nearest_levels="Nearest resistance sits near $85; base intrinsic value is $90.",
            primary_invalidation_risk="A material customer concentration loss would break the thesis.",
        ),
        important_levels=[
            {"type": "resistance", "source": "technical", "value": 85.0, "lower": 84.0, "upper": 86.0,
             "distance_percent": 0.06, "distance_atr": 1.2, "strength_0_100": 70, "status": "confirmed",
             "confirmation_rule": "close > 86.5 w/ volume", "invalidation_rule": "close back below 84"},
        ],
        valuation_scenarios=[
            {"label": "Bear", "growth": 0.02, "margin": 0.10, "value": 70.0},
            {"label": "Base", "growth": 0.06, "margin": 0.15, "value": 90.0},
            {"label": "Bull", "growth": 0.10, "margin": 0.20, "value": 110.0},
        ],
        notable_holders=[{"holder": "Example Capital", "shares": 1_000_000}],
        management_track_record=["CEO previously scaled a prior company to a successful acquisition."],
        insider_trades=[],
        thesis_killers=[{"risk": "Customer concentration", "impact": "high", "early_warning_metric": "top-customer revenue share"}],
        monitoring_triggers=monitoring_triggers or [{"trigger": "Q3 earnings release"}],
        missing_or_conflicted_data=[],
        audit={"packet_hashes": {"NVDA": "abc123"}, "formula_versions": ["FIN-2.0.0"], "validation_summary": {"passed": 10, "failed": 0}},
    )


# --- section headers present --------------------------------------------


def test_render_includes_all_required_section_headers(tmp_path):
    final = _minimal_report()
    out_dir = render(final, charts={}, out_dir=tmp_path / "out")
    md = (out_dir / "report.md").read_text(encoding="utf-8")

    required_headers = [
        "## Executive Summary",
        "## Research Classification",
        "## Category Scorecard",
        "## Price Scenario Ranges",
        "## Important Levels",
        "## Notable Holders & Management Track Record",
        "## Insider Activity",
        "## Thesis Killers & Monitoring Triggers",
        "## Profile Fit",
        "## Missing / Conflicted Data",
        "## Audit Appendix",
    ]
    for header in required_headers:
        assert header in md


def test_render_writes_seven_executive_summary_sentences(tmp_path):
    final = _minimal_report()
    out_dir = render(final, charts={}, out_dir=tmp_path / "out")
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    for i in range(1, 8):
        assert f"{i}. " in md


# --- avoid without revisit date raises ------------------------------------


def test_avoid_classification_without_revisit_raises(tmp_path):
    final = _minimal_report(label="Avoid/Wait", monitoring_triggers=[{"trigger": "generic, no date"}])
    with pytest.raises(RenderError, match="revisit"):
        render(final, charts={}, out_dir=tmp_path / "out")


def test_avoid_classification_with_revisit_succeeds(tmp_path):
    final = _minimal_report(label="Avoid/Wait", monitoring_triggers=[{"revisit_date": "2026-10-15", "reason": "next earnings"}])
    out_dir = render(final, charts={}, out_dir=tmp_path / "out")
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "2026-10-15" in md


# --- insider significance filter ----------------------------------------


def test_insider_filter_999999_not_flagged_1000001_flagged():
    trades = [
        {"reportingName": "A", "transactionType": "S-Sale", "shares": 999_999, "price": 1.0},
        {"reportingName": "B", "transactionType": "S-Sale", "shares": 1_000_001, "price": 1.0},
    ]
    sig = significant_insider_trades(trades)
    names = {t["reportingName"] for t in sig}
    assert "A" not in names
    assert "B" in names


# --- forbidden language ---------------------------------------------------


def test_forbidden_phrases_are_rejected(tmp_path):
    final = _minimal_report()
    final.executive_thesis.primary_invalidation_risk = "This is a guaranteed target for the stock."
    with pytest.raises(RenderError, match="forbidden phrase"):
        render(final, charts={}, out_dir=tmp_path / "out")


# --- report.json round-trips through the schema -----------------------


def test_report_json_round_trips_through_schema(tmp_path):
    final = _minimal_report()
    out_dir = render(final, charts={}, out_dir=tmp_path / "out")
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    reloaded = FinalReport.model_validate(payload)
    assert reloaded.security.ticker == "EXMPL"
    assert reloaded.profile.label == final.profile.label
