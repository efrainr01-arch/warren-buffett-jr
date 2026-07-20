"""Pydantic model for the final report — Cerebro/00_main_agent/
FINAL_REPORT_SCHEMA.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReportSecurity(BaseModel):
    ticker: str
    exchange: str
    currency: str
    analysis_timestamp: str
    knowledge_timestamp: str


class ReportProfile(BaseModel):
    label: str
    raw_score: float
    total_confidence: float
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[dict] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list)


class CategoryScore(BaseModel):
    points: float | None
    max: float
    confidence: float | None


class ExecutiveThesis(BaseModel):
    """The seven required executive-summary sentences (FINAL_REPORT_SCHEMA.md)."""

    business_quality: str
    growth_engine: str
    market_validation: str
    valuation_message: str
    primary_risk: str


class FinalReport(BaseModel):
    report_version: str = "2.0.0"
    security: ReportSecurity
    profile: ReportProfile
    category_scorecard: dict[str, CategoryScore]
    executive_thesis: ExecutiveThesis
    important_levels: list[dict] = Field(default_factory=list)
    valuation_scenarios: list[dict] = Field(default_factory=list)
    reverse_dcf: dict[str, Any] = Field(default_factory=dict)
    thesis_killers: list[dict] = Field(default_factory=list)
    monitoring_triggers: list[dict] = Field(default_factory=list)
    missing_or_conflicted_data: list[str] = Field(default_factory=list)
    audit: dict[str, Any] = Field(
        default_factory=lambda: {"packet_hashes": {}, "formula_versions": [], "validation_summary": {}}
    )
