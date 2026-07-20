"""Shared specialist output envelope, per Cerebro/shared/OUTPUT_CONTRACT.md
and the common envelope described in each specialist's OUTPUT_SCHEMA.md.

Every specialist module exposes `run(packet, overlay=None) -> <Agent>Output`
where `<Agent>Output` subclasses `SpecialistOutput` and adds its own
agent-specific extension fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["COMPLETE", "PARTIAL", "NOT_SCORABLE"]


class MetricRow(BaseModel):
    """One row of the output contract (Cerebro/shared/OUTPUT_CONTRACT.md):
    every scored metric must carry its formula id/version, value, unit,
    period, score, evidence class, source, confidence, and warnings."""

    metric_id: str
    value: float | None
    state: str | None = None
    unit: str = ""
    period: str | None = None
    formula: str
    score: float | str | None = None  # 0-10 float, or "NOT_SCORABLE"
    evidence_class: str | None = None
    source: str | None = None
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class JudgmentRequest(BaseModel):
    """A judgment-only metric the specialist could not score mechanically;
    merged later by `wbj.overlay.merge` (Task 20)."""

    request_id: str
    agent_id: str
    metric_id: str
    question: str
    schema_hint: str


class CategorySummary(BaseModel):
    max_points: float
    awarded_points: float | None = None
    score_10: float | None = None
    confidence: float | None = None


class SpecialistOutput(BaseModel):
    """The envelope every specialist output shares, per OUTPUT_CONTRACT.md
    and each agent's OUTPUT_SCHEMA.md `category`/`dimensions`/`metrics`
    block. Agent-specific fields (e.g. financial's `core_27_metrics`) live
    on subclasses."""

    agent_id: str
    version: str = "2.0.0"
    status: Status
    security: dict[str, Any] = Field(default_factory=dict)
    knowledge_timestamp: str | None = None
    category: CategorySummary
    coverage: float | None = None
    dimensions: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[MetricRow] = Field(default_factory=list)
    mandatory_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    judgment_requests: list[JudgmentRequest] = Field(default_factory=list)
    source_lineage: list[str] = Field(default_factory=list)
    validation_tests: dict[str, int] = Field(
        default_factory=lambda: {"passed": 0, "failed": 0, "warnings": 0}
    )
