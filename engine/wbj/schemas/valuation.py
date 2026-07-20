"""Pydantic models for the institutional valuation engine output.

Mirrors Cerebro/special_sauces/INSTITUTIONAL_VALUATION_ENGINE.md sections
6 (FCFF DCF) and 16 (scenarios).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DCFResult(BaseModel):
    """Output of `dcf_value` — enterprise value split into explicit and
    terminal present values, per INSTITUTIONAL_VALUATION_ENGINE.md 6.3/6.6."""

    ev: float | None
    pv_explicit: float | None = None
    pv_terminal: float | None = None
    terminal_share: float | None = None
    state: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ScenarioInput(BaseModel):
    """One Bear/Base/Bull scenario definition, per section 16.1."""

    label: str
    probability: float
    growth: float
    margin: float
    wacc: float
    tv_growth: float
    value: float
