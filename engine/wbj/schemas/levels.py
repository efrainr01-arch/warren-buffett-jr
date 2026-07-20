"""Pydantic models for the important-levels engine output.

Mirrors Cerebro/special_sauces/IMPORTANT_LEVELS_ENGINE.md section A/J.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ZoneType = Literal["support", "resistance"]
Timeframe = Literal["daily", "weekly"]
ZoneStatus = Literal["candidate", "confirmed", "strong", "broken", "role_reversed"]


class Touch(BaseModel):
    """A single independent touch of a zone (IMPORTANT_LEVELS_ENGINE.md A4)."""

    date: str
    pivot_price: float
    rejection_atr: float
    volume_ratio: float | None = None
    age_sessions: int


class Zone(BaseModel):
    """A clustered support/resistance zone (IMPORTANT_LEVELS_ENGINE.md A3/A6/A7)."""

    zone_id: str
    type: ZoneType
    lower: float
    center: float
    upper: float
    timeframe: Timeframe
    status: ZoneStatus = "candidate"
    strength_0_100: float = 0.0
    touches: list[Touch] = Field(default_factory=list)
    distance_percent: float | None = None
    distance_atr: float | None = None
    confirmation_rule: str = ""
    invalidation_rule: str = ""
