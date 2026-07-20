"""Tests for wbj.specialists.technical, per Cerebro/04_technical_momentum/
VALIDATION_TESTS.md. TECH-T003-T009/T011 (pivots, touches, breakouts, role
reversal) are already covered end-to-end in tests/engines/test_levels.py
(Task 12) since this specialist only orchestrates that engine; this file
covers TECH-T001/T002/T012 and the DECISION_RULES.md primary-trend anchors
this specialist adds on top.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wbj.engines.indicators import atr14, true_range
from wbj.schemas.packet import Packet
from wbj.specialists.technical import classify_primary_trend, run, volume_demand_signal

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


# --- TECH-T001/T002: base indicator sanity (guarded ratios) -------------


def test_TECH_T001_constant_price_series_atr_is_zero():
    df = pd.DataFrame({"open": [100.0] * 20, "high": [100.0] * 20, "low": [100.0] * 20, "close": [100.0] * 20, "volume": [1000] * 20})
    out = atr14(df)
    assert out.iloc[-1] == pytest.approx(0.0)


def test_TECH_T002_true_range():
    df = pd.DataFrame({"open": [11.0, 11.0], "high": [11.0, 12.0], "low": [11.0, 10.0], "close": [11.0, 11.5]})
    tr = true_range(df)
    assert tr.iloc[1] == pytest.approx(2.0)


# --- DECISION_RULES.md primary-trend anchors --------------------------


def test_trend_anchor_downtrend_stacked_below():
    score = classify_primary_trend(
        close=80, sma50=85, sma200=100, atr14=2.0, sma200_slope_atr=-1.5, sma50_slope_atr=-1.0, adx14=20, range_pos_52w=0.1
    )
    assert 0 <= score <= 2


def test_trend_anchor_below_sma200_mixed_stack():
    score = classify_primary_trend(
        close=90, sma50=95, sma200=100, atr14=2.0, sma200_slope_atr=0.1, sma50_slope_atr=0.1, adx14=20, range_pos_52w=0.3
    )
    assert score == pytest.approx(3.0)


def test_trend_anchor_near_sma200_flat():
    score = classify_primary_trend(
        close=100.5, sma50=101, sma200=100, atr14=2.0, sma200_slope_atr=0.1, sma50_slope_atr=0.0, adx14=18, range_pos_52w=0.5
    )
    assert 4 <= score <= 5


def test_trend_anchor_above_sma200_mixed_sma50():
    score = classify_primary_trend(
        close=110, sma50=108, sma200=100, atr14=2.0, sma200_slope_atr=0.1, sma50_slope_atr=-0.05, adx14=18, range_pos_52w=0.6
    )
    assert score == pytest.approx(6.0)


def test_trend_anchor_strong_uptrend_both_slopes_positive():
    score = classify_primary_trend(
        close=120, sma50=110, sma200=100, atr14=2.0, sma200_slope_atr=0.5, sma50_slope_atr=1.0, adx14=20, range_pos_52w=0.6
    )
    assert score == pytest.approx(8.0)


def test_trend_anchor_powerful_leadership_adx_and_52w_high():
    score = classify_primary_trend(
        close=120, sma50=110, sma200=100, atr14=2.0, sma200_slope_atr=0.5, sma50_slope_atr=1.0, adx14=28, range_pos_52w=0.85
    )
    assert 9 <= score <= 10


def test_trend_anchor_none_without_sma200():
    assert classify_primary_trend(close=100, sma50=100, sma200=None, atr14=2.0, sma200_slope_atr=None, sma50_slope_atr=None, adx14=None, range_pos_52w=None) is None


# --- TECH-T012: volume missing caps the volume dimension --------------


def test_TECH_T012_volume_demand_bad_signal_scores_low():
    score = volume_demand_signal(ud_ratio=0.4, cmf_value=-0.15, obv_slope=-1.0)
    assert score <= 3.0


def test_volume_demand_none_when_no_signal_available():
    assert volume_demand_signal(ud_ratio=None, cmf_value=None, obv_slope=None) is None


# --- schema-valid run against the golden NVDA packet -----------------------


def test_nvda_fixture_produces_schema_valid_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = run(packet)

    assert out.agent_id == "technical_momentum"
    assert out.category.max_points == 20
    assert len(out.dimensions) == 6
    assert sum(d["max_points"] for d in out.dimensions) == 20
    assert 0.0 <= out.coverage <= 1.0
    # No benchmark/sector series in this packet -> relative strength is NOT_SCORABLE.
    rs_dim = next(d for d in out.dimensions if d["name"] == "relative_strength")
    assert rs_dim["score_10"] is None

    dim_points_sum = sum(
        d["max_points"] * (d["score_10"] / 10.0) for d in out.dimensions if d["score_10"] is not None
    )
    assert abs(dim_points_sum - out.category.awarded_points) < 1e-6
