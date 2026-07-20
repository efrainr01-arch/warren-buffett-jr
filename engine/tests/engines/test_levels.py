"""Tests for wbj.engines.levels_engine, per Cerebro/special_sauces/
IMPORTANT_LEVELS_ENGINE.md.

Convention: DataFrames are oldest-first with a `date` column (ISO string)
plus `open/high/low/close/volume`, matching wbj.engines.indicators.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wbj.engines.indicators import atr14
from wbj.engines.levels_engine import (
    breakout_confirmed,
    cluster_zones,
    count_touches,
    classify,
    find_pivots,
    rank_levels,
    strength,
)
from wbj.schemas.levels import Touch, Zone


def _dates(n: int) -> list[str]:
    return [f"2026-01-{(i % 28) + 1:02d}" for i in range(n)]


def _flat_ohlcv(n: int, price: float = 100.0, volume: float = 1_000_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _dates(n),
            "open": [price] * n,
            "high": [price + 1] * n,
            "low": [price - 1] * n,
            "close": [price] * n,
            "volume": [volume] * n,
        }
    )


# --- pivots ------------------------------------------------------------


def test_symmetric_pivot_k3_detects_local_max():
    n = 20
    df = _flat_ohlcv(n)
    df.loc[10, "high"] = 150.0  # a clean local max at position 10, bars 7..13
    pivots = find_pivots(df, k=3)
    highs = [p for p in pivots if p["type"] == "high"]
    assert any(p["pos"] == 10 and p["price"] == 150.0 for p in highs)


def test_pivot_not_confirmed_without_k_future_bars():
    n = 20
    df = _flat_ohlcv(n)
    df.loc[n - 1, "high"] = 150.0  # last bar: no future bars to confirm it
    pivots = find_pivots(df, k=3)
    assert not any(p["pos"] == n - 1 for p in pivots)


# --- zone tolerance / clustering ----------------------------------------


def test_zone_tolerance_formula():
    # atr=2, price=100 -> tolerance = max(0.50*2, 0.0075*100) = max(1.0, 0.75) = 1.0
    df = _flat_ohlcv(30)
    atr = pd.Series([2.0] * 30)
    pivots = [{"type": "high", "pos": 15, "date": df["date"].iloc[15], "price": 100.0}]
    zones = cluster_zones(pivots, atr, current_pos=29, timeframe="daily")
    assert len(zones) == 1
    z = zones[0]
    assert z.upper - z.center == pytest.approx(1.0)
    assert z.center - z.lower == pytest.approx(1.0)


# --- touches -------------------------------------------------------------


def test_touches_5_sessions_apart():
    n = 40
    df = _flat_ohlcv(n)
    atr = pd.Series([1.0] * n)
    zone = Zone(zone_id="z1", type="resistance", lower=99.0, center=100.0, upper=101.0, timeframe="daily")

    # Two touches 3 sessions apart (too close -> only the first counts) and
    # a valid rejection each time (price drops >=0.5 ATR within 3 sessions).
    for pos in (10, 13, 25):
        df.loc[pos, "high"] = 100.5
        df.loc[pos + 1, "low"] = 99.0  # 1.0 ATR rejection, well over 0.5 ATR

    pivots = [
        {"type": "high", "pos": 10, "date": df["date"].iloc[10], "price": 100.5},
        {"type": "high", "pos": 13, "date": df["date"].iloc[13], "price": 100.5},
        {"type": "high", "pos": 25, "date": df["date"].iloc[25], "price": 100.5},
    ]
    updated = count_touches(zone, pivots, atr, df, min_gap_sessions=5)
    touched_positions = sorted(t.date for t in updated.touches)
    # pos 13 is within 5 sessions of pos 10 -> dropped; pos 25 is far enough.
    assert len(updated.touches) == 2
    assert df["date"].iloc[10] in touched_positions
    assert df["date"].iloc[25] in touched_positions
    assert df["date"].iloc[13] not in touched_positions


# --- strength --------------------------------------------------------------


def test_strength_formula_exact():
    # N_eff = 2 fresh touches (age 0 each) -> 30*min(2/4,1) = 15
    # median_reaction_ATR = 1.0 -> 20*min(1/2,1) = 10
    # median_volume_ratio = 1.5 -> 15*min(1.5/1.5,1) = 15
    # age_latest = 0 -> 15*exp(0) = 15
    # daily -> Timeframe = 5
    # confluence_count = 0 -> 0
    # total = 15+10+15+15+5+0 = 60
    zone = Zone(
        zone_id="z1",
        type="resistance",
        lower=99.0,
        center=100.0,
        upper=101.0,
        timeframe="daily",
        touches=[
            Touch(date="2026-01-01", pivot_price=100.5, rejection_atr=1.0, volume_ratio=1.5, age_sessions=0),
            Touch(date="2026-01-02", pivot_price=100.6, rejection_atr=1.0, volume_ratio=1.5, age_sessions=0),
        ],
    )
    assert strength(zone, confluence_count=0) == pytest.approx(60.0)


def test_strength_capped_at_100():
    touches = [
        Touch(date=f"2026-01-{i:02d}", pivot_price=100.5, rejection_atr=5.0, volume_ratio=5.0, age_sessions=0)
        for i in range(1, 9)
    ]
    zone = Zone(zone_id="z1", type="resistance", lower=99, center=100, upper=101, timeframe="weekly", touches=touches)
    assert strength(zone, confluence_count=10) == 100.0


# --- classify ----------------------------------------------------------


def test_classify_one_touch_is_candidate():
    zone = Zone(
        zone_id="z", type="resistance", lower=99, center=100, upper=101, timeframe="daily",
        touches=[Touch(date="d", pivot_price=100.5, rejection_atr=1.0, volume_ratio=1.0, age_sessions=0)],
    )
    assert classify(zone) == "candidate"


def test_classify_three_touches_is_strong():
    touches = [
        Touch(date=f"d{i}", pivot_price=100.5, rejection_atr=0.6, volume_ratio=1.0, age_sessions=i * 10)
        for i in range(3)
    ]
    zone = Zone(zone_id="z", type="resistance", lower=99, center=100, upper=101, timeframe="daily", touches=touches)
    assert classify(zone) == "strong"


def test_classify_two_touches_confirmed_unless_strong_conditions():
    touches = [
        Touch(date="d1", pivot_price=100.5, rejection_atr=0.6, volume_ratio=1.0, age_sessions=0),
        Touch(date="d2", pivot_price=100.5, rejection_atr=0.6, volume_ratio=1.0, age_sessions=20),
    ]
    zone = Zone(zone_id="z", type="resistance", lower=99, center=100, upper=101, timeframe="daily", touches=touches)
    assert classify(zone) == "confirmed"


# --- breakout ----------------------------------------------------------


def test_breakout_requires_volume_and_close():
    n = 60
    df = _flat_ohlcv(n)
    atr = pd.Series([1.0] * n)
    zone = Zone(zone_id="z", type="resistance", lower=99, center=100, upper=101, timeframe="daily")

    # Close clears the buffer but volume is not >=1.5x median -> not confirmed.
    df.loc[n - 1, "close"] = 102.0
    assert breakout_confirmed(df, zone, atr, pos=n - 1) is False

    # Now spike volume and give a second confirming close.
    df.loc[n - 1, "volume"] = 5_000_000
    df.loc[n - 2, "close"] = 101.6  # also above buffer -> two consecutive closes
    assert breakout_confirmed(df, zone, atr, pos=n - 1) is True


# --- earnings gap materiality -----------------------------------------


def test_gap_material_threshold():
    from wbj.engines.levels_engine import earnings_gaps

    n = 10
    df = _flat_ohlcv(n, price=100.0)
    atr = pd.Series([1.0] * n)

    # 2% gap on a $1 ATR / $100 close: not material (< max(1 ATR=1, 3%=3))
    df.loc[5, "open"] = 102.0
    small = earnings_gaps(df, [df["date"].iloc[5]], atr)
    assert small == []

    # 3.1% gap: material (> 3% and > 1 ATR)
    df.loc[5, "open"] = 103.1
    df.loc[5, "high"] = 103.1
    big = earnings_gaps(df, [df["date"].iloc[5]], atr)
    assert len(big) == 1
    assert big[0]["gap_open"] == pytest.approx(103.1)


# --- ranking -------------------------------------------------------------


def test_rank_levels_orders_by_relevance_desc():
    entries = [
        {"id": "a", "strength": 50, "recency": 50, "cross_lens_confluence": 0, "liquidity_confidence": 0},
        {"id": "b", "strength": 90, "recency": 90, "cross_lens_confluence": 90, "liquidity_confidence": 90},
    ]
    ranked = rank_levels(entries)
    assert [e["id"] for e in ranked] == ["b", "a"]
    assert ranked[0]["relevance"] > ranked[1]["relevance"]
