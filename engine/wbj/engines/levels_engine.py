"""Important-levels engine — Cerebro/special_sauces/IMPORTANT_LEVELS_ENGINE.md
and Cerebro/04_technical_momentum/FORMULAS.md TECH-022..035.

Convention: DataFrames are oldest-first (ascending date index) with a
`date` column (ISO string) plus `open/high/low/close/volume`, matching
`wbj.engines.indicators`. "Age" is measured in bar positions (sessions),
relative to the last row of the DataFrame passed to each function, not
wall-clock time — callers pass the frame sliced to "as of" whatever date
they're evaluating.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal

import numpy as np
import pandas as pd

from wbj.schemas.levels import Touch, Zone

_HALF_LIFE_SESSIONS = 126


def _recency_weight(age_sessions: float) -> float:
    return math.exp(-math.log(2) * age_sessions / _HALF_LIFE_SESSIONS)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return statistics.median(values)
    cum = 0.0
    for value, weight in pairs:
        cum += weight
        if cum >= total / 2:
            return value
    return pairs[-1][0]


# --- A1: swing detection -----------------------------------------------


def find_pivots(df: pd.DataFrame, k: int) -> list[dict]:
    """TECH-PIV-022: symmetric pivot highs/lows over a `2k+1`-bar window
    centered at `t` (bars `t-k..t+k`). A pivot is only returned once its
    `k` future confirmation bars exist — ties within a window resolve to
    the leftmost (canonical) occurrence of the extreme."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    pivots: list[dict] = []
    for t in range(k, n - k):
        high_window = highs[t - k : t + k + 1]
        if highs[t] == high_window.max() and int(np.argmax(high_window)) == k:
            pivots.append({"type": "high", "pos": t, "date": df["date"].iloc[t], "price": float(highs[t])})
        low_window = lows[t - k : t + k + 1]
        if lows[t] == low_window.min() and int(np.argmin(low_window)) == k:
            pivots.append({"type": "low", "pos": t, "date": df["date"].iloc[t], "price": float(lows[t])})
    return pivots


def zigzag_pivots(df: pd.DataFrame, atr: pd.Series) -> list[dict]:
    """TECH-ZIG-023: ATR zigzag — confirm a reversal once price retraces
    >=1.5*ATR14 (measured at the running extreme) from that extreme."""
    n = len(df)
    if n == 0:
        return []
    close = df["close"]
    pivots: list[dict] = []
    direction = 0  # 0 undetermined, 1 tracking a high, -1 tracking a low
    extreme_pos = 0
    extreme_price = float(close.iloc[0])
    for i in range(1, n):
        price = float(close.iloc[i])
        if direction >= 0 and price >= extreme_price:
            extreme_pos, extreme_price, direction = i, price, 1
            continue
        if direction <= 0 and price <= extreme_price:
            extreme_pos, extreme_price, direction = i, price, -1
            continue
        atr_at_extreme = atr.iloc[extreme_pos]
        if pd.isna(atr_at_extreme):
            continue
        threshold = 1.5 * atr_at_extreme
        if direction == 1 and (extreme_price - price) >= threshold:
            pivots.append(
                {"type": "high", "pos": extreme_pos, "date": df["date"].iloc[extreme_pos], "price": extreme_price}
            )
            direction, extreme_pos, extreme_price = -1, i, price
        elif direction == -1 and (price - extreme_price) >= threshold:
            pivots.append(
                {"type": "low", "pos": extreme_pos, "date": df["date"].iloc[extreme_pos], "price": extreme_price}
            )
            direction, extreme_pos, extreme_price = 1, i, price
    return pivots


# --- A2/A3: zone width and clustering -----------------------------------


def cluster_zones(pivots: list[dict], atr: pd.Series, current_pos: int, timeframe: Literal["daily", "weekly"]) -> list[Zone]:
    """TECH-ZTOL-024/TECH-ZONE-025: cluster overlapping pivot intervals
    (highs and lows separately) into zones; center/half-width are the
    recency-weighted median of the cluster's pivot prices/tolerances."""
    zones: list[Zone] = []
    for zone_type, ptype in (("resistance", "high"), ("support", "low")):
        enriched = []
        for p in pivots:
            if p["type"] != ptype:
                continue
            atr_i = atr.iloc[p["pos"]]
            if pd.isna(atr_i):
                continue
            tol = max(0.50 * atr_i, 0.0075 * p["price"])
            enriched.append({**p, "tolerance": tol, "lo": p["price"] - tol, "hi": p["price"] + tol})
        enriched.sort(key=lambda e: e["price"])

        clusters: list[list[dict]] = []
        for e in enriched:
            if clusters and e["lo"] <= clusters[-1][-1]["hi"]:
                clusters[-1].append(e)
            else:
                clusters.append([e])

        for idx, cluster in enumerate(clusters):
            weights = [_recency_weight(current_pos - c["pos"]) for c in cluster]
            center = _weighted_median([c["price"] for c in cluster], weights)
            half_width = _weighted_median([c["tolerance"] for c in cluster], weights)
            zones.append(
                Zone(
                    zone_id=f"{timeframe}_{zone_type}_{idx}",
                    type=zone_type,
                    lower=center - half_width,
                    center=center,
                    upper=center + half_width,
                    timeframe=timeframe,
                )
            )
    return zones


# --- A4/A5: independent touches and rejection magnitude --------------------


def count_touches(
    zone: Zone, pivots: list[dict], atr: pd.Series, df: pd.DataFrame, min_gap_sessions: int
) -> Zone:
    """TECH-NEFF-026/TECH-REJ-027: independent, validly-rejected touches of
    `zone` — a pivot of the matching type inside the zone bounds, spaced at
    least `min_gap_sessions` apart from the prior counted touch, with a
    rejection of >=0.5 ATR within 3 sessions."""
    ptype = "high" if zone.type == "resistance" else "low"
    candidates = sorted(
        (p for p in pivots if p["type"] == ptype and zone.lower <= p["price"] <= zone.upper),
        key=lambda p: p["pos"],
    )
    last_pos: int | None = None
    touches: list[Touch] = []
    for p in candidates:
        pos = p["pos"]
        if last_pos is not None and (pos - last_pos) < min_gap_sessions:
            continue
        atr_at_touch = atr.iloc[pos]
        if pd.isna(atr_at_touch) or atr_at_touch == 0 or pos + 1 >= len(df):
            continue
        window_end = min(pos + 3, len(df) - 1)
        if zone.type == "resistance":
            future_extreme = df["low"].iloc[pos + 1 : window_end + 1].min()
            reaction = (zone.center - future_extreme) / atr_at_touch
        else:
            future_extreme = df["high"].iloc[pos + 1 : window_end + 1].max()
            reaction = (future_extreme - zone.center) / atr_at_touch
        if reaction < 0.5:
            continue
        median_vol_prior = df["volume"].iloc[max(0, pos - 50) : pos].median() if pos > 0 else None
        volume_ratio = float(df["volume"].iloc[pos] / median_vol_prior) if median_vol_prior else None
        touches.append(
            Touch(
                date=df["date"].iloc[pos],
                pivot_price=p["price"],
                rejection_atr=float(reaction),
                volume_ratio=volume_ratio,
                age_sessions=len(df) - 1 - pos,
            )
        )
        last_pos = pos
    return zone.model_copy(update={"touches": touches})


# --- A6/A7: classification and strength -------------------------------------


def classify(zone: Zone) -> Literal["candidate", "confirmed", "strong"]:
    """TECH label per IMPORTANT_LEVELS_ENGINE.md A6 (touch-count part only
    — Broken/Role-reversed are assigned by the caller from
    `breakout_confirmed`/role-reversal logic, not from touch count)."""
    n = len(zone.touches)
    if n >= 3:
        return "strong"
    if n == 2:
        reactions = [t.rejection_atr for t in zone.touches]
        vol_ratios = [t.volume_ratio for t in zone.touches if t.volume_ratio is not None]
        if statistics.median(reactions) >= 1.0 and any(v >= 1.5 for v in vol_ratios):
            return "strong"
        return "confirmed"
    return "candidate"


def strength(zone: Zone, confluence_count: int = 0) -> float:
    """TECH-LSTR-028: level strength score, capped at 100."""
    if not zone.touches:
        return 0.0
    ages = [t.age_sessions for t in zone.touches]
    n_eff = sum(_recency_weight(a) for a in ages)
    median_reaction = statistics.median(t.rejection_atr for t in zone.touches)
    vol_ratios = [t.volume_ratio for t in zone.touches if t.volume_ratio is not None]
    median_volume_ratio = statistics.median(vol_ratios) if vol_ratios else 0.0
    age_latest = min(ages)

    touch_pts = 30 * min(n_eff / 4, 1)
    reaction_pts = 20 * min(median_reaction / 2, 1)
    volume_pts = 15 * min(median_volume_ratio / 1.5, 1)
    recency_pts = 15 * _recency_weight(age_latest)
    timeframe_pts = 10 if zone.timeframe == "weekly" else 5
    confluence_pts = 10 * min(confluence_count / 3, 1)

    return min(100.0, touch_pts + reaction_pts + volume_pts + recency_pts + timeframe_pts + confluence_pts)


# --- C: breakout / failed breakout / role reversal --------------------------


def breakout_confirmed(df: pd.DataFrame, zone: Zone, atr: pd.Series, pos: int | None = None) -> bool:
    """TECH-BCONF-031: confirmed upside breakout as of bar `pos` (defaults
    to the last bar)."""
    if pos is None:
        pos = len(df) - 1
    atr_pos = atr.iloc[pos]
    if pd.isna(atr_pos):
        return False
    buffer_level = zone.upper + 0.25 * atr_pos
    if df["close"].iloc[pos] <= buffer_level:
        return False

    window_start = max(0, pos - 50)
    median_vol_50 = df["volume"].iloc[window_start:pos].median() if pos > 0 else None
    if not median_vol_50 or df["volume"].iloc[pos] / median_vol_50 < 1.5:
        return False

    two_consecutive = False
    if pos >= 1 and pd.notna(atr.iloc[pos - 1]):
        prior_buffer = zone.upper + 0.25 * atr.iloc[pos - 1]
        two_consecutive = df["close"].iloc[pos - 1] > prior_buffer

    no_close_back_inside = False
    if pos + 3 < len(df):
        future_closes = df["close"].iloc[pos + 1 : pos + 4]
        no_close_back_inside = bool((future_closes >= zone.upper).all())

    return bool(two_consecutive or no_close_back_inside)


def failed_breakout(df: pd.DataFrame, zone: Zone, breakout_pos: int) -> bool:
    """TECH-FBRK-032: price closes back inside/below the zone within 3
    sessions of a breakout at `breakout_pos`."""
    end = min(breakout_pos + 3, len(df) - 1)
    if breakout_pos + 1 > end:
        return False
    window = df["close"].iloc[breakout_pos + 1 : end + 1]
    return bool((window <= zone.upper).any())


# --- D/E: AVWAP and volume profile ------------------------------------------


def avwap(df: pd.DataFrame, anchor_pos: int) -> float:
    """TECH-AVWAP-034: anchored VWAP from `anchor_pos` to the last row."""
    sub = df.iloc[anchor_pos:]
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3
    return float((typical * sub["volume"]).sum() / sub["volume"].sum())


def volume_profile(df: pd.DataFrame, atr_latest: float, price_latest: float) -> dict:
    """TECH-VP-035: approximate volume-at-price profile; POC/HVN/LVN."""
    bin_width = max(0.50 * atr_latest, 0.005 * price_latest)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    bin_index = ((typical - typical.min()) // bin_width).astype(int)
    grouped = df["volume"].groupby(bin_index).sum()
    poc_bin = grouped.idxmax()
    poc_price = float(typical.min() + poc_bin * bin_width + bin_width / 2)
    p75, p25 = grouped.quantile(0.75), grouped.quantile(0.25)
    return {
        "bin_width": float(bin_width),
        "poc_price": poc_price,
        "hvn_bins": grouped[grouped > p75].index.tolist(),
        "lvn_bins": grouped[grouped < p25].index.tolist(),
    }


# --- F: earnings gaps --------------------------------------------------


def earnings_gaps(df: pd.DataFrame, earnings_dates: list[str], atr: pd.Series) -> list[dict]:
    """TECH-GAP-020/TECH-GHOLD-021: material earnings gaps and their
    day-1/5/20 hold ratios. Material when |gap| >= max(1.0*ATR, 3% of
    prior close)."""
    dates = df["date"].tolist()
    gaps: list[dict] = []
    for ed in earnings_dates:
        if ed not in dates:
            continue
        pos = dates.index(ed)
        if pos == 0:
            continue
        prior_close = float(df["close"].iloc[pos - 1])
        gap_open = float(df["open"].iloc[pos])
        atr_pos = atr.iloc[pos]
        if pd.isna(atr_pos):
            continue
        threshold = max(1.0 * atr_pos, 0.03 * abs(prior_close))
        gap_size = gap_open - prior_close
        if abs(gap_size) < threshold:
            continue

        gap_high = max(float(df["high"].iloc[pos]), gap_open)
        gap_low = min(float(df["low"].iloc[pos]), gap_open)

        def _hold_ratio(k: int) -> float | None:
            if gap_size == 0:
                return None
            end = min(pos + k, len(df) - 1)
            return (float(df["close"].iloc[end]) - prior_close) / gap_size

        gaps.append(
            {
                "date": ed,
                "prior_close": prior_close,
                "gap_open": gap_open,
                "gap_high": gap_high,
                "gap_low": gap_low,
                "midpoint": (gap_high + gap_low) / 2,
                "day1_hold": _hold_ratio(1),
                "day5_hold": _hold_ratio(5),
                "day20_hold": _hold_ratio(20),
            }
        )
    return gaps


# --- I: ranking ----------------------------------------------------------


def rank_levels(entries: list[dict]) -> list[dict]:
    """IMPORTANT_LEVELS_ENGINE.md section I: Relevance = 0.45*strength +
    0.25*recency + 0.20*cross_lens_confluence + 0.10*liquidity_confidence,
    sorted descending."""

    def _relevance(e: dict) -> float:
        return (
            0.45 * e.get("strength", 0)
            + 0.25 * e.get("recency", 0)
            + 0.20 * e.get("cross_lens_confluence", 0)
            + 0.10 * e.get("liquidity_confidence", 0)
        )

    scored = [{**e, "relevance": _relevance(e)} for e in entries]
    return sorted(scored, key=lambda e: e["relevance"], reverse=True)
