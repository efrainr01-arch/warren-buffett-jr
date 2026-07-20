"""Technical & Momentum specialist (20 pts) — Cerebro/04_technical_momentum/.

Orchestrates the Task-11 indicator library and Task-12 levels engine
against the packet's daily OHLCV (reversed to oldest-first, the engines'
convention) into the five weighted dimensions (SCORING.md) using the
exact primary-trend anchors and technical-profile bands from
DECISION_RULES.md.

The packet built in Task 10 does not populate `market_data.benchmark` or
`market_data.sector` (no benchmark/sector series wired into the packet
builder yet), so Relative strength and the breadth half of Sector
breadth & volatility are correctly NOT_SCORABLE rather than fabricated.
Earnings-gap behavior needs earnings event dates, which aren't in the
Packet schema either (Task 10 only used them transiently for staleness);
`run()` takes an optional `earnings_dates` side-channel parameter,
mirroring how Financial/Business/Market take `wacc`/`overlay`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import CATEGORY_WEIGHTS, Category, Dimension, anchor_score
from wbj.engines import indicators as ind
from wbj.engines import levels_engine as lv
from wbj.specialists.common import CategorySummary, MetricRow, SpecialistOutput

AGENT_ID = "technical_momentum"
MAX_POINTS = float(CATEGORY_WEIGHTS["technical"])  # 20
MIN_SESSIONS_FOR_TREND = 200
MIN_EARNINGS_EVENTS = 4


def _daily_frame(packet) -> pd.DataFrame:
    """Packet OHLCV is newest-first; engines expect oldest-first."""
    rows = list(reversed(packet.market_data.daily))
    return pd.DataFrame(
        {
            "date": [r.date for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.adj_close for r in rows],
            "volume": [r.volume for r in rows],
        }
    )


def _ols_slope_atr(series: pd.Series, n: int, atr_latest: float | None) -> float | None:
    """TECH-SLOPE-004: OLS slope over the last `n` sessions, scaled by n
    and expressed in ATR units."""
    tail = series.dropna().iloc[-n:]
    if len(tail) < 2 or not atr_latest:
        return None
    x = np.arange(len(tail))
    slope, _ = np.polyfit(x, tail.to_numpy(), 1)
    return float(slope * n / atr_latest)


def classify_primary_trend(
    close: float, sma50: float | None, sma200: float | None, atr14: float | None,
    sma200_slope_atr: float | None, sma50_slope_atr: float | None, adx14: float | None, range_pos_52w: float | None,
) -> float | None:
    """DECISION_RULES.md's exact primary-trend anchors, returning the
    midpoint of each band."""
    if sma200 is None or atr14 is None:
        return None

    if sma50 is not None and close < sma50 < sma200 and sma200_slope_atr is not None and sma200_slope_atr < -1.0:
        return 1.0  # 0-2
    if close < sma200:
        return 3.0
    if abs(close - sma200) <= atr14 and sma200_slope_atr is not None and -0.25 <= sma200_slope_atr <= 0.25:
        return 4.5  # 4-5
    if sma50 is not None and close > sma50 > sma200 and sma50_slope_atr is not None and sma200_slope_atr is not None and sma50_slope_atr > 0 and sma200_slope_atr > 0:
        if adx14 is not None and adx14 >= 25 and range_pos_52w is not None and range_pos_52w >= 0.80:
            return 9.5  # 9-10
        return 8.0
    return 6.0  # close above SMA200, SMA50 mixed/flat


def volume_demand_signal(ud_ratio: float | None, cmf_value: float | None, obv_slope: float | None) -> float | None:
    if ud_ratio is None and cmf_value is None:
        return None
    score = 5.0
    if ud_ratio is not None:
        score = anchor_score(ud_ratio, [(0.5, 0), (1.0, 4), (1.2, 6.5), (2.0, 10)])
    if cmf_value is not None:
        if cmf_value > 0.10:
            score = max(score, 7.0)
        elif cmf_value < -0.10:
            score = min(score, 3.0)
    if obv_slope is not None and obv_slope > 0:
        score = min(10.0, score + 0.5)
    return score


class TechnicalOutput(SpecialistOutput):
    market_state: dict = {}
    indicators: dict = {}
    important_levels: dict = {}
    breakouts_and_failures: list = []


def run(packet, earnings_dates: list[str] | None = None) -> TechnicalOutput:
    df = _daily_frame(packet)
    n = len(df)

    sma20 = ind.sma(df["close"], 20)
    sma50 = ind.sma(df["close"], 50)
    sma100 = ind.sma(df["close"], 100)
    sma200 = ind.sma(df["close"], 200)
    atr = ind.atr14(df)
    rsi = ind.rsi14(df["close"])
    adx = ind.adx14(df)
    macd = ind.macd(df["close"])
    range_pos = ind.range_position_52w(df)

    close_latest = float(df["close"].iloc[-1])
    atr_latest = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None
    sma50_latest = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    sma200_latest = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
    adx_latest = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None
    range_pos_latest = float(range_pos.iloc[-1]) if pd.notna(range_pos.iloc[-1]) else None

    sma200_slope_atr = _ols_slope_atr(sma200, 50, atr_latest) if sma200_latest is not None else None
    sma50_slope_atr = _ols_slope_atr(sma50, 50, atr_latest) if sma50_latest is not None else None

    # --- Primary trend --------------------------------------------------
    trend_score = None
    if n >= MIN_SESSIONS_FOR_TREND and sma200_latest is not None:
        trend_score = classify_primary_trend(
            close_latest, sma50_latest, sma200_latest, atr_latest, sma200_slope_atr, sma50_slope_atr, adx_latest, range_pos_latest
        )
    elif sma200_latest is None:
        trend_score = None  # capped 6 without valid SMA200 handled below if partial data exists
    if trend_score is not None and sma200_latest is None:
        trend_score = min(trend_score, 6.0)

    # --- Relative strength: no benchmark data in this packet -----------
    rs_score = None
    if packet.market_data.benchmark:
        bench = pd.DataFrame({"close": [r.adj_close for r in reversed(packet.market_data.benchmark)]})
        rs21 = ind.relative_strength(df["close"], bench["close"], 21).iloc[-1]
        rs_score = anchor_score(rs21, [(-0.10, 0), (0.0, 4), (0.05, 7), (0.15, 10)]) if pd.notna(rs21) else None

    # --- Volume and institutional demand ---------------------------------
    ud_ratio = ind.up_down_volume_ratio(df, n=50).iloc[-1]
    cmf_series = ind.cmf(df, n=20)
    cmf_latest = float(cmf_series.iloc[-1]) if pd.notna(cmf_series.iloc[-1]) else None
    obv_series = ind.obv(df)
    obv_slope = _ols_slope_atr(obv_series, 20, 1.0)  # sign only; unit-normalization irrelevant here
    volume_score = volume_demand_signal(
        float(ud_ratio) if pd.notna(ud_ratio) else None, cmf_latest, obv_slope
    )
    if df["volume"].iloc[-50:].sum() == 0 and volume_score is not None:
        volume_score = min(volume_score, 5.0)

    # --- Earnings-gap behavior --------------------------------------
    gap_score = None
    gaps: list[dict] = []
    if earnings_dates and len(earnings_dates) >= MIN_EARNINGS_EVENTS:
        gaps = lv.earnings_gaps(df, earnings_dates, atr)
        if len(gaps) >= MIN_EARNINGS_EVENTS:
            hold_ratios = [g["day5_hold"] for g in gaps if g["day5_hold"] is not None]
            if hold_ratios:
                hold_rate = sum(1 for h in hold_ratios if h >= 0.70) / len(hold_ratios)
                gap_score = anchor_score(hold_rate, [(0.0, 2), (0.5, 5), (0.7, 7.5), (1.0, 10)])

    # --- Breakout & base quality (levels engine) ------------------------
    pivots = lv.find_pivots(df, k=3)
    zones = lv.cluster_zones(pivots, atr, current_pos=n - 1, timeframe="daily")
    scored_zones = []
    for z in zones:
        z2 = lv.count_touches(z, pivots, atr, df, min_gap_sessions=5)
        if z2.touches:
            s = lv.strength(z2, confluence_count=0)
            status = lv.classify(z2)
            scored_zones.append(z2.model_copy(update={"strength_0_100": s, "status": status}))

    breakout_score = None
    breakouts: list[dict] = []
    if scored_zones:
        best = max(scored_zones, key=lambda z: z.strength_0_100)
        breakout_score = anchor_score(best.strength_0_100, [(0, 2), (30, 5), (60, 7.5), (85, 10)])
        for z in scored_zones:
            if z.type == "resistance" and lv.breakout_confirmed(df, z, atr):
                breakouts.append({"zone_id": z.zone_id, "state": "confirmed_breakout"})

    # --- Sector breadth & volatility quality --------------------------
    realized_vol_series = ind.realized_vol(df["close"], 63)
    vol_latest = float(realized_vol_series.iloc[-1]) if pd.notna(realized_vol_series.iloc[-1]) else None
    liquidity = ind.median_dollar_volume(df, 63).iloc[-1]
    breadth_vol_score = None
    if vol_latest is not None:
        breadth_vol_score = anchor_score(vol_latest, [(1.0, 0), (0.60, 3), (0.35, 7), (0.15, 10)])
        if not packet.market_data.sector:
            breadth_vol_score = min(breadth_vol_score, 7.0)  # breadth half unavailable -> reserve top band

    dims_spec = [
        ("primary_price_trend", 4.0, trend_score),
        ("relative_strength", 4.0, rs_score),
        ("volume_and_institutional_demand", 3.0, volume_score),
        ("earnings_gap_behavior", 3.0, gap_score),
        ("breakout_and_base_quality", 3.0, breakout_score),
        ("sector_breadth_and_volatility_quality", 3.0, breadth_vol_score),
    ]
    # SCORING.md lists 6 dimensions summing 4+4+3+3+3+3=20, matching MAX_POINTS.
    dims = []
    for name, max_pts, score in dims_spec:
        if score is None:
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[]))
        else:
            v = Value.of(score, unit="score", evidence_class=EvidenceClass.C)
            dims.append(Dimension(name=name, max_points=max_pts, metric_scores=[(1.0, v)]))

    category = Category(name="technical", max_points=MAX_POINTS, dimensions=dims)
    awarded = category.points()
    score_10 = category.score10()
    coverage = category.coverage()

    metrics = [
        MetricRow(metric_id="sma200", value=sma200_latest, formula="TECH-SMA-002", unit="usd",
                  score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="atr14", value=atr_latest, formula="TECH-ATR-006", unit="usd",
                  score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="rsi14", value=float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None, formula="TECH-RSI-007",
                  unit="index", score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="adx14", value=adx_latest, formula="TECH-DMI-009", unit="index",
                  score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="range_position_52w", value=range_pos_latest, formula="TECH-52W-036", unit="ratio",
                  score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="realized_vol_63d", value=vol_latest, formula="TECH-VOL-018", unit="ratio",
                  score=None, evidence_class=str(EvidenceClass.C), source="packet.market_data.daily"),
        MetricRow(metric_id="median_dollar_volume_63d", value=float(liquidity) if pd.notna(liquidity) else None,
                  formula="TECH-LIQ-040", unit="usd", score=None, evidence_class=str(EvidenceClass.C),
                  source="packet.market_data.daily"),
    ]

    status = "COMPLETE" if coverage >= 0.70 else "PARTIAL"

    return TechnicalOutput(
        agent_id=AGENT_ID,
        status=status,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange, "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategorySummary(max_points=MAX_POINTS, awarded_points=awarded, score_10=score_10, confidence=coverage * 100),
        coverage=coverage,
        dimensions=[{"name": d.name, "max_points": d.max_points, "score_10": _dim_score10_or_none(d)} for d in dims],
        metrics=metrics,
        mandatory_flags=[],
        assumptions=[
            "Relative strength and sector breadth require benchmark/sector series not populated by "
            "the Task-10 packet builder yet -- correctly NOT_SCORABLE / partial rather than imputed.",
        ],
        source_lineage=["packet.market_data.daily"],
        market_state={
            "trend": trend_score,
            "relative_strength": rs_score,
            "demand": volume_score,
            "volatility": vol_latest,
        },
        indicators={
            "sma20": float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
            "sma50": sma50_latest,
            "sma100": float(sma100.iloc[-1]) if pd.notna(sma100.iloc[-1]) else None,
            "sma200": sma200_latest,
            "atr14": atr_latest,
            "rsi14": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
            "adx14": adx_latest,
            "macd": {
                "macd": float(macd["macd"].iloc[-1]) if pd.notna(macd["macd"].iloc[-1]) else None,
                "signal": float(macd["signal"].iloc[-1]) if pd.notna(macd["signal"].iloc[-1]) else None,
                "hist": float(macd["hist"].iloc[-1]) if pd.notna(macd["hist"].iloc[-1]) else None,
            },
        },
        important_levels={
            "support": [z.model_dump() for z in scored_zones if z.type == "support"][:3],
            "resistance": [z.model_dump() for z in scored_zones if z.type == "resistance"][:3],
            "moving_averages": [],
            "anchored_vwaps": [],
            "earnings_gaps": gaps,
            "volume_profile": [],
        },
        breakouts_and_failures=breakouts,
    )


def _dim_score10_or_none(d: Dimension) -> float | None:
    v = d.score10_value()
    return None if v.is_null else v.value
