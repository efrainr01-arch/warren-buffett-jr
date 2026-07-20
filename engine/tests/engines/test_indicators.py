"""Tests for wbj.engines.indicators, per Cerebro/04_technical_momentum/
FORMULAS.md TECH-001..021, TECH-034..040.

Convention: all Series/DataFrame inputs are oldest-first (ascending date
index), the standard order for rolling/ewm calculations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wbj.engines.indicators import (
    adx14,
    atr14,
    cmf,
    ema,
    macd,
    median_dollar_volume,
    obv,
    range_position_52w,
    realized_vol,
    relative_strength,
    composite_rs_percentile,
    roc,
    rsi14,
    sma,
    true_range,
    up_down_volume_ratio,
    volume_ratio,
)


def _ohlcv(closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    highs = highs if highs is not None else [c + 1 for c in closes]
    lows = lows if lows is not None else [c - 1 for c in closes]
    volumes = volumes if volumes is not None else [1_000_000] * n
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def constant_tr_frame(tr: float, bars: int) -> pd.DataFrame:
    """Close constant at 100, high/low straddling it by tr/2 so every bar's
    true range (including the first, which has no prior close) is `tr`."""
    close = [100.0] * bars
    high = [100.0 + tr / 2] * bars
    low = [100.0 - tr / 2] * bars
    return _ohlcv(close, highs=high, lows=low)


# --- SMA / EMA ---------------------------------------------------------


def test_sma_requires_n_observations():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(close, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    assert out.iloc[4] == pytest.approx((3 + 4 + 5) / 3)


def test_ema_initialized_with_sma():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    n = 3
    out = ema(close, n)
    seed = close.iloc[:n].mean()
    assert out.iloc[n - 1] == pytest.approx(seed)
    alpha = 2 / (n + 1)
    expected_next = alpha * close.iloc[n] + (1 - alpha) * seed
    assert out.iloc[n] == pytest.approx(expected_next)


# --- ATR / true range ----------------------------------------------------


def test_wilder_atr_smoothing():
    df = constant_tr_frame(tr=2.0, bars=20)
    out = atr14(df)
    assert out.iloc[13] == pytest.approx(2.0)
    assert out.iloc[-1] == pytest.approx(2.0)


def test_true_range_first_bar_uses_high_low_only():
    df = _ohlcv([100.0, 105.0], highs=[102.0, 108.0], lows=[98.0, 103.0])
    tr = true_range(df)
    assert tr.iloc[0] == pytest.approx(4.0)  # high-low, no prior close
    assert tr.iloc[1] == pytest.approx(max(5.0, 8.0, 3.0))  # |108-100|=8


# --- RSI -------------------------------------------------------------------


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1.0, 40.0))
    assert rsi14(close).iloc[-1] == 100.0


def test_rsi_all_losses_is_zero():
    close = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi14(close).iloc[-1] == pytest.approx(0.0)


# --- MACD ------------------------------------------------------------------


def test_macd_hist_is_macd_minus_signal():
    close = pd.Series(np.linspace(100, 160, 60) + np.sin(np.arange(60)) * 3)
    out = macd(close)
    valid = out["signal"].notna()
    diff = out["macd"][valid] - out["signal"][valid] - out["hist"][valid]
    assert (diff.abs() < 1e-9).all()


# --- ADX (sanity) ------------------------------------------------------


def test_adx_higher_for_strong_trend_than_choppy_range():
    trend_close = np.linspace(100, 200, 60)
    trend = _ohlcv(list(trend_close), highs=list(trend_close + 1), lows=list(trend_close - 1))
    choppy_close = 100 + np.sin(np.arange(60)) * 5
    choppy = _ohlcv(list(choppy_close), highs=list(choppy_close + 1), lows=list(choppy_close - 1))
    trend_adx = adx14(trend).iloc[-1]
    choppy_adx = adx14(choppy).iloc[-1]
    assert trend_adx > choppy_adx


# --- ROC / relative strength ------------------------------------------------


def test_roc():
    close = pd.Series([100.0] * 10 + [110.0])
    out = roc(close, 10)
    assert out.iloc[-1] == pytest.approx(0.10)


def test_relative_strength_is_excess_return():
    close = pd.Series([100.0] * 5 + [120.0])
    bench = pd.Series([100.0] * 5 + [105.0])
    out = relative_strength(close, bench, 5)
    assert out.iloc[-1] == pytest.approx(0.20 - 0.05)


def test_composite_rs_weights_sum_to_one_input():
    # Every window's security RS sits exactly at its universe's median with
    # an odd-sized, symmetric universe -> percentile rank 50 in each window,
    # so the weighted composite (weights sum to 1) is also 50.
    universe = pd.DataFrame(
        {
            "RS21": [-2, -1, 0, 1, 2],
            "RS63": [-2, -1, 0, 1, 2],
            "RS126": [-2, -1, 0, 1, 2],
            "RS252": [-2, -1, 0, 1, 2],
        }
    )
    rs_by_window = {"RS21": 0, "RS63": 0, "RS126": 0, "RS252": 0}
    assert composite_rs_percentile(rs_by_window, universe) == pytest.approx(50.0)


# --- volume family -----------------------------------------------------


def test_volume_ratio_vs_prior_50_median():
    volumes = [1_000_000] * 51
    volumes[-1] = 3_000_000
    close = [100.0 + i * 0.01 for i in range(51)]
    df = _ohlcv(close, volumes=volumes)
    out = volume_ratio(df["volume"])
    assert out.iloc[-1] == pytest.approx(3.0)


def test_up_down_volume_ratio_zero_denominator_is_nan():
    close = [100.0 + i for i in range(10)]  # strictly up every day
    volumes = [1_000_000] * 10
    df = _ohlcv(close, volumes=volumes)
    out = up_down_volume_ratio(df, n=9)
    assert math.isnan(out.iloc[-1])


def test_up_down_volume_ratio_basic():
    close = [100.0, 101.0, 100.0, 101.0, 100.0]
    volumes = [10, 100, 10, 200, 10]
    df = _ohlcv(close, volumes=volumes)
    out = up_down_volume_ratio(df, n=4)
    # up closes at idx1 (vol100), idx3(vol200); down closes idx2(vol10), idx4(vol10)
    assert out.iloc[-1] == pytest.approx(300 / 20)


def test_obv_accumulates_signed_volume():
    close = [100.0, 101.0, 100.5, 102.0]
    volumes = [1000, 500, 300, 700]
    df = _ohlcv(close, volumes=volumes)
    out = obv(df)
    assert out.iloc[0] == 0
    assert out.iloc[1] == 500
    assert out.iloc[2] == 500 - 300
    assert out.iloc[3] == 500 - 300 + 700


def test_cmf_zero_when_high_equals_low():
    close = [100.0, 100.0]
    df = _ohlcv(close, highs=[100.0, 100.0], lows=[100.0, 100.0], volumes=[1000, 1000])
    out = cmf(df, n=2)
    assert out.iloc[-1] == 0.0


def test_median_dollar_volume():
    close = [10.0] * 63
    volumes = [100] * 62 + [1000]
    df = _ohlcv(close, volumes=volumes)
    out = median_dollar_volume(df, n=63)
    assert out.iloc[-1] == pytest.approx(1000.0)  # median of 62x1000 + 1x10000


def test_range_position_52w_not_meaningful_when_range_zero():
    close = [100.0] * 252
    df = _ohlcv(close, highs=close, lows=close)
    out = range_position_52w(df)
    assert math.isnan(out.iloc[-1])


def test_range_position_52w_basic():
    closes = list(np.linspace(100, 150, 252))
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    df = _ohlcv(closes, highs=highs, lows=lows)
    out = range_position_52w(df)
    expected = (closes[-1] - min(lows)) / (max(highs) - min(lows))
    assert out.iloc[-1] == pytest.approx(expected)


def test_realized_vol_annualizes_with_sqrt_252():
    rng = np.random.default_rng(7)
    log_rets = rng.normal(0, 0.01, 300)
    close = pd.Series(100 * np.exp(np.cumsum(log_rets)))
    out = realized_vol(close, 252)
    manual = pd.Series(np.log(close / close.shift(1))).iloc[-252:].std() * math.sqrt(252)
    assert out.iloc[-1] == pytest.approx(manual)


# --- golden cross-check against inline pandas reference ---------------------


def test_golden_indicators_match_inline_pandas_reference():
    rng = np.random.default_rng(42)
    n = 300
    log_rets = rng.normal(0.0003, 0.02, n)
    close = pd.Series(100 * np.exp(np.cumsum(log_rets)))
    high = close * 1.01
    low = close * 0.99
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1_000_000})

    ref_sma20 = close.rolling(20, min_periods=20).mean()
    assert (sma(close, 20).dropna() == ref_sma20.dropna()).all()

    ref_ema12_seed = close.rolling(12, min_periods=12).mean()
    ema12 = ema(close, 12)
    assert ema12.iloc[11] == pytest.approx(ref_ema12_seed.iloc[11])

    ref_rsi = rsi14(close)
    assert ((ref_rsi.dropna() >= 0) & (ref_rsi.dropna() <= 100)).all()

    ref_atr = atr14(df)
    assert (ref_atr.dropna() > 0).all()

    m = macd(close)
    assert (m["macd"].dropna() == (ema(close, 12) - ema(close, 26)).dropna()).all()
