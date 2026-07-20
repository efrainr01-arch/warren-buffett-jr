"""Technical indicator library — Cerebro/04_technical_momentum/FORMULAS.md
TECH-001..021, TECH-034..040.

Convention: Series/DataFrame inputs are oldest-first (ascending date
index) — the standard order for rolling/ewm calculations. Callers
building from the packet's newest-first `market_data.daily` must reverse
before calling into this module. DataFrame inputs use lowercase OHLCV
column names: `open/high/low/close/volume`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

_RS_WEIGHTS = {"RS21": 0.35, "RS63": 0.25, "RS126": 0.25, "RS252": 0.15}


def sma(close: pd.Series, n: int) -> pd.Series:
    """TECH-SMA-002: simple moving average, NaN until N valid observations."""
    return close.rolling(window=n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """TECH-EMA-003: exponential moving average, alpha=2/(N+1), initialized
    with SMA_N (not the EWM-native seed)."""
    alpha = 2 / (n + 1)
    seed = close.rolling(window=n, min_periods=n).mean()
    out = pd.Series(np.nan, index=close.index, dtype=float)
    start = seed.first_valid_index()
    if start is None:
        return out
    start_pos = close.index.get_loc(start)
    out.iloc[start_pos] = seed.iloc[start_pos]
    for i in range(start_pos + 1, len(close)):
        out.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * out.iloc[i - 1]
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """TECH-TR-005: max(H-L, |H-prevC|, |L-prevC|). The first bar has no
    prior close, so it uses H-L only."""
    prior_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prior_close).abs()
    c = (df["low"] - prior_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1, skipna=True)
    tr.iloc[0] = a.iloc[0]
    return tr


def _wilder_smooth(s: pd.Series, n: int) -> pd.Series:
    """Generic Wilder smoothing: seed with the mean of the first N valid
    (non-NaN) values, then out_t = ((N-1)*out_{t-1} + s_t) / N. Works
    whether `s` has leading NaNs (e.g. a DX series) or none (e.g. true
    range) by operating on the compacted valid-only series and reindexing
    back to `s`'s original index."""
    valid = s.dropna()
    if len(valid) < n:
        return pd.Series(np.nan, index=s.index, dtype=float)
    out = pd.Series(np.nan, index=valid.index, dtype=float)
    out.iloc[n - 1] = valid.iloc[:n].mean()
    for i in range(n, len(valid)):
        out.iloc[i] = ((n - 1) * out.iloc[i - 1] + valid.iloc[i]) / n
    return out.reindex(s.index)


def atr14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """TECH-ATR-006: Wilder ATR."""
    return _wilder_smooth(true_range(df), n)


def rsi14(close: pd.Series, n: int = 14) -> pd.Series:
    """TECH-RSI-007: Wilder RSI. Zero average loss -> RSI 100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) < n + 1:
        return out

    def _rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    avg_gain = gain.iloc[1 : n + 1].mean()
    avg_loss = loss.iloc[1 : n + 1].mean()
    out.iloc[n] = _rsi(avg_gain, avg_loss)
    for i in range(n + 1, len(close)):
        avg_gain = (avg_gain * (n - 1) + gain.iloc[i]) / n
        avg_loss = (avg_loss * (n - 1) + loss.iloc[i]) / n
        out.iloc[i] = _rsi(avg_gain, avg_loss)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.Series]:
    """TECH-MACD-008: MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line.dropna(), signal).reindex(close.index)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def adx14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """TECH-DMI-009: Wilder ADX14 (average of DX=100*|+DI--DI|/(+DI+-DI))."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    smoothed_tr = _wilder_smooth(true_range(df), n)
    smoothed_plus_dm = _wilder_smooth(plus_dm, n)
    smoothed_minus_dm = _wilder_smooth(minus_dm, n)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    di_sum = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0)
    return _wilder_smooth(dx, n)


def roc(close: pd.Series, n: int) -> pd.Series:
    """TECH-ROC-010: rate of change over N sessions."""
    return close / close.shift(n) - 1


def relative_strength(close: pd.Series, bench: pd.Series, n: int) -> pd.Series:
    """TECH-RS-011: stock N-session return minus benchmark N-session return
    (percentage-point excess return — Cerebro's authoritative definition;
    a plain ratio would not be comparable across differently-scaled
    return magnitudes the way a subtraction is)."""
    return roc(close, n) - roc(bench, n)


def composite_rs_percentile(rs_by_window: dict[str, float], universe: pd.DataFrame) -> float:
    """TECH-RSC-013: 0.35*PctRank(RS21) + 0.25*PctRank(RS63) +
    0.25*PctRank(RS126) + 0.15*PctRank(RS252).

    `rs_by_window` holds the security's own RS value per window key
    ("RS21", "RS63", "RS126", "RS252"); `universe` holds the peer
    universe's RS values in matching columns, used to compute each
    window's percentile rank (mean of the weak/strict percentile, so a
    value sitting exactly at the sample median ranks at exactly 50).
    """
    total = 0.0
    for window, weight in _RS_WEIGHTS.items():
        peers = universe[window].dropna().tolist()
        pct = stats.percentileofscore(peers, rs_by_window[window], kind="mean")
        total += weight * pct
    return total


def volume_ratio(volume: pd.Series, n: int = 50) -> pd.Series:
    """TECH-VR-014: volume vs the median of the prior N sessions (current
    session excluded)."""
    median_prior = volume.rolling(window=n, min_periods=n).median().shift(1)
    return volume / median_prior


def up_down_volume_ratio(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """TECH-UDV-015: sum(volume on up closes) / sum(volume on down closes)
    over N sessions. NaN (not meaningful) when the down-volume sum is 0."""
    delta = df["close"].diff()
    up_vol = df["volume"].where(delta > 0, 0.0)
    down_vol = df["volume"].where(delta < 0, 0.0)
    up_sum = up_vol.rolling(window=n, min_periods=n).sum()
    down_sum = down_vol.rolling(window=n, min_periods=n).sum()
    return (up_sum / down_sum).where(down_sum != 0)


def obv(df: pd.DataFrame) -> pd.Series:
    """TECH-OBV-016: on-balance volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """TECH-CMF-017: Chaikin money flow. Multiplier is 0 for a bar where
    high == low."""
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    hl_range = high - low
    mfm = ((2 * close - high - low) / hl_range).where(hl_range != 0, 0.0)
    mfv = mfm * volume
    return mfv.rolling(window=n, min_periods=n).sum() / volume.rolling(window=n, min_periods=n).sum()


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """TECH-VOL-018: annualized realized volatility (stdev of log returns
    over N sessions, times sqrt(252))."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=n, min_periods=n).std() * math.sqrt(252)


def range_position_52w(df: pd.DataFrame, n: int = 252) -> pd.Series:
    """TECH-52W-036: (close - Nd low) / (Nd high - Nd low). NaN (not
    meaningful) when the range is zero."""
    high_n = df["high"].rolling(window=n, min_periods=n).max()
    low_n = df["low"].rolling(window=n, min_periods=n).min()
    rng = high_n - low_n
    return ((df["close"] - low_n) / rng).where(rng != 0)


def median_dollar_volume(df: pd.DataFrame, n: int = 63) -> pd.Series:
    """TECH-LIQ-040: median(close * volume) over N sessions."""
    return (df["close"] * df["volume"]).rolling(window=n, min_periods=n).median()
