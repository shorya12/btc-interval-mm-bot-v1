"""Feature engineering for OHLCV data and strike encoding."""

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_ohlcv_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from OHLCV candle DataFrame.

    Args:
        candles: DataFrame with columns: open, high, low, close, volume
                 indexed by timestamp (ascending)

    Returns:
        DataFrame of features (same index, NaN rows dropped)
    """
    df = candles.copy()
    close = df["close"]
    volume = df["volume"]

    # Log returns at multiple horizons
    for period in [1, 5, 15, 30, 60]:
        col = f"log_return_{period}"
        df[col] = np.log(close / close.shift(period))

    # Realized volatility: rolling std of 1-period log returns, annualized
    log_ret_1 = df["log_return_1"]
    # Annualization: assume 1h candles by default; will scale with candle count
    # Use sqrt(periods_per_year) where periods = 8760 for 1h, 525600 for 1m
    ann_factor = math.sqrt(8760)  # assume 1h; user can override
    for window in [5, 15, 60]:
        df[f"realized_vol_{window}"] = log_ret_1.rolling(window).std() * ann_factor

    # Volume ratio: current / rolling 20-period mean
    vol_mean_20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (vol_mean_20 + 1e-10)

    # RSI (14-period)
    df["rsi_14"] = _rsi(close, 14)

    # MACD (12/26/9)
    macd_line, macd_signal, macd_hist = _macd(close, 12, 26, 9)
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # High-low range as fraction of mid
    mid = (df["high"] + df["low"]) / 2
    df["hl_range_frac"] = (df["high"] - df["low"]) / (mid + 1e-10)

    # Vol regime ratio: 7-period vol / 30-period vol (short over long)
    vol_7d = log_ret_1.rolling(7).std() * ann_factor
    vol_30d = log_ret_1.rolling(30).std() * ann_factor
    df["vol_regime_ratio"] = vol_7d / (vol_30d + 1e-10)

    # Drop raw OHLCV columns and NaN rows
    feature_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    result = df[feature_cols].dropna()
    return result


def compute_strike_features(
    spot: float,
    strike: float,
    sigma_realized: float,
    time_remaining_frac: float,
) -> dict[str, float]:
    """
    Compute strike-relative features for a single prediction.

    Args:
        spot: Current BTC spot price
        strike: Market strike price (the threshold in P(close > X))
        sigma_realized: Annualized realized volatility (from LagModel or rolling)
        time_remaining_frac: Fraction of interval remaining [0, 1]

    Returns:
        Dict with keys: log_moneyness, vol_normalized_dist, time_remaining
    """
    log_moneyness = math.log(spot / strike) if spot > 0 and strike > 0 else 0.0
    denom = sigma_realized * math.sqrt(time_remaining_frac + 1e-8)
    vol_normalized_dist = log_moneyness / (denom + 1e-10)

    return {
        "log_moneyness": log_moneyness,
        "vol_normalized_dist": vol_normalized_dist,
        "time_remaining": time_remaining_frac,
    }


def build_feature_vector(
    ohlcv_features: dict[str, float],
    strike_features: dict[str, float],
) -> dict[str, float]:
    """Merge OHLCV and strike features into a single flat dict."""
    return {**ohlcv_features, **strike_features}


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature names."""
    ohlcv_names = []
    for period in [1, 5, 15, 30, 60]:
        ohlcv_names.append(f"log_return_{period}")
    for window in [5, 15, 60]:
        ohlcv_names.append(f"realized_vol_{window}")
    ohlcv_names += [
        "volume_ratio",
        "rsi_14",
        "macd_signal",
        "macd_hist",
        "hl_range_frac",
        "vol_regime_ratio",
    ]
    strike_names = ["log_moneyness", "vol_normalized_dist", "time_remaining"]
    return ohlcv_names + strike_names


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist
