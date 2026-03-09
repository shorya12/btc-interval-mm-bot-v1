"""Tests for feature engineering."""

import math
import pandas as pd
import numpy as np
import pytest

from src.probability_model.features import (
    compute_ohlcv_features,
    compute_strike_features,
    get_feature_names,
    build_feature_vector,
)


def make_candles(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV candles."""
    np.random.seed(42)
    close = 50000 * np.exp(np.cumsum(np.random.randn(n) * 0.001))
    high = close * (1 + np.abs(np.random.randn(n) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n) * 0.002))
    open_ = close * (1 + np.random.randn(n) * 0.001)
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestComputeOhlcvFeatures:
    def test_returns_dataframe(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_after_dropna(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        assert not result.isnull().any().any()

    def test_contains_expected_columns(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        expected = ["log_return_1", "log_return_5", "realized_vol_5", "rsi_14", "macd_hist", "vol_regime_ratio"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_in_valid_range(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_volume_ratio_positive(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        assert (result["volume_ratio"] >= 0).all()

    def test_fewer_rows_than_input_due_to_rolling(self):
        df = make_candles(100)
        result = compute_ohlcv_features(df)
        assert len(result) < len(df)


class TestComputeStrikeFeatures:
    def test_atm(self):
        feats = compute_strike_features(spot=50000, strike=50000, sigma_realized=0.8, time_remaining_frac=0.5)
        assert feats["log_moneyness"] == pytest.approx(0.0)
        assert feats["vol_normalized_dist"] == pytest.approx(0.0, abs=1e-6)
        assert feats["time_remaining"] == pytest.approx(0.5)

    def test_itm(self):
        feats = compute_strike_features(spot=51000, strike=50000, sigma_realized=0.8, time_remaining_frac=0.5)
        assert feats["log_moneyness"] > 0

    def test_otm(self):
        feats = compute_strike_features(spot=49000, strike=50000, sigma_realized=0.8, time_remaining_frac=0.5)
        assert feats["log_moneyness"] < 0

    def test_zero_vol_safe(self):
        # Should not raise even with sigma=0
        feats = compute_strike_features(spot=50000, strike=50000, sigma_realized=0.0, time_remaining_frac=0.5)
        assert math.isfinite(feats["vol_normalized_dist"])


class TestGetFeatureNames:
    def test_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 10

    def test_contains_strike_features(self):
        names = get_feature_names()
        assert "log_moneyness" in names
        assert "vol_normalized_dist" in names
        assert "time_remaining" in names


class TestBuildFeatureVector:
    def test_merges_dicts(self):
        ohlcv = {"log_return_1": 0.01, "rsi_14": 55.0}
        strike = {"log_moneyness": 0.02, "time_remaining": 0.5}
        result = build_feature_vector(ohlcv, strike)
        assert "log_return_1" in result
        assert "log_moneyness" in result
