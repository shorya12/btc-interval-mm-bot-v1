"""Tests for lag signal module."""

import pytest
import math
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from collections import deque

from src.lag_signal.price_feed import PriceFeed, PriceSnapshot, PriceHistory
from src.lag_signal.model import LagModel, AssetMetrics
from src.lag_signal.skew import SkewComputer, SkewSignal, AssetConfig


class TestPriceSnapshot:
    """Tests for PriceSnapshot."""

    def test_basic_creation(self):
        """Test basic snapshot creation."""
        snapshot = PriceSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
        )
        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.price == 50000.0
        assert snapshot.source == "ccxt"

    def test_with_optional_fields(self):
        """Test snapshot with all fields."""
        snapshot = PriceSnapshot(
            symbol="ETH/USDT",
            price=3000.0,
            timestamp=datetime.utcnow(),
            volume_24h=1000000.0,
            bid=2999.0,
            ask=3001.0,
        )
        assert snapshot.volume_24h == 1000000.0
        assert snapshot.bid == 2999.0
        assert snapshot.ask == 3001.0


class TestPriceHistory:
    """Tests for PriceHistory."""

    @pytest.fixture
    def history(self):
        """Create price history with sample data."""
        prices = deque([
            PriceSnapshot("BTC/USDT", 50000.0, datetime.utcnow()),
            PriceSnapshot("BTC/USDT", 50100.0, datetime.utcnow()),
            PriceSnapshot("BTC/USDT", 50200.0, datetime.utcnow()),
            PriceSnapshot("BTC/USDT", 50150.0, datetime.utcnow()),
            PriceSnapshot("BTC/USDT", 50300.0, datetime.utcnow()),
        ], maxlen=100)
        return PriceHistory(symbol="BTC/USDT", prices=prices)

    def test_latest(self, history):
        """Test getting latest price."""
        assert history.latest.price == 50300.0

    def test_latest_price(self, history):
        """Test getting latest price value."""
        assert history.latest_price == 50300.0

    def test_get_prices(self, history):
        """Test getting price list."""
        prices = history.get_prices()
        assert len(prices) == 5
        assert prices[-1] == 50300.0

    def test_get_prices_limited(self, history):
        """Test getting limited price list."""
        prices = history.get_prices(3)
        assert len(prices) == 3

    def test_get_returns(self, history):
        """Test computing returns."""
        returns = history.get_returns()
        assert len(returns) == 4  # n-1 returns for n prices
        # First return: log(50100/50000)
        assert returns[0] == pytest.approx(math.log(50100 / 50000), rel=1e-10)

    def test_empty_history(self):
        """Test empty history."""
        history = PriceHistory(symbol="BTC/USDT", prices=deque())
        assert history.latest is None
        assert history.latest_price is None
        assert history.get_prices() == []
        assert history.get_returns() == []


class TestLagModel:
    """Tests for LagModel."""

    @pytest.fixture
    def mock_price_feed(self):
        """Create mock price feed."""
        feed = MagicMock(spec=PriceFeed)
        feed.symbols = ["BTC/USDT", "ETH/USDT"]
        return feed

    @pytest.fixture
    def lag_model(self, mock_price_feed):
        """Create lag model with mock feed."""
        return LagModel(
            price_feed=mock_price_feed,
            vol_window=60,
            momentum_window=5,
        )

    def test_get_spot_price(self, lag_model, mock_price_feed):
        """Test getting spot price."""
        mock_price_feed.get_price.return_value = 50000.0
        price = lag_model.get_spot_price("BTC/USDT")
        assert price == 50000.0
        mock_price_feed.get_price.assert_called_with("BTC/USDT")

    def test_compute_realized_vol(self, lag_model, mock_price_feed):
        """Test realized vol computation."""
        # Create returns with known variance
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]  # Mean ~0
        mock_price_feed.get_returns.return_value = returns

        vol = lag_model.compute_realized_vol("BTC/USDT")
        assert vol > 0

    def test_compute_realized_vol_insufficient_data(self, lag_model, mock_price_feed):
        """Test vol with insufficient data."""
        mock_price_feed.get_returns.return_value = [0.01]
        vol = lag_model.compute_realized_vol("BTC/USDT")
        assert vol == 0.0

    def test_compute_lognormal_q(self, lag_model, mock_price_feed):
        """Test lognormal quantile computation."""
        # Prices where current is exactly at mean
        prices = [100, 101, 99, 100, 100]  # Current = 100, mean ~= 100
        mock_price_feed.get_prices.return_value = prices

        q = lag_model.compute_lognormal_q("BTC/USDT")
        # Should be close to 0 (at mean)
        assert abs(q) < 1.0

    def test_compute_lognormal_q_extreme(self, lag_model, mock_price_feed):
        """Test lognormal q with extreme current price."""
        # Current price much higher than historical
        prices = [100, 100, 100, 100, 150]
        mock_price_feed.get_prices.return_value = prices

        q = lag_model.compute_lognormal_q("BTC/USDT")
        # Should be positive (above mean)
        assert q > 0

    def test_compute_momentum(self, lag_model, mock_price_feed):
        """Test momentum computation."""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]  # Consistent positive
        mock_price_feed.get_returns.return_value = returns

        momentum = lag_model.compute_momentum("BTC/USDT")
        assert momentum == pytest.approx(0.01, rel=1e-10)

    def test_compute_metrics(self, lag_model, mock_price_feed):
        """Test full metrics computation."""
        mock_price_feed.get_price.return_value = 50000.0
        mock_price_feed.get_returns.return_value = [0.01, -0.01, 0.02, -0.02, 0.01]
        mock_price_feed.get_prices.return_value = [49000, 49500, 50000, 50500, 50000]

        metrics = lag_model.compute_metrics("BTC/USDT")

        assert metrics is not None
        assert metrics.symbol == "BTC/USDT"
        assert metrics.spot_price == 50000.0
        assert metrics.realized_vol > 0


class TestAssetMetrics:
    """Tests for AssetMetrics."""

    def test_vol_adjusted_momentum(self):
        """Test vol-adjusted momentum."""
        metrics = AssetMetrics(
            symbol="BTC/USDT",
            spot_price=50000.0,
            realized_vol=0.5,
            lognormal_q=0.5,
            log_return=0.01,
            momentum=0.05,
            timestamp=datetime.utcnow(),
        )

        # 0.05 / 0.5 = 0.1
        assert metrics.vol_adjusted_momentum == pytest.approx(0.1, rel=1e-10)

    def test_vol_adjusted_momentum_zero_vol(self):
        """Test vol-adjusted momentum with zero vol."""
        metrics = AssetMetrics(
            symbol="BTC/USDT",
            spot_price=50000.0,
            realized_vol=0.0,
            lognormal_q=0.5,
            log_return=0.01,
            momentum=0.05,
            timestamp=datetime.utcnow(),
        )

        assert metrics.vol_adjusted_momentum == 0.0


class TestSkewComputer:
    """Tests for SkewComputer."""

    @pytest.fixture
    def mock_lag_model(self):
        """Create mock lag model."""
        model = MagicMock(spec=LagModel)
        model.price_feed = MagicMock(spec=PriceFeed)
        return model

    @pytest.fixture
    def skew_computer(self, mock_lag_model):
        """Create skew computer."""
        return SkewComputer(
            lag_model=mock_lag_model,
            asset_configs=[
                AssetConfig("BTC/USDT", weight=0.5, signal_type="momentum"),
                AssetConfig("ETH/USDT", weight=0.5, signal_type="momentum"),
            ],
            skew_multiplier=1.0,
            max_skew=0.5,
        )

    def test_compute_weighted_skew_basic(self, skew_computer, mock_lag_model):
        """Test basic skew computation."""
        # Both assets have positive momentum
        mock_lag_model.compute_metrics.side_effect = [
            AssetMetrics(
                symbol="BTC/USDT",
                spot_price=50000,
                realized_vol=0.5,
                lognormal_q=0,
                log_return=0.01,
                momentum=0.02,
                timestamp=datetime.utcnow(),
            ),
            AssetMetrics(
                symbol="ETH/USDT",
                spot_price=3000,
                realized_vol=0.6,
                lognormal_q=0,
                log_return=0.01,
                momentum=0.02,
                timestamp=datetime.utcnow(),
            ),
        ]

        signal = skew_computer.compute_weighted_skew()

        assert isinstance(signal, SkewSignal)
        # 0.5 * 0.02 + 0.5 * 0.02 = 0.02
        assert signal.total_skew == pytest.approx(0.02, rel=0.01)
        assert signal.is_bullish

    def test_compute_weighted_skew_mixed(self, skew_computer, mock_lag_model):
        """Test skew with mixed signals."""
        mock_lag_model.compute_metrics.side_effect = [
            AssetMetrics(
                symbol="BTC/USDT",
                spot_price=50000,
                realized_vol=0.5,
                lognormal_q=0,
                log_return=0.01,
                momentum=0.04,  # Bullish
                timestamp=datetime.utcnow(),
            ),
            AssetMetrics(
                symbol="ETH/USDT",
                spot_price=3000,
                realized_vol=0.6,
                lognormal_q=0,
                log_return=-0.01,
                momentum=-0.04,  # Bearish
                timestamp=datetime.utcnow(),
            ),
        ]

        signal = skew_computer.compute_weighted_skew()

        # Should cancel out: 0.5 * 0.04 + 0.5 * (-0.04) = 0
        assert abs(signal.total_skew) < 0.01

    def test_compute_weighted_skew_clamped(self, skew_computer, mock_lag_model):
        """Test that skew is clamped to max."""
        # Very large momentum
        mock_lag_model.compute_metrics.side_effect = [
            AssetMetrics(
                symbol="BTC/USDT",
                spot_price=50000,
                realized_vol=0.5,
                lognormal_q=0,
                log_return=0.1,
                momentum=2.0,  # Huge momentum
                timestamp=datetime.utcnow(),
            ),
            AssetMetrics(
                symbol="ETH/USDT",
                spot_price=3000,
                realized_vol=0.6,
                lognormal_q=0,
                log_return=0.1,
                momentum=2.0,
                timestamp=datetime.utcnow(),
            ),
        ]

        signal = skew_computer.compute_weighted_skew()

        # Should be clamped to max_skew (0.5)
        assert signal.total_skew == pytest.approx(0.5, rel=0.01)

    def test_compute_weighted_skew_missing_data(self, skew_computer, mock_lag_model):
        """Test skew when some assets have no data."""
        mock_lag_model.compute_metrics.side_effect = [
            AssetMetrics(
                symbol="BTC/USDT",
                spot_price=50000,
                realized_vol=0.5,
                lognormal_q=0,
                log_return=0.01,
                momentum=0.02,
                timestamp=datetime.utcnow(),
            ),
            None,  # ETH has no data
        ]

        signal = skew_computer.compute_weighted_skew()

        # Should still compute with available data
        assert len(signal.components) == 2
        assert signal.components[1].metrics is None

    def test_skew_signal_properties(self):
        """Test SkewSignal properties."""
        signal = SkewSignal(
            total_skew=0.1,
            components=[],
        )

        assert signal.is_bullish
        assert not signal.is_bearish
        assert signal.strength == 0.1

        bearish_signal = SkewSignal(total_skew=-0.1, components=[])
        assert bearish_signal.is_bearish
        assert not bearish_signal.is_bullish
