"""Lag signal model: spot price, realized vol, lognormal q."""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.logging import get_logger
from .price_feed import PriceFeed

logger = get_logger(__name__)


@dataclass
class AssetMetrics:
    """Computed metrics for a single asset."""

    symbol: str
    spot_price: float
    realized_vol: float  # Annualized volatility
    lognormal_q: float  # Log-normal quantile (z-score of current price)
    log_return: float  # Most recent log return
    momentum: float  # Short-term momentum (mean of recent returns)
    timestamp: datetime

    @property
    def vol_adjusted_momentum(self) -> float:
        """Momentum divided by realized vol (standardized)."""
        if self.realized_vol == 0:
            return 0.0
        return self.momentum / self.realized_vol


class LagModel:
    """
    Computes lag signal metrics for crypto assets.

    For each asset, computes:
    - Spot price: Current price
    - Realized volatility: Rolling std dev of log returns (annualized)
    - Lognormal q: Z-score of current price relative to recent distribution
    """

    def __init__(
        self,
        price_feed: PriceFeed,
        vol_window: int = 60,  # Number of observations for vol calculation
        momentum_window: int = 5,  # Number of observations for momentum
        annualization_factor: float = 252 * 24 * 60,  # Minutes per year for 1-min data
    ) -> None:
        """
        Initialize lag model.

        Args:
            price_feed: Price feed instance
            vol_window: Window for realized vol calculation
            momentum_window: Window for momentum calculation
            annualization_factor: Factor to annualize volatility
        """
        self.price_feed = price_feed
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.annualization_factor = annualization_factor

    def get_spot_price(self, symbol: str) -> float | None:
        """
        Get current spot price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current price or None
        """
        return self.price_feed.get_price(symbol)

    def compute_realized_vol(self, symbol: str, window: int | None = None) -> float:
        """
        Compute realized volatility from log returns.

        Args:
            symbol: Trading pair
            window: Number of observations (default: self.vol_window)

        Returns:
            Annualized volatility (0 if insufficient data)
        """
        window = window or self.vol_window
        returns = self.price_feed.get_returns(symbol, window)

        if len(returns) < 2:
            return 0.0

        # Standard deviation of log returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        # Annualize
        annualized_vol = std_dev * math.sqrt(self.annualization_factor)
        return annualized_vol

    def compute_lognormal_q(self, symbol: str) -> float:
        """
        Compute log-normal quantile (z-score of current log price).

        This measures how extreme the current price is relative to
        the recent distribution of prices.

        Args:
            symbol: Trading pair

        Returns:
            Z-score (0 if insufficient data)
        """
        prices = self.price_feed.get_prices(symbol, self.vol_window)

        if len(prices) < 3:
            return 0.0

        # Work in log space
        log_prices = [math.log(p) for p in prices]
        current_log_price = log_prices[-1]

        # Compute mean and std of historical log prices (excluding current)
        historical = log_prices[:-1]
        mean_log = sum(historical) / len(historical)
        variance = sum((lp - mean_log) ** 2 for lp in historical) / (len(historical) - 1)
        std_log = math.sqrt(variance) if variance > 0 else 1e-10

        # Z-score
        z_score = (current_log_price - mean_log) / std_log
        return z_score

    def compute_momentum(self, symbol: str, window: int | None = None) -> float:
        """
        Compute short-term momentum (mean of recent returns).

        Args:
            symbol: Trading pair
            window: Number of returns to average

        Returns:
            Mean log return (0 if insufficient data)
        """
        window = window or self.momentum_window
        returns = self.price_feed.get_returns(symbol, window)

        if not returns:
            return 0.0

        return sum(returns) / len(returns)

    def compute_metrics(self, symbol: str) -> AssetMetrics | None:
        """
        Compute all metrics for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            AssetMetrics or None if insufficient data
        """
        spot = self.get_spot_price(symbol)
        if spot is None:
            return None

        returns = self.price_feed.get_returns(symbol, 1)
        log_return = returns[-1] if returns else 0.0

        metrics = AssetMetrics(
            symbol=symbol,
            spot_price=spot,
            realized_vol=self.compute_realized_vol(symbol),
            lognormal_q=self.compute_lognormal_q(symbol),
            log_return=log_return,
            momentum=self.compute_momentum(symbol),
            timestamp=datetime.utcnow(),
        )

        logger.debug(
            "metrics_computed",
            symbol=symbol,
            spot=round(spot, 2),
            vol=round(metrics.realized_vol, 4),
            q=round(metrics.lognormal_q, 2),
            momentum=round(metrics.momentum, 6),
        )

        return metrics

    def compute_all_metrics(self) -> dict[str, AssetMetrics]:
        """
        Compute metrics for all tracked symbols.

        Returns:
            Dict of symbol -> AssetMetrics
        """
        results: dict[str, AssetMetrics] = {}

        for symbol in self.price_feed.symbols:
            metrics = self.compute_metrics(symbol)
            if metrics is not None:
                results[symbol] = metrics

        return results
