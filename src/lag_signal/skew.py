"""Weighted skew computation from lag signal."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.common.logging import get_logger
from .model import LagModel, AssetMetrics

logger = get_logger(__name__)


@dataclass
class AssetSkewComponent:
    """Skew contribution from a single asset."""

    symbol: str
    weight: float
    raw_signal: float  # The underlying signal (e.g., momentum, q)
    weighted_signal: float  # weight * raw_signal
    metrics: AssetMetrics | None = None


@dataclass
class SkewSignal:
    """Combined skew signal from all assets."""

    total_skew: float  # Combined weighted skew in logit space
    components: list[AssetSkewComponent]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_bullish(self) -> bool:
        """Check if overall signal is bullish (positive skew)."""
        return self.total_skew > 0

    @property
    def is_bearish(self) -> bool:
        """Check if overall signal is bearish (negative skew)."""
        return self.total_skew < 0

    @property
    def strength(self) -> float:
        """Absolute strength of signal."""
        return abs(self.total_skew)

    def get_component(self, symbol: str) -> AssetSkewComponent | None:
        """Get component for a specific symbol."""
        for comp in self.components:
            if comp.symbol == symbol:
                return comp
        return None


@dataclass
class AssetConfig:
    """Configuration for a single asset in skew calculation."""

    symbol: str
    weight: float = 1.0
    signal_type: str = "momentum"  # "momentum", "q", "vol_adjusted"


class SkewComputer:
    """
    Computes weighted skew from crypto lag signals.

    The skew is used to adjust market-making quotes based on
    crypto market movements (e.g., if BTC is trending up,
    shift prediction market quotes higher).
    """

    def __init__(
        self,
        lag_model: LagModel,
        asset_configs: list[AssetConfig] | None = None,
        skew_multiplier: float = 1.0,
        max_skew: float = 0.5,  # Maximum skew in logit space
    ) -> None:
        """
        Initialize skew computer.

        Args:
            lag_model: Lag model for computing metrics
            asset_configs: Configuration for each asset
            skew_multiplier: Global multiplier for final skew
            max_skew: Maximum absolute skew (clamped)
        """
        self.lag_model = lag_model
        self.skew_multiplier = skew_multiplier
        self.max_skew = max_skew

        # Default configs if not provided
        if asset_configs is None:
            self.asset_configs = [
                AssetConfig("BTC/USDT", weight=0.5, signal_type="vol_adjusted"),
                AssetConfig("ETH/USDT", weight=0.3, signal_type="vol_adjusted"),
                AssetConfig("XRP/USDT", weight=0.2, signal_type="vol_adjusted"),
            ]
        else:
            self.asset_configs = asset_configs

        # Normalize weights
        total_weight = sum(c.weight for c in self.asset_configs)
        if total_weight > 0:
            for config in self.asset_configs:
                config.weight /= total_weight

    def _compute_raw_signal(
        self,
        metrics: AssetMetrics,
        signal_type: str,
    ) -> float:
        """
        Compute raw signal from metrics based on signal type.

        Args:
            metrics: Asset metrics
            signal_type: Type of signal to use

        Returns:
            Raw signal value
        """
        if signal_type == "momentum":
            return metrics.momentum
        elif signal_type == "q":
            # Lognormal q, scaled down
            return metrics.lognormal_q * 0.1
        elif signal_type == "vol_adjusted":
            return metrics.vol_adjusted_momentum
        else:
            return metrics.momentum

    def compute_weighted_skew(self) -> SkewSignal:
        """
        Compute combined weighted skew from all configured assets.

        Returns:
            SkewSignal with total skew and components
        """
        components: list[AssetSkewComponent] = []
        total_skew = 0.0

        for config in self.asset_configs:
            metrics = self.lag_model.compute_metrics(config.symbol)

            if metrics is None:
                # No data for this asset
                components.append(AssetSkewComponent(
                    symbol=config.symbol,
                    weight=config.weight,
                    raw_signal=0.0,
                    weighted_signal=0.0,
                    metrics=None,
                ))
                continue

            raw_signal = self._compute_raw_signal(metrics, config.signal_type)
            weighted_signal = config.weight * raw_signal

            components.append(AssetSkewComponent(
                symbol=config.symbol,
                weight=config.weight,
                raw_signal=raw_signal,
                weighted_signal=weighted_signal,
                metrics=metrics,
            ))

            total_skew += weighted_signal

        # Apply multiplier and clamp
        total_skew *= self.skew_multiplier
        total_skew = max(-self.max_skew, min(self.max_skew, total_skew))

        signal = SkewSignal(
            total_skew=total_skew,
            components=components,
            timestamp=datetime.utcnow(),
            metadata={
                "skew_multiplier": self.skew_multiplier,
                "max_skew": self.max_skew,
                "num_assets": len(self.asset_configs),
            },
        )

        logger.debug(
            "skew_computed",
            total_skew=round(total_skew, 4),
            components={c.symbol: round(c.weighted_signal, 4) for c in components},
        )

        return signal

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update asset weights.

        Args:
            new_weights: Dict of symbol -> new weight
        """
        for config in self.asset_configs:
            if config.symbol in new_weights:
                config.weight = new_weights[config.symbol]

        # Renormalize
        total_weight = sum(c.weight for c in self.asset_configs)
        if total_weight > 0:
            for config in self.asset_configs:
                config.weight /= total_weight

    def add_asset(
        self,
        symbol: str,
        weight: float = 1.0,
        signal_type: str = "vol_adjusted",
    ) -> None:
        """
        Add a new asset to track.

        Args:
            symbol: Trading pair
            weight: Initial weight
            signal_type: Signal type to use
        """
        # Check if already exists
        for config in self.asset_configs:
            if config.symbol == symbol:
                config.weight = weight
                config.signal_type = signal_type
                return

        self.asset_configs.append(AssetConfig(
            symbol=symbol,
            weight=weight,
            signal_type=signal_type,
        ))

        # Add to price feed if needed
        self.lag_model.price_feed.add_symbol(symbol)

    def remove_asset(self, symbol: str) -> bool:
        """
        Remove an asset from tracking.

        Args:
            symbol: Trading pair to remove

        Returns:
            True if removed, False if not found
        """
        for i, config in enumerate(self.asset_configs):
            if config.symbol == symbol:
                del self.asset_configs[i]
                return True
        return False
