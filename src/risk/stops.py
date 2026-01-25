"""Stop conditions for exiting positions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


class StopType(str, Enum):
    """Types of stop conditions."""

    PROBABILITY_LOW = "probability_low"
    PROBABILITY_HIGH = "probability_high"
    MAX_LOSS = "max_loss"
    TIME_TO_EXPIRY = "time_to_expiry"
    MAX_POSITION = "max_position"


@dataclass
class StopResult:
    """Result of stop condition check."""

    triggered: bool
    stop_type: StopType | None
    should_close: bool  # True if should close entire position
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def description(self) -> str:
        """Get human-readable description."""
        if not self.triggered:
            return "No stop triggered"
        return f"{self.stop_type.value}: {self.details}"


@dataclass
class StopConfig:
    """Configuration for stop conditions."""

    stop_prob_low: float = 0.02
    stop_prob_high: float = 0.98
    max_loss_pct: float = 0.10  # 10% max loss
    min_time_to_expiry_seconds: float = 300  # 5 minutes
    max_position_frac: float = 0.20  # 20% of bankroll


class StopChecker:
    """
    Checks for stop conditions that require position closure.

    Stops are more severe than vetoes - they indicate we should
    close our position and stop trading the market.
    """

    def __init__(self, config: StopConfig | None = None) -> None:
        """
        Initialize stop checker.

        Args:
            config: Stop configuration
        """
        self.config = config or StopConfig()

    def check_all(
        self,
        current_prob: float,
        position_size: float,
        position_pnl: float,
        bankroll: float,
        time_to_expiry_seconds: float | None = None,
    ) -> StopResult:
        """
        Check all stop conditions.

        Args:
            current_prob: Current market probability
            position_size: Current position size
            position_pnl: Current position PNL
            bankroll: Total bankroll
            time_to_expiry_seconds: Time until market expiry

        Returns:
            StopResult indicating if any stop is triggered
        """
        # Check probability stops
        prob_result = self.check_probability_stop(current_prob)
        if prob_result.triggered:
            return prob_result

        # Check loss stop
        loss_result = self.check_loss_stop(position_pnl, bankroll)
        if loss_result.triggered:
            return loss_result

        # Check time to expiry
        if time_to_expiry_seconds is not None:
            time_result = self.check_time_stop(time_to_expiry_seconds)
            if time_result.triggered:
                return time_result

        # Check position size
        position_result = self.check_position_stop(position_size, bankroll)
        if position_result.triggered:
            return position_result

        return StopResult(
            triggered=False,
            stop_type=None,
            should_close=False,
        )

    def check_probability_stop(self, current_prob: float) -> StopResult:
        """
        Check if probability has hit stop levels.

        At extreme probabilities (near 0 or 1), the market is likely
        resolved and we should exit.

        Args:
            current_prob: Current market probability

        Returns:
            StopResult
        """
        if current_prob <= self.config.stop_prob_low:
            logger.warning(
                "probability_stop_triggered",
                type="low",
                current=current_prob,
                threshold=self.config.stop_prob_low,
            )
            return StopResult(
                triggered=True,
                stop_type=StopType.PROBABILITY_LOW,
                should_close=True,
                details={
                    "current_prob": current_prob,
                    "threshold": self.config.stop_prob_low,
                },
            )

        if current_prob >= self.config.stop_prob_high:
            logger.warning(
                "probability_stop_triggered",
                type="high",
                current=current_prob,
                threshold=self.config.stop_prob_high,
            )
            return StopResult(
                triggered=True,
                stop_type=StopType.PROBABILITY_HIGH,
                should_close=True,
                details={
                    "current_prob": current_prob,
                    "threshold": self.config.stop_prob_high,
                },
            )

        return StopResult(triggered=False, stop_type=None, should_close=False)

    def check_loss_stop(
        self,
        position_pnl: float,
        bankroll: float,
    ) -> StopResult:
        """
        Check if losses have exceeded maximum.

        Args:
            position_pnl: Current position PNL (negative = loss)
            bankroll: Total bankroll

        Returns:
            StopResult
        """
        if bankroll <= 0:
            return StopResult(triggered=False, stop_type=None, should_close=False)

        loss_pct = -position_pnl / bankroll if position_pnl < 0 else 0

        if loss_pct >= self.config.max_loss_pct:
            logger.warning(
                "loss_stop_triggered",
                loss_pct=loss_pct,
                max_loss_pct=self.config.max_loss_pct,
                position_pnl=position_pnl,
            )
            return StopResult(
                triggered=True,
                stop_type=StopType.MAX_LOSS,
                should_close=True,
                details={
                    "loss_pct": loss_pct,
                    "max_loss_pct": self.config.max_loss_pct,
                    "position_pnl": position_pnl,
                },
            )

        return StopResult(triggered=False, stop_type=None, should_close=False)

    def check_time_stop(
        self,
        time_to_expiry_seconds: float,
    ) -> StopResult:
        """
        Check if too close to market expiry.

        Args:
            time_to_expiry_seconds: Seconds until market expires

        Returns:
            StopResult
        """
        if time_to_expiry_seconds <= self.config.min_time_to_expiry_seconds:
            logger.warning(
                "time_stop_triggered",
                time_to_expiry=time_to_expiry_seconds,
                min_time=self.config.min_time_to_expiry_seconds,
            )
            return StopResult(
                triggered=True,
                stop_type=StopType.TIME_TO_EXPIRY,
                should_close=True,
                details={
                    "time_to_expiry_seconds": time_to_expiry_seconds,
                    "min_time_seconds": self.config.min_time_to_expiry_seconds,
                },
            )

        return StopResult(triggered=False, stop_type=None, should_close=False)

    def check_position_stop(
        self,
        position_size: float,
        bankroll: float,
    ) -> StopResult:
        """
        Check if position size exceeds maximum.

        This is a soft stop - doesn't close but prevents increasing.

        Args:
            position_size: Absolute position size
            bankroll: Total bankroll

        Returns:
            StopResult (should_close=False, just prevents new orders)
        """
        if bankroll <= 0:
            return StopResult(triggered=False, stop_type=None, should_close=False)

        position_frac = abs(position_size) / bankroll

        if position_frac >= self.config.max_position_frac:
            logger.info(
                "position_stop_triggered",
                position_frac=position_frac,
                max_frac=self.config.max_position_frac,
            )
            return StopResult(
                triggered=True,
                stop_type=StopType.MAX_POSITION,
                should_close=False,  # Don't close, just prevent increasing
                details={
                    "position_frac": position_frac,
                    "max_frac": self.config.max_position_frac,
                    "position_size": position_size,
                },
            )

        return StopResult(triggered=False, stop_type=None, should_close=False)
