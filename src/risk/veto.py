"""Veto logic for jump and momentum detection."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.belief_state import BeliefState
from src.common.logging import get_logger

logger = get_logger(__name__)


class VetoReason(str, Enum):
    """Reasons for vetoing trading."""

    JUMP_DETECTED = "jump_detected"
    MOMENTUM_DETECTED = "momentum_detected"
    EXTREME_PRICE = "extreme_price"
    WIDE_SPREAD = "wide_spread"
    LOW_LIQUIDITY = "low_liquidity"
    CRYPTO_VOLATILITY = "crypto_volatility"


@dataclass
class VetoResult:
    """Result of veto check."""

    vetoed: bool
    reasons: list[VetoReason]
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def reason_str(self) -> str:
        """Get reasons as comma-separated string."""
        return ", ".join(r.value for r in self.reasons)


class VetoChecker:
    """
    Checks for conditions that should veto trading.

    Vetoes are temporary pauses in trading when market conditions
    are dangerous (e.g., price jumps, momentum regimes).
    """

    def __init__(
        self,
        jump_z: float = 3.0,
        momentum_z: float = 2.0,
        extreme_prob_threshold: float = 0.02,
        max_spread_bps: float = 500,
        min_liquidity: float = 100,
        crypto_vol_threshold: float = 0.5,  # 50% annualized
    ) -> None:
        """
        Initialize veto checker.

        Args:
            jump_z: Z-score threshold for jump detection
            momentum_z: Z-score threshold for momentum detection
            extreme_prob_threshold: Distance from 0/1 to consider extreme
            max_spread_bps: Maximum spread in bps before veto
            min_liquidity: Minimum depth required on each side
            crypto_vol_threshold: Crypto vol threshold for veto
        """
        self.jump_z = jump_z
        self.momentum_z = momentum_z
        self.extreme_prob_threshold = extreme_prob_threshold
        self.max_spread_bps = max_spread_bps
        self.min_liquidity = min_liquidity
        self.crypto_vol_threshold = crypto_vol_threshold

    def check(
        self,
        belief: BeliefState,
        spread_bps: float | None = None,
        bid_depth: float | None = None,
        ask_depth: float | None = None,
        crypto_vol: float | None = None,
    ) -> VetoResult:
        """
        Check for veto conditions.

        Args:
            belief: Current belief state
            spread_bps: Current spread in basis points
            bid_depth: Total bid depth (top 5 levels)
            ask_depth: Total ask depth (top 5 levels)
            crypto_vol: Current crypto realized volatility

        Returns:
            VetoResult indicating if trading should be vetoed
        """
        reasons: list[VetoReason] = []
        details: dict[str, Any] = {}

        # Check jump detection from belief state
        if belief.jump_detected:
            reasons.append(VetoReason.JUMP_DETECTED)
            details["jump_detected"] = True

        # Check momentum detection from belief state
        if belief.momentum_detected:
            reasons.append(VetoReason.MOMENTUM_DETECTED)
            details["momentum_detected"] = True

        # Check for extreme prices
        if belief.mid_prob < self.extreme_prob_threshold:
            reasons.append(VetoReason.EXTREME_PRICE)
            details["extreme_low"] = belief.mid_prob
        elif belief.mid_prob > (1 - self.extreme_prob_threshold):
            reasons.append(VetoReason.EXTREME_PRICE)
            details["extreme_high"] = belief.mid_prob

        # Check spread
        if spread_bps is not None and spread_bps > self.max_spread_bps:
            reasons.append(VetoReason.WIDE_SPREAD)
            details["spread_bps"] = spread_bps

        # Check liquidity
        if bid_depth is not None and bid_depth < self.min_liquidity:
            reasons.append(VetoReason.LOW_LIQUIDITY)
            details["bid_depth"] = bid_depth
        if ask_depth is not None and ask_depth < self.min_liquidity:
            reasons.append(VetoReason.LOW_LIQUIDITY)
            details["ask_depth"] = ask_depth

        # Check crypto volatility
        if crypto_vol is not None and crypto_vol > self.crypto_vol_threshold:
            reasons.append(VetoReason.CRYPTO_VOLATILITY)
            details["crypto_vol"] = crypto_vol

        vetoed = len(reasons) > 0

        if vetoed:
            logger.info(
                "trading_vetoed",
                reasons=[r.value for r in reasons],
                details=details,
            )

        return VetoResult(
            vetoed=vetoed,
            reasons=reasons,
            details=details,
        )

    def check_jump(
        self,
        price_change: float,
        sigma: float,
    ) -> bool:
        """
        Check if a price change constitutes a jump.

        Args:
            price_change: Price change in logit space
            sigma: Volatility in logit space

        Returns:
            True if jump detected
        """
        if sigma <= 0:
            return False
        z_score = abs(price_change) / sigma
        return z_score > self.jump_z

    def check_momentum(
        self,
        returns: list[float],
        sigma: float,
    ) -> bool:
        """
        Check if recent returns indicate momentum regime.

        Args:
            returns: Recent returns in logit space
            sigma: Volatility in logit space

        Returns:
            True if momentum detected
        """
        if not returns or sigma <= 0:
            return False

        import math
        mean_return = sum(returns) / len(returns)
        momentum_z = abs(mean_return) / (sigma / math.sqrt(len(returns)))
        return momentum_z > self.momentum_z
