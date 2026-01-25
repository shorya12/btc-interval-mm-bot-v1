"""Inventory management and position limits."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InventoryStatus:
    """Current inventory status."""

    position_size: float
    position_frac: float  # As fraction of bankroll
    max_position_frac: float
    at_max_long: bool
    at_max_short: bool
    can_buy: bool
    can_sell: bool
    available_to_buy: float
    available_to_sell: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class InventoryManager:
    """
    Manages inventory limits and position sizing.

    Enforces:
    - Maximum net position as fraction of bankroll
    - Asymmetric limits (can have different long/short limits)
    - Gradual position building/reduction
    """

    def __init__(
        self,
        max_net_frac: float = 0.20,
        max_long_frac: float | None = None,
        max_short_frac: float | None = None,
        target_position: float = 0.0,
        rebalance_threshold: float = 0.05,
    ) -> None:
        """
        Initialize inventory manager.

        Args:
            max_net_frac: Maximum net position as fraction of bankroll
            max_long_frac: Maximum long position (defaults to max_net_frac)
            max_short_frac: Maximum short position (defaults to max_net_frac)
            target_position: Target position to rebalance toward
            rebalance_threshold: Distance from target before rebalancing
        """
        self.max_net_frac = max_net_frac
        self.max_long_frac = max_long_frac or max_net_frac
        self.max_short_frac = max_short_frac or max_net_frac
        self.target_position = target_position
        self.rebalance_threshold = rebalance_threshold

    def get_status(
        self,
        position_size: float,
        bankroll: float,
    ) -> InventoryStatus:
        """
        Get current inventory status.

        Args:
            position_size: Current position (positive = long)
            bankroll: Total bankroll

        Returns:
            InventoryStatus
        """
        if bankroll <= 0:
            return InventoryStatus(
                position_size=position_size,
                position_frac=0,
                max_position_frac=self.max_net_frac,
                at_max_long=True,
                at_max_short=True,
                can_buy=False,
                can_sell=False,
                available_to_buy=0,
                available_to_sell=0,
            )

        position_frac = position_size / bankroll

        # Check limits
        max_long_size = bankroll * self.max_long_frac
        max_short_size = bankroll * self.max_short_frac

        at_max_long = position_size >= max_long_size
        at_max_short = position_size <= -max_short_size

        # Calculate available capacity
        available_to_buy = max(0, max_long_size - position_size)
        available_to_sell = max(0, position_size + max_short_size)

        can_buy = not at_max_long
        can_sell = not at_max_short

        return InventoryStatus(
            position_size=position_size,
            position_frac=position_frac,
            max_position_frac=self.max_net_frac,
            at_max_long=at_max_long,
            at_max_short=at_max_short,
            can_buy=can_buy,
            can_sell=can_sell,
            available_to_buy=available_to_buy,
            available_to_sell=available_to_sell,
        )

    def get_order_size_limit(
        self,
        side: str,
        position_size: float,
        bankroll: float,
        desired_size: float,
    ) -> float:
        """
        Get the maximum order size allowed given current position.

        Args:
            side: "BUY" or "SELL"
            position_size: Current position
            bankroll: Total bankroll
            desired_size: Desired order size

        Returns:
            Maximum allowed size (may be less than desired)
        """
        status = self.get_status(position_size, bankroll)

        if side == "BUY":
            max_size = status.available_to_buy
        else:
            max_size = status.available_to_sell

        limited_size = min(desired_size, max_size)

        if limited_size < desired_size:
            logger.debug(
                "order_size_limited",
                side=side,
                desired=desired_size,
                limited=limited_size,
                reason="inventory_limit",
            )

        return limited_size

    def should_rebalance(self, position_size: float, bankroll: float) -> bool:
        """
        Check if position should be rebalanced toward target.

        Args:
            position_size: Current position
            bankroll: Total bankroll

        Returns:
            True if rebalancing is recommended
        """
        if bankroll <= 0:
            return False

        current_frac = position_size / bankroll
        target_frac = self.target_position / bankroll if bankroll > 0 else 0

        distance = abs(current_frac - target_frac)
        return distance > self.rebalance_threshold

    def get_rebalance_direction(self, position_size: float) -> str | None:
        """
        Get direction to rebalance toward target.

        Args:
            position_size: Current position

        Returns:
            "BUY", "SELL", or None if at target
        """
        if position_size > self.target_position:
            return "SELL"
        elif position_size < self.target_position:
            return "BUY"
        return None

    def compute_inventory_skew(
        self,
        position_size: float,
        bankroll: float,
        max_skew: float = 0.5,
    ) -> float:
        """
        Compute inventory-based skew for A-S model.

        Returns a value that should be subtracted from the reservation price
        to encourage reducing inventory.

        Args:
            position_size: Current position (positive = long)
            bankroll: Total bankroll
            max_skew: Maximum skew in logit space

        Returns:
            Skew value (positive if long, to lower quotes and encourage selling)
        """
        if bankroll <= 0:
            return 0.0

        position_frac = position_size / bankroll

        # Linear mapping from position fraction to skew
        # At max long, skew = +max_skew (lower quotes)
        # At max short, skew = -max_skew (raise quotes)
        skew = (position_frac / self.max_net_frac) * max_skew

        # Clamp
        skew = max(-max_skew, min(max_skew, skew))

        return skew

    def update_limits(
        self,
        max_net_frac: float | None = None,
        max_long_frac: float | None = None,
        max_short_frac: float | None = None,
    ) -> None:
        """Update position limits."""
        if max_net_frac is not None:
            self.max_net_frac = max_net_frac
        if max_long_frac is not None:
            self.max_long_frac = max_long_frac
        if max_short_frac is not None:
            self.max_short_frac = max_short_frac
