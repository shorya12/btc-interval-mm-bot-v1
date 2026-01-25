"""Parameters and data structures for Avellaneda-Stoikov quoting."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ASParams:
    """
    Avellaneda-Stoikov market-making parameters.

    The A-S model produces optimal bid/ask quotes that balance:
    - Inventory risk (gamma parameter)
    - Adverse selection from informed traders
    - Price volatility

    Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"
    """

    gamma: float
    """Risk aversion parameter (0-1). Higher = more aggressive inventory management."""

    base_spread_x: float
    """Base spread multiplier. Scales the minimum spread."""

    sigma_b: float
    """Belief volatility (std dev in logit space). From BeliefState."""

    time_remaining: float
    """Time remaining to horizon (0-1). 1 = full period, 0 = end."""

    inventory: float
    """Current inventory position. Positive = long, negative = short."""

    kappa: float | None = None
    """Order arrival rate (optional). If None, use simplified model."""

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.base_spread_x < 0:
            raise ValueError(f"base_spread_x must be non-negative, got {self.base_spread_x}")
        if self.sigma_b <= 0:
            raise ValueError(f"sigma_b must be positive, got {self.sigma_b}")
        if not 0 <= self.time_remaining <= 1:
            raise ValueError(f"time_remaining must be in [0, 1], got {self.time_remaining}")


@dataclass
class Quote:
    """
    Generated bid/ask quote.

    All prices are in probability space [0, 1].
    Logit values are provided for reference.
    """

    bid_price: float
    """Bid price (probability)."""

    ask_price: float
    """Ask price (probability)."""

    bid_logit: float
    """Bid price in logit space."""

    ask_logit: float
    """Ask price in logit space."""

    reservation_price: float
    """Reservation/indifference price (probability)."""

    reservation_logit: float
    """Reservation price in logit space."""

    spread: float
    """Total spread (ask - bid) in probability space."""

    spread_logit: float
    """Total spread in logit space."""

    half_spread_logit: float
    """Half-spread in logit space (symmetric around reservation)."""

    inventory_skew: float
    """Inventory-induced skew in logit space."""

    signal_skew: float
    """External signal skew applied (e.g., from lag signal)."""

    metadata: dict[str, Any] | None = None
    """Optional metadata for debugging/logging."""

    @property
    def mid_price(self) -> float:
        """Mid-price in probability space."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid."""
        if self.mid_price == 0:
            return 0
        return (self.spread / self.mid_price) * 10000

    def is_valid(self) -> bool:
        """Check if quote is valid (bid < ask, both in [0,1])."""
        return (
            0 < self.bid_price < self.ask_price < 1
            and self.spread > 0
        )

    def with_skew(self, skew_logit: float) -> "Quote":
        """
        Create a new quote with additional skew applied.

        Args:
            skew_logit: Skew to add in logit space (positive = shift up)

        Returns:
            New Quote with skew applied
        """
        from src.belief_state.logit import sigmoid

        new_bid_logit = self.bid_logit + skew_logit
        new_ask_logit = self.ask_logit + skew_logit

        return Quote(
            bid_price=sigmoid(new_bid_logit),
            ask_price=sigmoid(new_ask_logit),
            bid_logit=new_bid_logit,
            ask_logit=new_ask_logit,
            reservation_price=sigmoid(self.reservation_logit + skew_logit),
            reservation_logit=self.reservation_logit + skew_logit,
            spread=sigmoid(new_ask_logit) - sigmoid(new_bid_logit),
            spread_logit=self.spread_logit,
            half_spread_logit=self.half_spread_logit,
            inventory_skew=self.inventory_skew,
            signal_skew=self.signal_skew + skew_logit,
            metadata=self.metadata,
        )
