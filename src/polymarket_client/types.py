"""Data types for Polymarket CLOB client."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    GTC = "GTC"  # Good-til-cancelled
    GTD = "GTD"  # Good-til-date
    FOK = "FOK"  # Fill-or-kill


class OrderStatus(str, Enum):
    """Order status from CLOB."""

    LIVE = "LIVE"
    MATCHED = "MATCHED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float  # Probability 0-1
    size: float  # Size in outcome tokens

    def __post_init__(self) -> None:
        if not 0 <= self.price <= 1:
            raise ValueError(f"Price must be in [0, 1], got {self.price}")
        if self.size < 0:
            raise ValueError(f"Size must be non-negative, got {self.size}")


@dataclass
class OrderBook:
    """L2 order book snapshot."""

    token_id: str
    bids: list[OrderBookLevel]  # Sorted by price descending
    asks: list[OrderBookLevel]  # Sorted by price ascending
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> OrderBookLevel | None:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    @property
    def best_bid_price(self) -> float | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask_price(self) -> float | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Get mid price (arithmetic mean of best bid/ask)."""
        if self.best_bid_price is None or self.best_ask_price is None:
            return None
        return (self.best_bid_price + self.best_ask_price) / 2

    @property
    def spread(self) -> float | None:
        """Get bid-ask spread."""
        if self.best_bid_price is None or self.best_ask_price is None:
            return None
        return self.best_ask_price - self.best_bid_price

    @property
    def spread_bps(self) -> float | None:
        """Get spread in basis points relative to mid."""
        if self.mid_price is None or self.mid_price == 0 or self.spread is None:
            return None
        return (self.spread / self.mid_price) * 10000

    def bid_depth(self, levels: int = 5) -> float:
        """Get total bid size for top N levels."""
        return sum(level.size for level in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        """Get total ask size for top N levels."""
        return sum(level.size for level in self.asks[:levels])

    def is_valid(self) -> bool:
        """Check if order book is valid (has both sides, no crossed book)."""
        if not self.bids or not self.asks:
            return False
        return self.best_bid_price < self.best_ask_price


@dataclass
class Order:
    """Order placed on CLOB."""

    id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    original_size: float
    order_type: OrderType = OrderType.GTC
    status: OrderStatus = OrderStatus.LIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expiration: datetime | None = None
    maker_address: str | None = None
    salt: str | None = None
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filled_size(self) -> float:
        """Get filled size."""
        return self.original_size - self.size

    @property
    def is_live(self) -> bool:
        """Check if order is still live."""
        return self.status == OrderStatus.LIVE

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.size == 0 or self.status == OrderStatus.MATCHED


@dataclass
class Fill:
    """Fill/trade execution."""

    id: str
    order_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tx_hash: str | None = None
    taker_order_id: str | None = None
    maker_address: str | None = None
    taker_address: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Get fill value (price * size)."""
        return self.price * self.size

    @property
    def net_value(self) -> float:
        """Get net value after fees."""
        if self.side == OrderSide.BUY:
            return -(self.value + self.fee)
        else:
            return self.value - self.fee


@dataclass
class Position:
    """Current position for a token."""

    token_id: str
    size: float  # Positive = long, negative = short
    avg_entry_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    condition_id: str | None = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.size == 0

    @property
    def notional_value(self) -> float | None:
        """Get notional value of position."""
        if self.avg_entry_price is None:
            return None
        return abs(self.size) * self.avg_entry_price

    def compute_unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized PNL at current price."""
        if self.avg_entry_price is None or self.size == 0:
            return 0.0
        if self.size > 0:
            # Long: profit when price goes up
            return self.size * (current_price - self.avg_entry_price)
        else:
            # Short: profit when price goes down
            return abs(self.size) * (self.avg_entry_price - current_price)


@dataclass
class Market:
    """Market/condition information."""

    condition_id: str
    question_id: str | None = None
    question: str = ""
    description: str = ""
    outcomes: list[str] = field(default_factory=list)
    outcome_prices: list[float] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)  # Token IDs for each outcome
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: datetime | None = None
    resolved: bool = False
    winning_outcome: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
