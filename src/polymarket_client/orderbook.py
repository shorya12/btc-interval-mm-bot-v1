"""Order book management and analysis."""

from dataclasses import dataclass
from datetime import datetime
from collections import deque

from src.belief_state.logit import logit_midpoint, logit
from src.common.logging import get_logger
from .types import OrderBook, OrderBookLevel

logger = get_logger(__name__)


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state for analysis."""

    timestamp: datetime
    best_bid: float
    best_ask: float
    mid_price: float
    mid_logit: float
    spread: float
    spread_bps: float
    bid_depth_5: float
    ask_depth_5: float
    imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth)


class OrderBookManager:
    """
    Manages order book state and provides analysis.

    Tracks:
    - Current order book
    - Historical snapshots for analysis
    - Mid-price in logit space
    - Depth and imbalance metrics
    """

    def __init__(
        self,
        token_id: str,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize order book manager.

        Args:
            token_id: Token ID this manager tracks
            max_history: Maximum historical snapshots to keep
        """
        self.token_id = token_id
        self._current: OrderBook | None = None
        self._history: deque[OrderBookSnapshot] = deque(maxlen=max_history)

    def update(self, book: OrderBook) -> OrderBookSnapshot | None:
        """
        Update with new order book data.

        Args:
            book: New order book

        Returns:
            Snapshot if book is valid, None otherwise
        """
        if book.token_id != self.token_id:
            logger.warning(
                "token_id_mismatch",
                expected=self.token_id,
                received=book.token_id,
            )
            return None

        if not book.is_valid():
            logger.warning("invalid_orderbook", token_id=self.token_id[:16] + "...")
            return None

        self._current = book

        # Create snapshot
        best_bid = book.best_bid_price
        best_ask = book.best_ask_price

        if best_bid is None or best_ask is None:
            return None

        mid_price = logit_midpoint(best_bid, best_ask)
        spread = best_ask - best_bid
        mid_for_bps = (best_bid + best_ask) / 2

        bid_depth = book.bid_depth(5)
        ask_depth = book.ask_depth(5)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        snapshot = OrderBookSnapshot(
            timestamp=book.timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            mid_logit=logit(mid_price),
            spread=spread,
            spread_bps=(spread / mid_for_bps) * 10000 if mid_for_bps > 0 else 0,
            bid_depth_5=bid_depth,
            ask_depth_5=ask_depth,
            imbalance=imbalance,
        )

        self._history.append(snapshot)
        return snapshot

    @property
    def current(self) -> OrderBook | None:
        """Get current order book."""
        return self._current

    @property
    def latest_snapshot(self) -> OrderBookSnapshot | None:
        """Get latest snapshot."""
        return self._history[-1] if self._history else None

    @property
    def best_bid(self) -> float | None:
        """Get current best bid price."""
        return self._current.best_bid_price if self._current else None

    @property
    def best_ask(self) -> float | None:
        """Get current best ask price."""
        return self._current.best_ask_price if self._current else None

    @property
    def mid_price(self) -> float | None:
        """Get current mid price in logit space."""
        if self._current is None:
            return None
        bid = self._current.best_bid_price
        ask = self._current.best_ask_price
        if bid is None or ask is None:
            return None
        return logit_midpoint(bid, ask)

    @property
    def mid_logit(self) -> float | None:
        """Get current mid price as logit."""
        mid = self.mid_price
        return logit(mid) if mid is not None else None

    def get_price_for_size(self, side: str, size: float) -> float | None:
        """
        Get average execution price for a given size.

        Walks the book to compute VWAP for the size.

        Args:
            side: "BUY" or "SELL"
            size: Size to execute

        Returns:
            VWAP or None if insufficient liquidity
        """
        if self._current is None:
            return None

        levels = self._current.asks if side == "BUY" else self._current.bids
        remaining = size
        total_value = 0.0

        for level in levels:
            if remaining <= 0:
                break
            fill_size = min(remaining, level.size)
            total_value += fill_size * level.price
            remaining -= fill_size

        if remaining > 0:
            # Insufficient liquidity
            return None

        return total_value / size

    def get_depth_at_price(self, price: float, side: str) -> float:
        """
        Get total depth at or better than a price.

        Args:
            price: Price level
            side: "BUY" (for ask side) or "SELL" (for bid side)

        Returns:
            Total size available
        """
        if self._current is None:
            return 0.0

        if side == "BUY":
            # Sum asks at or below price
            return sum(
                level.size
                for level in self._current.asks
                if level.price <= price
            )
        else:
            # Sum bids at or above price
            return sum(
                level.size
                for level in self._current.bids
                if level.price >= price
            )

    def get_average_spread(self, n: int = 10) -> float | None:
        """
        Get average spread over last N snapshots.

        Args:
            n: Number of snapshots to average

        Returns:
            Average spread or None
        """
        if len(self._history) == 0:
            return None

        recent = list(self._history)[-n:]
        return sum(s.spread for s in recent) / len(recent)

    def get_average_imbalance(self, n: int = 10) -> float | None:
        """
        Get average order book imbalance over last N snapshots.

        Args:
            n: Number of snapshots to average

        Returns:
            Average imbalance (-1 to 1) or None
        """
        if len(self._history) == 0:
            return None

        recent = list(self._history)[-n:]
        return sum(s.imbalance for s in recent) / len(recent)

    def is_crossed(self) -> bool:
        """Check if book is crossed (bid >= ask)."""
        if self._current is None:
            return False
        bid = self._current.best_bid_price
        ask = self._current.best_ask_price
        if bid is None or ask is None:
            return False
        return bid >= ask

    def has_sufficient_liquidity(self, min_depth: float = 100) -> bool:
        """
        Check if book has sufficient liquidity.

        Args:
            min_depth: Minimum depth required on each side

        Returns:
            True if both sides have at least min_depth
        """
        if self._current is None:
            return False
        return (
            self._current.bid_depth(5) >= min_depth
            and self._current.ask_depth(5) >= min_depth
        )
