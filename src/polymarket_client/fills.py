"""Fill tracking and position computation."""

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Any

from src.common.logging import get_logger
from .types import Fill, Position, OrderSide
from .client import PolymarketClient

logger = get_logger(__name__)


@dataclass
class FillStats:
    """Statistics about fills."""

    total_fills: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    avg_fill_price: float | None = None
    last_fill_time: datetime | None = None


class FillTracker:
    """
    Tracks fills and computes positions.

    Maintains:
    - Fill history per token
    - Current position per token
    - Fill statistics
    """

    def __init__(
        self,
        client: PolymarketClient,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize fill tracker.

        Args:
            client: Polymarket client
            max_history: Maximum fills to keep per token
        """
        self.client = client
        self.max_history = max_history

        self._fills: dict[str, deque[Fill]] = {}  # token_id -> fills
        self._positions: dict[str, Position] = {}  # token_id -> position
        self._last_fetch: datetime | None = None

    async def sync_fills(self, token_id: str | None = None) -> list[Fill]:
        """
        Fetch and sync fills from exchange.

        Args:
            token_id: Optional token to filter

        Returns:
            New fills since last sync
        """
        new_fills = await self.client.get_fills(
            token_id=token_id,
            since=self._last_fetch,
        )

        self._last_fetch = datetime.utcnow()

        for fill in new_fills:
            self._process_fill(fill)

        if new_fills:
            logger.info(
                "fills_synced",
                count=len(new_fills),
                token_id=token_id,
            )

        return new_fills

    def _process_fill(self, fill: Fill) -> None:
        """Process a single fill and update position."""
        token_id = fill.token_id

        # Add to fill history
        if token_id not in self._fills:
            self._fills[token_id] = deque(maxlen=self.max_history)
        self._fills[token_id].append(fill)

        # Update position
        self._update_position_from_fill(fill)

        logger.debug(
            "fill_processed",
            fill_id=fill.id[:16] + "..." if fill.id else "unknown",
            token_id=token_id[:16] + "...",
            side=fill.side.value,
            price=round(fill.price, 4),
            size=fill.size,
        )

    def _update_position_from_fill(self, fill: Fill) -> None:
        """Update position based on fill."""
        token_id = fill.token_id

        if token_id not in self._positions:
            self._positions[token_id] = Position(token_id=token_id, size=0.0)

        pos = self._positions[token_id]
        old_size = pos.size

        if fill.side == OrderSide.BUY:
            # Buying increases position
            new_size = old_size + fill.size

            if old_size >= 0:
                # Adding to long or opening long
                old_cost = (pos.avg_entry_price or 0) * old_size
                new_cost = old_cost + fill.price * fill.size
                pos.avg_entry_price = new_cost / new_size if new_size > 0 else None
            else:
                # Closing short
                closed_size = min(fill.size, abs(old_size))
                if pos.avg_entry_price is not None:
                    # Realize PNL on closed portion
                    pnl = closed_size * (pos.avg_entry_price - fill.price)
                    pos.realized_pnl += pnl

                if new_size > 0:
                    # Flipped to long
                    pos.avg_entry_price = fill.price
                elif new_size == 0:
                    pos.avg_entry_price = None
                # If still short, keep existing avg entry

        else:  # SELL
            # Selling decreases position
            new_size = old_size - fill.size

            if old_size <= 0:
                # Adding to short or opening short
                old_cost = (pos.avg_entry_price or 0) * abs(old_size)
                new_cost = old_cost + fill.price * fill.size
                pos.avg_entry_price = new_cost / abs(new_size) if new_size < 0 else None
            else:
                # Closing long
                closed_size = min(fill.size, old_size)
                if pos.avg_entry_price is not None:
                    # Realize PNL on closed portion
                    pnl = closed_size * (fill.price - pos.avg_entry_price)
                    pos.realized_pnl += pnl

                if new_size < 0:
                    # Flipped to short
                    pos.avg_entry_price = fill.price
                elif new_size == 0:
                    pos.avg_entry_price = None
                # If still long, keep existing avg entry

        pos.size = new_size
        pos.updated_at = datetime.utcnow()

    def get_position(self, token_id: str) -> Position:
        """
        Get current position for a token.

        Args:
            token_id: Token ID

        Returns:
            Position (may be flat if no position)
        """
        if token_id not in self._positions:
            return Position(token_id=token_id, size=0.0)
        return self._positions[token_id]

    def get_all_positions(self) -> list[Position]:
        """Get all positions."""
        return list(self._positions.values())

    def get_nonzero_positions(self) -> list[Position]:
        """Get all non-zero positions."""
        return [p for p in self._positions.values() if not p.is_flat]

    def get_fills(self, token_id: str, n: int | None = None) -> list[Fill]:
        """
        Get recent fills for a token.

        Args:
            token_id: Token ID
            n: Number of fills (None for all)

        Returns:
            List of fills (most recent last)
        """
        if token_id not in self._fills:
            return []

        fills = list(self._fills[token_id])
        if n is not None:
            fills = fills[-n:]
        return fills

    def get_fill_stats(self, token_id: str) -> FillStats:
        """
        Get fill statistics for a token.

        Args:
            token_id: Token ID

        Returns:
            FillStats
        """
        if token_id not in self._fills:
            return FillStats()

        fills = list(self._fills[token_id])
        if not fills:
            return FillStats()

        total_volume = sum(f.size * f.price for f in fills)
        total_fees = sum(f.fee for f in fills)
        buy_volume = sum(f.size * f.price for f in fills if f.side == OrderSide.BUY)
        sell_volume = sum(f.size * f.price for f in fills if f.side == OrderSide.SELL)
        total_size = sum(f.size for f in fills)

        return FillStats(
            total_fills=len(fills),
            total_volume=total_volume,
            total_fees=total_fees,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            avg_fill_price=total_volume / total_size if total_size > 0 else None,
            last_fill_time=fills[-1].timestamp if fills else None,
        )

    def update_unrealized_pnl(self, token_id: str, current_price: float) -> float:
        """
        Update and return unrealized PNL for a position.

        Args:
            token_id: Token ID
            current_price: Current market price

        Returns:
            Unrealized PNL
        """
        pos = self.get_position(token_id)
        pnl = pos.compute_unrealized_pnl(current_price)
        pos.unrealized_pnl = pnl
        return pnl

    def get_total_pnl(self, token_id: str) -> float:
        """
        Get total PNL (realized + unrealized) for a position.

        Args:
            token_id: Token ID

        Returns:
            Total PNL
        """
        pos = self.get_position(token_id)
        return pos.realized_pnl + pos.unrealized_pnl

    def add_fill_manually(self, fill: Fill) -> None:
        """
        Add a fill manually (e.g., from database on startup).

        Args:
            fill: Fill to add
        """
        self._process_fill(fill)

    def reset_position(self, token_id: str) -> None:
        """
        Reset position tracking for a token.

        Args:
            token_id: Token ID
        """
        if token_id in self._positions:
            del self._positions[token_id]
        if token_id in self._fills:
            del self._fills[token_id]

        logger.info("position_reset", token_id=token_id[:16] + "...")
