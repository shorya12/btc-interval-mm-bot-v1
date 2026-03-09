"""Dry run / paper trading adapter."""

import asyncio
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.polymarket_client.types import (
    OrderBook,
    OrderBookLevel,
    Order,
    Fill,
    OrderSide,
    OrderStatus,
)
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatedPosition:
    """Simulated position for paper trading."""

    token_id: str
    size: float = 0.0
    avg_entry_price: float | None = None
    realized_pnl: float = 0.0


@dataclass
class DryRunState:
    """State for dry run simulation."""

    balance: float
    positions: dict[str, SimulatedPosition] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    total_fees: float = 0.0


@dataclass
class FillStats:
    """Statistics about fill simulation for analysis."""
    
    total_attempts: int = 0
    fills_executed: int = 0
    fills_rejected_probability: int = 0
    fills_rejected_price: int = 0
    avg_fill_probability: float = 0.0


class DryRunAdapter:
    """
    Paper trading adapter that simulates order execution.

    Provides the same interface as PolymarketClient but simulates
    fills based on order book depth, spread, and realistic probability.
    
    Fill probability is calculated based on:
    - Order size relative to available liquidity at that price level
    - Distance from mid price (better prices fill faster)
    - Time decay (orders closer to market fill sooner)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fill_rate: float = 0.5,  # Base fill rate, now adjusted by volume
        fee_rate: float = 0.001,  # 0.1% fee
        slippage_bps: float = 5.0,  # 5 bps slippage
        realistic_fills: bool = True,  # Use volume-based fill probability
    ) -> None:
        """
        Initialize dry run adapter.

        Args:
            initial_balance: Starting paper balance
            fill_rate: Base probability of simulated fill (0-1), adjusted by volume
            fee_rate: Simulated fee rate
            slippage_bps: Simulated slippage in basis points
            realistic_fills: If True, use volume-based fill probability
        """
        self.base_fill_rate = fill_rate
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.realistic_fills = realistic_fills

        self.state = DryRunState(balance=initial_balance)
        self._last_books: dict[str, OrderBook] = {}
        self._fill_stats = FillStats()

        logger.info(
            "dry_run_initialized",
            initial_balance=initial_balance,
            fill_rate=fill_rate,
            realistic_fills=realistic_fills,
        )

    def set_order_book(self, book: OrderBook) -> None:
        """
        Update cached order book for price reference.

        Args:
            book: Latest order book
        """
        self._last_books[book.token_id] = book

    async def place_limit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> Order:
        """
        Simulate placing a limit order.

        May immediately fill based on fill_rate and book prices.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price
            size: Order size

        Returns:
            Simulated Order
        """
        order_id = f"dry_{uuid.uuid4().hex[:16]}"

        order = Order(
            id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            original_size=size,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        )

        self.state.orders[order_id] = order

        logger.info(
            "dry_run_order_placed",
            order_id=order_id[:12],
            side=side.value,
            price=round(price, 4),
            size=size,
        )

        # Simulate immediate fill based on fill rate
        await self._try_fill_order(order)

        return order

    def _calculate_fill_probability(self, order: Order, book: OrderBook | None) -> float:
        """
        Calculate realistic fill probability for MARKET-MAKING orders.

        Market makers PROVIDE liquidity with passive limit orders.
        Fill probability depends on how attractive our quote is:
        - If our BUY price > best bid, we're at top of queue = HIGH fill prob
        - If our SELL price < best ask, we're at top of queue = HIGH fill prob
        - Orders that IMPROVE the spread get filled by incoming market orders

        Args:
            order: The order to evaluate
            book: Current order book (if available)

        Returns:
            Fill probability between 0 and 1
        """
        if not self.realistic_fills or book is None:
            return self.base_fill_rate

        best_bid = book.best_bid_price
        best_ask = book.best_ask_price
        mid_price = book.mid_price or 0.5
        spread = book.spread or 0.02

        if best_bid is None or best_ask is None:
            return self.base_fill_rate * 0.5

        # For market makers, check if our order IMPROVES the current spread
        # Better prices = higher chance of getting hit by incoming market orders
        if order.side == OrderSide.BUY:
            # BUY order: compare to best bid
            # If our bid > best_bid, we're improving the market = high fill prob
            price_improvement = order.price - best_bid
            # Also check if we're crossing the spread (bid >= ask = immediate fill)
            if order.price >= best_ask:
                return 0.95  # Crossing spread = almost certain fill
        else:
            # SELL order: compare to best ask
            # If our ask < best_ask, we're improving the market = high fill prob
            price_improvement = best_ask - order.price
            # Also check if we're crossing the spread (ask <= bid = immediate fill)
            if order.price <= best_bid:
                return 0.95  # Crossing spread = almost certain fill

        # Normalize improvement by spread width
        # improvement_ratio > 0 means we're improving the spread
        # improvement_ratio = 1.0 means we've moved to the mid price
        improvement_ratio = price_improvement / (spread / 2) if spread > 0 else 0

        # Calculate fill probability based on how much we improve the spread
        if improvement_ratio >= 1.0:
            # At or past mid price = very high fill probability
            fill_prob = 0.7 + min(0.25, improvement_ratio * 0.1)
        elif improvement_ratio > 0:
            # Improving spread but not at mid = good fill probability
            # Linear from 0.4 (at best bid/ask) to 0.7 (at mid)
            fill_prob = 0.4 + improvement_ratio * 0.3
        elif improvement_ratio > -0.5:
            # Slightly behind best = moderate fill probability
            fill_prob = 0.2 + (improvement_ratio + 0.5) * 0.4
        else:
            # Far from best = low fill probability
            fill_prob = max(0.05, 0.2 + improvement_ratio * 0.2)

        # Factor in order size (smaller orders fill easier)
        # Use total book depth at our side for comparison
        if order.side == OrderSide.BUY:
            total_depth = sum(level.size for level in book.bids) if book.bids else 100
        else:
            total_depth = sum(level.size for level in book.asks) if book.asks else 100

        size_ratio = order.size / max(total_depth, 1)
        size_factor = 1.0 - min(0.5, size_ratio * 0.5)  # Small penalty for large orders

        # Apply base fill rate as a scaling factor
        fill_prob = fill_prob * self.base_fill_rate * 2 * size_factor  # 2x because base is 0.5

        # Clamp to reasonable range
        return min(0.90, max(0.05, fill_prob))

    async def _try_fill_order(self, order: Order) -> Fill | None:
        """
        Try to fill an order based on realistic fill probability and prices.

        Args:
            order: Order to try to fill

        Returns:
            Fill if filled, None otherwise
        """
        self._fill_stats.total_attempts += 1
        
        # Get reference book
        book = self._last_books.get(order.token_id)
        
        # Calculate fill probability based on volume/depth
        fill_prob = self._calculate_fill_probability(order, book)
        self._fill_stats.avg_fill_probability = (
            (self._fill_stats.avg_fill_probability * (self._fill_stats.total_attempts - 1) + fill_prob)
            / self._fill_stats.total_attempts
        )
        
        # Check fill probability
        if random.random() > fill_prob:
            self._fill_stats.fills_rejected_probability += 1
            return None

        # Determine fill price
        if book is None:
            # No book, use order price
            fill_price = order.price
        else:
            # Apply slippage
            slippage = self.slippage_bps / 10000
            if order.side == OrderSide.BUY:
                # Buy at slightly higher price
                ref_price = book.best_ask_price or order.price
                fill_price = min(order.price, ref_price * (1 + slippage))
            else:
                # Sell at slightly lower price
                ref_price = book.best_bid_price or order.price
                fill_price = max(order.price, ref_price * (1 - slippage))

        # Check if order would fill at this price
        if order.side == OrderSide.BUY and fill_price > order.price:
            self._fill_stats.fills_rejected_price += 1
            return None
        if order.side == OrderSide.SELL and fill_price < order.price:
            self._fill_stats.fills_rejected_price += 1
            return None

        # Create fill
        fill_id = f"fill_{uuid.uuid4().hex[:16]}"
        fee = order.size * fill_price * self.fee_rate

        fill_size = order.size  # Capture size before zeroing
        
        fill = Fill(
            id=fill_id,
            order_id=order.id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size=fill_size,
            fee=fee,
            timestamp=datetime.utcnow(),
        )

        # Update order status
        order.size = 0
        order.status = OrderStatus.MATCHED

        # Update state
        self.state.fills.append(fill)
        self.state.total_fees += fee
        self._update_position(fill)
        self._update_balance(fill)
        self._fill_stats.fills_executed += 1

        logger.info(
            "dry_run_fill",
            fill_id=fill_id[:12],
            order_id=order.id[:12],
            side=order.side.value,
            price=round(fill_price, 4),
            size=fill_size,
            fee=round(fee, 4),
        )

        return fill

    def _update_position(self, fill: Fill) -> None:
        """Update position from fill."""
        token_id = fill.token_id

        if token_id not in self.state.positions:
            self.state.positions[token_id] = SimulatedPosition(token_id=token_id)

        pos = self.state.positions[token_id]
        old_size = pos.size

        if fill.side == OrderSide.BUY:
            new_size = old_size + fill.size
            if old_size >= 0:
                # Increasing long
                old_cost = (pos.avg_entry_price or 0) * old_size
                new_cost = old_cost + fill.price * fill.size
                pos.avg_entry_price = new_cost / new_size if new_size > 0 else None
            else:
                # Closing short
                if new_size >= 0:
                    pnl = abs(old_size) * ((pos.avg_entry_price or 0) - fill.price)
                    pos.realized_pnl += pnl
                    pos.avg_entry_price = fill.price if new_size > 0 else None
        else:
            new_size = old_size - fill.size
            if old_size <= 0:
                # Increasing short
                old_cost = (pos.avg_entry_price or 0) * abs(old_size)
                new_cost = old_cost + fill.price * fill.size
                pos.avg_entry_price = new_cost / abs(new_size) if new_size < 0 else None
            else:
                # Closing long
                if new_size <= 0:
                    pnl = old_size * (fill.price - (pos.avg_entry_price or 0))
                    pos.realized_pnl += pnl
                    pos.avg_entry_price = fill.price if new_size < 0 else None

        pos.size = new_size

    def _update_balance(self, fill: Fill) -> None:
        """Update balance from fill."""
        value = fill.price * fill.size

        if fill.side == OrderSide.BUY:
            # Buying: reduce balance
            self.state.balance -= value + fill.fee
        else:
            # Selling: increase balance
            self.state.balance += value - fill.fee

    async def cancel_order(self, order_id: str) -> bool:
        """
        Simulate cancelling an order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled
        """
        if order_id in self.state.orders:
            order = self.state.orders[order_id]
            if order.status == OrderStatus.LIVE:
                order.status = OrderStatus.CANCELLED
                logger.info("dry_run_order_cancelled", order_id=order_id[:12])
                return True
        return False

    async def cancel_all_orders(self, token_id: str | None = None) -> int:
        """Cancel all open orders."""
        count = 0
        for order in self.state.orders.values():
            if order.status == OrderStatus.LIVE:
                if token_id is None or order.token_id == token_id:
                    order.status = OrderStatus.CANCELLED
                    count += 1
        return count

    def get_open_orders(self, token_id: str | None = None) -> list[Order]:
        """Get all open orders."""
        orders = []
        for order in self.state.orders.values():
            if order.status == OrderStatus.LIVE:
                if token_id is None or order.token_id == token_id:
                    orders.append(order)
        return orders

    def get_position(self, token_id: str) -> SimulatedPosition:
        """Get position for a token."""
        if token_id not in self.state.positions:
            return SimulatedPosition(token_id=token_id)
        return self.state.positions[token_id]

    def get_balance(self) -> float:
        """Get current balance."""
        return self.state.balance

    def get_equity(self, prices: dict[str, float]) -> float:
        """
        Get total equity (balance + position values).

        Args:
            prices: Current prices for each token

        Returns:
            Total equity
        """
        equity = self.state.balance

        for token_id, pos in self.state.positions.items():
            if pos.size != 0 and token_id in prices:
                equity += pos.size * prices[token_id]

        return equity

    def get_stats(self) -> dict[str, Any]:
        """Get paper trading statistics."""
        fill_rate = (
            self._fill_stats.fills_executed / self._fill_stats.total_attempts
            if self._fill_stats.total_attempts > 0 else 0
        )
        
        return {
            "balance": self.state.balance,
            "total_fills": len(self.state.fills),
            "total_fees": self.state.total_fees,
            "open_orders": len(self.get_open_orders()),
            "positions": {
                tid: {"size": p.size, "pnl": p.realized_pnl}
                for tid, p in self.state.positions.items()
                if p.size != 0
            },
            "fill_stats": {
                "total_attempts": self._fill_stats.total_attempts,
                "fills_executed": self._fill_stats.fills_executed,
                "actual_fill_rate": round(fill_rate, 3),
                "avg_fill_probability": round(self._fill_stats.avg_fill_probability, 3),
                "rejected_probability": self._fill_stats.fills_rejected_probability,
                "rejected_price": self._fill_stats.fills_rejected_price,
            },
        }
