"""Order management and lifecycle tracking."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.common.logging import get_logger
from .types import Order, OrderSide, OrderStatus
from .client import PolymarketClient

logger = get_logger(__name__)


@dataclass
class OrderState:
    """Extended order state for management."""

    order: Order
    created_at: datetime
    last_update: datetime
    cancel_attempts: int = 0
    reprice_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get order age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def is_stale(self) -> bool:
        """Check if order needs repricing (default 30s)."""
        return self.age_seconds > 30


class OrderManager:
    """
    Manages order lifecycle.

    Handles:
    - Order placement with validation
    - Order cancellation with cooldown
    - Stale order detection
    - Order repricing logic
    """

    def __init__(
        self,
        client: PolymarketClient,
        cancel_cooldown_seconds: float = 2.0,
        reprice_threshold_ticks: int = 2,
        order_lifetime_seconds: float = 30.0,
    ) -> None:
        """
        Initialize order manager.

        Args:
            client: Polymarket client
            cancel_cooldown_seconds: Minimum time between cancels for same order
            reprice_threshold_ticks: Price ticks before repricing
            order_lifetime_seconds: Maximum order age before forced cancel
        """
        self.client = client
        self.cancel_cooldown_seconds = cancel_cooldown_seconds
        self.reprice_threshold_ticks = reprice_threshold_ticks
        self.order_lifetime_seconds = order_lifetime_seconds

        self._orders: dict[str, OrderState] = {}
        self._last_cancel_time: dict[str, datetime] = {}
        self._tick_size = 0.001  # 0.1% tick size for Polymarket

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        metadata: dict[str, Any] | None = None,
    ) -> Order | None:
        """
        Place a new order with validation.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price
            size: Order size
            metadata: Optional metadata to track

        Returns:
            Order if successful, None if failed
        """
        # Validate price bounds
        if not 0.001 <= price <= 0.999:
            logger.warning("price_out_of_bounds", price=price)
            return None

        # Round price to tick size
        price = round(price / self._tick_size) * self._tick_size

        try:
            order = await self.client.place_limit_order(
                token_id=token_id,
                side=side,
                price=price,
                size=size,
            )

            # Track order state
            self._orders[order.id] = OrderState(
                order=order,
                created_at=datetime.utcnow(),
                last_update=datetime.utcnow(),
                metadata=metadata or {},
            )

            logger.debug(
                "order_placed",
                order_id=order.id[:16] + "...",
                side=side.value,
                price=round(price, 4),
                size=size,
            )

            return order

        except Exception as e:
            logger.error("place_order_failed", error=str(e))
            return None

    async def cancel_order(self, order_id: str, force: bool = False) -> bool:
        """
        Cancel an order with cooldown check.

        Args:
            order_id: Order to cancel
            force: Skip cooldown check

        Returns:
            True if cancelled, False otherwise
        """
        if order_id not in self._orders:
            # Order not tracked, try to cancel anyway
            return await self.client.cancel_order(order_id)

        state = self._orders[order_id]

        # Check cooldown
        if not force and order_id in self._last_cancel_time:
            elapsed = (datetime.utcnow() - self._last_cancel_time[order_id]).total_seconds()
            if elapsed < self.cancel_cooldown_seconds:
                logger.debug(
                    "cancel_cooldown",
                    order_id=order_id[:16] + "...",
                    remaining=self.cancel_cooldown_seconds - elapsed,
                )
                return False

        # Attempt cancel
        success = await self.client.cancel_order(order_id)

        if success:
            state.order.status = OrderStatus.CANCELLED
            state.last_update = datetime.utcnow()
            state.cancel_attempts += 1
            self._last_cancel_time[order_id] = datetime.utcnow()

            logger.debug("order_cancelled", order_id=order_id[:16] + "...")
        else:
            state.cancel_attempts += 1
            logger.warning(
                "cancel_failed",
                order_id=order_id[:16] + "...",
                attempts=state.cancel_attempts,
            )

        return success

    async def cancel_all(self, token_id: str | None = None) -> int:
        """
        Cancel all orders, optionally filtered by token.

        Args:
            token_id: Optional token to filter

        Returns:
            Number of orders cancelled
        """
        count = await self.client.cancel_all_orders(token_id)

        # Update local state
        for order_id, state in self._orders.items():
            if state.order.status == OrderStatus.LIVE:
                if token_id is None or state.order.token_id == token_id:
                    state.order.status = OrderStatus.CANCELLED
                    state.last_update = datetime.utcnow()

        return count

    def get_open_orders(self, token_id: str | None = None) -> list[Order]:
        """
        Get all open orders from local cache.

        Args:
            token_id: Optional token to filter

        Returns:
            List of open orders
        """
        orders = []
        for state in self._orders.values():
            if state.order.status == OrderStatus.LIVE:
                if token_id is None or state.order.token_id == token_id:
                    orders.append(state.order)
        return orders

    def get_stale_orders(self) -> list[Order]:
        """
        Get orders that have exceeded their lifetime.

        Returns:
            List of stale orders
        """
        stale = []
        now = datetime.utcnow()

        for state in self._orders.values():
            if state.order.status != OrderStatus.LIVE:
                continue

            age = (now - state.created_at).total_seconds()
            if age > self.order_lifetime_seconds:
                stale.append(state.order)

        return stale

    def needs_reprice(
        self,
        order: Order,
        new_price: float,
    ) -> bool:
        """
        Check if order needs repricing.

        Args:
            order: Current order
            new_price: Proposed new price

        Returns:
            True if should reprice
        """
        price_diff = abs(order.price - new_price)
        ticks_diff = price_diff / self._tick_size

        if ticks_diff >= self.reprice_threshold_ticks:
            return True

        # Also reprice if order is stale
        if order.id in self._orders:
            state = self._orders[order.id]
            if state.age_seconds > self.order_lifetime_seconds:
                return True

        return False

    async def reprice_order(
        self,
        order_id: str,
        new_price: float,
        new_size: float | None = None,
    ) -> Order | None:
        """
        Cancel and replace an order at a new price.

        Args:
            order_id: Order to reprice
            new_price: New price
            new_size: New size (or keep existing)

        Returns:
            New order if successful, None otherwise
        """
        if order_id not in self._orders:
            return None

        state = self._orders[order_id]
        old_order = state.order

        # Cancel old order
        cancelled = await self.cancel_order(order_id, force=True)
        if not cancelled:
            logger.warning("reprice_cancel_failed", order_id=order_id[:16] + "...")
            # Try to place new order anyway

        # Place new order
        size = new_size if new_size is not None else old_order.size
        new_order = await self.place_order(
            token_id=old_order.token_id,
            side=old_order.side,
            price=new_price,
            size=size,
            metadata=state.metadata,
        )

        if new_order:
            # Track reprice count
            if new_order.id in self._orders:
                self._orders[new_order.id].reprice_count = state.reprice_count + 1

            logger.info(
                "order_repriced",
                old_order_id=order_id[:16] + "...",
                new_order_id=new_order.id[:16] + "...",
                old_price=round(old_order.price, 4),
                new_price=round(new_price, 4),
            )

        return new_order

    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """
        Remove old cancelled/filled orders from tracking.

        Args:
            max_age_hours: Maximum age to keep

        Returns:
            Number of orders removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for order_id, state in self._orders.items():
            if state.order.status != OrderStatus.LIVE:
                if state.last_update < cutoff:
                    to_remove.append(order_id)

        for order_id in to_remove:
            del self._orders[order_id]
            if order_id in self._last_cancel_time:
                del self._last_cancel_time[order_id]

        return len(to_remove)
