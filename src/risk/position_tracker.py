"""Position and exposure tracking for live trading."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import defaultdict

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpenOrder:
    """Represents an open order being tracked."""
    
    order_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    notional: float  # price * size
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class Position:
    """Current position in a market."""
    
    token_id: str
    size: float = 0.0  # Positive = long
    avg_entry_price: float | None = None
    realized_pnl: float = 0.0
    last_fill_time: datetime | None = None


@dataclass
class ExposureSummary:
    """Summary of total exposure."""
    
    # Position-based
    total_position_value: float  # Sum of |position| * estimated_price
    net_position: float  # Net position across all markets
    
    # Order-based  
    open_buy_notional: float  # Total $ at risk in open buy orders
    open_sell_notional: float  # Total $ at risk in open sell orders
    total_open_notional: float  # Sum of all open order notional
    
    # Combined
    total_exposure: float  # Position value + open order notional
    exposure_pct: float  # As percentage of bankroll
    
    # Limits
    available_to_buy: float  # $ available for new buy orders
    available_to_sell: float  # $ available for new sell orders
    at_max_exposure: bool


class PositionTracker:
    """
    Tracks positions and open orders to manage total exposure.
    
    Unlike the existing InventoryManager which works on reported position_size,
    this tracker maintains state of:
    - Open orders that haven't filled
    - Positions from fills
    - Total exposure for risk management
    """
    
    def __init__(
        self,
        max_exposure_pct: float = 0.20,  # Max 20% of bankroll
        max_open_order_value: float = 50.0,  # Max $ in open orders per market
        max_position_value: float = 100.0,  # Max $ in position per market
    ) -> None:
        """
        Initialize position tracker.
        
        Args:
            max_exposure_pct: Maximum total exposure as % of bankroll
            max_open_order_value: Maximum open order value per market
            max_position_value: Maximum position value per market
        """
        self.max_exposure_pct = max_exposure_pct
        self.max_open_order_value = max_open_order_value
        self.max_position_value = max_position_value
        
        # State per market
        self._open_orders: dict[str, dict[str, OpenOrder]] = defaultdict(dict)  # token_id -> order_id -> order
        self._positions: dict[str, Position] = {}  # token_id -> position
        
        # Global state
        self._bankroll: float = 10000.0
        
    def set_bankroll(self, bankroll: float) -> None:
        """Update the current bankroll."""
        self._bankroll = max(0.0, bankroll)
        
    def record_order_placed(
        self,
        order_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> None:
        """
        Record an order being placed.
        
        Args:
            order_id: Order ID from exchange
            token_id: Market token ID
            side: "BUY" or "SELL"
            price: Order price
            size: Order size
        """
        notional = price * size
        order = OpenOrder(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            notional=notional,
        )
        self._open_orders[token_id][order_id] = order
        
        logger.debug(
            "order_tracked",
            order_id=order_id[:16] + "...",
            token_id=token_id[:16] + "...",
            side=side,
            notional=round(notional, 2),
        )
        
    def record_order_cancelled(self, order_id: str, token_id: str | None = None) -> None:
        """
        Record an order being cancelled.
        
        Args:
            order_id: Order ID
            token_id: Optional token ID (searches all if not provided)
        """
        if token_id and token_id in self._open_orders:
            if order_id in self._open_orders[token_id]:
                del self._open_orders[token_id][order_id]
                logger.debug("order_untracked", order_id=order_id[:16] + "...", reason="cancelled")
                return
                
        # Search all markets
        for tid, orders in list(self._open_orders.items()):
            if order_id in orders:
                del orders[order_id]
                logger.debug("order_untracked", order_id=order_id[:16] + "...", reason="cancelled")
                return
    
    def record_all_orders_cancelled(self, token_id: str | None = None) -> None:
        """
        Record all orders being cancelled for a market (or all markets).
        
        Args:
            token_id: Optional market to cancel orders for
        """
        if token_id:
            count = len(self._open_orders.get(token_id, {}))
            self._open_orders[token_id] = {}
            logger.debug("orders_cleared", token_id=token_id[:16] + "...", count=count)
        else:
            total = sum(len(orders) for orders in self._open_orders.values())
            self._open_orders.clear()
            logger.debug("all_orders_cleared", total_count=total)
                
    def record_fill(
        self,
        order_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> None:
        """
        Record an order fill.
        
        Updates position and removes filled portion from open orders.
        
        Args:
            order_id: Order ID
            token_id: Market token ID
            side: "BUY" or "SELL"  
            price: Fill price
            size: Fill size
        """
        # Update position
        if token_id not in self._positions:
            self._positions[token_id] = Position(token_id=token_id)
            
        pos = self._positions[token_id]
        
        if side == "BUY":
            # Buying increases position
            if pos.size >= 0:
                # Adding to long
                total_value = (pos.size * (pos.avg_entry_price or price)) + (size * price)
                pos.size += size
                pos.avg_entry_price = total_value / pos.size if pos.size > 0 else None
            else:
                # Reducing short
                # This is realized PnL
                old_size = abs(pos.size)
                fill_size = min(size, old_size)
                pnl = fill_size * ((pos.avg_entry_price or price) - price)
                pos.realized_pnl += pnl
                pos.size += size
                if pos.size > 0:
                    pos.avg_entry_price = price
                elif pos.size == 0:
                    pos.avg_entry_price = None
        else:
            # Selling decreases position
            if pos.size <= 0:
                # Adding to short (or opening short)
                if pos.size < 0:
                    total_value = abs(pos.size) * (pos.avg_entry_price or price) + size * price
                    pos.size -= size
                    pos.avg_entry_price = total_value / abs(pos.size)
                else:
                    pos.size = -size
                    pos.avg_entry_price = price
            else:
                # Reducing long - realized PnL
                fill_size = min(size, pos.size)
                pnl = fill_size * (price - (pos.avg_entry_price or price))
                pos.realized_pnl += pnl
                pos.size -= size
                if pos.size < 0:
                    pos.avg_entry_price = price
                elif pos.size == 0:
                    pos.avg_entry_price = None
                    
        pos.last_fill_time = datetime.utcnow()
        
        # Update/remove open order
        if token_id in self._open_orders and order_id in self._open_orders[token_id]:
            order = self._open_orders[token_id][order_id]
            order.size -= size
            order.notional = order.price * order.size
            if order.size <= 0:
                del self._open_orders[token_id][order_id]
                
        logger.info(
            "fill_recorded",
            token_id=token_id[:16] + "...",
            side=side,
            price=round(price, 4),
            size=round(size, 2),
            new_position=round(pos.size, 2),
            realized_pnl=round(pos.realized_pnl, 4),
        )
        
    def get_position(self, token_id: str) -> Position:
        """Get current position for a market."""
        if token_id not in self._positions:
            self._positions[token_id] = Position(token_id=token_id)
        return self._positions[token_id]
    
    def get_open_orders(self, token_id: str) -> list[OpenOrder]:
        """Get open orders for a market."""
        return list(self._open_orders.get(token_id, {}).values())
    
    def get_open_order_notional(self, token_id: str, side: str | None = None) -> float:
        """
        Get total notional value of open orders for a market.
        
        Args:
            token_id: Market token ID
            side: Optional filter by side ("BUY" or "SELL")
            
        Returns:
            Total notional value
        """
        orders = self._open_orders.get(token_id, {}).values()
        if side:
            orders = [o for o in orders if o.side == side]
        return sum(o.notional for o in orders)
    
    def get_exposure_summary(self, current_prices: dict[str, float] | None = None) -> ExposureSummary:
        """
        Get summary of total exposure.
        
        Args:
            current_prices: Optional map of token_id -> current price for position valuation
            
        Returns:
            ExposureSummary
        """
        # Calculate position value
        total_position_value = 0.0
        net_position = 0.0
        
        for token_id, pos in self._positions.items():
            price = 0.5  # Default
            if current_prices and token_id in current_prices:
                price = current_prices[token_id]
            position_value = abs(pos.size) * price
            total_position_value += position_value
            net_position += pos.size
            
        # Calculate open order notional
        open_buy_notional = 0.0
        open_sell_notional = 0.0
        
        for token_id, orders in self._open_orders.items():
            for order in orders.values():
                if order.side == "BUY":
                    open_buy_notional += order.notional
                else:
                    open_sell_notional += order.notional
                    
        total_open_notional = open_buy_notional + open_sell_notional
        total_exposure = total_position_value + total_open_notional
        
        # Calculate percentages and limits
        exposure_pct = total_exposure / self._bankroll if self._bankroll > 0 else 0.0
        max_exposure = self._bankroll * self.max_exposure_pct
        
        available_to_buy = max(0.0, max_exposure - total_exposure)
        available_to_sell = max(0.0, max_exposure - total_exposure)
        at_max_exposure = exposure_pct >= self.max_exposure_pct
        
        return ExposureSummary(
            total_position_value=total_position_value,
            net_position=net_position,
            open_buy_notional=open_buy_notional,
            open_sell_notional=open_sell_notional,
            total_open_notional=total_open_notional,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            available_to_buy=available_to_buy,
            available_to_sell=available_to_sell,
            at_max_exposure=at_max_exposure,
        )
    
    def can_place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> tuple[bool, str]:
        """
        Check if an order can be placed within exposure limits.
        
        Args:
            token_id: Market token ID
            side: "BUY" or "SELL"
            price: Order price
            size: Order size
            
        Returns:
            (can_place, reason)
        """
        notional = price * size
        
        # Check per-market open order limit
        current_open = self.get_open_order_notional(token_id)
        if current_open + notional > self.max_open_order_value:
            return False, f"would exceed max open order value (${self.max_open_order_value})"
            
        # Check per-market position limit
        pos = self.get_position(token_id)
        current_position_value = abs(pos.size) * price
        new_size = pos.size + size if side == "BUY" else pos.size - size
        new_position_value = abs(new_size) * price
        if new_position_value > self.max_position_value:
            return False, f"would exceed max position value (${self.max_position_value})"
            
        # Check total exposure limit
        exposure = self.get_exposure_summary()
        if exposure.total_exposure + notional > self._bankroll * self.max_exposure_pct:
            return False, f"would exceed max total exposure ({self.max_exposure_pct*100}%)"
            
        return True, "ok"
    
    def get_allowed_order_size(
        self,
        token_id: str,
        side: str,
        price: float,
        desired_size: float,
    ) -> float:
        """
        Get the maximum allowed order size respecting limits.
        
        Args:
            token_id: Market token ID
            side: "BUY" or "SELL"
            price: Order price
            desired_size: Desired order size
            
        Returns:
            Maximum allowed size (may be 0, may be less than desired)
        """
        if price <= 0 or desired_size <= 0:
            return 0.0
            
        # Check per-market open order limit
        current_open = self.get_open_order_notional(token_id)
        available_notional = max(0.0, self.max_open_order_value - current_open)
        max_size_open_limit = available_notional / price
        
        # Check per-market position limit  
        pos = self.get_position(token_id)
        current_value = abs(pos.size) * price
        available_position_value = max(0.0, self.max_position_value - current_value)
        max_size_position_limit = available_position_value / price
        
        # Check total exposure limit
        exposure = self.get_exposure_summary()
        max_total_exposure = self._bankroll * self.max_exposure_pct
        available_total = max(0.0, max_total_exposure - exposure.total_exposure)
        max_size_total_limit = available_total / price
        
        # Take the most restrictive limit
        max_allowed = min(
            desired_size,
            max_size_open_limit,
            max_size_position_limit, 
            max_size_total_limit,
        )
        
        if max_allowed < desired_size:
            logger.debug(
                "order_size_reduced",
                token_id=token_id[:16] + "...",
                side=side,
                desired=round(desired_size, 2),
                allowed=round(max_allowed, 2),
                open_limit=round(max_size_open_limit, 2),
                position_limit=round(max_size_position_limit, 2),
                total_limit=round(max_size_total_limit, 2),
            )
            
        return max(0.0, max_allowed)
    
    def get_status(self) -> dict[str, Any]:
        """Get current tracker status for logging."""
        exposure = self.get_exposure_summary()
        
        positions = {}
        for token_id, pos in self._positions.items():
            if pos.size != 0:
                positions[token_id[:16]] = {
                    "size": round(pos.size, 2),
                    "avg_entry": round(pos.avg_entry_price, 4) if pos.avg_entry_price else None,
                    "realized_pnl": round(pos.realized_pnl, 4),
                }
                
        open_orders = {}
        for token_id, orders in self._open_orders.items():
            if orders:
                open_orders[token_id[:16]] = {
                    "count": len(orders),
                    "buy_notional": round(sum(o.notional for o in orders.values() if o.side == "BUY"), 2),
                    "sell_notional": round(sum(o.notional for o in orders.values() if o.side == "SELL"), 2),
                }
                
        return {
            "bankroll": round(self._bankroll, 2),
            "total_exposure": round(exposure.total_exposure, 2),
            "exposure_pct": round(exposure.exposure_pct * 100, 1),
            "open_buy_notional": round(exposure.open_buy_notional, 2),
            "open_sell_notional": round(exposure.open_sell_notional, 2),
            "positions": positions,
            "open_orders": open_orders,
            "at_max": exposure.at_max_exposure,
        }
