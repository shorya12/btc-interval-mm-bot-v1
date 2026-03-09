"""Repository pattern for database CRUD operations."""

from datetime import datetime, timedelta
from typing import Any

from .database import Database
from .models import (
    Order,
    OrderStatus,
    Fill,
    Position,
    PnlSnapshot,
    CryptoPrice,
    OptionsSignal,
    EventLog,
    EventSeverity,
    Side,
)
from src.common.logging import get_logger

logger = get_logger(__name__)


class Repository:
    """Repository for all database operations."""

    def __init__(self, db: Database) -> None:
        """
        Initialize repository with database connection.

        Args:
            db: Database instance
        """
        self.db = db

    # ==================== Orders ====================

    async def create_order(self, order: Order) -> Order:
        """Insert a new order."""
        data = order.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO orders ({columns}) VALUES ({placeholders})"
        await self.db.execute(sql, data)
        await self.db.commit()
        logger.debug("order_created", order_id=order.id, token_id=order.token_id)
        return order

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM orders WHERE id = :id",
            {"id": order_id},
        )
        return Order.from_row(row) if row else None

    async def update_order(
        self,
        order_id: str,
        status: OrderStatus | None = None,
        filled_size: float | None = None,
        avg_fill_price: float | None = None,
        cancelled_at: datetime | None = None,
    ) -> None:
        """Update order fields."""
        updates: dict[str, Any] = {"updated_at": datetime.utcnow().isoformat()}
        if status is not None:
            updates["status"] = status.value
        if filled_size is not None:
            updates["filled_size"] = filled_size
        if avg_fill_price is not None:
            updates["avg_fill_price"] = avg_fill_price
        if cancelled_at is not None:
            updates["cancelled_at"] = cancelled_at.isoformat()

        set_clause = ", ".join(f"{k} = :{k}" for k in updates.keys())
        updates["id"] = order_id
        sql = f"UPDATE orders SET {set_clause} WHERE id = :id"
        await self.db.execute(sql, updates)
        await self.db.commit()
        logger.debug("order_updated", order_id=order_id, updates=list(updates.keys()))

    async def get_open_orders(self, token_id: str | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by token."""
        if token_id:
            rows = await self.db.fetch_all(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'OPEN', 'PARTIALLY_FILLED') AND token_id = :token_id ORDER BY created_at",
                {"token_id": token_id},
            )
        else:
            rows = await self.db.fetch_all(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'OPEN', 'PARTIALLY_FILLED') ORDER BY created_at"
            )
        return [Order.from_row(row) for row in rows]

    async def get_orders_by_status(self, status: OrderStatus) -> list[Order]:
        """Get all orders with a specific status."""
        rows = await self.db.fetch_all(
            "SELECT * FROM orders WHERE status = :status ORDER BY created_at DESC",
            {"status": status.value},
        )
        return [Order.from_row(row) for row in rows]

    async def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Get most recent orders."""
        rows = await self.db.fetch_all(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT :limit",
            {"limit": limit},
        )
        return [Order.from_row(row) for row in rows]

    # ==================== Fills ====================

    async def create_fill(self, fill: Fill) -> Fill:
        """Insert a new fill."""
        data = fill.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO fills ({columns}) VALUES ({placeholders})"
        await self.db.execute(sql, data)
        await self.db.commit()
        logger.debug("fill_created", fill_id=fill.id, order_id=fill.order_id)
        return fill

    async def get_fill(self, fill_id: str) -> Fill | None:
        """Get fill by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM fills WHERE id = :id",
            {"id": fill_id},
        )
        return Fill.from_row(row) if row else None

    async def get_fills_by_order(self, order_id: str) -> list[Fill]:
        """Get all fills for an order."""
        rows = await self.db.fetch_all(
            "SELECT * FROM fills WHERE order_id = :order_id ORDER BY created_at",
            {"order_id": order_id},
        )
        return [Fill.from_row(row) for row in rows]

    async def get_fills_since(self, since: datetime) -> list[Fill]:
        """Get all fills since a timestamp."""
        rows = await self.db.fetch_all(
            "SELECT * FROM fills WHERE created_at >= :since ORDER BY created_at",
            {"since": since.isoformat()},
        )
        return [Fill.from_row(row) for row in rows]

    async def get_recent_fills(self, token_id: str | None = None, limit: int = 100) -> list[Fill]:
        """Get most recent fills."""
        if token_id:
            rows = await self.db.fetch_all(
                "SELECT * FROM fills WHERE token_id = :token_id ORDER BY created_at DESC LIMIT :limit",
                {"token_id": token_id, "limit": limit},
            )
        else:
            rows = await self.db.fetch_all(
                "SELECT * FROM fills ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )
        return [Fill.from_row(row) for row in rows]

    # ==================== Positions ====================

    async def upsert_position(self, position: Position) -> Position:
        """Insert or update a position."""
        data = position.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        update_clause = ", ".join(f"{k} = excluded.{k}" for k in data.keys() if k != "token_id")

        sql = f"""
        INSERT INTO positions ({columns}) VALUES ({placeholders})
        ON CONFLICT (token_id) DO UPDATE SET {update_clause}
        """
        await self.db.execute(sql, data)
        await self.db.commit()
        logger.debug("position_upserted", token_id=position.token_id, size=position.size)
        return position

    async def get_position(self, token_id: str) -> Position | None:
        """Get position by token ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM positions WHERE token_id = :token_id",
            {"token_id": token_id},
        )
        return Position.from_row(row) if row else None

    async def get_all_positions(self) -> list[Position]:
        """Get all positions."""
        rows = await self.db.fetch_all("SELECT * FROM positions ORDER BY token_id")
        return [Position.from_row(row) for row in rows]

    async def get_nonzero_positions(self) -> list[Position]:
        """Get all positions with non-zero size."""
        rows = await self.db.fetch_all(
            "SELECT * FROM positions WHERE size != 0 ORDER BY token_id"
        )
        return [Position.from_row(row) for row in rows]

    async def update_position_from_fill(
        self,
        token_id: str,
        fill: Fill,
        current_price: float,
    ) -> Position:
        """Update position based on a fill."""
        position = await self.get_position(token_id)
        if position is None:
            position = Position(token_id=token_id, condition_id=fill.metadata.get("condition_id"))

        old_size = position.size
        fill_value = fill.price * fill.size

        if fill.side == Side.BUY:
            # Buying increases position
            new_size = old_size + fill.size
            position.total_bought += fill_value

            # Update average entry price
            if old_size >= 0:
                # Adding to long or opening long
                old_cost = (position.avg_entry_price or 0) * old_size
                position.avg_entry_price = (old_cost + fill_value) / new_size if new_size > 0 else None
            else:
                # Closing short
                if new_size >= 0:
                    # Fully closed or flipped to long
                    realized = (-old_size) * ((position.avg_entry_price or 0) - fill.price)
                    position.realized_pnl += realized
                    position.avg_entry_price = fill.price if new_size > 0 else None
        else:
            # Selling decreases position
            new_size = old_size - fill.size
            position.total_sold += fill_value

            if old_size <= 0:
                # Adding to short or opening short
                old_cost = (position.avg_entry_price or 0) * abs(old_size)
                position.avg_entry_price = (old_cost + fill_value) / abs(new_size) if new_size < 0 else None
            else:
                # Closing long
                if new_size <= 0:
                    realized = old_size * (fill.price - (position.avg_entry_price or 0))
                    position.realized_pnl += realized
                    position.avg_entry_price = fill.price if new_size < 0 else None

        position.size = new_size
        position.num_trades += 1
        position.updated_at = datetime.utcnow()

        # Update unrealized PNL
        if position.size != 0 and position.avg_entry_price is not None:
            if position.size > 0:
                position.unrealized_pnl = position.size * (current_price - position.avg_entry_price)
            else:
                position.unrealized_pnl = abs(position.size) * (position.avg_entry_price - current_price)
        else:
            position.unrealized_pnl = 0

        return await self.upsert_position(position)

    # ==================== PNL Snapshots ====================

    async def create_pnl_snapshot(self, snapshot: PnlSnapshot) -> PnlSnapshot:
        """Insert a PNL snapshot."""
        data = snapshot.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO pnl_snapshots ({columns}) VALUES ({placeholders})"
        cursor = await self.db.execute(sql, data)
        await self.db.commit()
        snapshot.id = cursor.lastrowid
        return snapshot

    async def get_recent_snapshots(self, limit: int = 100) -> list[PnlSnapshot]:
        """Get most recent PNL snapshots."""
        rows = await self.db.fetch_all(
            "SELECT * FROM pnl_snapshots ORDER BY timestamp DESC LIMIT :limit",
            {"limit": limit},
        )
        return [PnlSnapshot.from_row(row) for row in rows]

    async def get_snapshots_since(self, since: datetime) -> list[PnlSnapshot]:
        """Get snapshots since a timestamp."""
        rows = await self.db.fetch_all(
            "SELECT * FROM pnl_snapshots WHERE timestamp >= :since ORDER BY timestamp",
            {"since": since.isoformat()},
        )
        return [PnlSnapshot.from_row(row) for row in rows]

    # ==================== Crypto Prices ====================

    async def create_crypto_price(self, price: CryptoPrice) -> CryptoPrice:
        """Insert a crypto price record."""
        data = price.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO crypto_prices ({columns}) VALUES ({placeholders})"
        cursor = await self.db.execute(sql, data)
        await self.db.commit()
        price.id = cursor.lastrowid
        return price

    async def get_recent_crypto_prices(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[CryptoPrice]:
        """Get most recent prices for a symbol."""
        rows = await self.db.fetch_all(
            "SELECT * FROM crypto_prices WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit",
            {"symbol": symbol, "limit": limit},
        )
        return [CryptoPrice.from_row(row) for row in rows]

    async def get_crypto_prices_window(
        self,
        symbol: str,
        window_seconds: int,
    ) -> list[CryptoPrice]:
        """Get prices within a time window."""
        cutoff = (datetime.utcnow() - timedelta(seconds=window_seconds)).isoformat()
        rows = await self.db.fetch_all(
            "SELECT * FROM crypto_prices WHERE symbol = :symbol AND timestamp >= :cutoff ORDER BY timestamp",
            {"symbol": symbol, "cutoff": cutoff},
        )
        return [CryptoPrice.from_row(row) for row in rows]

    async def cleanup_old_crypto_prices(self, keep_hours: int = 24) -> int:
        """Delete crypto prices older than specified hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=keep_hours)).isoformat()
        cursor = await self.db.execute(
            "DELETE FROM crypto_prices WHERE timestamp < :cutoff",
            {"cutoff": cutoff},
        )
        await self.db.commit()
        return cursor.rowcount

    # ==================== Event Log ====================

    async def log_event(
        self,
        event_type: str,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        data: dict[str, Any] | None = None,
        token_id: str | None = None,
        order_id: str | None = None,
    ) -> EventLog:
        """Create an event log entry."""
        event = EventLog(
            id=None,
            event_type=event_type,
            message=message,
            severity=severity,
            data=data or {},
            token_id=token_id,
            order_id=order_id,
        )
        event_data = event.to_dict()
        columns = ", ".join(event_data.keys())
        placeholders = ", ".join(f":{k}" for k in event_data.keys())
        sql = f"INSERT INTO event_log ({columns}) VALUES ({placeholders})"
        cursor = await self.db.execute(sql, event_data)
        await self.db.commit()
        event.id = cursor.lastrowid
        return event

    async def get_recent_events(
        self,
        event_type: str | None = None,
        severity: EventSeverity | None = None,
        limit: int = 100,
    ) -> list[EventLog]:
        """Get recent events with optional filters."""
        conditions = []
        params: dict[str, Any] = {"limit": limit}

        if event_type:
            conditions.append("event_type = :event_type")
            params["event_type"] = event_type
        if severity:
            conditions.append("severity = :severity")
            params["severity"] = severity.value

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM event_log {where_clause} ORDER BY timestamp DESC LIMIT :limit"
        rows = await self.db.fetch_all(sql, params)
        return [EventLog.from_row(row) for row in rows]

    async def cleanup_old_events(self, keep_days: int = 7) -> int:
        """Delete events older than specified days."""
        cutoff = (datetime.utcnow() - timedelta(days=keep_days)).isoformat()
        cursor = await self.db.execute(
            "DELETE FROM event_log WHERE timestamp < :cutoff",
            {"cutoff": cutoff},
        )
        await self.db.commit()
        return cursor.rowcount

    # ==================== Options Signals ====================

    async def create_options_signal(self, signal: OptionsSignal) -> OptionsSignal:
        """Insert an options signal record (INSERT OR IGNORE for safe re-backfill)."""
        data = signal.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT OR IGNORE INTO options_signals ({columns}) VALUES ({placeholders})"
        cursor = await self.db.execute(sql, data)
        await self.db.commit()
        signal.id = cursor.lastrowid
        return signal

    async def get_options_signals(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[OptionsSignal]:
        """Fetch options signals for a symbol in the given time range."""
        rows = await self.db.fetch_all(
            "SELECT * FROM options_signals WHERE symbol = :symbol AND timestamp >= :start AND timestamp < :end ORDER BY timestamp",
            {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()},
        )
        return [OptionsSignal.from_row(row) for row in rows]
