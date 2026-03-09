"""Data models for persistence layer."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class Side(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class EventSeverity(str, Enum):
    """Event severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Order:
    """Order record."""

    id: str
    token_id: str
    side: Side
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    condition_id: str | None = None
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    cancelled_at: datetime | None = None
    strategy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Order":
        """Create Order from database row."""
        return cls(
            id=row["id"],
            token_id=row["token_id"],
            side=Side(row["side"]),
            price=row["price"],
            size=row["size"],
            status=OrderStatus(row["status"]),
            condition_id=row.get("condition_id"),
            filled_size=row.get("filled_size", 0.0),
            avg_fill_price=row.get("avg_fill_price"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.utcnow(),
            cancelled_at=datetime.fromisoformat(row["cancelled_at"]) if row.get("cancelled_at") else None,
            strategy_id=row.get("strategy_id"),
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "token_id": self.token_id,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "status": self.status.value,
            "condition_id": self.condition_id,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "strategy_id": self.strategy_id,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }


@dataclass
class Fill:
    """Fill/execution record."""

    id: str
    order_id: str
    token_id: str
    side: Side
    price: float
    size: float
    fee: float = 0.0
    realized_pnl: float | None = None
    position_after: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    tx_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Fill":
        """Create Fill from database row."""
        return cls(
            id=row["id"],
            order_id=row["order_id"],
            token_id=row["token_id"],
            side=Side(row["side"]),
            price=row["price"],
            size=row["size"],
            fee=row.get("fee", 0.0),
            realized_pnl=row.get("realized_pnl"),
            position_after=row.get("position_after"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.utcnow(),
            tx_hash=row.get("tx_hash"),
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "token_id": self.token_id,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "fee": self.fee,
            "realized_pnl": self.realized_pnl,
            "position_after": self.position_after,
            "created_at": self.created_at.isoformat(),
            "tx_hash": self.tx_hash,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }


@dataclass
class Position:
    """Current position state for a token."""

    token_id: str
    size: float = 0.0  # Positive = long, negative = short
    avg_entry_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_bought: float = 0.0
    total_sold: float = 0.0
    num_trades: int = 0
    condition_id: str | None = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Position":
        """Create Position from database row."""
        return cls(
            token_id=row["token_id"],
            size=row.get("size", 0.0),
            avg_entry_price=row.get("avg_entry_price"),
            realized_pnl=row.get("realized_pnl", 0.0),
            unrealized_pnl=row.get("unrealized_pnl", 0.0),
            total_bought=row.get("total_bought", 0.0),
            total_sold=row.get("total_sold", 0.0),
            num_trades=row.get("num_trades", 0),
            condition_id=row.get("condition_id"),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.utcnow(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "token_id": self.token_id,
            "size": self.size,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_bought": self.total_bought,
            "total_sold": self.total_sold,
            "num_trades": self.num_trades,
            "condition_id": self.condition_id,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class PnlSnapshot:
    """Periodic PNL/equity snapshot."""

    id: int | None
    total_equity: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    position_value: float
    cash_balance: float
    num_open_orders: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    market_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PnlSnapshot":
        """Create PnlSnapshot from database row."""
        return cls(
            id=row.get("id"),
            total_equity=row["total_equity"],
            total_realized_pnl=row["total_realized_pnl"],
            total_unrealized_pnl=row["total_unrealized_pnl"],
            position_value=row["position_value"],
            cash_balance=row["cash_balance"],
            num_open_orders=row.get("num_open_orders", 0),
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.utcnow(),
            market_data=json.loads(row["market_data"]) if row.get("market_data") else {},
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "total_equity": self.total_equity,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "position_value": self.position_value,
            "cash_balance": self.cash_balance,
            "num_open_orders": self.num_open_orders,
            "timestamp": self.timestamp.isoformat(),
            "market_data": json.dumps(self.market_data) if self.market_data else None,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }


@dataclass
class CryptoPrice:
    """Crypto price record for lag signal."""

    id: int | None
    symbol: str
    price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "ccxt"
    volume_24h: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CryptoPrice":
        """Create CryptoPrice from database row."""
        return cls(
            id=row.get("id"),
            symbol=row["symbol"],
            price=row["price"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.utcnow(),
            source=row.get("source", "ccxt"),
            volume_24h=row.get("volume_24h"),
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "volume_24h": self.volume_24h,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }


@dataclass
class OptionsSignal:
    """Options market signal record (Deribit DVOL + put/call ratio)."""

    id: int | None
    timestamp: datetime
    symbol: str = "BTC"
    dvol: float | None = None
    put_call_ratio: float | None = None
    source: str = "deribit"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OptionsSignal":
        """Create OptionsSignal from database row."""
        return cls(
            id=row.get("id"),
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.utcnow(),
            symbol=row.get("symbol", "BTC"),
            dvol=row.get("dvol"),
            put_call_ratio=row.get("put_call_ratio"),
            source=row.get("source", "deribit"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "dvol": self.dvol,
            "put_call_ratio": self.put_call_ratio,
            "source": self.source,
        }


@dataclass
class EventLog:
    """Event log entry for audit trail."""

    id: int | None
    event_type: str
    message: str
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)
    token_id: str | None = None
    order_id: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "EventLog":
        """Create EventLog from database row."""
        return cls(
            id=row.get("id"),
            event_type=row["event_type"],
            message=row["message"],
            severity=EventSeverity(row.get("severity", "INFO")),
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.utcnow(),
            data=json.loads(row["data"]) if row.get("data") else {},
            token_id=row.get("token_id"),
            order_id=row.get("order_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "event_type": self.event_type,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "data": json.dumps(self.data) if self.data else None,
            "token_id": self.token_id,
            "order_id": self.order_id,
        }
