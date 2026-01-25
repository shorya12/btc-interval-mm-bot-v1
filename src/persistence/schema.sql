-- Polybot Database Schema
-- SQLite DDL statements

-- Orders table: tracks all orders placed
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    token_id TEXT NOT NULL,
    condition_id TEXT,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price REAL NOT NULL CHECK (price >= 0 AND price <= 1),
    size REAL NOT NULL CHECK (size > 0),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'OPEN', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'EXPIRED', 'REJECTED')),
    filled_size REAL NOT NULL DEFAULT 0,
    avg_fill_price REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    cancelled_at TEXT,
    strategy_id TEXT,
    metadata TEXT  -- JSON blob for strategy-specific data
);

CREATE INDEX IF NOT EXISTS idx_orders_token_id ON orders(token_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);

-- Fills table: execution records
CREATE TABLE IF NOT EXISTS fills (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    token_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price REAL NOT NULL,
    size REAL NOT NULL CHECK (size > 0),
    fee REAL NOT NULL DEFAULT 0,
    realized_pnl REAL,
    position_after REAL,  -- Position size after this fill
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    tx_hash TEXT,
    metadata TEXT  -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_token_id ON fills(token_id);
CREATE INDEX IF NOT EXISTS idx_fills_created_at ON fills(created_at);

-- Positions table: current position state per token
CREATE TABLE IF NOT EXISTS positions (
    token_id TEXT PRIMARY KEY,
    condition_id TEXT,
    size REAL NOT NULL DEFAULT 0,  -- Positive = long, negative = short
    avg_entry_price REAL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    total_bought REAL NOT NULL DEFAULT 0,
    total_sold REAL NOT NULL DEFAULT 0,
    num_trades INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- PNL snapshots: periodic equity snapshots for analysis
CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    total_equity REAL NOT NULL,
    total_realized_pnl REAL NOT NULL,
    total_unrealized_pnl REAL NOT NULL,
    position_value REAL NOT NULL,
    cash_balance REAL NOT NULL,
    num_open_orders INTEGER NOT NULL DEFAULT 0,
    market_data TEXT,  -- JSON blob with mid prices, spreads, etc.
    metadata TEXT  -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_timestamp ON pnl_snapshots(timestamp);

-- Crypto prices: lag signal price history
CREATE TABLE IF NOT EXISTS crypto_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,  -- e.g., "BTC/USDT"
    price REAL NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    source TEXT NOT NULL DEFAULT 'ccxt',  -- Exchange source
    volume_24h REAL,
    metadata TEXT  -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol ON crypto_prices(symbol);
CREATE INDEX IF NOT EXISTS idx_crypto_prices_timestamp ON crypto_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol_timestamp ON crypto_prices(symbol, timestamp);

-- Event log: audit trail for all significant events
CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,  -- ORDER_PLACED, FILL_RECEIVED, RISK_VETO, STOP_TRIGGERED, etc.
    severity TEXT NOT NULL DEFAULT 'INFO' CHECK (severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    data TEXT,  -- JSON blob with event-specific data
    token_id TEXT,
    order_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_log_event_type ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_event_log_severity ON event_log(severity);

-- Configuration snapshots: track config changes
CREATE TABLE IF NOT EXISTS config_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    config_hash TEXT NOT NULL,
    config_data TEXT NOT NULL  -- Full config as JSON
);

-- Migrations tracking
CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Insert initial migration record
INSERT OR IGNORE INTO migrations (version, name) VALUES (1, 'initial_schema');
