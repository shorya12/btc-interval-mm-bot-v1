# CLAUDE.md - Project Context

## Project Summary
**Polybot** - Production-grade Python market-making bot for Polymarket CLOB on Polygon. Targets BTC hourly up/down prediction markets with ~1s trading cadence.

### Core Features
- **Avellaneda-Stoikov quoting** in logit space (probability-aware spreads)
- **Crypto lag signal** - BTC/ETH/XRP momentum skews quotes
- **Auto market discovery** - Finds current BTC hourly market via Gamma API
- **Risk management** - Jump/momentum vetoes, probability stops, inventory caps
- **Dry run mode** - Paper trading with synthetic order books

## Tech Stack
- **Python 3.11+** with asyncio
- **py-clob-client** - Polymarket CLOB API
- **ccxt** - Crypto price feeds (Binance default)
- **web3.py** - Polygon wallet/approvals
- **aiosqlite** - Async SQLite persistence
- **Pydantic** - Config validation
- **Typer/Rich** - CLI
- **structlog** - Structured logging

## Architecture
```
src/
├── belief_state/     # Logit transforms, BeliefManager (rolling window estimation)
├── quoting/          # A-S algorithm, QuoteCalculator with gamma danger zones
├── lag_signal/       # PriceFeed (ccxt), LagModel (vol, lognormal q), SkewComputer
├── risk/             # VetoChecker, StopChecker, InventoryManager, RiskManager, PositionTracker
├── polymarket_client/# PolymarketClient, OrderBook/Order/Fill types, MarketDiscovery
├── wallet_approval/  # Polygon USDC/CTF approvals
├── persistence/      # SQLite: orders, fills, positions, pnl_snapshots, events
├── main_loop/        # TradingLoop, DryRunAdapter, CLI
└── common/           # Config, logging, errors

scripts/
├── close_all_positions.py  # Fetch positions, sell active, show redemption info
├── approve_allowances.py   # On-chain token approvals
└── allowances_example.py   # Example allowance sync
```

## CLI Commands
```bash
polybot run --dry-run          # Paper trading
polybot run                    # Live trading
polybot approve                # Set token approvals
polybot status                 # Show positions/PNL
polybot events                 # View event log
```

## Utility Scripts
```bash
# Close all positions and show redemption info
python scripts/close_all_positions.py --dry-run   # Preview
python scripts/close_all_positions.py             # Execute

# Approve token allowances on-chain
python scripts/approve_allowances.py
```

## Config Structure (config.yaml)
Key sections: `network`, `markets`, `avellaneda_stoikov`, `belief`, `execution`, `risk`, `lag_signal`, `dry_run`, `database`

**Auto-discovery for BTC markets:**
```yaml
markets:
  - auto_discover: true
    outcome: "YES"  # or "NO"
```

## Code Conventions
- **Type hints** everywhere (Python 3.11+ syntax: `list[str]`, `dict[str, Any]`, `X | None`)
- **Dataclasses** for data structures, **Pydantic BaseModel** for config
- **Async/await** for all I/O operations
- **structlog** for logging with context binding
- Prices in **probability space [0,1]**, math in **logit space (-∞, +∞)**
- Position: positive = long, negative = short

## Key Algorithms

### Avellaneda-Stoikov (src/quoting/avellaneda_stoikov.py)
- Reservation price: `r = mid_logit - γ * σ² * T * q` (inventory adjustment)
- Optimal spread: `δ = γ * σ² * T + base_spread`
- Gamma increases in "danger zone" (near 0 or 1 probability)

### Belief State (src/belief_state/belief.py)
- Rolling window of price observations in logit space
- Robust estimation (median/mean/ewma/huber)
- Jump detection: single-tick z-score > jump_z
- Momentum detection: average return z-score > momentum_z

### Lag Signal (src/lag_signal/)
- Fetches BTC/ETH/XRP from Binance via ccxt
- Computes realized vol, lognormal quantile, momentum
- Weighted skew shifts quotes (bullish signal → raise quotes)

## Known Issues / Bugs
1. ~~**Order size hardcoded**~~ - ✅ FIXED: Dynamic sizing with min $1 value
2. **No position close logic** - When stop triggers, `close_position=True` but no actual close orders placed
3. ~~**Market refresh clears state**~~ - ✅ FIXED: Managers cleared on refresh, invalid orderbook detection triggers refresh
4. **Resolved positions** - Winning positions must be redeemed on Polymarket website (not via API)
5. ~~**No sell orders placed**~~ - ✅ FIXED: Fill timestamp parsing was broken (Unix timestamp vs ISO format)
6. ~~**15m fallback instead of 1h**~~ - ✅ FIXED: Market discovery no longer falls back to different intervals

## TODOs (Priority Order)
1. ~~**Add order sizing**~~ - ✅ DONE: Dynamic sizing based on price with min $1 value
2. **Implement position close** - Market order to flatten when stop triggered
3. **Add WebSocket support** - Real-time order book instead of polling
4. ~~**Live testing**~~ - ✅ DONE: Tested with real orders on live markets
5. **PNL snapshots** - Periodic snapshots not yet triggered in main loop
6. ~~**Cleanup old managers**~~ - ✅ DONE: Cleared on market refresh

## Test Coverage
**Completed:**
- `tests/unit/test_logit.py` - Logit/sigmoid transforms ✓
- `tests/unit/test_avellaneda_stoikov.py` - A-S algorithm, Quote, ASParams ✓
- `tests/unit/test_lag_signal.py` - PriceFeed, LagModel, SkewComputer ✓
- `tests/unit/test_risk.py` - Veto, stops, inventory, RiskManager ✓
- `tests/unit/test_position_sizing.py` - InventoryManager, RiskManager sizing ✓
- `tests/unit/test_order_placement.py` - OrderManager, validation, lifecycle ✓
- `tests/integration/test_order_integration.py` - Live CLOB API diagnostics ✓

**Missing:**
- Integration tests for persistence (Database, Repository)
- End-to-end dry run test
- Market discovery tests

## API References
- **Polymarket CLOB**: `https://clob.polymarket.com`
- **Gamma API** (market discovery): `https://gamma-api.polymarket.com/events?slug=...`
- **py-clob-client docs**: https://github.com/Polymarket/py-clob-client

## Environment Variables (.env)
```
POLYBOT_PRIVATE_KEY=<wallet_private_key>
POLYBOT_SIGNATURE_TYPE=0          # 0 for EOA/Phantom wallet, 1 for Magic/email wallet
POLYBOT_FUNDER_ADDRESS=<funder>   # Required for proxy wallet setup (Phantom)
POLYBOT_API_KEY=<optional>
POLYBOT_API_SECRET=<optional>
POLYBOT_API_PASSPHRASE=<optional>
```

## Recent Changes

### USDC Balance & Exposure Limits (Latest)
- **Added real USDC balance checking** (`src/polymarket_client/client.py`):
  - Fetches actual on-chain USDC balance before trading
  - Tries multiple methods: Gamma API, CLOB API, on-chain RPC
  - Bot refuses to start if balance < $5
- **Dynamic exposure limits** (`src/main_loop/runner.py`):
  - Limits now based on ACTUAL balance, not hardcoded $10,000
  - `max_net_frac` (15%): Total exposure cap as % of balance
  - `max_open_order_pct` (10%): Per-market open order limit
  - `max_position_pct` (20%): Per-market position limit
- **Periodic balance refresh**: Checks balance every 30 seconds
- **Auto-stop on low balance**: Cancels orders and stops if balance < $5
- **Configurable limits** in `config.yaml`:
  ```yaml
  risk:
    max_net_frac: 0.15           # 15% total exposure
    max_open_order_pct: 0.10     # 10% per market
    max_position_pct: 0.20       # 20% per market
    min_balance_to_trade: 5.0    # Stop if below $5
  ```

### Fill Tracking & Market Interval Fixes
- **Fixed fill timestamp parsing** (`src/polymarket_client/client.py`):
  - API returns Unix timestamps, not ISO strings
  - Now handles both formats correctly
  - Fills are synced properly, enabling sell orders when positions exist
- **Fixed market interval fallback** (`src/polymarket_client/market_discovery.py`):
  - No longer falls back to 15m when 1h is requested
  - Returns None instead of wrong interval market
  - Prevents trading unexpected market types
- **Updated config** - Default interval now 15m (only currently available BTC up/down markets)
- **Fixed Position initialization** - Added required `size=0.0` parameter

### Position & Exposure Management
- **Added `PositionTracker`** (`src/risk/position_tracker.py`) for live trading:
  - Tracks open orders (pending notional value)
  - Tracks filled positions and realized PnL
  - Enforces exposure limits:
    - Max 20% of bankroll total exposure
    - Max $50 in open orders per market
    - Max $100 position value per market
  - Logs `live_status` every 10 seconds with exposure info
  - Logs `at_max_exposure` when limits reached

### Market Expiry Detection
- **Invalid orderbook detection** - Tracks consecutive invalid orderbooks
- Auto-refreshes markets after 10 consecutive failures (market expired)
- Logs `market_likely_expired` when detecting stale market

### Dynamic Order Sizing
- **Replaced hardcoded sizes** with `_calculate_order_size()` method
- Ensures minimum $1 order value (Polymarket requirement)
- Applies risk manager inventory limits
- Sizes based on price: `size = target_value / price`

### Close Positions Script
- **New `scripts/close_all_positions.py`**:
  - Fetches positions from Polymarket Data API
  - Categorizes: Active (can sell), Winning (redeem), Losing (worthless)
  - Places sell orders for active positions at best bid
  - Shows redemption instructions for resolved winning positions
  - Run with `--dry-run` to preview

### Previous Changes
- Added `MarketDiscovery` for auto-discovering BTC markets from Gamma API
- Updated discovery to use `/markets` endpoint (hourly markets no longer exist)
- Added `auto_discover` and `outcome` fields to MarketConfig
- Runner refreshes markets every 5 minutes for market rotation
- Synthetic order book generation for dry run mode
- Dry run falls back to mock market only if API discovery fails
- **Fixed "not enough balance/allowance" error** - Added CLOB API allowance sync
  - On-chain approvals alone are not enough; CLOB API maintains its own cache
  - `polybot approve` now calls `update_balance_allowance` to sync with CLOB API
  - Runner automatically syncs allowances on startup for live trading
- Added `POLYBOT_SIGNATURE_TYPE` and `POLYBOT_FUNDER_ADDRESS` env vars for Phantom wallet