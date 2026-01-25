# Session Summary - Polybot Implementation

## Project Overview
**Polybot** - Production-grade Python market-making bot for Polymarket CLOB on Polygon. Targets BTC hourly up/down prediction markets with ~1s trading cadence.

## Key Design Choices

### 1. Logit Space for Pricing
All probability math done in logit space for better numerical stability:
- `logit(p) = log(p/(1-p))`
- `sigmoid(x) = 1/(1+exp(-x))`

### 2. Avellaneda-Stoikov Algorithm
Modified for prediction markets with inventory skew:
- Reservation price: `r = mid_logit - γ * σ² * T * q`
- Optimal spread: `δ = γ * σ² * T + base_spread`
- Gamma increases in "danger zone" (near 0 or 1 probability)

### 3. Auto-Discovery for Market Rotation
Markets rotate hourly, so bot auto-discovers current market via Gamma API:
- `MarketDiscovery` class fetches from `gamma-api.polymarket.com`
- Refreshes every 5 minutes for hourly market rotation
- Config supports `auto_discover: true` with `outcome: "YES"` or `"NO"`

### 4. Dry Run with Synthetic Books
Generates random-walk order books for testing:
- 2% spread, 5 levels each side
- Random walk from last price
- Configurable fill rate simulation

### 5. Modular Architecture
Separated concerns:
```
src/
├── belief_state/     # Logit transforms, BeliefManager
├── quoting/          # A-S algorithm, QuoteCalculator
├── lag_signal/       # PriceFeed, LagModel, SkewComputer
├── risk/             # Veto, stops, inventory, RiskManager, PositionTracker
├── polymarket_client/# Client, OrderBook, MarketDiscovery
├── persistence/      # SQLite async storage
└── main_loop/        # TradingLoop, CLI

scripts/
├── close_all_positions.py  # Fetch and close all positions
├── approve_allowances.py   # On-chain approvals
└── allowances_example.py   # Example for allowance sync
```

### 6. Position & Exposure Tracking
Live trading exposure management via `PositionTracker`:
- Tracks open orders (pending notional)
- Tracks filled positions
- Enforces limits: 20% total exposure, $50 open orders/market, $100 position/market
- Auto-refresh on market expiry (10 consecutive invalid orderbooks)

## Critical Files

### `src/polymarket_client/market_discovery.py`
Discovers active BTC hourly markets:
```python
class MarketDiscovery:
    async def find_btc_hourly_market(self) -> DiscoveredMarket | None:
        markets = await self.search_markets("bitcoin up or down")
        active_markets = [m for m in markets if m.end_date and m.end_date > now]
        active_markets.sort(key=lambda m: m.end_date)
        return active_markets[0]
```

### `src/main_loop/runner.py`
Core trading loop with market refresh and synthetic book generation.

### `src/quoting/avellaneda_stoikov.py`
Core MM algorithm with inventory adjustment.

### `src/belief_state/logit.py`
Probability transforms.

### `src/lag_signal/skew.py`
Crypto signal → quote skew computation.

## Configuration

```yaml
markets:
  - auto_discover: true
    outcome: "YES"
    description: "BTC Hourly Auto"

avellaneda_stoikov:
  gamma: 0.1
  base_spread_x: 0.01
  kappa: null

belief:
  window_seconds: 300
  sigma_b_floor: 0.01
  robust_method: "median"

lag_signal:
  exchange: "binance"
  skew_multiplier: 1.0
  assets:
    - symbol: "BTC/USDT"
      weight: 0.5
```

## Errors Fixed During Implementation
1. **ImportError: QuoteContext** - Added to `src/quoting/__init__.py`
2. **ImportError: AssetConfig** - Added to `src/lag_signal/__init__.py`
3. **Dry run no order book** - Added `_generate_synthetic_book()` method
4. **"not enough balance/allowance"** - Added CLOB API allowance sync
5. **Market 404 errors** - Added invalid orderbook detection with auto-refresh
6. **Unlimited order placement** - Added PositionTracker with exposure limits
7. **Hardcoded order sizes** - Dynamic sizing with min $1 value requirement
8. **SELL errors without position** - Added `can_sell` check (position_size > 0)

## Known Issues
1. ~~Live trading untested~~ - ✅ FIXED: Tested with real orders
2. ~~Order size hardcoded~~ - ✅ FIXED: Dynamic sizing with min $1 value
3. No position close logic - Stop triggers but no close orders
4. ~~Market refresh clears state~~ - ✅ FIXED: Managers cleared, invalid orderbook detection
5. Resolved positions need manual redemption on Polymarket website

## TODOs (Priority Order)
1. ~~Add configurable order sizing~~ - ✅ DONE
2. Implement position close logic
3. Add WebSocket support for real-time order books
4. ~~Live testing with small amounts~~ - ✅ DONE
5. PNL snapshots implementation
6. ~~Cleanup old managers on market refresh~~ - ✅ DONE

## Environment Variables
```
POLYBOT_PRIVATE_KEY=<wallet_private_key>
POLYBOT_SIGNATURE_TYPE=0          # 0 for EOA/Phantom, 1 for Magic/email
POLYBOT_FUNDER_ADDRESS=<funder>   # Required for Phantom wallet
POLYBOT_API_KEY=<optional>
POLYBOT_API_SECRET=<optional>
POLYBOT_API_PASSPHRASE=<optional>
```

## CLI Commands
```bash
polybot run --dry-run    # Paper trading
polybot run              # Live trading
polybot approve          # Set token approvals
polybot status           # Show positions/PNL
polybot events           # View event log
```

## Utility Scripts
```bash
# Close all positions
python scripts/close_all_positions.py --dry-run   # Preview
python scripts/close_all_positions.py             # Execute

# Approve allowances
python scripts/approve_allowances.py
```

## Position Types
When closing positions, the script categorizes them:
- **Active** - Can be sold on the market (places limit order at best bid)
- **Winning** - Resolved in your favor, need to redeem on Polymarket website
- **Losing** - Resolved against you, worthless (no action needed)
