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
├── belief_state/       # Logit transforms, BeliefManager (rolling window + ML estimate)
├── quoting/            # A-S algorithm, QuoteCalculator with gamma danger zones
├── lag_signal/         # PriceFeed (ccxt), LagModel (vol, lognormal q), SkewComputer
├── risk/               # VetoChecker, StopChecker, InventoryManager, RiskManager, PositionTracker
├── polymarket_client/  # PolymarketClient, OrderBook/Order/Fill types, MarketDiscovery
├── wallet_approval/    # Polygon USDC/CTF approvals
├── persistence/        # SQLite: orders, fills, positions, pnl_snapshots, events
├── data_pipeline/      # Binance OHLCV backfill, gap detection        ← NEW
├── probability_model/  # ML probability models (XGBoost → Transformer) ← NEW
│   ├── base.py         # Abstract ProbabilityModel interface
│   ├── features.py     # OHLCV feature engineering + strike encoding
│   ├── xgboost_model.py
│   ├── transformer_model.py  # Phase 2 (stub — see TODOs)
│   ├── calibrator.py   # Isotonic regression post-hoc calibration
│   ├── trainer.py      # Walk-forward training pipeline
│   ├── evaluator.py    # Brier/BSS/log-loss + reliability diagrams
│   └── model_adapter.py # Live vs mock inference for dry-run
├── main_loop/          # TradingLoop, DryRunAdapter, CLI
└── common/             # Config, logging, errors

scripts/
├── close_all_positions.py  # Fetch positions, sell active, show redemption info
├── approve_allowances.py   # On-chain token approvals
└── allowances_example.py   # Example allowance sync

Key modules:
- polymarket_client/positions.py    # Shared position close logic (used by bot and script)
- probability_model/model_adapter.py # ModelAdapter(live=True/False) wraps real vs mock model
- belief_state/belief.py            # estimate_fair_value() — ML model with rolling-window fallback
```

## Active Feature Branch

**Branch:** `feature/probability-model`
**Worktree:** `.worktrees/probability-model/` (isolated from `main`)
**Status:** Phases 1–6 implemented, committed. Phase 7 (Transformer) pending.

```bash
# Work on the feature branch
cd .worktrees/probability-model

# Run all tests (206 passing, 3 pre-existing failures unrelated to ML work)
python3 -m pytest tests/ -q

# Backfill historical data (required before training)
polybot backfill --symbol BTC/USDT --start-date 2024-01-01 --end-date 2024-12-31 --timeframe 1h

# Train the model with walk-forward validation
polybot train --output models/btc_prob_model.pkl

# Run dry-run with ML model active
polybot run --dry-run
```

## CLI Commands
```bash
polybot run --dry-run          # Paper trading
polybot run                    # Live trading
polybot approve                # Set token approvals
polybot status                 # Show positions/PNL
polybot events                 # View event log
polybot backfill               # Backfill OHLCV from Binance (NEW)
polybot train                  # Walk-forward train probability model (NEW)
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

The `belief` section now includes ML model settings (added in `feature/probability-model`):

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
- **NEW** `estimate_fair_value(spot, strike, candles, time_remaining, model_adapter)`:
  - Delegates to ML model when `ModelAdapter` is loaded and candles available
  - Falls back to rolling-window mid estimate automatically
  - Returns `{probability, confidence, low_confidence, source, latency_ms}`

### Probability Model (src/probability_model/)
- **Interface:** `ProbabilityModel.fit / predict / calibrate / predict_calibrated / predict_one`
- **XGBoost:** `binary:logistic` trained on OHLCV features + strike encoding
- **Calibration:** Isotonic regression on held-out calibration fold (last 2 weeks of training window)
- **Strike encoding:**
  - `log_moneyness = log(spot / X)` — symmetric, scale-invariant
  - `vol_normalized_dist = log_moneyness / (σ_realized · √T)` — BSM d2 analog
- **Walk-forward:** 6mo train / 2wk cal / 4wk val / 2wk step, ≥3 folds required
- **Evaluation:** BSS per moneyness bucket (deep ITM → deep OTM), reliability diagram
- **Retraining triggers:** calendar (every 7d) or BSS < 0 for 3 consecutive days
- **Regime signal:** `vol_regime_ratio = vol_7d / vol_30d > 2.0` → `low_confidence` flag

### Lag Signal (src/lag_signal/)
- Fetches BTC/ETH/XRP from Binance via ccxt
- Computes realized vol, lognormal quantile, momentum
- Weighted skew shifts quotes (bullish signal → raise quotes)

## Known Issues / Bugs
1. ~~**Order size hardcoded**~~ - ✅ FIXED: Dynamic sizing with min $1 value
2. ~~**No position close logic**~~ - ✅ FIXED: Uses `close_all_positions()` from Data API at expiry
3. ~~**Market refresh clears state**~~ - ✅ FIXED: Managers cleared on refresh, invalid orderbook detection triggers refresh
4. **Resolved positions** - Winning positions must be redeemed on Polymarket website (not via API)
5. ~~**No sell orders placed**~~ - ✅ FIXED: Fill timestamp parsing was broken (Unix timestamp vs ISO format)
6. ~~**15m fallback instead of 1h**~~ - ✅ FIXED: Market discovery no longer falls back to different intervals
7. **PnL tracking may be inaccurate** - PositionTracker doesn't persist between sessions; investigation needed

## TODOs (Priority Order)

### Probability Model — In-Progress (branch: `feature/probability-model`)
1. **Backfill historical data** — Run `polybot backfill --symbol BTC/USDT --start-date 2024-01-01 --end-date 2024-12-31 --timeframe 1h` to populate `CryptoPrice` table before training
2. **First walk-forward train** — Run `polybot train` and verify ≥3 folds, BSS > 0 in ATM bucket, reliability diagram within ±5% of perfect calibration
3. **Real strike prices** — Current trainer uses `current_close` as a synthetic strike proxy. Replace with actual Polymarket strike prices stored alongside fills/positions in the DB. See `src/persistence/models.py` — add `strike_price` field to `Fill` or a separate `MarketMetadata` table
4. **Alpha signal verification** — Confirm `model_implied_prob` vs `market_implied_prob` spread appears in `EventLog` during dry-run; check that spread diverges on ≥20% of ticks
5. **Latency validation** — Add timing log around `estimate_fair_value()` in dry-run; verify p99 < 50ms
6. **Transformer model (Phase 7)** — Implement `src/probability_model/transformer_model.py`:
   - Same `ProbabilityModel` interface
   - Input: rolling sequence of OHLCV feature vectors (not flat)
   - Calibration: temperature scaling instead of isotonic regression
   - Compare BSS vs XGBoost on same walk-forward folds; only ship if meaningfully better
7. **Merge to main** — After dry-run validation passes success criteria (BSS > 0, latency < 50ms, ≥20% tick divergence), merge `feature/probability-model` → `main`

### Viability Checks Before Merging
- [ ] Walk-forward BSS > 0 in ≥3 folds across ≥2 moneyness buckets
- [ ] Reliability diagram within ±5% in ATM bucket
- [ ] `model_implied_prob` diverges from Polymarket price on ≥20% of dry-run ticks
- [ ] `estimate_fair_value()` p99 latency < 50ms in dry-run
- [ ] `CryptoPrice` table populated with ≥6 months of 1h OHLCV data
- [ ] No new test failures vs baseline (currently 3 pre-existing failures)
- [ ] `EventLog` shows `REGIME_CHANGE_DETECTED` and `MODEL_RETRAINED` events after a simulated BSS drop

### Bot Infrastructure
8. ~~**Add order sizing**~~ - ✅ DONE
9. ~~**Implement position close**~~ - ✅ DONE
10. **Add WebSocket support** - Real-time order book instead of polling
11. ~~**Live testing**~~ - ✅ DONE
12. **PNL snapshots** - Periodic snapshots not yet triggered in main loop
13. ~~**Cleanup old managers**~~ - ✅ DONE
14. **Investigate PnL tracking** - PositionTracker doesn't persist between sessions

### Data & Model Quality
15. **OHLCV schema gap** — `CryptoPrice` stores OHLCV in `metadata` JSON, not dedicated columns. Consider a migration adding `open`, `high`, `low`, `volume` columns for query performance once the model is validated
16. **Gap exclusion in trainer** — `filter_candles_with_gaps()` in `gap_detector.py` exists but is not yet wired into `WalkForwardTrainer._build_dataset()`. Wire it in before production training
17. **Annualization factor** — `features.py` assumes 1h candles (8760 periods/year). Add config param or auto-detect from candle spacing
18. **Funding rates / order book depth** — Evaluate as separate feature experiments with before/after BSS comparison; only add if BSS improves

## Test Coverage
**Completed:**
- `tests/unit/test_logit.py` - Logit/sigmoid transforms ✓
- `tests/unit/test_avellaneda_stoikov.py` - A-S algorithm, Quote, ASParams ✓
- `tests/unit/test_lag_signal.py` - PriceFeed, LagModel, SkewComputer ✓
- `tests/unit/test_risk.py` - Veto, stops, inventory, RiskManager ✓
- `tests/unit/test_position_sizing.py` - InventoryManager, RiskManager sizing ✓
- `tests/unit/test_order_placement.py` - OrderManager, validation, lifecycle ✓
- `tests/integration/test_order_integration.py` - Live CLOB API diagnostics ✓
- `tests/unit/test_features.py` - OHLCV features, strike encoding ✓ (NEW)
- `tests/unit/test_calibrator.py` - IsotonicCalibrator fit/transform/save/load ✓ (NEW)
- `tests/unit/test_xgboost_model.py` - XGBoostModel fit/calibrate/predict/save/load ✓ (NEW)
- `tests/unit/test_evaluator.py` - BSS, Brier, log loss, reliability diagram ✓ (NEW)
- `tests/unit/test_gap_detector.py` - Gap detection edge cases ✓ (NEW)

**Pre-existing failures (not caused by ML work):**
- `tests/integration/test_order_integration.py::test_order_signature_generation` — requires live API keys
- `tests/unit/test_avellaneda_stoikov.py::test_optimal_spread_increases_with_time` — pre-existing edge case
- `tests/unit/test_logit.py::test_sigmoid_extreme_positive` — float precision edge case

**Missing:**
- Integration tests for persistence (Database, Repository)
- End-to-end dry run test with ML model active
- Market discovery tests
- Walk-forward trainer integration test (needs real or synthetic candle DB)
- `ModelAdapter` integration test (live=True path with a saved model)

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

### ML Probability Model — Phases 1–6 (branch: `feature/probability-model`)
All work lives in `.worktrees/probability-model/`. Branch is `feature/probability-model`, **not yet merged to `main`**.

**Data Pipeline (`src/data_pipeline/`)**
- `binance_fetcher.py`: async CCXT OHLCV backfill + incremental fetch. Stores candles in `CryptoPrice` table with OHLC in `metadata` JSON field
- `gap_detector.py`: detects gaps > 2× interval, writes to `EventLog` with severity WARNING, provides `filter_candles_with_gaps()` for training exclusion

**Feature Engineering (`src/probability_model/features.py`)**
- Log returns: 1/5/15/30/60-period
- Realized vol: 5/15/60-period rolling std (annualized, assumes 1h candles)
- Volume ratio, RSI-14, MACD signal + histogram, HL range fraction
- `vol_regime_ratio = vol_7d / vol_30d` for confidence flagging
- Strike features: `log_moneyness`, `vol_normalized_dist`, `time_remaining`

**Model Interface & XGBoost (`src/probability_model/`)**
- `base.py`: abstract `ProbabilityModel` with full interface
- `calibrator.py`: `IsotonicCalibrator` (sklearn) with save/load
- `xgboost_model.py`: `binary:logistic`, isotonic calibration, pickle serialization
- `model_adapter.py`: `ModelAdapter(live=True/False)` — live calls real model, `live=False` returns `fixed_prob` for dry-run/testing

**Training & Evaluation**
- `trainer.py`: `WalkForwardTrainer` — 6mo/2wk/4wk/2wk schedule, regime coverage warning, synthetic strike proxy (replace with real strikes later)
- `evaluator.py`: BSS, Brier, log loss per moneyness bucket, reliability diagram data, rich table output

**Integration**
- `belief.py`: added `estimate_fair_value()` — calls ML model, falls back to rolling-window mid, logs latency
- `runner.py`: `ModelAdapter` initialized at startup, `_check_retrain_schedule()` checks calendar + BSS performance triggers, `_background_retrain()` runs non-blocking via `asyncio.create_task`
- `config.py` + `config.example.yaml`: 6 new `belief:` keys with defaults (`model_type`, `model_path`, `retrain_interval_days`, `bss_retrain_threshold`, `bss_window_days`, `vol_regime_ratio_threshold`)
- CLI: `polybot backfill` and `polybot train` commands added

**Config additions under `belief:`**
```yaml
belief:
  model_type: "xgboost"
  model_path: "models/btc_prob_model.pkl"
  retrain_interval_days: 7
  bss_retrain_threshold: 0.0
  bss_window_days: 14
  vol_regime_ratio_threshold: 2.0
```

**Known gaps / not yet wired:**
- `filter_candles_with_gaps()` exists but not called inside `WalkForwardTrainer._build_dataset()` — wire before production training
- Strike prices in trainer are synthetic (`current_close` proxy); real Polymarket strikes needed
- Alpha signal logging (`model_implied_prob` vs `market_implied_prob` spread) not yet emitted per tick
- `transformer_model.py` is a stub — Phase 7 not started

---

### Position Close at Expiry (Latest)
- **Implemented automatic position closing** when `time_to_expiry` stop triggers:
  - New `src/polymarket_client/positions.py` module with shared logic
  - `get_positions_from_data_api()` - Fetches actual positions from Polymarket Data API
  - `close_all_positions(client)` - Cancels orders, fetches positions, sells all at best bid
  - Bot now uses this proven logic (same as `close_all_positions.py` script) at market expiry
  - Fixes "not enough balance/allowance" errors when selling positions
- **How it works**:
  1. When `time_to_expiry <= 300 seconds` (5 min), stop triggers
  2. Bot calls `close_all_positions(self.client._client)`
  3. Function cancels all orders, fetches real positions from Data API
  4. Places sell orders at best bid for all active positions
  5. Market refresh is forced to move to next market

### USDC Balance & Exposure Limits
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