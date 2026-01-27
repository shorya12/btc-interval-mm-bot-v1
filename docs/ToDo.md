# Outstanding TODOs

## High Priority

### 1. Investigate PnL Tracking Accuracy
**Status**: Needs investigation  
**Issue**: PnL and position values displayed in logs may be inaccurate (~$7 shown vs ~$100 actual reported by user)

**Possible causes**:
- `PositionTracker` starts with empty state each session
- Fills may not be tracked correctly
- Multiple markets/positions not aggregated properly
- Calculation bug in PnL formula

**Files to investigate**:
- `src/risk/position_tracker.py` - Position and PnL tracking
- `src/polymarket_client/fills.py` - Fill tracking
- `src/main_loop/runner.py` - Status logging (`_maybe_log_live_status`)

**Context**: User reports positions are sold between runs, so pre-existing positions shouldn't affect tracking within a session.

---

### 2. PNL Snapshots Implementation
**Status**: Not started  
**Issue**: Periodic PnL snapshots not yet triggered in main loop

**Files**:
- `src/persistence/models.py` - `PnlSnapshot` model exists
- `src/persistence/schema.sql` - `pnl_snapshots` table exists
- `src/main_loop/runner.py` - Need to call snapshot logic

---

## Medium Priority

### 3. WebSocket Support for Order Books
**Status**: Not started  
**Issue**: Currently polling for order book updates instead of real-time WebSocket

**Benefits**:
- Lower latency order book updates
- Reduced API rate limiting concerns
- Better fill detection

---

### 4. Integration Tests
**Status**: Partially complete

**Missing tests**:
- Persistence integration tests (Database, Repository)
- End-to-end dry run test
- Market discovery tests
- Position close tests

---

## Low Priority

### 5. Refactor close_all_positions.py Script
**Status**: Optional  
**Issue**: Script now duplicates logic from `src/polymarket_client/positions.py`

**Action**: Update script to import and use shared module instead of duplicating code.

---

## Completed (Reference)

- ✅ Dynamic order sizing with min $1 value
- ✅ Position close logic at market expiry (uses `close_all_positions()`)
- ✅ Live testing with real orders
- ✅ Manager cleanup on market refresh
- ✅ Invalid orderbook detection and auto-refresh
- ✅ USDC balance checking and exposure limits
- ✅ CLOB API allowance sync
