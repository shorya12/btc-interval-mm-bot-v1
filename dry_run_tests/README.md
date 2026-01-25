# Dry Run Tests

This folder contains tools for running dry-run (paper trading) tests to measure expected PnL had the bot been running live.

## Quick Start

```bash
# Run a 5-minute test with default settings
python dry_run_tests/run_test.py --duration 300

# Run a 1-hour test with hourly markets
python dry_run_tests/run_test.py --duration 3600 --interval 1h

# Run overnight test (8 hours)
python dry_run_tests/run_test.py --duration 28800 --output results/overnight_test.json
```

## Scripts

### `run_test.py`

Runs the polybot in dry-run mode for a specified duration and captures performance metrics.

**Arguments:**
- `--config`: Path to config file (default: `config.yaml`)
- `--duration`: Test duration in seconds (default: 300)
- `--interval`: Market interval to test - `15m`, `1h`, `4h`, `1d`, `1w`, `1M` (overrides config)
- `--output`: Output path for results JSON (default: auto-generated)
- `--snapshot-interval`: Seconds between metric snapshots (default: 30)

**Example:**
```bash
# Test different intervals
python dry_run_tests/run_test.py --duration 600 --interval 15m --output results/test_15m.json
python dry_run_tests/run_test.py --duration 3600 --interval 1h --output results/test_1h.json
python dry_run_tests/run_test.py --duration 14400 --interval 4h --output results/test_4h.json
```

### `analyze_results.py`

Analyzes test results and compares multiple runs.

**Arguments:**
- `files`: One or more result JSON files to analyze
- `--compare`: Compare multiple test results side-by-side
- `--csv`: Export comparison to CSV file

**Example:**
```bash
# Analyze a single test
python dry_run_tests/analyze_results.py results/test_001.json

# Compare multiple tests
python dry_run_tests/analyze_results.py results/*.json --compare

# Export comparison to CSV
python dry_run_tests/analyze_results.py results/*.json --compare --csv comparison.csv
```

## Output Metrics

### Performance Metrics
- **Net PnL**: Total profit/loss after fees
- **PnL Percent**: Return as percentage of initial balance
- **Annualized Return**: Projected yearly return based on test period
- **Sharpe Proxy**: Risk-adjusted return measure

### Trading Activity
- **Total Fills**: Number of orders that were filled
- **Fill Rate**: Percentage of orders that filled (realistic based on volume)
- **Fill Imbalance**: Buy/sell ratio (positive = net buyer)
- **Final Position**: Net position at end of test

### Fill Statistics (Realistic Mode)
- **Actual Fill Rate**: Observed fill probability
- **Avg Fill Probability**: Mean calculated fill probability
- **Rejected (prob)**: Orders rejected due to probability check
- **Rejected (price)**: Orders rejected due to price check

## Results Format

Results are saved as JSON with the following structure:

```json
{
  "test_info": {
    "start_time": "2026-01-24T00:00:00",
    "end_time": "2026-01-24T01:00:00",
    "duration_seconds": 3600,
    "interval": "1h"
  },
  "config": {
    "avellaneda_stoikov": {...},
    "risk": {...},
    "dry_run": {...}
  },
  "final_metrics": {
    "initial_balance": 10000.0,
    "final_balance": 9998.5,
    "realized_pnl": 0.15,
    "unrealized_pnl": 0.02,
    "total_fees": 1.67,
    "net_pnl": -1.50,
    "pnl_percent": -0.015,
    "total_fills": 45,
    "buy_fills": 24,
    "sell_fills": 21,
    "fill_stats": {...}
  },
  "snapshots": [
    {"timestamp": "...", "tick": 30, "balance": 10000.0, ...},
    {"timestamp": "...", "tick": 60, "balance": 9999.5, ...}
  ]
}
```

## Tips

1. **Long Tests**: For meaningful results, run tests for at least 30 minutes
2. **Multiple Runs**: Run the same config multiple times to account for randomness
3. **Compare Intervals**: Test different market intervals (15m vs 1h) to find optimal settings
4. **Monitor Fill Rate**: Low fill rates may indicate spreads are too wide

## Realistic Fill Simulation

The dry-run adapter now uses volume-based fill probability:

- **Size Factor**: Smaller orders relative to book depth fill more easily
- **Price Aggressiveness**: Orders closer to/crossing the spread fill faster
- **Market Activity**: Tighter spreads indicate more activity = higher fills

This provides more realistic PnL estimates compared to a fixed fill rate.
