# Polybot — ML Directional Trading Bot for Polymarket

Python bot for Polymarket's CLOB on Polygon, targeting BTC hourly up/down prediction markets.

**How it works:** Every tick, an XGBoost model trained on 1h BTC OHLCV features produces `model_prob` — the estimated probability that BTC closes higher than its current price in the next hour. When `|model_prob - 0.5| > direction_threshold` (default 0.05), the bot places a directional taker order: buy YES shares if the model says UP, buy NO shares if it says DOWN. It holds the position to market expiry and rotates to the next 1h market automatically. Position size is a fixed fraction of bankroll (`ml_bet_fraction`, default 10%).

## Backtest Results (Jan 2024 – Mar 2025, 18 folds, post-fee)

| Threshold | Bets | Win Rate | Return | Sharpe | Fees Paid |
|-----------|------|----------|--------|--------|-----------|
| 0.00 | 6,036 | 50.9% | +16.3% | 0.25 | $8,770 |
| **0.05 ★** | **2,295** | **51.5%** | **+41.4%** | **1.69** | **$2,959** |
| 0.10 | 1,187 | 51.6% | +24.1% | 1.90 | $1,292 |
| 0.15 | 571 | 52.0% | +18.6% | 3.06 | $437 |

*Flat-bet simulation, $100/bet, $10k starting balance. Fees use Polymarket crypto taker formula: `fee = bet × 0.25 × (p × (1-p))²`. Threshold=0.05 chosen for best absolute return.*

## Installation

```bash
git clone <repo-url>
cd btc-latency
pip install -e ".[dev]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `.env`:
```
POLYBOT_PRIVATE_KEY=your_private_key
```

## Workflow

```bash
# 1. Backfill historical candles (14+ months recommended)
polybot backfill --symbol BTC/USDT --start-date 2024-01-01 --end-date 2025-03-09 --timeframe 1h

# 2. Backfill Deribit DVOL (optional, improves calibration)
polybot backfill-options --symbol BTC --start-date 2024-01-01 --end-date 2025-03-09

# 3. Train the model
polybot train --output models/btc_prob_model.pkl

# 4. Run backtest + significance tests + cumulative PnL chart
polybot backtest --config config.yaml

# 5. Check model status and backtest summary
polybot summary --config config.yaml

# 6. Paper trade
polybot run --dry-run --config config.yaml --log-level DEBUG

# 7. Live
polybot run --config config.yaml
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `polybot run` | Start trading (add `--dry-run` for paper mode) |
| `polybot approve` | Set on-chain USDC/CTF token approvals |
| `polybot status` | Show positions and recent fills |
| `polybot events` | View event log |
| `polybot backfill` | Backfill OHLCV candles from Binance |
| `polybot backfill-options` | Backfill Deribit DVOL implied vol index |
| `polybot train` | Walk-forward train XGBoost probability model |
| `polybot backtest` | Flat-bet backtest + significance tests + `models/backtest_summary.json` |
| `polybot summary` | Dashboard: model status, backtest perf, live DVOL, paper P&L |

## Architecture

```
src/
├── belief_state/        # estimate_fair_value() — ML model inference + rolling fallback
├── probability_model/   # XGBoost model, calibrator, walk-forward trainer, evaluator
├── data_pipeline/       # Binance OHLCV backfill + gap detection + Deribit DVOL fetcher
├── risk/                # VetoChecker, StopChecker, InventoryManager, PositionTracker
├── polymarket_client/   # CLOB API, order book, fills, market discovery
├── wallet_approval/     # Polygon USDC/CTF approvals
├── persistence/         # SQLite: orders, fills, positions, events, options_signals
├── lag_signal/          # BTC price feed (informational; not used in order decisions)
├── main_loop/           # TradingLoop (ML directional logic), DryRunAdapter, CLI
└── common/              # Config (Pydantic), logging, errors

scripts/
└── backtest_flat_bets.py   # Flat-bet simulation with fees, bootstrap CIs, MC p-values

models/
├── btc_prob_model.pkl       # Trained model (after polybot train)
├── backtest_summary.json    # Backtest results (after polybot backtest)
├── significance_tests.json  # Bootstrap CIs + Monte Carlo p-values per threshold
└── cumulative_pnl.png       # Cumulative PnL chart per threshold
```

## Key Config Parameters (`config.yaml`)

```yaml
markets:
  - auto_discover: true
    outcome: "YES"     # buys YES when model says UP
    interval: "1h"
  - auto_discover: true
    outcome: "NO"      # buys NO when model says DOWN
    interval: "1h"

belief:
  direction_threshold: 0.05   # min |model_prob - 0.5| to enter (backtest-validated)
  ml_bet_fraction: 0.10       # fraction of bankroll per entry (10%)
  model_path: "models/btc_prob_model.pkl"

risk:
  max_net_frac: 0.70          # max total exposure as % of balance
  max_position_pct: 0.50      # max per-market position
  stop_prob_low: 0.02
  stop_prob_high: 0.98
```

## Testing

```bash
pytest tests/ -q          # 206 passing, 3 known pre-existing failures
pytest tests/unit/ -v     # unit tests only
```

## License

MIT
