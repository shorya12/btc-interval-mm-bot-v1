# Polybot - Polymarket CLOB Market Making Bot

Production-grade Python market-making bot for Polymarket CLOB on Polygon.

## Features

- **Avellaneda-Stoikov Market Making**: Optimal quoting in logit space with inventory management
- **Crypto Lag Signal**: BTC/ETH/XRP momentum-based quote skew
- **Risk Management**: Jump/momentum vetoes, probability stops, inventory caps
- **Dry Run Mode**: Paper trading for testing strategies
- **SQLite Persistence**: Full order/fill/position history

## Installation

```bash
# Clone repository
git clone <repo-url>
cd btc-latency

# Install with pip
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Configuration

1. Copy example config:
```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

2. Edit `config.yaml` with your markets and parameters

3. Edit `.env` with your credentials:
```
POLYBOT_PRIVATE_KEY=your_private_key
POLYBOT_API_KEY=your_api_key
POLYBOT_API_SECRET=your_api_secret
POLYBOT_API_PASSPHRASE=your_passphrase
```

## Usage

### Start Trading (Dry Run)
```bash
polybot run --dry-run --config config.yaml
```

### Start Trading (Live)
```bash
polybot run --config config.yaml
```

### Set Token Approvals
```bash
polybot approve --config config.yaml
```

### Check Status
```bash
polybot status --config config.yaml
```

### View Events
```bash
polybot events --config config.yaml --limit 50
```

## Architecture

```
src/
├── belief_state/     # Logit-space belief estimation
├── quoting/          # Avellaneda-Stoikov market making
├── lag_signal/       # Crypto price feed and skew
├── risk/             # Vetos, stops, inventory management
├── polymarket_client/# CLOB API integration
├── wallet_approval/  # Polygon approvals
├── persistence/      # SQLite database
├── main_loop/        # Trading loop and CLI
└── common/           # Config, logging, errors
```

## Key Parameters

### Avellaneda-Stoikov
- `gamma`: Risk aversion (0-1). Higher = wider spreads
- `base_spread_x`: Base spread multiplier

### Belief State
- `window_seconds`: Rolling window for volatility
- `sigma_b_floor`: Minimum belief volatility
- `jump_z`: Z-score for jump detection
- `momentum_z`: Z-score for momentum detection

### Risk
- `stop_prob_low/high`: Probability bounds for trading
- `max_net_frac`: Maximum position as bankroll fraction
- `gamma_danger`: Increase gamma near price extremes

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=src tests/
```

## License

MIT
