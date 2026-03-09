# How Polybot Works

Polybot is a **market-making bot** for Polymarket prediction markets. It profits by providing liquidity and capturing the bid-ask spread.

> **Note:** This is market-making, not arbitrage. Arbitrage exploits price differences across markets. Market-making earns the spread by facilitating trades.

---

## Core Concept

The bot places **both BUY and SELL orders** simultaneously, slightly away from the fair price:

```
Fair price: 50¢

Bot places:  BUY at 49.5¢    SELL at 50.5¢
             ←───── 1% spread ─────→

If both fill: Buy 10 shares at 49.5¢  = pay $4.95
              Sell 10 shares at 50.5¢ = receive $5.05
              Profit = $0.10 (the spread)
```

---

## The Trading Loop

Every ~1 second, the bot:

1. **Fetches order book** → Current best bid/ask prices
2. **Estimates fair price** → Adjusted for inventory and signals
3. **Evaluates risk** → Check for vetoes, stops, exposure limits
4. **Calculates quotes** → Optimal bid/ask using Avellaneda-Stoikov
5. **Places orders** → BUY below fair price, SELL above
6. **Tracks fills** → Update positions, calculate PnL

---

## Fair Price (Reservation Price)

The "fair price" isn't just the market midpoint. It's adjusted based on your inventory using the **Avellaneda-Stoikov model**:

```
reservation_price = mid_price - γ × σ² × T × inventory
```

| Term | Meaning |
|------|---------|
| `mid_price` | Current market midpoint |
| `γ` (gamma) | Risk aversion parameter |
| `σ²` | Volatility squared |
| `T` | Time remaining |
| `inventory` | Current position (+ long, - short) |

### Example

```
Mid = 50¢, γ = 0.1, σ = 0.1, T = 1, inventory = +100 shares

reservation = 0.50 - (0.1 × 0.01 × 1 × 100) = 40¢
```

**Long 100 shares → fair price drops → quotes shift down → encourages selling to reduce risk.**

---

## Gamma (γ) - Risk Aversion

Gamma controls how aggressive vs conservative the bot is:

| γ Value | Behavior |
|---------|----------|
| **Low (0.05)** | Tight spreads, slow inventory adjustment, more risk |
| **High (0.5)** | Wide spreads, fast inventory reduction, less risk |

### Gamma affects spread width

```
spread = γ × σ² × T + base_spread
```

Higher gamma → wider spread → fewer fills but safer.

### Danger Zone Adjustment

Near extreme prices (close to 0% or 100%), gamma increases automatically:

```python
if price < threshold or price > (1 - threshold):
    gamma = base_gamma × gamma_multiplier
```

**Why?** Near extremes:
- Small price moves = large percentage changes
- Higher chance of total loss
- Widen spreads to protect yourself

---

## Inventory Management

The bot constantly rebalances inventory:

```
                    INVENTORY EFFECT

     Long (+)                      Short (-)
        │                              │
        ▼                              ▼
   Lower quotes                  Raise quotes
   (want to sell)               (want to buy)

        ◄───── Reservation Price ─────►
```

| Position | Quote Adjustment | Goal |
|----------|------------------|------|
| Long (holding shares) | Lower prices | Encourage selling |
| Short (owe shares) | Raise prices | Encourage buying |
| Flat (no position) | Centered on mid | No bias |

---

## Lag Signal (Crypto Skew)

The bot monitors BTC/ETH/XRP prices on Binance. When crypto moves, it skews quotes:

| Crypto Signal | Quote Adjustment |
|---------------|------------------|
| BTC pumping | Raise both bid and ask (bullish skew) |
| BTC dumping | Lower both bid and ask (bearish skew) |

**Why?** BTC price movements often predict Polymarket BTC up/down market outcomes.

```yaml
lag_signal:
  skew_multiplier: 1.0  # Higher = stronger directional bets
  assets:
    - symbol: "BTC/USDT"
      weight: 0.5
```

---

## Risk Management

### Vetoes (Temporary Pause)

The bot stops quoting when:

| Condition | Why |
|-----------|-----|
| Jump detected | Large sudden price move, wait for stability |
| Momentum detected | Strong trend, don't fade it |
| Wide spread | Illiquid market, bad fills likely |
| Extreme price | Near 0% or 100%, high risk |

### Stops (Close Position)

| Trigger | Action |
|---------|--------|
| Price below `stop_prob_low` | Close position, stop trading |
| Price above `stop_prob_high` | Close position, stop trading |
| Near market expiry | Close position, wait for next market |

### Exposure Limits

| Limit | Purpose |
|-------|---------|
| `max_net_frac` | Max total exposure as % of balance |
| `max_open_order_pct` | Max open order value per market |
| `max_position_pct` | Max position value per market |

---

## When Does the Bot Buy vs Sell?

| Action | Condition | Purpose |
|--------|-----------|---------|
| **BUY** | Always (if risk allows) | Provide bid liquidity |
| **SELL** | Only if holding shares | Can't sell what you don't own |

---

## Configuration Reference

### Aggressive Settings (More Risk)

```yaml
avellaneda_stoikov:
  gamma: 0.05              # Tighter spreads

risk:
  stop_prob_low: 0.01      # Trade closer to 0%
  stop_prob_high: 0.99     # Trade closer to 100%
  max_net_frac: 0.70       # Larger positions allowed
  gamma_danger:
    threshold: 0.05        # Only widen near 5%/95%
    gamma_multiplier: 1.3  # Less spread widening

lag_signal:
  skew_multiplier: 1.5     # Stronger directional bets
```

### Conservative Settings (Less Risk)

```yaml
avellaneda_stoikov:
  gamma: 0.2               # Wider spreads

risk:
  stop_prob_low: 0.05      # Stop further from extremes
  stop_prob_high: 0.95
  max_net_frac: 0.20       # Smaller positions
  gamma_danger:
    threshold: 0.15        # Widen earlier
    gamma_multiplier: 3.0  # Much wider near extremes

lag_signal:
  skew_multiplier: 0.5     # Weaker directional bets
```

---

## Profit Sources

1. **Bid-ask spread** → Primary income from providing liquidity
2. **Inventory gains** → If held positions move in your favor
3. **Signal alpha** → Crypto lag signal predicts market direction

## Loss Sources

1. **Adverse selection** → Informed traders pick you off
2. **Inventory losses** → Held positions move against you
3. **Fees** → Polymarket charges 0.1% per trade

---

## Quick Reference

| Term | Meaning |
|------|---------|
| **Market making** | Providing liquidity by quoting both sides |
| **Spread** | Difference between bid and ask price |
| **Inventory** | Your current position (shares held) |
| **Gamma (γ)** | Risk aversion parameter |
| **Reservation price** | Your personal fair price adjusted for inventory |
| **Veto** | Temporary pause in quoting |
| **Stop** | Close position and halt trading |
