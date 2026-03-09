# Probability Model for BTC Interval Markets — Design Doc

**Date:** 2026-03-09
**Status:** Approved

---

## Context

The existing `BeliefManager` estimates fair-value probability using a rolling window of order book midpoints. This works well near ATM but is poorly calibrated at strike distances, regime-blind, and has no ability to generalize from historical outcomes. The goal is to replace it with a proper ML probability model — trained on historical BTC price data, calibrated per moneyness bucket, and evaluated rigorously with walk-forward validation spanning both bull and bear regimes.

---

## Architecture: Option A — Clean Interface Replacement

A new `src/probability_model/` module with an abstract `ProbabilityModel` base class. `BeliefManager` becomes a thin adapter calling the model. XGBoost ships first; the Transformer slots in behind the same interface for apples-to-apples comparison. If the Transformer doesn't meaningfully beat XGBoost on calibration and Brier score after proper tuning, we ship the simpler model.

```
src/
├── data_pipeline/
│   ├── binance_fetcher.py      # backfill + incremental OHLCV ingestion via CCXT
│   └── gap_detector.py         # flag missing candles, never silent-fill
├── probability_model/
│   ├── base.py                 # abstract ProbabilityModel interface
│   ├── features.py             # OHLCV feature engineering + strike encoding
│   ├── xgboost_model.py        # XGBoost implementation
│   ├── transformer_model.py    # Transformer (same interface, added later)
│   ├── calibrator.py           # isotonic regression post-hoc calibration
│   ├── trainer.py              # walk-forward training pipeline
│   └── evaluator.py            # Brier, BSS, log loss, reliability diagrams
└── belief_state/
    └── belief.py               # refactored to call ProbabilityModel
```

---

## Strike Price Encoding

Two features, both required — feed separately so the model can learn which matters more at different distances:

| Feature | Formula | Why |
|---------|---------|-----|
| Log-moneyness | `log(spot / X)` | Symmetric, scale-invariant raw distance |
| Vol-normalized distance | `log(spot / X) / (σ_realized · √T)` | Regime-invariant; BSM d2 analog |

Where `σ_realized` uses the same realized vol window as `LagModel`, and `T` is the fraction of the interval remaining (also a separate feature).

---

## Feature Engineering (OHLCV-first)

Base features derived from OHLCV candles:
- Returns: 1, 5, 15, 30, 60-period log returns
- Realized volatility: 5, 15, 60-period rolling std of log returns (annualized)
- Volume ratios: current volume / rolling mean volume
- RSI (14-period)
- MACD signal line and histogram
- High-low range as fraction of mid (realized range)
- Strike features: log-moneyness, vol-normalized distance, time remaining

Each new data source (funding rates, order book depth, on-chain) added as a separate experiment with before/after BSS comparison. Drop if no improvement.

---

## Calibration

- **Training objective**: XGBoost trained with log loss (pushes toward calibrated probabilities inherently)
- **Post-hoc calibration**: Isotonic regression fit on a held-out calibration fold (last 2 weeks of the training window, excluded from XGBoost training)
- **Interface**: `ProbabilityModel` exposes `calibrate(cal_fold)` and `predict_calibrated(features)` separately from raw `predict()`
- **Transformer**: Temperature scaling instead of isotonic regression — same interface, different implementation

---

## Walk-Forward Validation

| Component | Value |
|-----------|-------|
| Training window | 6 months rolling |
| Calibration fold | Last 2 weeks of training window |
| Validation fold | Next 4 weeks (fully held out) |
| Step size | Advance 2 weeks, retrain from scratch |
| Minimum folds | 3 folds spanning ≥1 bull + ≥1 bear regime |

**Moneyness buckets for stratified evaluation:**

| Bucket | log(spot/X) range | Typical base rate |
|--------|-------------------|-------------------|
| Deep ITM | > +3% | ~85% |
| Near ITM | +1% to +3% | ~65–75% |
| ATM | ±1% | ~45–55% |
| Near OTM | -1% to -3% | ~25–35% |
| Deep OTM | < -3% | ~15% |

---

## Evaluation Suite

Per fold, per moneyness bucket:

1. **Brier Score** and **Brier Skill Score (BSS)** vs. climatology baseline:
   `BSS = 1 - (Brier_model / Brier_climatology)`
   BSS > 0 = beats baseline. BSS ≤ 0 = no value, do not ship.
2. **Log loss**
3. **Reliability diagram** (predicted probability vs. observed frequency, 10 bins)
4. **Sharpness histogram** (distribution of predictions — collapsed near 50% = model has no edge)

**Alpha signal check**: log the spread between `model_probability` and Polymarket's implied probability (current YES token price). This divergence is the actual trading edge.

---

## Regime Detection & Retraining

**Calendar trigger**: Retrain every 7 days regardless of performance.

**Performance trigger**: Rolling 14-day BSS computed on live predictions. If BSS < `0.0` for 3 consecutive days → immediate retrain + `REGIME_CHANGE_DETECTED` logged to `EventLog`.

**Inference-time confidence signal**: `vol_regime_ratio = vol_7d / vol_30d`. If > 2.0, prediction tagged as `low_confidence` in metadata. `QuoteCalculator` uses this to widen spreads (analogous to existing `VetoChecker` jump detection). No model switching.

---

## Data Pipeline

**Backfill job** (`binance_fetcher.py`):
- Fetches 1-minute OHLCV from Binance REST API via existing CCXT dependency
- Aggregates to configurable resolution at load time
- Stores in existing `CryptoPrice` SQLite table

**Gap detection** (`gap_detector.py`):
- Any span > 2× candle interval is flagged in `EventLog` and excluded from training windows
- Never silently filled — gaps are logged, not interpolated

**Live inference**:
- Returns `(probability: float, confidence: str, metadata: dict)` per tick
- Must complete < 50ms (XGBoost on tabular features is ~1ms; timeout guard added)
- `DryRunAdapter` unchanged — `ModelAdapter` wraps live vs. mock inference for testing

---

## Config Changes

New keys under `belief:` in `config.yaml`:

```yaml
belief:
  model_type: xgboost           # or: transformer
  model_path: models/btc_prob_model.pkl
  retrain_interval_days: 7
  bss_retrain_threshold: 0.0
  bss_window_days: 14
  vol_regime_ratio_threshold: 2.0
```

---

## Success Criteria

- BSS > 0 in ≥ 3 walk-forward folds across at least 2 moneyness buckets
- Reliability diagram within ±5% of perfect calibration in ATM bucket
- Model probability diverges meaningfully from Polymarket implied probability on ≥20% of ticks (otherwise we have no edge)
- Inference latency < 50ms p99 in dry-run

---

## What We Are Not Building (Yet)

- Online learning / gradient updates during live trading
- Order book depth or on-chain features (evaluated as separate experiments later)
- Ensemble of XGBoost + Transformer (compare first, blend only if both add value)
