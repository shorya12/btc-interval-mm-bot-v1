"""Belief state estimation and management."""

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Any, Literal
import math
import statistics

from .logit import logit, sigmoid, logit_midpoint
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BeliefState:
    """
    Current belief state for a market.

    Stores the estimated fair value and uncertainty in logit space.
    """

    token_id: str
    mid_prob: float  # Current mid-price as probability
    mid_logit: float  # Current mid-price in logit space
    sigma_b: float  # Belief volatility (std dev in logit space)
    jump_detected: bool = False  # True if recent jump detected
    momentum_detected: bool = False  # True if momentum regime detected
    last_update: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_extreme(self) -> bool:
        """Check if price is in extreme territory (near 0 or 1)."""
        return self.mid_prob < 0.05 or self.mid_prob > 0.95

    @property
    def confidence_band(self) -> tuple[float, float]:
        """Return 1-sigma confidence band in probability space."""
        low_logit = self.mid_logit - self.sigma_b
        high_logit = self.mid_logit + self.sigma_b
        return sigmoid(low_logit), sigmoid(high_logit)


@dataclass
class PriceObservation:
    """Single price observation for belief estimation."""

    timestamp: datetime
    bid: float
    ask: float
    mid_prob: float
    mid_logit: float


class BeliefManager:
    """
    Manages belief state estimation for a market.

    Uses a rolling window of price observations to estimate:
    - Fair value (robust central estimate)
    - Belief volatility (sigma_b)
    - Jump and momentum detection
    """

    def __init__(
        self,
        token_id: str,
        window_seconds: int = 300,
        sigma_b_floor: float = 0.01,
        robust_method: Literal["median", "mean", "ewma", "huber"] = "median",
        jump_z: float = 3.0,
        momentum_z: float = 2.0,
        max_observations: int = 1000,
    ) -> None:
        """
        Initialize belief manager.

        Args:
            token_id: Token identifier
            window_seconds: Rolling window for estimation
            sigma_b_floor: Minimum belief volatility
            robust_method: Method for robust central estimate
            jump_z: Z-score threshold for jump detection
            momentum_z: Z-score threshold for momentum detection
            max_observations: Maximum observations to store
        """
        self.token_id = token_id
        self.window_seconds = window_seconds
        self.sigma_b_floor = sigma_b_floor
        self.robust_method = robust_method
        self.jump_z = jump_z
        self.momentum_z = momentum_z

        self._observations: deque[PriceObservation] = deque(maxlen=max_observations)
        self._current_state: BeliefState | None = None

    def update(self, bid: float, ask: float, timestamp: datetime | None = None) -> BeliefState:
        """
        Update belief state with new price observation.

        Args:
            bid: Best bid price
            ask: Best ask price
            timestamp: Observation timestamp (defaults to now)

        Returns:
            Updated BeliefState
        """
        timestamp = timestamp or datetime.utcnow()

        # Compute mid in logit space
        mid_prob = logit_midpoint(bid, ask)
        mid_logit = logit(mid_prob)

        # Create observation
        obs = PriceObservation(
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            mid_prob=mid_prob,
            mid_logit=mid_logit,
        )
        self._observations.append(obs)

        # Prune old observations
        self._prune_old_observations(timestamp)

        # Estimate belief state
        self._current_state = self._estimate_state(obs)
        return self._current_state

    def _prune_old_observations(self, current_time: datetime) -> None:
        """Remove observations outside the rolling window."""
        cutoff = current_time.timestamp() - self.window_seconds
        while (
            self._observations
            and self._observations[0].timestamp.timestamp() < cutoff
        ):
            self._observations.popleft()

    def _estimate_state(self, latest: PriceObservation) -> BeliefState:
        """Estimate belief state from observations."""
        if len(self._observations) < 2:
            # Not enough data, use latest observation
            return BeliefState(
                token_id=self.token_id,
                mid_prob=latest.mid_prob,
                mid_logit=latest.mid_logit,
                sigma_b=self.sigma_b_floor,
                last_update=latest.timestamp,
            )

        logits = [obs.mid_logit for obs in self._observations]

        # Compute robust central estimate
        if self.robust_method == "median":
            central_logit = statistics.median(logits)
        elif self.robust_method == "mean":
            central_logit = statistics.mean(logits)
        elif self.robust_method == "ewma":
            central_logit = self._compute_ewma(logits)
        elif self.robust_method == "huber":
            central_logit = self._compute_huber_mean(logits)
        else:
            central_logit = statistics.median(logits)

        # Compute volatility (standard deviation in logit space)
        if len(logits) >= 3:
            sigma_b = statistics.stdev(logits)
        else:
            sigma_b = abs(logits[-1] - logits[0]) / 2

        # Apply floor
        sigma_b = max(sigma_b, self.sigma_b_floor)

        # Detect jump (large single-tick move)
        jump_detected = False
        if len(logits) >= 2:
            tick_change = abs(logits[-1] - logits[-2])
            if sigma_b > 0 and tick_change / sigma_b > self.jump_z:
                jump_detected = True
                logger.info(
                    "jump_detected",
                    token_id=self.token_id,
                    tick_change=tick_change,
                    sigma_b=sigma_b,
                    z_score=tick_change / sigma_b,
                )

        # Detect momentum (consistent directional move)
        momentum_detected = False
        if len(logits) >= 5:
            # Use recent 5 observations to detect trend
            recent = logits[-5:]
            changes = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            avg_change = sum(changes) / len(changes)
            if sigma_b > 0:
                momentum_z_score = abs(avg_change) / (sigma_b / math.sqrt(len(changes)))
                if momentum_z_score > self.momentum_z:
                    momentum_detected = True
                    logger.debug(
                        "momentum_detected",
                        token_id=self.token_id,
                        avg_change=avg_change,
                        z_score=momentum_z_score,
                    )

        return BeliefState(
            token_id=self.token_id,
            mid_prob=sigmoid(central_logit),
            mid_logit=central_logit,
            sigma_b=sigma_b,
            jump_detected=jump_detected,
            momentum_detected=momentum_detected,
            last_update=latest.timestamp,
        )

    def _compute_ewma(self, values: list[float], alpha: float = 0.1) -> float:
        """Compute exponentially weighted moving average."""
        if not values:
            return 0.0
        ewma = values[0]
        for v in values[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        return ewma

    def _compute_huber_mean(self, values: list[float], delta: float = 1.0) -> float:
        """
        Compute Huber robust mean (less sensitive to outliers).

        Uses iterative reweighted least squares.
        """
        if not values:
            return 0.0

        # Start with median
        estimate = statistics.median(values)

        # Iterate to refine
        for _ in range(10):
            weights = []
            for v in values:
                residual = abs(v - estimate)
                if residual <= delta:
                    weights.append(1.0)
                else:
                    weights.append(delta / residual)

            # Weighted mean
            total_weight = sum(weights)
            if total_weight == 0:
                break
            estimate = sum(w * v for w, v in zip(weights, values)) / total_weight

        return estimate

    @property
    def state(self) -> BeliefState | None:
        """Get current belief state."""
        return self._current_state

    @property
    def observations_count(self) -> int:
        """Get number of observations in window."""
        return len(self._observations)

    def get_returns(self, n: int | None = None) -> list[float]:
        """
        Get log-returns in logit space.

        Args:
            n: Number of returns (None for all)

        Returns:
            List of returns (logit[t] - logit[t-1])
        """
        logits = [obs.mid_logit for obs in self._observations]
        if len(logits) < 2:
            return []

        returns = [logits[i + 1] - logits[i] for i in range(len(logits) - 1)]
        if n is not None:
            returns = returns[-n:]
        return returns

    def estimate_fair_value(
        self,
        spot: float,
        strike: float,
        candles: Any,  # pd.DataFrame or None
        time_remaining_frac: float,
        model_adapter: Any | None = None,
        sigma_realized: float | None = None,
    ) -> dict[str, Any]:
        """
        Estimate fair-value probability P(close > strike at interval end).

        When a ModelAdapter is provided and ready, delegates to the ML model.
        Falls back to the rolling-window mid-price estimate otherwise.

        Args:
            spot: Current BTC spot price
            strike: Market strike (threshold)
            candles: OHLCV DataFrame (required for ML model features)
            time_remaining_frac: Fraction of interval remaining [0, 1]
            model_adapter: Optional ModelAdapter instance
            sigma_realized: Realized vol override (uses rolling estimate if None)

        Returns:
            Dict with keys: probability, confidence, low_confidence (bool), source
        """
        import time

        t0 = time.perf_counter()

        # Try ML model path first
        if model_adapter is not None and model_adapter.is_ready and candles is not None:
            try:
                import pandas as pd
                from src.probability_model.features import compute_ohlcv_features

                if isinstance(candles, pd.DataFrame) and len(candles) >= 30:
                    ohlcv_feats_df = compute_ohlcv_features(candles)
                    if not ohlcv_feats_df.empty:
                        # Use last row as the current feature vector
                        ohlcv_feats = ohlcv_feats_df.iloc[-1].to_dict()

                        # Use provided or estimated sigma
                        if sigma_realized is None:
                            sigma_realized = float(ohlcv_feats.get("realized_vol_60", 0.3) or 0.3)

                        result = model_adapter.predict(
                            ohlcv_features=ohlcv_feats,
                            spot=spot,
                            strike=strike,
                            sigma_realized=sigma_realized,
                            time_remaining_frac=time_remaining_frac,
                        )

                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        vol_ratio = ohlcv_feats.get("vol_regime_ratio", 1.0)
                        low_confidence = result.confidence == "low"

                        logger.debug(
                            "estimate_fair_value_model",
                            prob=round(result.probability, 4),
                            confidence=result.confidence,
                            latency_ms=round(elapsed_ms, 2),
                        )

                        return {
                            "probability": result.probability,
                            "confidence": result.confidence,
                            "low_confidence": low_confidence,
                            "source": "ml_model",
                            "latency_ms": elapsed_ms,
                            "vol_regime_ratio": vol_ratio,
                        }
            except Exception as exc:
                logger.warning("estimate_fair_value_model_error", error=str(exc))

        # Fallback: rolling-window mid estimate
        state = self._current_state
        prob = state.mid_prob if state is not None else 0.5
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.debug("estimate_fair_value_fallback", prob=round(prob, 4))

        return {
            "probability": prob,
            "confidence": "medium",
            "low_confidence": False,
            "source": "rolling_window",
            "latency_ms": elapsed_ms,
        }
