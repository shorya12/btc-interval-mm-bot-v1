"""Walk-forward training pipeline for probability models."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Type

import numpy as np
import pandas as pd

from .base import ProbabilityModel
from .features import compute_ohlcv_features, compute_strike_features, get_feature_names
from .evaluator import evaluate, EvalResult
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    eval_result: EvalResult
    model: Any  # ProbabilityModel instance
    n_train: int
    n_val: int


class WalkForwardTrainer:
    """
    Walk-forward training pipeline.

    Split schedule (per fold):
    - Training window: 6 months
    - Calibration fold: last 2 weeks of training window
    - Validation fold: next 4 weeks after training window
    - Step size: advance 2 weeks

    Minimum 3 folds required.
    """

    def __init__(
        self,
        model_class: Type[ProbabilityModel],
        config: Any = None,
        train_months: int = 6,
        cal_weeks: int = 2,
        val_weeks: int = 4,
        step_weeks: int = 2,
    ) -> None:
        self.model_class = model_class
        self.config = config
        self.train_months = train_months
        self.cal_weeks = cal_weeks
        self.val_weeks = val_weeks
        self.step_weeks = step_weeks

    def run(self, candles: pd.DataFrame) -> list[FoldResult]:
        """
        Execute walk-forward training.

        Args:
            candles: OHLCV DataFrame indexed by timestamp (ascending)

        Returns:
            List of FoldResult per fold
        """
        candles = candles.sort_index()

        # Deduplicate: keep last record per timestamp (handles overlapping backfills)
        n_before_dedup = len(candles)
        candles = candles[~candles.index.duplicated(keep="last")]
        n_deduped = n_before_dedup - len(candles)
        if n_deduped > 0:
            logger.info("deduplication_applied", n_dropped=n_deduped, n_remaining=len(candles))

        # Filter candles that follow a gap > 2× the timeframe interval (1h = 3600s)
        timeframe_seconds = 3600
        if len(candles) > 1:
            ts_seconds = candles.index.astype(np.int64) // 10**9
            diffs = np.diff(ts_seconds, prepend=ts_seconds[0])
            gap_mask = diffs > timeframe_seconds * 2
            n_before = len(candles)
            candles = candles[~gap_mask]
            n_dropped = n_before - len(candles)
            if n_dropped > 0:
                logger.info("gap_filter_applied", n_dropped=n_dropped, n_remaining=len(candles))

        index = candles.index

        train_window = timedelta(days=self.train_months * 30)
        cal_window = timedelta(weeks=self.cal_weeks)
        val_window = timedelta(weeks=self.val_weeks)
        step = timedelta(weeks=self.step_weeks)

        # Regime coverage check
        self._check_regime_coverage(candles)

        # Compute features once (no strike; use synthetic strike = current close)
        # Strike features will be computed per-sample using the close price as proxy
        ohlcv_feats = compute_ohlcv_features(candles)
        if ohlcv_feats.empty:
            logger.error("no_ohlcv_features", n_candles=len(candles))
            return []

        # Align candles with feature index
        candles_aligned = candles.loc[ohlcv_feats.index]

        feature_names = get_feature_names()
        fold_results: list[FoldResult] = []

        # First possible training start
        data_start = ohlcv_feats.index[0]
        data_end = ohlcv_feats.index[-1]

        fold_start = data_start
        fold_idx = 0

        while True:
            train_start = fold_start
            train_end = fold_start + train_window
            cal_start = train_end - cal_window
            val_start = train_end
            val_end = val_start + val_window

            if val_end > data_end:
                break

            # Slice data
            train_mask = (ohlcv_feats.index >= train_start) & (ohlcv_feats.index < cal_start)
            cal_mask = (ohlcv_feats.index >= cal_start) & (ohlcv_feats.index < train_end)
            val_mask = (ohlcv_feats.index >= val_start) & (ohlcv_feats.index < val_end)

            X_train, y_train = self._build_dataset(ohlcv_feats[train_mask], candles_aligned[train_mask], feature_names)
            X_cal, y_cal = self._build_dataset(ohlcv_feats[cal_mask], candles_aligned[cal_mask], feature_names)
            X_val, y_val = self._build_dataset(ohlcv_feats[val_mask], candles_aligned[val_mask], feature_names)

            if len(X_train) < 50 or len(X_val) < 10:
                logger.warning("fold_skipped_insufficient_data", fold=fold_idx, n_train=len(X_train), n_val=len(X_val))
                fold_start += step
                fold_idx += 1
                continue

            # Train model
            model = self.model_class()
            model.set_feature_names(feature_names) if hasattr(model, "set_feature_names") else None
            model.fit(X_train, y_train)

            # Calibrate
            if len(X_cal) > 10:
                model.calibrate(X_cal, y_cal)

            # Evaluate
            y_pred = model.predict_calibrated(X_val)
            log_moneyness_val = ohlcv_feats.loc[ohlcv_feats.index[val_mask]]["log_moneyness"] if "log_moneyness" in ohlcv_feats.columns else np.zeros(len(y_val))

            eval_result = evaluate(y_val, y_pred, np.zeros(len(y_val)))

            fold_result = FoldResult(
                fold_index=fold_idx,
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                val_start=pd.Timestamp(val_start),
                val_end=pd.Timestamp(val_end),
                eval_result=eval_result,
                model=model,
                n_train=len(X_train),
                n_val=len(X_val),
            )
            fold_results.append(fold_result)

            logger.info(
                "fold_complete",
                fold=fold_idx,
                n_train=len(X_train),
                n_val=len(X_val),
                bss_overall=eval_result.bss_overall,
            )

            fold_start += step
            fold_idx += 1

        logger.info("walk_forward_complete", n_folds=len(fold_results))
        return fold_results

    def _build_dataset(
        self,
        features: pd.DataFrame,
        candles: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) arrays for a data slice.

        Target: 1 if next-period close > current close (synthetic "strike = current close")
        This is a proxy until real Polymarket strike prices are available.
        """
        if features.empty or candles.empty:
            return np.array([]).reshape(0, len(feature_names)), np.array([])

        # Add synthetic strike features using current close as strike
        feat_df = features.copy()
        close = candles["close"].reindex(features.index)

        # Use realized_vol_60 as sigma; fall back to realized_vol_15
        if "realized_vol_60" in feat_df.columns:
            sigma = feat_df["realized_vol_60"].fillna(0.3)
        elif "realized_vol_15" in feat_df.columns:
            sigma = feat_df["realized_vol_15"].fillna(0.3)
        else:
            sigma = pd.Series(0.3, index=feat_df.index)

        # Synthetic: predict if next bar close > current close
        # log_moneyness = 0 (spot == strike = current close)
        # vol_normalized_dist = 0
        # time_remaining = 0.5 (mid-interval approximation)
        feat_df["log_moneyness"] = 0.0
        feat_df["vol_normalized_dist"] = 0.0
        feat_df["time_remaining"] = 0.5

        # Build y: 1 if next bar close > current close
        close_next = candles["close"].reindex(features.index).shift(-1)
        y = (close_next > close).astype(int)

        # Drop last row (no next close)
        valid = y.notna()
        feat_df = feat_df[valid]
        y = y[valid]

        # Select features in canonical order
        available = [f for f in feature_names if f in feat_df.columns]
        X = feat_df[available].values.astype(np.float32)
        y_arr = y.values.astype(np.int32)

        return X, y_arr

    def _check_regime_coverage(self, candles: pd.DataFrame) -> None:
        """Warn if training data lacks both bull and bear regimes."""
        if len(candles) < 60:
            return

        close = candles["close"]
        # 30-day return at rolling points
        returns_30d = close.pct_change(periods=min(30 * 24, len(close) - 1))

        has_bull = (returns_30d > 0.20).any()
        has_bear = (returns_30d < -0.20).any()

        if not has_bull:
            logger.warning("regime_coverage_warning", missing="bull", note="No +20% 30d return in training data")
        if not has_bear:
            logger.warning("regime_coverage_warning", missing="bear", note="No -20% 30d return in training data")
