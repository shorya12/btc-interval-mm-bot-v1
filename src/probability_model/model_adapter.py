"""ModelAdapter: wraps live or mock probability model for dry-run support."""

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import ProbabilityModel, PredictionResult
from .features import compute_strike_features, get_feature_names
from src.common.logging import get_logger

logger = get_logger(__name__)


class ModelAdapter:
    """
    Adapter that provides a unified interface for live or mock inference.

    - live=True: loads and calls the real ProbabilityModel
    - live=False: returns fixed_prob for all predictions (dry-run/testing)
    """

    def __init__(
        self,
        live: bool = True,
        model_path: str | None = None,
        model_type: str = "xgboost",
        fixed_prob: float = 0.5,
        vol_regime_ratio_threshold: float = 2.0,
    ) -> None:
        self.live = live
        self.model_path = model_path
        self.model_type = model_type
        self.fixed_prob = fixed_prob
        self.vol_regime_ratio_threshold = vol_regime_ratio_threshold
        self._model: ProbabilityModel | None = None
        self._loaded = False

    def load(self) -> bool:
        """
        Attempt to load the model from disk.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not self.live:
            return True

        if self.model_path and Path(self.model_path).exists():
            try:
                model = self._create_model()
                model.load(self.model_path)
                self._model = model
                self._loaded = True
                logger.info("model_loaded", path=self.model_path, model_type=self.model_type)
                return True
            except Exception as exc:
                logger.warning("model_load_failed", path=self.model_path, error=str(exc))
                return False
        else:
            logger.warning("model_not_found", path=self.model_path)
            return False

    def _create_model(self) -> ProbabilityModel:
        """Instantiate the configured model type."""
        if self.model_type == "xgboost":
            from .xgboost_model import XGBoostModel
            return XGBoostModel()
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(
        self,
        ohlcv_features: dict[str, float],
        spot: float,
        strike: float,
        sigma_realized: float,
        time_remaining_frac: float,
    ) -> PredictionResult:
        """
        Make a calibrated probability prediction.

        Args:
            ohlcv_features: Dict of OHLCV-derived features
            spot: Current BTC price
            strike: Market strike price
            sigma_realized: Annualized realized vol
            time_remaining_frac: Fraction of interval remaining

        Returns:
            PredictionResult
        """
        if not self.live or self._model is None:
            return PredictionResult(
                probability=self.fixed_prob,
                confidence="low",
                raw_score=self.fixed_prob,
                metadata={"source": "mock"},
            )

        strike_feats = compute_strike_features(spot, strike, sigma_realized, time_remaining_frac)
        all_features = {**ohlcv_features, **strike_feats}

        # Determine vol regime confidence
        vol_ratio = ohlcv_features.get("vol_regime_ratio", 1.0)
        if vol_ratio > self.vol_regime_ratio_threshold:
            confidence = "low"
        elif vol_ratio > self.vol_regime_ratio_threshold * 0.75:
            confidence = "medium"
        else:
            confidence = "high"

        try:
            result = self._model.predict_one(all_features)
            result.confidence = confidence
            result.metadata["vol_regime_ratio"] = vol_ratio
            return result
        except Exception as exc:
            logger.error("model_predict_failed", error=str(exc))
            return PredictionResult(
                probability=self.fixed_prob,
                confidence="low",
                raw_score=self.fixed_prob,
                metadata={"error": str(exc)},
            )

    @property
    def is_ready(self) -> bool:
        """True if model is available for inference."""
        return not self.live or self._loaded
