"""XGBoost implementation of ProbabilityModel."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .base import ProbabilityModel, PredictionResult
from .calibrator import IsotonicCalibrator
from src.common.logging import get_logger

logger = get_logger(__name__)


class XGBoostModel(ProbabilityModel):
    """
    XGBoost binary classifier for P(BTC_close > strike).

    Training objective: binary:logistic (log loss).
    Post-hoc calibration: isotonic regression on calibration fold.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_lambda: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            **kwargs,
        }
        self._booster: xgb.XGBClassifier | None = None
        self._calibrator = IsotonicCalibrator()
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost on training fold."""
        params = {k: v for k, v in self._params.items() if k != "use_label_encoder"}
        self._booster = xgb.XGBClassifier(**params, verbosity=0)
        self._booster.fit(X, y)
        logger.info("xgboost_fit", n_samples=len(y), pos_rate=float(y.mean()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw (uncalibrated) probabilities."""
        if self._booster is None:
            raise RuntimeError("Model not fitted — call fit() first")
        return self._booster.predict_proba(X)[:, 1]

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Fit isotonic calibrator on calibration fold."""
        raw_scores = self.predict(X_cal)
        self._calibrator.fit(raw_scores, y_cal)
        logger.info("xgboost_calibrated", n_cal=len(y_cal))

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_scores = self.predict(X)
        return self._calibrator.transform(raw_scores)

    def set_feature_names(self, names: list[str]) -> None:
        """Store feature names for predict_one()."""
        self._feature_names = names

    def save(self, path: str) -> None:
        """Serialize model + calibrator to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "booster": self._booster,
            "calibrator": self._calibrator,
            "feature_names": self._feature_names,
            "params": self._params,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("xgboost_saved", path=path)

    def load(self, path: str) -> None:
        """Deserialize model + calibrator from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._booster = payload["booster"]
        self._calibrator = payload["calibrator"]
        self._feature_names = payload.get("feature_names", [])
        self._params = payload.get("params", self._params)
        logger.info("xgboost_loaded", path=path, n_features=len(self._feature_names))
