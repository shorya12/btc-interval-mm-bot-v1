"""Abstract base class for probability models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictionResult:
    """Result from a single model prediction."""

    probability: float  # Calibrated probability P(close > strike)
    confidence: str  # "high", "medium", "low"
    raw_score: float  # Uncalibrated model output
    metadata: dict[str, Any] = field(default_factory=dict)


class ProbabilityModel(ABC):
    """
    Abstract interface for BTC interval probability models.

    Given features derived from OHLCV + strike encoding, outputs
    P(BTC_close > strike at interval end).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on features X and binary labels y."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw (uncalibrated) scores for each sample."""
        ...

    @abstractmethod
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Fit post-hoc calibration on a held-out calibration fold."""
        ...

    @abstractmethod
    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for each sample."""
        ...

    def predict_one(self, features: dict[str, float]) -> PredictionResult:
        """
        Make a single prediction from a feature dict.

        Args:
            features: Flat feature dict (must match training feature names)

        Returns:
            PredictionResult with probability, confidence, and metadata
        """
        import numpy as np

        feature_names = self.feature_names
        X = np.array([[features.get(f, 0.0) for f in feature_names]], dtype=np.float32)

        raw_score = float(self.predict(X)[0])
        calibrated_prob = float(self.predict_calibrated(X)[0])

        # Confidence based on vol_regime_ratio in metadata
        vol_ratio = features.get("vol_regime_ratio", 1.0)
        if vol_ratio > 2.0:
            confidence = "low"
        elif vol_ratio > 1.5:
            confidence = "medium"
        else:
            confidence = "high"

        return PredictionResult(
            probability=calibrated_prob,
            confidence=confidence,
            raw_score=raw_score,
            metadata={"vol_regime_ratio": vol_ratio},
        )

    @property
    def feature_names(self) -> list[str]:
        """Return ordered list of feature names used at training time."""
        return getattr(self, "_feature_names", [])

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Deserialize model from disk."""
        ...
