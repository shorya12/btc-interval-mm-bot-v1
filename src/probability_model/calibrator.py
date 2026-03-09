"""Isotonic regression calibrator for post-hoc probability calibration."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Post-hoc calibration using isotonic regression.

    Fits a monotone non-decreasing mapping from raw model scores
    to calibrated probabilities.
    """

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit isotonic regression on calibration fold.

        Args:
            raw_scores: Uncalibrated model probabilities [0, 1]
            labels: Binary ground truth labels (0 or 1)
        """
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(raw_scores, labels)
        self._fitted = True

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map raw scores to calibrated probabilities.

        Args:
            raw_scores: Uncalibrated scores

        Returns:
            Calibrated probabilities in [0, 1]
        """
        if not self._fitted or self._model is None:
            # If not fitted, return scores unchanged
            return np.clip(raw_scores, 0.0, 1.0)
        return self._model.predict(raw_scores)

    def save(self, path: str) -> None:
        """Serialize calibrator to disk."""
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        """Deserialize calibrator from disk."""
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = self._model is not None
