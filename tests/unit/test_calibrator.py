"""Tests for IsotonicCalibrator."""

import numpy as np
import pytest
import tempfile
import os

from src.probability_model.calibrator import IsotonicCalibrator


class TestIsotonicCalibrator:
    def setup_method(self):
        np.random.seed(42)

    def _make_data(self, n=200):
        raw = np.random.uniform(0, 1, n)
        labels = (raw + np.random.randn(n) * 0.2 > 0.5).astype(float)
        return raw, labels

    def test_fit_and_transform(self):
        cal = IsotonicCalibrator()
        raw, labels = self._make_data()
        cal.fit(raw, labels)
        calibrated = cal.transform(raw)
        assert calibrated.shape == raw.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_monotone_increasing(self):
        """Calibrated output should be non-decreasing with raw scores."""
        cal = IsotonicCalibrator()
        raw, labels = self._make_data()
        cal.fit(raw, labels)

        sorted_raw = np.sort(raw)
        calibrated = cal.transform(sorted_raw)
        # Non-decreasing
        assert np.all(np.diff(calibrated) >= -1e-10)

    def test_unfitted_returns_clipped(self):
        cal = IsotonicCalibrator()
        raw = np.array([-0.1, 0.5, 1.2])
        result = cal.transform(raw)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_save_and_load(self):
        cal = IsotonicCalibrator()
        raw, labels = self._make_data()
        cal.fit(raw, labels)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            cal.save(path)
            cal2 = IsotonicCalibrator()
            cal2.load(path)

            result1 = cal.transform(raw)
            result2 = cal2.transform(raw)
            np.testing.assert_array_almost_equal(result1, result2)
        finally:
            os.unlink(path)
