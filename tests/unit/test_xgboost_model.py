"""Tests for XGBoostModel."""

import numpy as np
import pytest
import tempfile
import os

from src.probability_model.xgboost_model import XGBoostModel
from src.probability_model.features import get_feature_names


def make_classification_data(n=500, n_features=None):
    np.random.seed(42)
    if n_features is None:
        n_features = len(get_feature_names())
    X = np.random.randn(n, n_features).astype(np.float32)
    # Simple linear decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    return X, y


class TestXGBoostModel:
    def test_fit_predict(self):
        X, y = make_classification_data(300)
        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (300,)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_predict_calibrated_unfitted_calibrator(self):
        """Without calibration, predict_calibrated should still return valid probabilities."""
        X, y = make_classification_data(300)
        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict_calibrated(X)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_calibrate(self):
        X, y = make_classification_data(500)
        X_train, y_train = X[:300], y[:300]
        X_cal, y_cal = X[300:400], y[300:400]
        X_val = X[400:]

        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)
        preds = model.predict_calibrated(X_val)
        assert preds.shape == (100,)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_predict_one(self):
        X, y = make_classification_data(300)
        names = get_feature_names()
        model = XGBoostModel(n_estimators=10)
        model.set_feature_names(names)
        model.fit(X, y)

        features = {name: 0.0 for name in names}
        result = model.predict_one(features)
        assert 0.0 <= result.probability <= 1.0
        assert result.confidence in {"high", "medium", "low"}

    def test_save_and_load(self):
        X, y = make_classification_data(300)
        names = get_feature_names()
        model = XGBoostModel(n_estimators=10)
        model.set_feature_names(names)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            model2 = XGBoostModel()
            model2.load(path)

            preds1 = model.predict(X)
            preds2 = model2.predict(X)
            np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)
        finally:
            os.unlink(path)

    def test_predict_without_fit_raises(self):
        model = XGBoostModel()
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError):
            model.predict(X)
