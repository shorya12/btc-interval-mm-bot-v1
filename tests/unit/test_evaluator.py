"""Tests for model evaluator."""

import numpy as np
import pytest

from src.probability_model.evaluator import (
    brier_score,
    brier_skill_score,
    log_loss,
    reliability_diagram_data,
    evaluate,
    EvalResult,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0], dtype=float)
        pred = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier_score(y, pred) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y = np.array([1, 0], dtype=float)
        pred = np.array([0.0, 1.0])
        assert brier_score(y, pred) == pytest.approx(1.0)

    def test_uniform_predictions(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        pred = np.full(4, 0.5)
        assert brier_score(y, pred) == pytest.approx(0.25)


class TestBrierSkillScore:
    def test_bss_perfect_model(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        # Perfect predictions beat climatology
        pred = y.copy()
        bss = brier_skill_score(y, pred)
        assert bss == pytest.approx(1.0)

    def test_bss_climatology_baseline(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        # Climatology: always predict base rate
        pred = np.full(4, 0.5)
        bss = brier_skill_score(y, pred)
        assert bss == pytest.approx(0.0, abs=1e-6)

    def test_bss_worse_than_baseline_negative(self):
        y = np.array([1, 1, 0, 0], dtype=float)
        pred = np.array([0.0, 0.0, 1.0, 1.0])  # Always wrong
        bss = brier_skill_score(y, pred)
        assert bss < 0


class TestLogLoss:
    def test_near_perfect_predictions_low_loss(self):
        y = np.array([1, 0, 1], dtype=float)
        pred = np.array([0.99, 0.01, 0.99])
        ll = log_loss(y, pred)
        assert ll < 0.02

    def test_uniform_predictions(self):
        y = np.array([1, 0], dtype=float)
        pred = np.full(2, 0.5)
        ll = log_loss(y, pred)
        assert ll == pytest.approx(np.log(2), rel=1e-4)


class TestReliabilityDiagram:
    def test_returns_list_of_dicts(self):
        y = np.random.randint(0, 2, 200).astype(float)
        pred = np.random.uniform(0, 1, 200)
        result = reliability_diagram_data(y, pred)
        assert isinstance(result, list)
        for item in result:
            assert "bin_center" in item
            assert "frac_pos" in item
            assert "count" in item

    def test_frac_pos_in_range(self):
        y = np.random.randint(0, 2, 500).astype(float)
        pred = np.random.uniform(0, 1, 500)
        result = reliability_diagram_data(y, pred)
        for item in result:
            assert 0.0 <= item["frac_pos"] <= 1.0


class TestEvaluate:
    def test_empty_input(self):
        result = evaluate(np.array([]), np.array([]), np.array([]))
        assert isinstance(result, EvalResult)
        assert result.bss_overall == 0.0

    def test_basic_evaluation(self):
        np.random.seed(42)
        y = np.random.randint(0, 2, 200).astype(float)
        pred = np.clip(y + np.random.randn(200) * 0.3, 0, 1)
        moneyness = np.random.uniform(-0.05, 0.05, 200)

        result = evaluate(y, pred, moneyness)
        assert result.n_samples == 200
        assert isinstance(result.bss_overall, float)
        assert isinstance(result.brier_overall, float)

    def test_bucket_metrics_populated(self):
        np.random.seed(42)
        y = np.random.randint(0, 2, 1000).astype(float)
        pred = np.clip(y + np.random.randn(1000) * 0.3, 0, 1)
        # Spread moneyness across all buckets
        moneyness = np.random.uniform(-0.05, 0.05, 1000)

        result = evaluate(y, pred, moneyness)
        # ATM bucket should have samples
        assert len(result.bucket_metrics) > 0
