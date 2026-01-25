"""Tests for logit/sigmoid transformations."""

import pytest
import math
from src.belief_state.logit import (
    logit,
    sigmoid,
    prob_to_logit,
    logit_to_prob,
    logit_midpoint,
    logit_spread,
    logit_distance,
    EPS,
)


class TestLogit:
    """Tests for logit function."""

    def test_logit_half(self):
        """logit(0.5) should be 0."""
        assert logit(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_logit_quarter(self):
        """logit(0.25) should be negative."""
        result = logit(0.25)
        assert result < 0
        assert result == pytest.approx(math.log(0.25 / 0.75), abs=1e-10)

    def test_logit_three_quarters(self):
        """logit(0.75) should be positive and symmetric with 0.25."""
        assert logit(0.75) == pytest.approx(-logit(0.25), abs=1e-10)

    def test_logit_extreme_low(self):
        """logit near 0 should be very negative but finite."""
        result = logit(0.001)
        assert result < -5
        assert math.isfinite(result)

    def test_logit_extreme_high(self):
        """logit near 1 should be very positive but finite."""
        result = logit(0.999)
        assert result > 5
        assert math.isfinite(result)

    def test_logit_at_zero_clamped(self):
        """logit(0) should be clamped and return finite value."""
        result = logit(0)
        assert math.isfinite(result)
        assert result < -10

    def test_logit_at_one_clamped(self):
        """logit(1) should be clamped and return finite value."""
        result = logit(1)
        assert math.isfinite(result)
        assert result > 10

    def test_logit_custom_epsilon(self):
        """Custom epsilon should work."""
        result = logit(0, eps=0.1)
        assert math.isfinite(result)


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be 0.5."""
        assert sigmoid(0) == pytest.approx(0.5, abs=1e-10)

    def test_sigmoid_positive(self):
        """sigmoid of positive value should be > 0.5."""
        assert sigmoid(1) > 0.5
        assert sigmoid(1) < 1

    def test_sigmoid_negative(self):
        """sigmoid of negative value should be < 0.5."""
        assert sigmoid(-1) < 0.5
        assert sigmoid(-1) > 0

    def test_sigmoid_extreme_positive(self):
        """sigmoid of large positive should approach 1."""
        assert sigmoid(100) > 0.99
        assert sigmoid(100) < 1

    def test_sigmoid_extreme_negative(self):
        """sigmoid of large negative should approach 0."""
        assert sigmoid(-100) < 0.01
        assert sigmoid(-100) > 0

    def test_sigmoid_overflow_protection(self):
        """sigmoid should handle extreme values without overflow."""
        assert sigmoid(1000) < 1
        assert sigmoid(-1000) > 0


class TestRoundTrip:
    """Tests for logit/sigmoid round-trip."""

    @pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_round_trip(self, p):
        """sigmoid(logit(p)) should return p."""
        assert sigmoid(logit(p)) == pytest.approx(p, rel=1e-10)

    @pytest.mark.parametrize("x", [-5, -1, 0, 1, 5])
    def test_inverse_round_trip(self, x):
        """logit(sigmoid(x)) should return x."""
        assert logit(sigmoid(x)) == pytest.approx(x, rel=1e-10)


class TestAliases:
    """Tests for function aliases."""

    def test_prob_to_logit_is_logit(self):
        """prob_to_logit should be same as logit."""
        assert prob_to_logit(0.75) == logit(0.75)

    def test_logit_to_prob_is_sigmoid(self):
        """logit_to_prob should be same as sigmoid."""
        assert logit_to_prob(1.0) == sigmoid(1.0)


class TestLogitMidpoint:
    """Tests for logit midpoint calculation."""

    def test_symmetric_spread(self):
        """Symmetric spread around 0.5 should give 0.5."""
        result = logit_midpoint(0.45, 0.55)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_asymmetric_spread(self):
        """Asymmetric spread should not give arithmetic mean."""
        bid, ask = 0.20, 0.30
        arithmetic_mean = (bid + ask) / 2
        logit_mean = logit_midpoint(bid, ask)
        # Logit midpoint should differ from arithmetic mean
        assert logit_mean != pytest.approx(arithmetic_mean, abs=0.001)

    def test_extreme_low(self):
        """Low probability spread."""
        result = logit_midpoint(0.05, 0.10)
        assert 0.05 < result < 0.10

    def test_extreme_high(self):
        """High probability spread."""
        result = logit_midpoint(0.90, 0.95)
        assert 0.90 < result < 0.95


class TestLogitSpread:
    """Tests for logit spread calculation."""

    def test_spread_positive(self):
        """Spread should always be positive."""
        assert logit_spread(0.45, 0.55) > 0

    def test_spread_increases_near_extremes(self):
        """Same prob spread should have larger logit spread near extremes."""
        # 10% spread centered at 0.5
        spread_mid = logit_spread(0.45, 0.55)
        # 10% spread centered at 0.1
        spread_low = logit_spread(0.05, 0.15)
        # Spread in logit space should be larger at extremes
        assert spread_low > spread_mid


class TestLogitDistance:
    """Tests for logit distance calculation."""

    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        assert logit_distance(0.3, 0.7) == logit_distance(0.7, 0.3)

    def test_distance_zero_same_prob(self):
        """Distance between same probability should be 0."""
        assert logit_distance(0.5, 0.5) == pytest.approx(0, abs=1e-10)

    def test_distance_always_positive(self):
        """Distance should always be non-negative."""
        assert logit_distance(0.2, 0.8) >= 0
