"""Tests for Avellaneda-Stoikov market-making algorithm."""

import pytest
from src.quoting.avellaneda_stoikov import AvellanedaStoikov
from src.quoting.params import ASParams, Quote
from src.belief_state.logit import logit, sigmoid


class TestAvellanedaStoikov:
    """Tests for A-S market maker."""

    @pytest.fixture
    def as_model(self):
        """Default A-S model for testing."""
        return AvellanedaStoikov(gamma=0.1, base_spread_x=0.01)

    def test_initialization(self, as_model):
        """Test model initializes with correct parameters."""
        assert as_model.gamma == 0.1
        assert as_model.base_spread_x == 0.01
        assert as_model.kappa is None

    def test_initialization_with_kappa(self):
        """Test model initializes with kappa."""
        model = AvellanedaStoikov(gamma=0.1, base_spread_x=0.01, kappa=1.5)
        assert model.kappa == 1.5

    def test_reservation_price_no_inventory(self, as_model):
        """With no inventory, reservation should equal mid."""
        mid_logit = logit(0.5)
        reservation = as_model.compute_reservation_price(
            mid_logit=mid_logit,
            inventory=0,
            sigma_b=0.05,
            time_remaining=1.0,
        )
        assert reservation == pytest.approx(mid_logit, abs=1e-10)

    def test_reservation_price_long_inventory(self, as_model):
        """With long inventory, reservation should be below mid."""
        mid_logit = logit(0.5)
        reservation = as_model.compute_reservation_price(
            mid_logit=mid_logit,
            inventory=10,  # Long 10 units
            sigma_b=0.05,
            time_remaining=1.0,
        )
        # Should be lower to encourage selling
        assert reservation < mid_logit

    def test_reservation_price_short_inventory(self, as_model):
        """With short inventory, reservation should be above mid."""
        mid_logit = logit(0.5)
        reservation = as_model.compute_reservation_price(
            mid_logit=mid_logit,
            inventory=-10,  # Short 10 units
            sigma_b=0.05,
            time_remaining=1.0,
        )
        # Should be higher to encourage buying
        assert reservation > mid_logit

    def test_reservation_price_time_decay(self, as_model):
        """Inventory adjustment should decrease as time remaining decreases."""
        mid_logit = logit(0.5)
        inventory = 10

        # Full time remaining
        res_full = as_model.compute_reservation_price(
            mid_logit, inventory, sigma_b=0.05, time_remaining=1.0
        )

        # Half time remaining
        res_half = as_model.compute_reservation_price(
            mid_logit, inventory, sigma_b=0.05, time_remaining=0.5
        )

        # No time remaining
        res_zero = as_model.compute_reservation_price(
            mid_logit, inventory, sigma_b=0.05, time_remaining=0.0
        )

        # Adjustment should be smaller with less time
        assert abs(mid_logit - res_half) < abs(mid_logit - res_full)
        assert res_zero == pytest.approx(mid_logit, abs=1e-10)

    def test_optimal_spread_positive(self, as_model):
        """Spread should always be positive."""
        spread = as_model.compute_optimal_spread(sigma_b=0.05, time_remaining=1.0)
        assert spread > 0

    def test_optimal_spread_increases_with_volatility(self, as_model):
        """Spread should increase with volatility."""
        spread_low = as_model.compute_optimal_spread(sigma_b=0.02, time_remaining=1.0)
        spread_high = as_model.compute_optimal_spread(sigma_b=0.10, time_remaining=1.0)
        assert spread_high > spread_low

    def test_optimal_spread_increases_with_time(self, as_model):
        """Spread should increase with more time remaining."""
        spread_short = as_model.compute_optimal_spread(sigma_b=0.05, time_remaining=0.1)
        spread_long = as_model.compute_optimal_spread(sigma_b=0.05, time_remaining=1.0)
        assert spread_long > spread_short

    def test_compute_quotes_basic(self, as_model):
        """Test basic quote computation."""
        quote = as_model.compute_quotes(
            mid_logit=logit(0.5),
            inventory=0,
            sigma_b=0.05,
            time_remaining=1.0,
        )

        assert isinstance(quote, Quote)
        assert quote.is_valid()
        assert 0 < quote.bid_price < quote.ask_price < 1
        assert quote.spread > 0

    def test_compute_quotes_symmetric_at_mid(self, as_model):
        """With no inventory at mid, quotes should be symmetric."""
        quote = as_model.compute_quotes(
            mid_logit=logit(0.5),
            inventory=0,
            sigma_b=0.05,
            time_remaining=1.0,
        )

        # Should be approximately symmetric around 0.5
        assert quote.reservation_price == pytest.approx(0.5, abs=0.01)
        mid = (quote.bid_price + quote.ask_price) / 2
        assert mid == pytest.approx(0.5, abs=0.01)

    def test_compute_quotes_inventory_skew(self, as_model):
        """Long inventory should skew quotes down."""
        quote_neutral = as_model.compute_quotes(
            mid_logit=logit(0.5), inventory=0, sigma_b=0.05, time_remaining=1.0
        )
        quote_long = as_model.compute_quotes(
            mid_logit=logit(0.5), inventory=10, sigma_b=0.05, time_remaining=1.0
        )

        # Long inventory: quotes should be lower
        assert quote_long.bid_price < quote_neutral.bid_price
        assert quote_long.ask_price < quote_neutral.ask_price

    def test_compute_quotes_signal_skew(self, as_model):
        """Positive signal skew should shift quotes up."""
        quote_neutral = as_model.compute_quotes(
            mid_logit=logit(0.5),
            inventory=0,
            sigma_b=0.05,
            time_remaining=1.0,
            signal_skew=0,
        )
        quote_bullish = as_model.compute_quotes(
            mid_logit=logit(0.5),
            inventory=0,
            sigma_b=0.05,
            time_remaining=1.0,
            signal_skew=0.2,  # Bullish signal
        )

        # Bullish skew: quotes should be higher
        assert quote_bullish.bid_price > quote_neutral.bid_price
        assert quote_bullish.ask_price > quote_neutral.ask_price

    def test_compute_quotes_min_spread(self, as_model):
        """Quotes should respect minimum spread."""
        quote = as_model.compute_quotes(
            mid_logit=logit(0.5),
            inventory=0,
            sigma_b=0.001,  # Very low volatility
            time_remaining=0.01,  # Almost no time
            min_spread_prob=0.01,
        )

        assert quote.spread >= 0.01

    def test_compute_quotes_extreme_price(self, as_model):
        """Quotes should be valid at extreme prices."""
        # Near 0
        quote_low = as_model.compute_quotes(
            mid_logit=logit(0.05),
            inventory=0,
            sigma_b=0.1,
            time_remaining=1.0,
        )
        assert quote_low.is_valid()
        assert quote_low.bid_price > 0

        # Near 1
        quote_high = as_model.compute_quotes(
            mid_logit=logit(0.95),
            inventory=0,
            sigma_b=0.1,
            time_remaining=1.0,
        )
        assert quote_high.is_valid()
        assert quote_high.ask_price < 1

    def test_update_params(self, as_model):
        """Test parameter updates."""
        as_model.update_params(gamma=0.2)
        assert as_model.gamma == 0.2

        as_model.update_params(base_spread_x=0.02, kappa=1.0)
        assert as_model.base_spread_x == 0.02
        assert as_model.kappa == 1.0

    def test_update_params_invalid_gamma(self, as_model):
        """Invalid gamma should raise error."""
        with pytest.raises(ValueError):
            as_model.update_params(gamma=1.5)

        with pytest.raises(ValueError):
            as_model.update_params(gamma=-0.1)


class TestQuote:
    """Tests for Quote dataclass."""

    @pytest.fixture
    def sample_quote(self):
        """Sample valid quote."""
        return Quote(
            bid_price=0.45,
            ask_price=0.55,
            bid_logit=logit(0.45),
            ask_logit=logit(0.55),
            reservation_price=0.50,
            reservation_logit=logit(0.50),
            spread=0.10,
            spread_logit=logit(0.55) - logit(0.45),
            half_spread_logit=(logit(0.55) - logit(0.45)) / 2,
            inventory_skew=0,
            signal_skew=0,
        )

    def test_quote_is_valid(self, sample_quote):
        """Valid quote should pass validation."""
        assert sample_quote.is_valid()

    def test_quote_invalid_bid_above_ask(self):
        """Quote with bid >= ask should be invalid."""
        quote = Quote(
            bid_price=0.55,
            ask_price=0.45,  # Invalid: ask < bid
            bid_logit=0,
            ask_logit=0,
            reservation_price=0.5,
            reservation_logit=0,
            spread=-0.1,
            spread_logit=0,
            half_spread_logit=0,
            inventory_skew=0,
            signal_skew=0,
        )
        assert not quote.is_valid()

    def test_quote_mid_price(self, sample_quote):
        """Mid price should be average of bid and ask."""
        assert sample_quote.mid_price == pytest.approx(0.50, abs=1e-10)

    def test_quote_spread_bps(self, sample_quote):
        """Spread in bps should be calculated correctly."""
        expected_bps = (0.10 / 0.50) * 10000  # 2000 bps
        assert sample_quote.spread_bps == pytest.approx(expected_bps, abs=1)

    def test_quote_with_skew(self, sample_quote):
        """with_skew should shift quote correctly."""
        skewed = sample_quote.with_skew(0.1)

        # Prices should be higher
        assert skewed.bid_price > sample_quote.bid_price
        assert skewed.ask_price > sample_quote.ask_price

        # Signal skew should be recorded
        assert skewed.signal_skew == 0.1


class TestASParams:
    """Tests for ASParams dataclass."""

    def test_valid_params(self):
        """Valid parameters should work."""
        params = ASParams(
            gamma=0.1,
            base_spread_x=0.01,
            sigma_b=0.05,
            time_remaining=0.5,
            inventory=0,
        )
        assert params.gamma == 0.1

    def test_invalid_gamma_high(self):
        """Gamma > 1 should raise error."""
        with pytest.raises(ValueError):
            ASParams(
                gamma=1.5,
                base_spread_x=0.01,
                sigma_b=0.05,
                time_remaining=0.5,
                inventory=0,
            )

    def test_invalid_sigma_zero(self):
        """Zero sigma should raise error."""
        with pytest.raises(ValueError):
            ASParams(
                gamma=0.1,
                base_spread_x=0.01,
                sigma_b=0,  # Invalid
                time_remaining=0.5,
                inventory=0,
            )

    def test_invalid_time_remaining(self):
        """Time remaining > 1 should raise error."""
        with pytest.raises(ValueError):
            ASParams(
                gamma=0.1,
                base_spread_x=0.01,
                sigma_b=0.05,
                time_remaining=1.5,  # Invalid
                inventory=0,
            )
