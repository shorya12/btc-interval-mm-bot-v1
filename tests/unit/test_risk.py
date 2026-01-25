"""Tests for risk management module."""

import pytest
from datetime import datetime

from src.belief_state import BeliefState
from src.risk.veto import VetoChecker, VetoResult, VetoReason
from src.risk.stops import StopChecker, StopResult, StopConfig, StopType
from src.risk.inventory import InventoryManager, InventoryStatus
from src.risk.risk_manager import RiskManager, RiskDecision


class TestVetoChecker:
    """Tests for VetoChecker."""

    @pytest.fixture
    def checker(self):
        """Create veto checker with default config."""
        return VetoChecker(
            jump_z=3.0,
            momentum_z=2.0,
            extreme_prob_threshold=0.02,
            max_spread_bps=500,
            min_liquidity=100,
        )

    @pytest.fixture
    def normal_belief(self):
        """Create normal belief state."""
        return BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
            jump_detected=False,
            momentum_detected=False,
        )

    def test_no_veto_normal_conditions(self, checker, normal_belief):
        """Test no veto under normal conditions."""
        result = checker.check(
            belief=normal_belief,
            spread_bps=100,
            bid_depth=200,
            ask_depth=200,
        )

        assert not result.vetoed
        assert len(result.reasons) == 0

    def test_veto_on_jump(self, checker):
        """Test veto when jump detected."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
            jump_detected=True,
            momentum_detected=False,
        )

        result = checker.check(belief=belief)

        assert result.vetoed
        assert VetoReason.JUMP_DETECTED in result.reasons

    def test_veto_on_momentum(self, checker):
        """Test veto when momentum detected."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
            jump_detected=False,
            momentum_detected=True,
        )

        result = checker.check(belief=belief)

        assert result.vetoed
        assert VetoReason.MOMENTUM_DETECTED in result.reasons

    def test_veto_on_extreme_low_price(self, checker):
        """Test veto on extremely low price."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.01,  # Below 0.02 threshold
            mid_logit=-4.6,
            sigma_b=0.05,
        )

        result = checker.check(belief=belief)

        assert result.vetoed
        assert VetoReason.EXTREME_PRICE in result.reasons

    def test_veto_on_extreme_high_price(self, checker):
        """Test veto on extremely high price."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.99,  # Above 0.98 threshold
            mid_logit=4.6,
            sigma_b=0.05,
        )

        result = checker.check(belief=belief)

        assert result.vetoed
        assert VetoReason.EXTREME_PRICE in result.reasons

    def test_veto_on_wide_spread(self, checker, normal_belief):
        """Test veto on wide spread."""
        result = checker.check(
            belief=normal_belief,
            spread_bps=600,  # Above 500 threshold
        )

        assert result.vetoed
        assert VetoReason.WIDE_SPREAD in result.reasons

    def test_veto_on_low_liquidity(self, checker, normal_belief):
        """Test veto on low liquidity."""
        result = checker.check(
            belief=normal_belief,
            bid_depth=50,  # Below 100 threshold
            ask_depth=200,
        )

        assert result.vetoed
        assert VetoReason.LOW_LIQUIDITY in result.reasons

    def test_multiple_veto_reasons(self, checker):
        """Test multiple veto reasons accumulated."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.01,
            mid_logit=-4.6,
            sigma_b=0.05,
            jump_detected=True,
        )

        result = checker.check(
            belief=belief,
            spread_bps=600,
        )

        assert result.vetoed
        assert len(result.reasons) >= 2


class TestStopChecker:
    """Tests for StopChecker."""

    @pytest.fixture
    def checker(self):
        """Create stop checker with default config."""
        return StopChecker(StopConfig(
            stop_prob_low=0.02,
            stop_prob_high=0.98,
            max_loss_pct=0.10,
            min_time_to_expiry_seconds=300,
            max_position_frac=0.20,
        ))

    def test_no_stop_normal_conditions(self, checker):
        """Test no stop under normal conditions."""
        result = checker.check_all(
            current_prob=0.5,
            position_size=100,
            position_pnl=10,
            bankroll=1000,
            time_to_expiry_seconds=3600,
        )

        assert not result.triggered
        assert not result.should_close

    def test_stop_on_low_probability(self, checker):
        """Test stop on low probability."""
        result = checker.check_probability_stop(0.01)

        assert result.triggered
        assert result.stop_type == StopType.PROBABILITY_LOW
        assert result.should_close

    def test_stop_on_high_probability(self, checker):
        """Test stop on high probability."""
        result = checker.check_probability_stop(0.99)

        assert result.triggered
        assert result.stop_type == StopType.PROBABILITY_HIGH
        assert result.should_close

    def test_stop_on_max_loss(self, checker):
        """Test stop on maximum loss."""
        result = checker.check_loss_stop(
            position_pnl=-150,  # -15% loss
            bankroll=1000,
        )

        assert result.triggered
        assert result.stop_type == StopType.MAX_LOSS
        assert result.should_close

    def test_no_stop_within_loss_limit(self, checker):
        """Test no stop within loss limit."""
        result = checker.check_loss_stop(
            position_pnl=-50,  # -5% loss, within 10% limit
            bankroll=1000,
        )

        assert not result.triggered

    def test_stop_on_time_to_expiry(self, checker):
        """Test stop near expiry."""
        result = checker.check_time_stop(100)  # 100s, below 300s

        assert result.triggered
        assert result.stop_type == StopType.TIME_TO_EXPIRY
        assert result.should_close

    def test_position_stop_doesnt_close(self, checker):
        """Test position stop doesn't force close."""
        result = checker.check_position_stop(
            position_size=250,  # 25% of bankroll
            bankroll=1000,
        )

        assert result.triggered
        assert result.stop_type == StopType.MAX_POSITION
        assert not result.should_close  # Just prevents increasing


class TestInventoryManager:
    """Tests for InventoryManager."""

    @pytest.fixture
    def manager(self):
        """Create inventory manager."""
        return InventoryManager(
            max_net_frac=0.20,
            max_long_frac=0.20,
            max_short_frac=0.20,
        )

    def test_get_status_flat(self, manager):
        """Test status with no position."""
        status = manager.get_status(position_size=0, bankroll=1000)

        assert status.position_size == 0
        assert status.position_frac == 0
        assert status.can_buy
        assert status.can_sell
        assert not status.at_max_long
        assert not status.at_max_short

    def test_get_status_long(self, manager):
        """Test status with long position."""
        status = manager.get_status(position_size=100, bankroll=1000)

        assert status.position_frac == 0.10
        assert status.can_buy  # Still below 20%
        assert status.can_sell

    def test_get_status_at_max_long(self, manager):
        """Test status at max long."""
        status = manager.get_status(position_size=200, bankroll=1000)

        assert status.at_max_long
        assert not status.can_buy
        assert status.can_sell

    def test_get_status_at_max_short(self, manager):
        """Test status at max short."""
        status = manager.get_status(position_size=-200, bankroll=1000)

        assert status.at_max_short
        assert status.can_buy
        assert not status.can_sell

    def test_get_order_size_limit_buy(self, manager):
        """Test order size limit for buy."""
        # Position is 100, max is 200, so can buy 100 more
        limited = manager.get_order_size_limit(
            side="BUY",
            position_size=100,
            bankroll=1000,
            desired_size=150,
        )

        assert limited == 100  # Limited to available capacity

    def test_get_order_size_limit_sell(self, manager):
        """Test order size limit for sell."""
        # Position is 100, max short is -200, so can sell 300 total
        limited = manager.get_order_size_limit(
            side="SELL",
            position_size=100,
            bankroll=1000,
            desired_size=150,
        )

        assert limited == 150  # Within limit

    def test_compute_inventory_skew_long(self, manager):
        """Test inventory skew when long."""
        # At 10% long, should have positive skew to lower quotes
        skew = manager.compute_inventory_skew(
            position_size=100,
            bankroll=1000,
            max_skew=0.5,
        )

        assert skew > 0  # Positive = lower quotes to encourage selling

    def test_compute_inventory_skew_short(self, manager):
        """Test inventory skew when short."""
        skew = manager.compute_inventory_skew(
            position_size=-100,
            bankroll=1000,
            max_skew=0.5,
        )

        assert skew < 0  # Negative = raise quotes to encourage buying


class TestRiskManager:
    """Tests for unified RiskManager."""

    @pytest.fixture
    def manager(self):
        """Create risk manager."""
        return RiskManager(
            jump_z=3.0,
            momentum_z=2.0,
            extreme_prob_threshold=0.02,
            stop_prob_low=0.02,
            stop_prob_high=0.98,
            max_loss_pct=0.10,
            max_net_frac=0.20,
            gamma_danger_threshold=0.10,
            gamma_danger_multiplier=2.0,
        )

    @pytest.fixture
    def normal_belief(self):
        """Create normal belief state."""
        return BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
        )

    def test_evaluate_normal_conditions(self, manager, normal_belief):
        """Test evaluation under normal conditions."""
        decision = manager.evaluate(
            belief=normal_belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )

        assert decision.allow_trading
        assert decision.allow_buy
        assert decision.allow_sell
        assert not decision.close_position

    def test_evaluate_vetoed(self, manager):
        """Test evaluation when vetoed."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
            jump_detected=True,
        )

        decision = manager.evaluate(
            belief=belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )

        assert not decision.allow_trading
        assert decision.veto.vetoed

    def test_evaluate_stop_triggered(self, manager):
        """Test evaluation when stop triggered."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.01,  # Extreme low
            mid_logit=-4.6,
            sigma_b=0.05,
        )

        decision = manager.evaluate(
            belief=belief,
            position_size=100,
            position_pnl=-50,
            bankroll=1000,
        )

        assert decision.close_position
        assert decision.stop.triggered

    def test_evaluate_inventory_restricted(self, manager, normal_belief):
        """Test evaluation with inventory restriction."""
        decision = manager.evaluate(
            belief=normal_belief,
            position_size=200,  # At max long
            position_pnl=0,
            bankroll=1000,
        )

        assert decision.allow_trading
        assert not decision.allow_buy  # Can't increase long
        assert decision.allow_sell

    def test_gamma_multiplier_normal(self, manager, normal_belief):
        """Test gamma multiplier at normal price."""
        decision = manager.evaluate(
            belief=normal_belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )

        assert decision.gamma_multiplier == 1.0

    def test_gamma_multiplier_danger_zone(self, manager):
        """Test gamma multiplier in danger zone."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.05,  # Within 0.10 of extreme
            mid_logit=-2.9,
            sigma_b=0.05,
        )

        decision = manager.evaluate(
            belief=belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )

        assert decision.gamma_multiplier > 1.0

    def test_restriction_reasons(self, manager):
        """Test restriction reasons are accumulated."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
            jump_detected=True,
        )

        decision = manager.evaluate(
            belief=belief,
            position_size=200,  # At max
            position_pnl=0,
            bankroll=1000,
        )

        assert len(decision.restriction_reasons) >= 1
        assert "jump_detected" in decision.restriction_reasons
