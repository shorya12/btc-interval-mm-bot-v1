"""Tests for position sizing logic."""

import pytest
from src.risk.inventory import InventoryManager, InventoryStatus
from src.risk.risk_manager import RiskManager, RiskDecision
from src.belief_state import BeliefState


class TestPositionSizing:
    """Tests for position sizing calculations."""

    @pytest.fixture
    def inventory_manager(self):
        """Create inventory manager with typical config."""
        return InventoryManager(
            max_net_frac=0.20,  # Max 20% of bankroll
            max_long_frac=0.20,
            max_short_frac=0.20,
        )

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with typical config."""
        return RiskManager(
            max_net_frac=0.20,
            gamma_danger_threshold=0.10,
            gamma_danger_multiplier=2.0,
        )

    # =========================================================================
    # Inventory Manager Tests
    # =========================================================================

    def test_max_order_size_from_flat(self, inventory_manager):
        """Test maximum order size when starting from flat position."""
        # With $1000 bankroll and 20% max, can go up to $200 long or short
        max_buy = inventory_manager.get_order_size_limit(
            side="BUY",
            position_size=0,
            bankroll=1000,
            desired_size=500,  # Want to buy $500 worth
        )
        assert max_buy == 200  # Limited to 20% of bankroll

        max_sell = inventory_manager.get_order_size_limit(
            side="SELL",
            position_size=0,
            bankroll=1000,
            desired_size=500,
        )
        assert max_sell == 200

    def test_max_order_size_with_existing_long(self, inventory_manager):
        """Test order sizing with existing long position."""
        # Already long $100, max is $200, so can only buy $100 more
        max_buy = inventory_manager.get_order_size_limit(
            side="BUY",
            position_size=100,
            bankroll=1000,
            desired_size=150,
        )
        assert max_buy == 100  # Can only buy 100 more (200 - 100)

        # Can sell up to $300 (go from +100 to -200)
        max_sell = inventory_manager.get_order_size_limit(
            side="SELL",
            position_size=100,
            bankroll=1000,
            desired_size=500,
        )
        assert max_sell == 300  # Can sell 300 to go from +100 to -200

    def test_max_order_size_with_existing_short(self, inventory_manager):
        """Test order sizing with existing short position."""
        # Already short $100, max is -$200
        max_sell = inventory_manager.get_order_size_limit(
            side="SELL",
            position_size=-100,
            bankroll=1000,
            desired_size=150,
        )
        assert max_sell == 100  # Can only sell 100 more

        # Can buy up to $300 (go from -100 to +200)
        max_buy = inventory_manager.get_order_size_limit(
            side="BUY",
            position_size=-100,
            bankroll=1000,
            desired_size=500,
        )
        assert max_buy == 300

    def test_at_max_long_no_more_buys(self, inventory_manager):
        """Test that no more buys are allowed at max long."""
        status = inventory_manager.get_status(position_size=200, bankroll=1000)
        
        assert status.at_max_long
        assert not status.can_buy
        assert status.can_sell

        max_buy = inventory_manager.get_order_size_limit(
            side="BUY",
            position_size=200,
            bankroll=1000,
            desired_size=50,
        )
        assert max_buy == 0  # Cannot buy any more

    def test_at_max_short_no_more_sells(self, inventory_manager):
        """Test that no more sells are allowed at max short."""
        status = inventory_manager.get_status(position_size=-200, bankroll=1000)
        
        assert status.at_max_short
        assert status.can_buy
        assert not status.can_sell

        max_sell = inventory_manager.get_order_size_limit(
            side="SELL",
            position_size=-200,
            bankroll=1000,
            desired_size=50,
        )
        assert max_sell == 0  # Cannot sell any more

    def test_inventory_skew_long_position(self, inventory_manager):
        """Test inventory skew calculation for long position."""
        # When long, skew should be positive (lower quotes to encourage selling)
        skew = inventory_manager.compute_inventory_skew(
            position_size=100,
            bankroll=1000,
            max_skew=0.5,
        )
        assert skew > 0
        assert skew <= 0.5  # Should not exceed max_skew

    def test_inventory_skew_short_position(self, inventory_manager):
        """Test inventory skew calculation for short position."""
        # When short, skew should be negative (raise quotes to encourage buying)
        skew = inventory_manager.compute_inventory_skew(
            position_size=-100,
            bankroll=1000,
            max_skew=0.5,
        )
        assert skew < 0
        assert skew >= -0.5

    def test_inventory_skew_flat_position(self, inventory_manager):
        """Test inventory skew is zero when flat."""
        skew = inventory_manager.compute_inventory_skew(
            position_size=0,
            bankroll=1000,
            max_skew=0.5,
        )
        assert skew == 0.0

    # =========================================================================
    # Risk Manager Integration Tests
    # =========================================================================

    def test_risk_manager_position_limit_enforcement(self, risk_manager):
        """Test risk manager enforces position limits."""
        belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
        )

        # At max long position
        decision = risk_manager.evaluate(
            belief=belief,
            position_size=200,  # At 20% of $1000
            position_pnl=0,
            bankroll=1000,
        )

        assert decision.allow_trading
        assert not decision.allow_buy  # Can't increase long
        assert decision.allow_sell  # Can reduce position

    def test_risk_manager_order_size_limit(self, risk_manager):
        """Test risk manager returns correct order size limits."""
        # With 10% position, can buy 10% more
        limit = risk_manager.get_order_size_limit(
            side="BUY",
            position_size=100,  # 10% of bankroll
            bankroll=1000,
            desired_size=500,
        )
        assert limit == 100  # Limited to remaining 10% (200 - 100)

    def test_risk_manager_gamma_adjustment_near_extremes(self, risk_manager):
        """Test gamma is increased near price extremes."""
        # Normal price
        normal_belief = BeliefState(
            token_id="test",
            mid_prob=0.5,
            mid_logit=0.0,
            sigma_b=0.05,
        )
        normal_decision = risk_manager.evaluate(
            belief=normal_belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )
        assert normal_decision.gamma_multiplier == 1.0

        # Near extreme (within danger threshold)
        extreme_belief = BeliefState(
            token_id="test",
            mid_prob=0.05,  # Close to 0
            mid_logit=-2.9,
            sigma_b=0.05,
        )
        extreme_decision = risk_manager.evaluate(
            belief=extreme_belief,
            position_size=0,
            position_pnl=0,
            bankroll=1000,
        )
        assert extreme_decision.gamma_multiplier > 1.0


class TestPositionSizingEdgeCases:
    """Edge case tests for position sizing."""

    @pytest.fixture
    def inventory_manager(self):
        return InventoryManager(
            max_net_frac=0.20,
            max_long_frac=0.20,
            max_short_frac=0.20,
        )

    def test_zero_bankroll(self, inventory_manager):
        """Test behavior with zero bankroll."""
        status = inventory_manager.get_status(position_size=0, bankroll=0)
        # Should handle gracefully without division by zero
        assert status.position_size == 0

    def test_negative_bankroll_rejected(self, inventory_manager):
        """Test behavior with negative bankroll."""
        # This should be handled gracefully
        status = inventory_manager.get_status(position_size=0, bankroll=-100)
        # Depending on implementation, may return safe defaults
        assert status.position_size == 0

    def test_very_small_positions(self, inventory_manager):
        """Test handling of very small position sizes."""
        status = inventory_manager.get_status(position_size=0.001, bankroll=1000)
        assert status.can_buy
        assert status.can_sell

    def test_position_exceeds_max(self, inventory_manager):
        """Test when position already exceeds max (e.g., from price movement)."""
        # Position is 25% of bankroll (exceeds 20% max)
        status = inventory_manager.get_status(position_size=250, bankroll=1000)
        
        assert status.at_max_long
        assert not status.can_buy
        assert status.can_sell  # Should still be able to reduce

    def test_fractional_positions(self, inventory_manager):
        """Test handling of fractional position sizes."""
        max_buy = inventory_manager.get_order_size_limit(
            side="BUY",
            position_size=100.5,
            bankroll=1000,
            desired_size=99.5,
        )
        # Should still work with fractional values
        assert max_buy == 99.5  # Within limit
