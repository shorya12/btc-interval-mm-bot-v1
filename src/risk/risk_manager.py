"""Unified risk management interface."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.belief_state import BeliefState
from src.common.logging import get_logger
from .veto import VetoChecker, VetoResult, VetoReason
from .stops import StopChecker, StopResult, StopConfig, StopType
from .inventory import InventoryManager, InventoryStatus

logger = get_logger(__name__)


@dataclass
class RiskDecision:
    """
    Combined risk decision for trading.

    Aggregates veto, stop, and inventory decisions into a single
    actionable decision.
    """

    allow_trading: bool  # Can we trade at all?
    allow_buy: bool  # Can we place buy orders?
    allow_sell: bool  # Can we place sell orders?
    close_position: bool  # Should we close entire position?

    veto: VetoResult
    stop: StopResult
    inventory: InventoryStatus

    gamma_multiplier: float = 1.0  # Risk aversion multiplier
    max_order_size: float | None = None  # Maximum order size allowed

    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_restricted(self) -> bool:
        """Check if any trading restrictions are in place."""
        return not self.allow_trading or not self.allow_buy or not self.allow_sell

    @property
    def restriction_reasons(self) -> list[str]:
        """Get list of restriction reasons."""
        reasons = []
        if self.veto.vetoed:
            reasons.extend([r.value for r in self.veto.reasons])
        if self.stop.triggered:
            reasons.append(self.stop.stop_type.value if self.stop.stop_type else "stop")
        if not self.inventory.can_buy:
            reasons.append("inventory_max_long")
        if not self.inventory.can_sell:
            reasons.append("inventory_max_short")
        return reasons


class RiskManager:
    """
    Unified risk management for trading.

    Combines:
    - Veto checking (temporary pauses)
    - Stop checking (position closure)
    - Inventory management (position limits)
    - Gamma adjustment (risk aversion scaling)
    """

    def __init__(
        self,
        # Veto parameters
        jump_z: float = 3.0,
        momentum_z: float = 2.0,
        extreme_prob_threshold: float = 0.02,
        max_spread_bps: float = 500,
        min_liquidity: float = 100,
        # Stop parameters
        stop_prob_low: float = 0.02,
        stop_prob_high: float = 0.98,
        max_loss_pct: float = 0.10,
        min_time_to_expiry_seconds: float = 300,
        # Inventory parameters
        max_net_frac: float = 0.20,
        # Gamma danger zone
        gamma_danger_threshold: float = 0.10,
        gamma_danger_multiplier: float = 2.0,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            jump_z: Z-score for jump detection
            momentum_z: Z-score for momentum detection
            extreme_prob_threshold: Probability distance from 0/1 for extreme
            max_spread_bps: Maximum spread before veto
            min_liquidity: Minimum liquidity before veto
            stop_prob_low: Low probability stop
            stop_prob_high: High probability stop
            max_loss_pct: Maximum loss percentage stop
            min_time_to_expiry_seconds: Minimum time before expiry
            max_net_frac: Maximum position fraction
            gamma_danger_threshold: Distance from extreme for gamma increase
            gamma_danger_multiplier: Gamma multiplier in danger zone
        """
        self.veto_checker = VetoChecker(
            jump_z=jump_z,
            momentum_z=momentum_z,
            extreme_prob_threshold=extreme_prob_threshold,
            max_spread_bps=max_spread_bps,
            min_liquidity=min_liquidity,
        )

        self.stop_checker = StopChecker(StopConfig(
            stop_prob_low=stop_prob_low,
            stop_prob_high=stop_prob_high,
            max_loss_pct=max_loss_pct,
            min_time_to_expiry_seconds=min_time_to_expiry_seconds,
            max_position_frac=max_net_frac,
        ))

        self.inventory_manager = InventoryManager(
            max_net_frac=max_net_frac,
        )

        self.gamma_danger_threshold = gamma_danger_threshold
        self.gamma_danger_multiplier = gamma_danger_multiplier

    def evaluate(
        self,
        belief: BeliefState,
        position_size: float,
        position_pnl: float,
        bankroll: float,
        spread_bps: float | None = None,
        bid_depth: float | None = None,
        ask_depth: float | None = None,
        crypto_vol: float | None = None,
        time_to_expiry_seconds: float | None = None,
    ) -> RiskDecision:
        """
        Evaluate all risk conditions and return unified decision.

        Args:
            belief: Current belief state
            position_size: Current position size
            position_pnl: Current position PNL
            bankroll: Total bankroll
            spread_bps: Current spread in basis points
            bid_depth: Total bid depth
            ask_depth: Total ask depth
            crypto_vol: Crypto realized volatility
            time_to_expiry_seconds: Time until market expiry

        Returns:
            RiskDecision with trading allowances and actions
        """
        # Check vetoes
        veto = self.veto_checker.check(
            belief=belief,
            spread_bps=spread_bps,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            crypto_vol=crypto_vol,
        )

        # Check stops
        stop = self.stop_checker.check_all(
            current_prob=belief.mid_prob,
            position_size=position_size,
            position_pnl=position_pnl,
            bankroll=bankroll,
            time_to_expiry_seconds=time_to_expiry_seconds,
        )

        # Check inventory
        inventory = self.inventory_manager.get_status(position_size, bankroll)

        # Compute gamma multiplier for danger zone
        gamma_multiplier = self._compute_gamma_multiplier(belief.mid_prob)

        # Determine trading allowances
        allow_trading = not veto.vetoed and not stop.should_close
        allow_buy = allow_trading and inventory.can_buy
        allow_sell = allow_trading and inventory.can_sell

        # Position stop doesn't close but restricts increasing
        if stop.stop_type == StopType.MAX_POSITION:
            if position_size > 0:
                allow_buy = False  # Can't increase long
            else:
                allow_sell = False  # Can't increase short

        decision = RiskDecision(
            allow_trading=allow_trading,
            allow_buy=allow_buy,
            allow_sell=allow_sell,
            close_position=stop.should_close,
            veto=veto,
            stop=stop,
            inventory=inventory,
            gamma_multiplier=gamma_multiplier,
            metadata={
                "belief_mid": belief.mid_prob,
                "position_frac": inventory.position_frac,
                "gamma_multiplier": gamma_multiplier,
            },
        )

        if decision.is_restricted:
            logger.info(
                "risk_restricted",
                allow_trading=allow_trading,
                allow_buy=allow_buy,
                allow_sell=allow_sell,
                close_position=stop.should_close,
                reasons=decision.restriction_reasons,
            )

        return decision

    def _compute_gamma_multiplier(self, mid_prob: float) -> float:
        """
        Compute gamma multiplier based on price extremity.

        Near 0 or 1, increase gamma to be more conservative.

        Args:
            mid_prob: Current mid-price probability

        Returns:
            Gamma multiplier (>= 1.0)
        """
        distance_from_extreme = min(mid_prob, 1 - mid_prob)

        if distance_from_extreme < self.gamma_danger_threshold:
            # In danger zone - increase gamma
            danger_ratio = 1 - (distance_from_extreme / self.gamma_danger_threshold)
            gamma_increase = (self.gamma_danger_multiplier - 1) * danger_ratio
            return 1.0 + gamma_increase

        return 1.0

    def get_order_size_limit(
        self,
        side: str,
        position_size: float,
        bankroll: float,
        desired_size: float,
    ) -> float:
        """
        Get maximum allowed order size.

        Args:
            side: "BUY" or "SELL"
            position_size: Current position
            bankroll: Total bankroll
            desired_size: Desired order size

        Returns:
            Maximum allowed size
        """
        return self.inventory_manager.get_order_size_limit(
            side=side,
            position_size=position_size,
            bankroll=bankroll,
            desired_size=desired_size,
        )

    def update_config(
        self,
        max_net_frac: float | None = None,
        gamma_danger_threshold: float | None = None,
        gamma_danger_multiplier: float | None = None,
    ) -> None:
        """Update risk configuration."""
        if max_net_frac is not None:
            self.inventory_manager.update_limits(max_net_frac=max_net_frac)
        if gamma_danger_threshold is not None:
            self.gamma_danger_threshold = gamma_danger_threshold
        if gamma_danger_multiplier is not None:
            self.gamma_danger_multiplier = gamma_danger_multiplier
