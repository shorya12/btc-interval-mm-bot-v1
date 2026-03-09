"""High-level quote calculator integrating belief state and A-S model."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.belief_state import BeliefState
from .avellaneda_stoikov import AvellanedaStoikov
from .params import Quote
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuoteContext:
    """Context for quote calculation."""

    belief: BeliefState
    inventory: float
    time_remaining: float
    signal_skew: float = 0.0
    gamma_multiplier: float = 1.0  # For danger zone adjustment
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class QuoteDecision:
    """
    Quote decision with optional suppression reasons.

    Contains the computed quote and metadata about why quoting
    may be suppressed (vetoes, stops, etc.).
    """

    quote: Quote | None
    should_quote: bool
    suppression_reasons: list[str]
    context: QuoteContext
    metadata: dict[str, Any]

    @property
    def is_suppressed(self) -> bool:
        return not self.should_quote


class QuoteCalculator:
    """
    High-level quote calculator.

    Integrates:
    - Belief state from market data
    - Avellaneda-Stoikov optimal quoting
    - Signal skew from lag model
    - Risk-based suppression
    """

    def __init__(
        self,
        gamma: float = 0.1,
        base_spread_x: float = 0.01,
        kappa: float | None = None,
        min_spread_prob: float = 0.001,
        gamma_danger_threshold: float = 0.1,
        gamma_danger_multiplier: float = 2.0,
    ) -> None:
        """
        Initialize quote calculator.

        Args:
            gamma: Base risk aversion parameter
            base_spread_x: Base spread multiplier
            kappa: Order arrival rate (optional)
            min_spread_prob: Minimum spread in probability space
            gamma_danger_threshold: Distance from 0/1 to trigger gamma increase
            gamma_danger_multiplier: Multiplier for gamma in danger zone
        """
        self.base_gamma = gamma
        self.as_model = AvellanedaStoikov(gamma, base_spread_x, kappa)
        self.min_spread_prob = min_spread_prob
        self.gamma_danger_threshold = gamma_danger_threshold
        self.gamma_danger_multiplier = gamma_danger_multiplier

    def calculate(self, context: QuoteContext) -> QuoteDecision:
        """
        Calculate optimal quotes given context.

        Args:
            context: Quote calculation context

        Returns:
            QuoteDecision with quote and metadata
        """
        suppression_reasons: list[str] = []
        metadata: dict[str, Any] = {}

        # Check for jump/momentum suppression
        if context.belief.jump_detected:
            suppression_reasons.append("jump_detected")
        if context.belief.momentum_detected:
            suppression_reasons.append("momentum_detected")

        # Adjust gamma for danger zone (near 0 or 1)
        effective_gamma = self._compute_effective_gamma(context)
        metadata["effective_gamma"] = effective_gamma
        metadata["base_gamma"] = self.base_gamma

        # Temporarily update A-S model gamma
        original_gamma = self.as_model.gamma
        self.as_model.gamma = effective_gamma * context.gamma_multiplier

        try:
            # Compute quote
            quote = self.as_model.compute_quotes(
                mid_logit=context.belief.mid_logit,
                inventory=context.inventory,
                sigma_b=context.belief.sigma_b,
                time_remaining=context.time_remaining,
                signal_skew=context.signal_skew,
                min_spread_prob=self.min_spread_prob,
            )

            # Validate quote
            if not quote.is_valid():
                suppression_reasons.append("invalid_quote")
                logger.warning(
                    "invalid_quote_generated",
                    bid=quote.bid_price,
                    ask=quote.ask_price,
                )

            should_quote = len(suppression_reasons) == 0 and quote.is_valid()

            return QuoteDecision(
                quote=quote if quote.is_valid() else None,
                should_quote=should_quote,
                suppression_reasons=suppression_reasons,
                context=context,
                metadata=metadata,
            )

        finally:
            # Restore original gamma
            self.as_model.gamma = original_gamma

    def _compute_effective_gamma(self, context: QuoteContext) -> float:
        """
        Compute effective gamma.

        NOTE: Danger zone adjustment is handled by RiskManager via gamma_multiplier
        in the QuoteContext. We just return base_gamma here to avoid double-adjustment.

        The final gamma = base_gamma * context.gamma_multiplier (applied in calculate())
        """
        # Don't apply danger zone adjustment here - RiskManager handles it
        # via context.gamma_multiplier to avoid double-adjustment
        return self.base_gamma

    def calculate_two_sided(
        self,
        context: QuoteContext,
        allow_buy: bool = True,
        allow_sell: bool = True,
    ) -> QuoteDecision:
        """
        Calculate quotes with directional restrictions.

        Args:
            context: Quote calculation context
            allow_buy: Whether to generate bid
            allow_sell: Whether to generate ask

        Returns:
            QuoteDecision with potentially one-sided quote
        """
        decision = self.calculate(context)

        if decision.quote is None:
            return decision

        # Apply directional restrictions
        if not allow_buy and not allow_sell:
            return QuoteDecision(
                quote=None,
                should_quote=False,
                suppression_reasons=decision.suppression_reasons + ["no_direction_allowed"],
                context=context,
                metadata=decision.metadata,
            )

        # For now, return full quote - order placement will handle restrictions
        # This preserves quote information for logging/analysis
        metadata = decision.metadata.copy()
        metadata["allow_buy"] = allow_buy
        metadata["allow_sell"] = allow_sell

        return QuoteDecision(
            quote=decision.quote,
            should_quote=decision.should_quote,
            suppression_reasons=decision.suppression_reasons,
            context=context,
            metadata=metadata,
        )
