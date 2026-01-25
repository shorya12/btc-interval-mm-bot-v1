"""Avellaneda-Stoikov market-making algorithm in logit space."""

import math

from src.belief_state.logit import logit, sigmoid
from .params import ASParams, Quote
from src.common.logging import get_logger

logger = get_logger(__name__)


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov optimal market-making in logit space.

    The classic A-S model optimizes quotes to maximize expected utility
    while managing inventory risk. This implementation works in logit space
    which is more appropriate for prediction market probabilities.

    Key equations (in logit space):
    - Reservation price: r = mid - gamma * sigma^2 * (T-t) * q
    - Optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

    Where:
    - mid: current mid-price in logit space
    - gamma: risk aversion
    - sigma: volatility in logit space
    - T-t: time remaining
    - q: inventory
    - kappa: order arrival rate
    """

    def __init__(
        self,
        gamma: float = 0.1,
        base_spread_x: float = 0.01,
        kappa: float | None = None,
    ) -> None:
        """
        Initialize A-S market maker.

        Args:
            gamma: Risk aversion parameter (0-1)
            base_spread_x: Base spread multiplier
            kappa: Order arrival rate (optional)
        """
        self.gamma = gamma
        self.base_spread_x = base_spread_x
        self.kappa = kappa

    def compute_reservation_price(
        self,
        mid_logit: float,
        inventory: float,
        sigma_b: float,
        time_remaining: float,
    ) -> float:
        """
        Compute the reservation (indifference) price in logit space.

        The reservation price is where the market maker is indifferent
        between buying and selling. It's shifted from mid based on inventory.

        Args:
            mid_logit: Mid-price in logit space
            inventory: Current inventory (positive = long)
            sigma_b: Belief volatility in logit space
            time_remaining: Time remaining to horizon (0-1)

        Returns:
            Reservation price in logit space
        """
        # Inventory adjustment: shift away from current position
        # If long (q > 0), lower reservation to encourage selling
        # If short (q < 0), raise reservation to encourage buying
        inventory_adjustment = self.gamma * (sigma_b ** 2) * time_remaining * inventory

        reservation = mid_logit - inventory_adjustment
        return reservation

    def compute_optimal_spread(
        self,
        sigma_b: float,
        time_remaining: float,
    ) -> float:
        """
        Compute the optimal total spread in logit space.

        Args:
            sigma_b: Belief volatility in logit space
            time_remaining: Time remaining to horizon (0-1)

        Returns:
            Optimal spread in logit space
        """
        # Base spread from volatility and time
        volatility_term = self.gamma * (sigma_b ** 2) * time_remaining

        # Arrival rate term (if kappa is specified)
        if self.kappa is not None and self.kappa > 0:
            arrival_term = (2 / self.gamma) * math.log(1 + self.gamma / self.kappa)
        else:
            # Simplified: use base_spread_x as minimum spread
            arrival_term = 0

        # Total spread
        spread = volatility_term + arrival_term

        # Apply base spread multiplier as floor
        min_spread = self.base_spread_x * sigma_b * 2  # Scale by volatility
        spread = max(spread, min_spread)

        return spread

    def compute_quotes(
        self,
        mid_logit: float,
        inventory: float,
        sigma_b: float,
        time_remaining: float,
        signal_skew: float = 0.0,
        min_spread_prob: float = 0.001,
    ) -> Quote:
        """
        Compute optimal bid/ask quotes.

        Args:
            mid_logit: Mid-price in logit space
            inventory: Current inventory
            sigma_b: Belief volatility in logit space
            time_remaining: Time remaining to horizon (0-1)
            signal_skew: External signal skew in logit space
            min_spread_prob: Minimum spread in probability space

        Returns:
            Quote with bid/ask prices
        """
        # Compute reservation price (shifted by inventory)
        reservation = self.compute_reservation_price(
            mid_logit, inventory, sigma_b, time_remaining
        )

        # Apply signal skew to reservation
        # Positive skew = expect price to go up = raise quotes
        reservation_with_skew = reservation + signal_skew

        # Compute optimal spread
        spread = self.compute_optimal_spread(sigma_b, time_remaining)
        half_spread = spread / 2

        # Compute bid/ask in logit space
        bid_logit = reservation_with_skew - half_spread
        ask_logit = reservation_with_skew + half_spread

        # Convert to probability space
        bid_price = sigmoid(bid_logit)
        ask_price = sigmoid(ask_logit)

        # Ensure minimum spread in probability space
        if ask_price - bid_price < min_spread_prob:
            mid_prob = (bid_price + ask_price) / 2
            bid_price = max(0.001, mid_prob - min_spread_prob / 2)
            ask_price = min(0.999, mid_prob + min_spread_prob / 2)
            bid_logit = logit(bid_price)
            ask_logit = logit(ask_price)

        # Compute inventory skew for logging
        inventory_skew = mid_logit - self.compute_reservation_price(
            mid_logit, inventory, sigma_b, time_remaining
        )

        quote = Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_logit=bid_logit,
            ask_logit=ask_logit,
            reservation_price=sigmoid(reservation_with_skew),
            reservation_logit=reservation_with_skew,
            spread=ask_price - bid_price,
            spread_logit=spread,
            half_spread_logit=half_spread,
            inventory_skew=inventory_skew,
            signal_skew=signal_skew,
            metadata={
                "gamma": self.gamma,
                "sigma_b": sigma_b,
                "time_remaining": time_remaining,
                "inventory": inventory,
            },
        )

        logger.debug(
            "quote_computed",
            bid=round(bid_price, 4),
            ask=round(ask_price, 4),
            spread_bps=round(quote.spread_bps, 1),
            inventory_skew=round(inventory_skew, 4),
            signal_skew=round(signal_skew, 4),
        )

        return quote

    def update_params(
        self,
        gamma: float | None = None,
        base_spread_x: float | None = None,
        kappa: float | None = None,
    ) -> None:
        """Update A-S parameters."""
        if gamma is not None:
            if not 0 <= gamma <= 1:
                raise ValueError(f"gamma must be in [0, 1], got {gamma}")
            self.gamma = gamma
        if base_spread_x is not None:
            if base_spread_x < 0:
                raise ValueError(f"base_spread_x must be non-negative")
            self.base_spread_x = base_spread_x
        if kappa is not None:
            self.kappa = kappa
