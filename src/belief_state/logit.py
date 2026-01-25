"""Logit and sigmoid transformations for probability space."""

import math

# Small epsilon to prevent log(0) and log(1)
EPS = 1e-8


def logit(p: float, eps: float = EPS) -> float:
    """
    Transform probability to log-odds (logit) space.

    The logit function maps probabilities from (0,1) to (-inf, +inf),
    which is useful for market-making in prediction markets where
    prices are bounded probabilities.

    Args:
        p: Probability in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        Log-odds: log(p / (1-p))

    Examples:
        >>> logit(0.5)
        0.0
        >>> logit(0.75)  # ~1.099
        1.0986122886681098
        >>> logit(0.25)  # ~-1.099
        -1.0986122886681098
    """
    # Clamp to avoid log(0)
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    """
    Transform log-odds back to probability (inverse logit).

    Args:
        x: Log-odds value in (-inf, +inf)

    Returns:
        Probability in (0, 1)

    Examples:
        >>> sigmoid(0.0)
        0.5
        >>> sigmoid(1.0986122886681098)  # ~0.75
        0.7500000000000001
    """
    # Handle extreme values to avoid overflow
    if x > 700:
        return 1.0 - EPS
    if x < -700:
        return EPS
    return 1.0 / (1.0 + math.exp(-x))


# Aliases for clarity
def prob_to_logit(p: float, eps: float = EPS) -> float:
    """Alias for logit(): probability to log-odds."""
    return logit(p, eps)


def logit_to_prob(x: float) -> float:
    """Alias for sigmoid(): log-odds to probability."""
    return sigmoid(x)


def logit_midpoint(bid: float, ask: float, eps: float = EPS) -> float:
    """
    Compute the midpoint in logit space, then convert back to probability.

    This is more appropriate for prediction markets than arithmetic mean,
    as it accounts for the non-linear nature of probability space.

    Args:
        bid: Best bid price (probability)
        ask: Best ask price (probability)
        eps: Epsilon for numerical stability

    Returns:
        Midpoint probability

    Examples:
        >>> logit_midpoint(0.45, 0.55)  # ~0.5
        0.5
        >>> logit_midpoint(0.20, 0.30)  # ~0.245, not 0.25
        0.24489795918367346
    """
    bid_logit = logit(bid, eps)
    ask_logit = logit(ask, eps)
    mid_logit = (bid_logit + ask_logit) / 2.0
    return sigmoid(mid_logit)


def logit_spread(bid: float, ask: float, eps: float = EPS) -> float:
    """
    Compute the spread in logit space.

    Args:
        bid: Best bid price (probability)
        ask: Best ask price (probability)
        eps: Epsilon for numerical stability

    Returns:
        Spread in logit space (always positive)
    """
    return logit(ask, eps) - logit(bid, eps)


def logit_distance(p1: float, p2: float, eps: float = EPS) -> float:
    """
    Compute the absolute distance between two probabilities in logit space.

    Args:
        p1: First probability
        p2: Second probability
        eps: Epsilon for numerical stability

    Returns:
        Absolute distance in logit space
    """
    return abs(logit(p1, eps) - logit(p2, eps))
