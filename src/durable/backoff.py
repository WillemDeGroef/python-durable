"""
Backoff strategies for task retries.

Usage:
    exponential(base=2, max=60)   → 2s, 4s, 8s, 16s... capped at 60s
    constant(5)                   → always 5s
    linear(start=2, step=3)       → 2s, 5s, 8s, 11s...
"""

from typing import Callable

# A strategy takes the attempt number (0-indexed) and returns seconds to wait.
BackoffStrategy = Callable[[int], float]


def exponential(base: float = 2.0, max: float = 60.0) -> BackoffStrategy:
    """Exponential backoff: base^attempt, capped at max seconds."""

    def strategy(attempt: int) -> float:
        return min(base**attempt, max)

    return strategy


def constant(seconds: float) -> BackoffStrategy:
    """Always wait the same number of seconds."""

    def strategy(attempt: int) -> float:
        return seconds

    return strategy


def linear(start: float = 1.0, step: float = 1.0) -> BackoffStrategy:
    """Linearly increasing wait: start, start+step, start+2*step..."""

    def strategy(attempt: int) -> float:
        return start + step * attempt

    return strategy
