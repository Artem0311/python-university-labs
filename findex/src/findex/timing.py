
"""Timing decorator for index operations."""

import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter

logger = logging.getLogger(__name__)


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Measure execution time while preserving function metadata."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            logger.debug(
                "[timed] %s: %.6f s",
                func.__name__,
                elapsed,
            )

    return wrapper
