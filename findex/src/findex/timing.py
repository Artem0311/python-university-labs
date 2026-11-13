
"""Timing decorator for index operations."""

from collections.abc import Callable
from functools import wraps
from time import perf_counter


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Measure execution time while preserving function metadata."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            print(f"[timed] {func.__name__}: {elapsed:.6f} s")

    return wrapper
