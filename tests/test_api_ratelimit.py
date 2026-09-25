"""Tests for the in-process sliding-window limiter used by the write endpoints."""

from qorgan.api_ratelimit import SlidingWindowLimiter


def test_allows_up_to_max_then_blocks_within_window():
    clock = [1000.0]
    limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60, clock=lambda: clock[0])
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("b") is True  # independent buckets per key


def test_window_slides():
    clock = [1000.0]
    limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60, clock=lambda: clock[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    clock[0] += 61
    assert limiter.allow("a") is True


def test_reset_clears_state():
    limiter = SlidingWindowLimiter(max_requests=1, window_seconds=60, clock=lambda: 0.0)
    assert limiter.allow("a") and not limiter.allow("a")
    limiter.reset()
    assert limiter.allow("a")
