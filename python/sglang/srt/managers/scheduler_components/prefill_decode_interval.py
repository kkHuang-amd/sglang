"""Decode rounds a prefill batch buys before the next prefill (--prefill-decode-interval)."""

from typing import Optional


def prefill_decode_interval_rounds(
    *,
    interval: int,
    extend_tokens: Optional[int],
    chunk_tokens: Optional[int],
    fixed_tokens: int,
) -> int:
    # The rounds scale with the batch's cost, fixed_tokens + one per new token. A full
    # chunk keeps the whole interval, which bounds the share of time decode stalls.
    if (
        fixed_tokens <= 0
        or extend_tokens is None
        or not chunk_tokens
        or chunk_tokens <= 0
    ):
        return interval
    rounds = -(
        -interval * (extend_tokens + fixed_tokens) // (chunk_tokens + fixed_tokens)
    )
    return max(1, min(interval, rounds))
