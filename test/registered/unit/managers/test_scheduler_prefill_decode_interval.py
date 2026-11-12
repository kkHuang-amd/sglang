"""Tests for gating the prefill/decode interval on the running batch."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *,
    interval: int,
    min_bs: int,
    require_mlp_sync: bool = False,
    fixed_tokens: int = 0,
    chunk_tokens: int = 16384,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.prefill_decode_interval_fixed_tokens = fixed_tokens
    scheduler.chunked_prefill_size = chunk_tokens
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _arm(scheduler: Scheduler, extend_tokens: int = 16384) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
            extend_num_tokens=extend_tokens,
        )
    )


def _make_running_batch(batch_size: int, *, prefill_only: bool = False):
    return SimpleNamespace(
        batch_size=lambda: batch_size,
        is_empty=lambda: batch_size == 0,
        is_prefill_only=prefill_only,
    )


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillDecodeIntervalNothingDecodes(unittest.TestCase):
    # These use the default min_bs=0, where the interval applies at any batch size.

    def test_empty_running_batch_does_not_defer(self):
        # The next chunk of a long prompt on an idle server runs right away.
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))

    def test_prefill_only_running_batch_does_not_defer(self):
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        running = _make_running_batch(3, prefill_only=True)
        self.assertFalse(scheduler._should_defer_prefill(running))

    def test_single_decoding_request_keeps_interval(self):
        scheduler = _make_scheduler(interval=2, min_bs=0)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(1)))

    def test_dp_defers_without_local_decode(self):
        # Under DP attention, peer ranks may still be decoding.
        scheduler = _make_scheduler(interval=2, min_bs=0, require_mlp_sync=True)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(0)))
        running = _make_running_batch(3, prefill_only=True)
        self.assertTrue(scheduler._should_defer_prefill(running))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))


class TestPrefillDecodeIntervalFixedTokens(unittest.TestCase):
    # interval 32, chunk 16384, fixed cost 1600 tokens: rounds = ceil(32 * (n + 1600) / 17984).

    def test_rounds_scale_with_batch_cost_and_a_full_chunk_keeps_the_interval(self):
        # A full chunk must keep the whole interval: that is what bounds the share of
        # time decode stalls behind prefill at the fixed-interval level.
        for extend_tokens, rounds in ((16384, 32), (20000, 32), (3500, 10), (1, 3)):
            scheduler = _make_scheduler(interval=32, min_bs=0, fixed_tokens=1600)
            _arm(scheduler, extend_tokens=extend_tokens)
            self.assertEqual(
                scheduler._prefill_decode_interval_remaining, rounds, extend_tokens
            )

    def test_never_drops_below_one_round(self):
        scheduler = _make_scheduler(interval=4, min_bs=0, fixed_tokens=1)
        _arm(scheduler, extend_tokens=1)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)

    def test_fixed_interval_kept_when_disabled_dp_or_unchunked(self):
        cases = (
            dict(fixed_tokens=0),
            # DP ranks run different token counts; their cadence must stay identical.
            dict(fixed_tokens=1600, require_mlp_sync=True),
            dict(fixed_tokens=1600, chunk_tokens=-1),
        )
        for kwargs in cases:
            scheduler = _make_scheduler(interval=32, min_bs=0, **kwargs)
            _arm(scheduler, extend_tokens=100)
            self.assertEqual(scheduler._prefill_decode_interval_remaining, 32, kwargs)


if __name__ == "__main__":
    unittest.main()
