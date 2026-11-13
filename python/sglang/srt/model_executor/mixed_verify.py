"""One target forward over [extend rows | verify rows].

A mixed-verify forward runs the layer stack once over the prompt rows of new
requests followed by the speculative verify rows of running requests, so the
FFN weights are read once for both. Attention runs per segment, each with its
own ForwardBatch and attention backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MixedVerifySegments(msgspec.Struct, frozen=True, kw_only=True):
    """Row segments of a mixed-verify forward, in row order: extend, verify."""

    extend: ForwardBatch
    verify: ForwardBatch

    @property
    def num_extend_tokens(self) -> int:
        return self.extend.input_ids.shape[0]

    def with_backends(
        self, backend: AttentionBackend
    ) -> Iterator[Tuple[ForwardBatch, AttentionBackend, slice]]:
        """Each segment's batch, backend and rows of the merged tensor."""
        from sglang.srt.layers.attention.mixed_verify_backend import (
            MixedVerifyAttnBackend,
        )

        assert isinstance(backend, MixedVerifyAttnBackend), type(backend)
        split = self.num_extend_tokens
        yield self.extend, backend.extend, slice(0, split)
        yield self.verify, backend.verify, slice(split, None)


class MixedVerifyOutput(msgspec.Struct, kw_only=True):
    extend: LogitsProcessorOutput
    verify: LogitsProcessorOutput
