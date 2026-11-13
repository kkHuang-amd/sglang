"""Eager runner of a mixed-verify forward (model_executor/mixed_verify.py)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.mixed_verify_backend import MixedVerifyAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.mixed_verify import (
    MixedVerifyOutput,
    MixedVerifySegments,
)
from sglang.srt.utils.device_timer import device_timer_ctx

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class MixedVerifyRunner:
    """Runs new requests' extend rows and running requests' verify rows as one
    target forward. Takes the live runner as the phase runners do."""

    def __init__(self, model_runner: ModelRunner) -> None:
        # Imports ModelRunner at module load.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        self.model_runner = model_runner
        # The verify segment plans on a backend of its own: the runner's
        # backend holds the extend segment's metadata during the forward.
        verify_backend = model_runner._get_attention_backend()
        model_runner.runner_ids.bind_and_verify_backends([verify_backend])
        if (
            isinstance(verify_backend, HybridLinearAttnBackend)
            and verify_backend.linear_attn_backend.accept_lens_pool is not None
        ):
            # The next verify, on the runner's backend, would not see it.
            raise ValueError(
                "--speculative-mixed-verify needs the linear-attention verify "
                "state in the state pool, not on the attention backend."
            )
        self._attn_backend = MixedVerifyAttnBackend(
            extend=model_runner.attn_backend, verify=verify_backend
        )

    @property
    def verify_attn_backend(self):
        """The backend holding the verify segment's metadata after a forward:
        the post-verify state commit reads it there, not on the runner's."""
        return self._attn_backend.verify

    def execute(
        self, *, extend: ForwardBatch, verify: ForwardBatch
    ) -> MixedVerifyOutput:
        return self.forward(self.plan(extend=extend, verify=verify))

    @torch.no_grad()
    def plan(self, *, extend: ForwardBatch, verify: ForwardBatch) -> ForwardBatch:
        """Plan each segment's attention metadata on its backend; return the
        merged batch the forward runs."""
        assert extend.forward_mode == ForwardMode.EXTEND, extend.forward_mode
        assert verify.forward_mode.is_target_verify(), verify.forward_mode
        assert not extend.contains_mm_inputs(), "multimodal extend rows not supported"
        model_runner = self.model_runner
        backend = self._attn_backend
        with (
            model_runner.eager_runner.model_context(),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            for segment in (extend, verify):
                model_runner._prepare_eager_forward_batch(segment)
            model_runner._maybe_execute_deferred_mamba_cow_and_clear(extend)
            model_runner.eager_runner.plan_extend_metadata(extend, backend.extend)
            model_runner.eager_runner.plan_extend_metadata(verify, backend.verify)
            parent = self._merged_batch(extend=extend, verify=verify)
            model_runner._prepare_eager_forward_batch(parent)
        return parent

    @torch.no_grad()
    def forward(self, parent: ForwardBatch) -> MixedVerifyOutput:
        model_runner = self.model_runner
        with (
            model_runner.eager_runner.model_context(),
            device_timer_ctx(model_runner.device_timer, "mixed_verify"),
            forward_context(
                ForwardContext(attn_backend=self._attn_backend), forward_batch=parent
            ),
        ):
            return model_runner.model.forward(
                parent.input_ids, parent.positions, parent
            )

    def _merged_batch(
        self, *, extend: ForwardBatch, verify: ForwardBatch
    ) -> ForwardBatch:
        """The batch the layer stack outside attention reads: every row of both
        segments, labeled as the extend it is shaped like."""
        parent = replace(
            extend,
            batch_size=extend.batch_size + verify.batch_size,
            input_ids=torch.cat([extend.input_ids, verify.input_ids]),
            positions=torch.cat([extend.positions, verify.positions]),
            mixed_verify=MixedVerifySegments(extend=extend, verify=verify),
            residual_stream=None,
            global_num_token_non_padded=None,
            num_token_non_padded=None,
        )
        parent._init_num_token_non_padded(
            device=self.model_runner.device, is_draft_worker=False
        )
        return parent
