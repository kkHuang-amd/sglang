from __future__ import annotations

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class MixedVerifyAttnBackend(AttentionBackend):
    """Dispatcher of a mixed-verify forward (model_executor/mixed_verify.py).

    Each row segment runs attention on its own backend, which holds that
    segment's metadata; the layer selects it through the forward context.
    """

    def __init__(self, *, extend: AttentionBackend, verify: AttentionBackend):
        super().__init__()
        self.extend = extend
        self.verify = verify
        # Pool readers outside attention resolve through the dispatcher.
        self.token_to_kv_pool = extend.token_to_kv_pool
        self.req_to_token_pool = extend.req_to_token_pool
        self.runner_ids = extend.runner_ids

    def init_forward_metadata(self, forward_batch):
        raise RuntimeError(
            "a mixed-verify forward plans each segment on its own backend"
        )
