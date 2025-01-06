from sglang.srt.utils import is_hip


if is_hip():
    from sgl_kernel.ops._kernels import moe_fused_experts as _moe_fused_experts

    def moe_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_tokens_post_pad,
        out,
        block_m,
        fused_qunt,
        gate_only,
    
    ):
        _moe_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_tokens_post_pad,
            out,
            block_m,
            fused_qunt,
            gate_only,
        )

else:
    from sgl_kernel.ops._kernels import all_reduce as _all_reduce
    from sgl_kernel.ops._kernels import dispose as _dispose
    from sgl_kernel.ops._kernels import init_custom_ar as _init_custom_ar
    from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
    
    def init_custom_reduce(rank_id, num_devices, buffers, barrier_in, barrier_out):
        return _init_custom_ar(rank_id, num_devices, buffers, barrier_in, barrier_out)
    
    def custom_dispose(fa):
        _dispose(fa)
    
    def custom_reduce(fa, inp, out):
        _all_reduce(fa, inp, out)
    
    def moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    ):
        _moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            token_cnts_buffer,
            cumsum_buffer,
        )


