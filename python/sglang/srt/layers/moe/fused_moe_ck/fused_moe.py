# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

import functools
import logging
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from sgl_kernel import moe_fused_experts as sgl_moe_fused_experts

logger = logging.getLogger(__name__)

def fused_experts_ck(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
):

    block_m = 32
    tokens = hidden_states.shape[0]
    experts = w1.shape[0]

    topk = topk_ids.shape[1]

    out_hidden_states = torch.empty_like(hidden_states)
    max_num_tokens_padded = topk * tokens + experts * block_m - topk

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_weight = torch.empty(
        (max_num_tokens_padded,), dtype=topk_weights.dtype, device=topk_ids.device
    )
    
    max_num_m_blocks = math.floor((max_num_tokens_padded + block_m - 1) / block_m)

    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )

    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    #if use_fp8_w8a8:
    #    assert B_scale is not None
    #    if block_shape is None:
    #        padded_size = padding_size
    #        A, A_scale = ops.scaled_fp8_quant(A, A_scale)
    #    else:
    #        assert len(block_shape) == 2
    #        block_n, block_k = block_shape[0], block_shape[1]
    #        A, A_scale = per_token_group_quant_fp8(A, block_k)
    #        assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    #        assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
    #        assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]

    #if a1_scale == None:
    #    a1_scale = torch.empty(1)
    #if a2_scale == None:
    #    a2_scale = torch.empty(1)

    sgl_moe_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        sorted_token_ids,
        sorted_weight,
        sorted_expert_ids,
        num_tokens_post_pad,
        out_hidden_states,
        32,
        1,
        0)

    return out_hidden_states
