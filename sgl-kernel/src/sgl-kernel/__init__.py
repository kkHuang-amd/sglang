from sglang.srt.utils import is_hip

if is_hip():
    from sgl_kernel.ops import (
        moe_fused_experts,
    )

    __all__ = [
        "moe_fused_experts",
    ]
else:
    from sgl_kernel.ops import (
        custom_dispose,
        custom_reduce,
        init_custom_reduce,
        moe_align_block_size,
    )
    
    __all__ = [
        "moe_align_block_size",
        "init_custom_reduce",
        "custom_dispose",
        "custom_reduce",
    ]
