#include "utils.hpp"

// trt_reduce
using fptr_t = int64_t;
fptr_t init_custom_ar(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                      const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out);
void dispose(fptr_t _fa);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);

// moe_align_block_size
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer);

void moe_fused_experts(torch::Tensor hidden_states, torch::Tensor w1, torch::Tensor w2,
		          torch::Tensor topk_weights, torch::Tensor topk_ids,
			  torch::Tensor w1_scale, torch::Tensor w2_scale,
			  torch::Tensor a1_scale, torch::Tensor a2_scale,
                          torch::Tensor sorted_ids, torch::Tensor sorted_weights,
			  torch::Tensor sorted_expert_ids, torch::Tensor num_tokens_post_pad,
			  torch::Tensor out, int64_t block_m)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // trt_reduce
  m.def("init_custom_ar", &init_custom_ar, "init custom allreduce meta (CUDA)");
  m.def("dispose", &dispose, "dispose custom allreduce meta");
  m.def("all_reduce", &all_reduce, "custom all reduce (CUDA)");
  // moe_align_block_size
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (CUDA)");
  m.def("moe_fused_experts", &moe_fused_experts, "MOE implementation by ck")
}
