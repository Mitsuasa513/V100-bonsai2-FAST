#include "common.cuh"

// state_wb: when the graph's state write-back CPY was elided by the conv-state matcher
// (decode only), the last conv_kernel-1 rows of the conv input are written to this
// destination view from inside the conv kernel -- see the note in ssm-conv.cu.
void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr,
                           ggml_tensor * silu_dst = nullptr, const ggml_cuda_state_rows_read * state_wb = nullptr);
