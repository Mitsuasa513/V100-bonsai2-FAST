#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_add_rms_norm_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

void ggml_cuda_op_add_rms_norm_scale_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, float * row_scale);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// EXPERIMENTAL (Codex, V100): fuse "rms_norm -> ggml_scale(scalar)" into a single kernel. The SSM
// layers of the hybrid models apply a constant scale to the q/k conv norms, which costs one extra
// kernel launch per tensor (96 launches per decoded token in the test model).
void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
