#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused delta-net conv-state gather+concat (Codex, 2026-09-28): the graph does
//   GET_ROWS(live conv state row) -> RESHAPE -> CONCAT(transposed qkv) -> [views] -> CPY(new state)
// all of which is pure data movement. Read the live state from the recurrent cache row
// `rows[seq]` instead of a gathered copy and build the concatenation (conv_input) the
// SSM_CONV op consumes, in one launch.
//
// The write-back CPY is intentionally *left in the graph*: folding it into the same
// launch means reading and writing the same cache rows concurrently, and for T < KS the
// new state row X is built from old row X+T -- an unsynchronised cross-block
// read-after-write that made the decode logits differ run to run. See concat.cu.
void ggml_cuda_op_conv_state_concat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_cat,        // concat output (conv_kernel-1 + T, C, n_seqs)
        const ggml_tensor * src_qkv_t,      // concat src[1]: transposed qkv (T, C, n_seqs)
        const float *   state_base,         // base of the recurrent conv-state cache rows
        const int32_t * rows,               // cache row index per sequence
        int64_t         state_row_stride);  // floats per cache row (== C*(conv_kernel-1))
