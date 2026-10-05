#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// EXPERIMENTAL (Codex, V100): per-graph cache that lets several MMVQ nodes share the q8_1
// quantization of the same activation tensor. See the comment in mmvq.cu. The caller brackets one
// graph pass (a full node loop) with begin/end; outside of that the cache is inactive.
void ggml_cuda_q8_1_cache_begin(int device);
void ggml_cuda_q8_1_cache_end  (int device);

// Books (or looks up) the q8_1 slot that holds the quantized copy of src1 for this pass. A non-null
// return with *needs_quantize == true is a fresh slot the caller must fill; with false the slot
// already holds valid bytes. Used by the FWHT fusion to hand the quantized activation to its MMVQ
// consumers so they do not launch a quantizer of their own.
char * ggml_cuda_q8_1_cache_get (int device, cudaStream_t stream, const ggml_tensor * src1,
                                 size_t nbytes, bool * needs_quantize);
void   ggml_cuda_q8_1_cache_drop(int device, const ggml_tensor * src1);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
