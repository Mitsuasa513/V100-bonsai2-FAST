#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data        = nullptr; // rollback slot 0
    int64_t slot_stride = 0;       // between rollback slots (0 when K==1)

    // rows-mode state read: the live state of sequence `seq` lives in the recurrent
    // cache row state_rows[seq] (row stride state_row_stride floats), so the kernel
    // reads it directly instead of a gathered copy. nullptr = gathered scratch.
    // The values are the same ones the per-layer GET_ROWS would have copied, so the
    // result is bit-identical to the gathered path.
    const float *   state_base       = nullptr; // base of the cache rows (gather source)
    const int32_t * state_rows       = nullptr;
    int64_t         state_row_stride = 0;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
