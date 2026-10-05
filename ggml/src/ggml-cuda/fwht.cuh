#include "common.cuh"

// Optional pre-quantized q8_1 activation output (Codex, V100, 2026-10-01).
//
// The MMVQ consumers of a Hadamard rotation re-quantize the transform's output to q8_1 in a node
// of their own, one launch for an activation they just read. The transform can produce those bytes
// itself: it already holds every row in registers, a q8_1 block is 32 consecutive elements, and
// with NT = 256 that block is exactly one warp's slice of one register. The reduction and the
// rounding are the ones quantize_q8_1 uses, so the bytes are identical to what the matvec would
// have written.
//
// The buffer must be booked through ggml_cuda_q8_1_cache_get() under the consumer's tensor: the
// matvec then finds the slot filled and skips its quantization launch entirely.
struct ggml_cuda_fwht_q8 {
    char * buf            = nullptr;  // block_q8_1 buffer, rows GGML_PAD(ne10, MATRIX_ROW_PADDING) wide
    int    blocks_per_row = 0;        // GGML_PAD(ne10, MATRIX_ROW_PADDING) / QK8_1
    int    chunks_per_row = 0;        // ne10 / N: how many transform rows make up one activation row
};

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst,
                              const ggml_cuda_fwht_q8 * q8 = nullptr);

// same, but reads a non-contiguous (permuted) source view through its own strides, so the
// graph's CONT in front of the Hadamard rotation can be elided (see ggml-cuda.cu)
bool ggml_cuda_op_fwht_signed_view(ggml_backend_cuda_context & ctx, const ggml_tensor * src_view,
                                   const ggml_tensor * signs, ggml_tensor * dst,
                                   const ggml_cuda_fwht_q8 * q8 = nullptr);

// Whether this pattern is one the FWHT handles at all (so a caller can decide up front whether to
// prepare the q8_1 slot), and whether it is one that can also emit the q8_1 copy (block kernel).
bool ggml_cuda_fwht_supported(const ggml_tensor * src, const ggml_tensor * dst,
                              const ggml_tensor * signs);
bool ggml_cuda_fwht_q8_supported(const ggml_tensor * src, const ggml_tensor * dst,
                                 const ggml_tensor * signs);
