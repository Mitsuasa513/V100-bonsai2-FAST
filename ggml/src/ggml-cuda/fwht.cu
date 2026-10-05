#include "common.cuh"
#include "fwht.cuh"

#include <cstdlib>

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
// One block per row. Measured 2026-10-01: the kernel's cost is latency, not bandwidth or barrier
// count - NT=64 (one shared-memory stage, 16 registers/thread) is ~27% SLOWER than NT=256 (three
// stages, 4 registers/thread), and NT=512 is the same as NT=256, so the wider block wins by having
// more warps to hide the shared-memory round trips.
#define FWHT_BLOCK_THREADS 256

// Quantize one finished row into the shared q8_1 activation buffer (Codex, V100, 2026-10-01).
//
// Element (i*NT + tid) of the row is block (i*NT + tid)/QK8_1; with NT a multiple of QK8_1 every
// 32-element block therefore lives inside a single warp, and the warp reduction below covers
// exactly the elements of one block - the same reduction, in the same order, as quantize_q8_1 in
// quantize.cu, so the resulting bytes are the ones the matvec would have written itself.
template <int N, int NT, int NE>
static __device__ __forceinline__ void fwht_store_q8(const float (&reg)[NE], const int64_t r, const int tid,
        block_q8_1 * GGML_CUDA_RESTRICT q8, const int q8_blocks_per_row, const int q8_chunks_per_row) {
    static_assert(NT % QK8_1 == 0 && N % QK8_1 == 0, "bad q8_1 block shape");

    const int warp  = tid / QK8_1;  // which of the NT/QK8_1 blocks held per register step
    const int lane8 = tid % QK8_1;

    const int64_t row   = r / q8_chunks_per_row;
    const int     chunk = (int) (r % q8_chunks_per_row);
    block_q8_1 * q8_row = q8 + row * (int64_t) q8_blocks_per_row + (int64_t) chunk * (N / QK8_1) + warp;

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const float xi = reg[i];
        float amax = fabsf(xi);
        float sum  = xi;

        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);

        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

        block_q8_1 * y = q8_row + i * (NT / QK8_1);
        y->qs[lane8] = q;
        if (lane8 == 0) {
            y->ds = make_half2(d, sum);
        }
    }
}

template <int N, int NT, typename T, bool has_signs, bool with_q8 = false>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk,
                                block_q8_1 * GGML_CUDA_RESTRICT q8, const int q8_blocks_per_row,
                                const int q8_chunks_per_row) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fwht_load(src[i * NT + tid]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }

    if constexpr (with_q8) {
        fwht_store_q8<N, NT, NE>(reg, r, tid, q8, q8_blocks_per_row, q8_chunks_per_row);
    }
}

// One block per row, FWHT_BLOCK_THREADS threads, N >= 512. This is the only kernel that can also
// emit the q8_1 copy (see fwht_store_q8): a 32-element q8_1 block is one warp's slice of a register.
template <int N, typename T>
static void fwht_launch_block_case(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                                   const int64_t rows, const float scale, const float * signs,
                                   const int n_blk, const ggml_cuda_fwht_q8 * q8) {
    const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream());

    block_q8_1 * q8_buf = q8 != nullptr ? (block_q8_1 *) q8->buf : nullptr;
    const int    q8_bpr = q8 != nullptr ? q8->blocks_per_row : 1;
    const int    q8_cpr = q8 != nullptr ? q8->chunks_per_row : 1;

    if (q8 != nullptr) {
        if (signs) {
            ggml_cuda_kernel_launch(fwht_cuda_block<N, FWHT_BLOCK_THREADS, T, true,  true>,
                                    lp, src_d, dst_d, rows, scale, signs,   n_blk, q8_buf, q8_bpr, q8_cpr);
        } else {
            ggml_cuda_kernel_launch(fwht_cuda_block<N, FWHT_BLOCK_THREADS, T, false, true>,
                                    lp, src_d, dst_d, rows, scale, nullptr, 1,     q8_buf, q8_bpr, q8_cpr);
        }
        return;
    }

    if (signs) {
        ggml_cuda_kernel_launch(fwht_cuda_block<N, FWHT_BLOCK_THREADS, T, true,  false>,
                                lp, src_d, dst_d, rows, scale, signs,   n_blk, nullptr, 0, 1);
    } else {
        ggml_cuda_kernel_launch(fwht_cuda_block<N, FWHT_BLOCK_THREADS, T, false, false>,
                                lp, src_d, dst_d, rows, scale, nullptr, 1,     nullptr, 0, 1);
    }
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk, const ggml_cuda_fwht_q8 * q8) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;

    // Only the block kernel (N >= 512, one block per row) can emit the q8_1 copy alongside the
    // transform. Decline instead of booking a slot that would never be filled.
    if (q8 != nullptr) {
        const bool block_shape = n == 512 || n == 1024 || n == 2048 || n == 4096 || n == 8192;
        if (legacy || !block_shape || q8->buf == nullptr || q8->blocks_per_row <= 0 || q8->chunks_per_row <= 0) {
            return false;
        }
    }

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
#define FWHT_BLOCK_CASE(NN) \
        case NN: \
            fwht_launch_block_case<NN, T>(ctx, src_d, dst_d, rows, scale, signs, n_blk, q8); \
            return true;
    if (legacy) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
            FWHT_SMEM_CASE(8192)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
        FWHT_BLOCK_CASE(8192)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
        default:
            return false;
    }
}

// Shared validation. Kept separate from fwht_dispatch so that a caller can ask up front whether the
// pattern will be handled (and whether it can carry the q8_1 copy) before it books anything.
static bool fwht_check(const ggml_tensor * src, const ggml_tensor * dst, const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];

    if (n <= 0 || (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
    }

    return true;
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t, const ggml_cuda_fwht_q8 * q8) {
    if (!fwht_check(src, dst, signs_t)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk, q8);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk, q8);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst,
                              const ggml_cuda_fwht_q8 * q8) {
    return fwht_dispatch(ctx, src, dst, signs, q8);
}

// Whether the transform handles this pattern, and whether it can also emit the q8_1 copy: the
// block kernel (N >= 512, one block per row) only, and only for a 32-element-aligned width.
bool ggml_cuda_fwht_supported(const ggml_tensor * src, const ggml_tensor * dst,
                              const ggml_tensor * signs) {
    return fwht_check(src, dst, signs);
}

bool ggml_cuda_fwht_q8_supported(const ggml_tensor * src, const ggml_tensor * dst,
                                 const ggml_tensor * signs) {
    if (!fwht_check(src, dst, signs) || getenv("GGML_CUDA_FWHT_LEGACY") != nullptr) {
        return false;
    }
    switch (dst->ne[0]) {
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Strided-source variant (Codex, 2026-09-28)
//
// build_lora_mm() materialises the permuted activation with a CONT right before the sign
// flip + Hadamard rotation (which the CUDA backend already replaces by this FWHT), i.e. a
// 24 KB copy per layer that exists only to make the tensor contiguous. Read the permuted
// view directly through its own strides instead: element g of the CONT's flat order maps to
// (g % ne0, (g / ne0) % ne1, g / (ne0*ne1)) of the view, exactly like ggml_cont produces it.
//
// Only the model's Hadamard block size (1024) is supported; the butterfly is byte-for-byte
// the same code as fwht_cuda_block so the result stays bit-identical.
#define FWHT_STRIDED_N 1024

template <typename T, bool with_q8 = false>
__launch_bounds__(FWHT_BLOCK_THREADS, 1)
static __global__ void fwht_cuda_block_strided(const char * __restrict__ src, float * __restrict__ dst,
        const int64_t n_rows, const float scale, const float * __restrict__ signs, const int n_blk,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
        block_q8_1 * GGML_CUDA_RESTRICT q8, const int q8_blocks_per_row, const int q8_chunks_per_row) {
    constexpr int N  = FWHT_STRIDED_N;
    constexpr int NT = FWHT_BLOCK_THREADS;
    constexpr int NE = N / NT;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();
    const float * signs_row = signs != nullptr ? signs + (r % n_blk)*N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        // 32-bit index math on purpose: this kernel runs with a handful of blocks, so its
        // duration is the per-thread latency -- 64-bit divisions here cost several microseconds
        const int g   = (int) r*N + i*NT + tid;
        const int n0  = (int) ne0;
        const int n01 = n0 * (int) ne1;
        const int i0  = g % n0;
        const int i1  = (g / n0) % (int) ne1;
        const int i2  = (g / n01) % (int) ne2;
        const int i3  = g / (n01 * (int) ne2);
        const T v = *(const T *) (src + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
        reg[i] = fwht_load(v) * scale;
        if (signs_row != nullptr) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    // same butterfly (and therefore the same rounding order) as fwht_cuda_block
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[r*N + i * NT + tid] = reg[i];
    }

    if constexpr (with_q8) {
        fwht_store_q8<N, NT, NE>(reg, r, tid, q8, q8_blocks_per_row, q8_chunks_per_row);
    }
}

bool ggml_cuda_op_fwht_signed_view(ggml_backend_cuda_context & ctx, const ggml_tensor * src_view,
                                   const ggml_tensor * signs_t, ggml_tensor * dst,
                                   const ggml_cuda_fwht_q8 * q8) {
    constexpr int N = FWHT_STRIDED_N;

    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != N ||
        (src_view->type != GGML_TYPE_F32 && src_view->type != GGML_TYPE_F16) ||
        src_view->nb[0] != ggml_type_size(src_view->type) ||
        ggml_nelements(src_view) != ggml_nelements(dst) ||
        ggml_nelements(src_view) % N != 0) {
        return false;
    }

    if (q8 != nullptr && (q8->buf == nullptr || q8->blocks_per_row <= 0 || q8->chunks_per_row <= 0)) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t != nullptr) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % N != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = (int) (signs_t->ne[0] / N);
    }

    const int64_t n_rows = ggml_nelements(dst) / N;
    const float   scale  = 1 / sqrtf((float) N);

    const dim3 g((unsigned) n_rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream());

    const int64_t nb0 = (int64_t) src_view->nb[0];
    const int64_t nb1 = (int64_t) src_view->nb[1];
    const int64_t nb2 = (int64_t) src_view->nb[2];
    const int64_t nb3 = (int64_t) src_view->nb[3];

    block_q8_1 * q8_buf = q8 != nullptr ? (block_q8_1 *) q8->buf : nullptr;
    const int q8_bpr = q8 != nullptr ? q8->blocks_per_row : 1;
    const int q8_cpr = q8 != nullptr ? q8->chunks_per_row : 1;

    if (q8 != nullptr) {
        if (src_view->type == GGML_TYPE_F32) {
            ggml_cuda_kernel_launch(fwht_cuda_block_strided<float, true>, lp,
                    (const char *) src_view->data, (float *) dst->data, n_rows, scale, signs, n_blk,
                    src_view->ne[0], src_view->ne[1], src_view->ne[2], nb0, nb1, nb2, nb3,
                    q8_buf, q8_bpr, q8_cpr);
        } else {
            ggml_cuda_kernel_launch(fwht_cuda_block_strided<half, true>, lp,
                    (const char *) src_view->data, (float *) dst->data, n_rows, scale, signs, n_blk,
                    src_view->ne[0], src_view->ne[1], src_view->ne[2], nb0, nb1, nb2, nb3,
                    q8_buf, q8_bpr, q8_cpr);
        }
        return true;
    }

    if (src_view->type == GGML_TYPE_F32) {
        ggml_cuda_kernel_launch(fwht_cuda_block_strided<float>, lp,
                (const char *) src_view->data, (float *) dst->data, n_rows, scale, signs, n_blk,
                src_view->ne[0], src_view->ne[1], src_view->ne[2], nb0, nb1, nb2, nb3,
                nullptr, 0, 1);
    } else {
        ggml_cuda_kernel_launch(fwht_cuda_block_strided<half>, lp,
                (const char *) src_view->data, (float *) dst->data, n_rows, scale, signs, n_blk,
                src_view->ne[0], src_view->ne[1], src_view->ne[2], nb0, nb1, nb2, nb3,
                nullptr, 0, 1);
    }

    return true;
}

#undef FWHT_STRIDED_N
