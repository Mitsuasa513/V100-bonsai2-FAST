// Checks where ggml_backend_sched places an op whose weight lives in a host (CPU) buffer when the op is offloaded to
// a GPU for large batches: it must go to the device that already holds the op's activations, not blindly to the
// first device. Also checks that the result computed through the resulting splits matches a CPU-only computation.
//
// Requires at least one CUDA device; the interesting cases need two. Skips (exit 0) otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int n_embd        = 64;
static const int n_ff          = 128;
static const int n_expert      = 8;
static const int n_expert_used = 2;

struct moe_case {
    std::vector<ggml_backend_t> backends; // last one must be the CPU backend
    ggml_backend_t act_backend;           // where the activations (and the norm weight) live
    int  n_tokens;
    int  expect;                          // expected index into backends for the expert mul_mat_id ops
    const char * desc;
};

static void fill_f32(ggml_tensor * t, uint32_t seed) {
    std::vector<float> v(ggml_nelements(t));
    uint32_t s = seed;
    for (auto & x : v) {
        s = s * 1664525u + 1013904223u;
        x = ((s >> 8) & 0xffff) / 65535.0f - 0.5f;
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// returns 0 on success, 1 on wrong placement; out receives the result of the down projection
static int run_case(const moe_case & c, std::vector<float> & out) {
    ggml_backend_t cpu = c.backends.back();

    ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };

    // expert weights in a host buffer, marked as weights (as the model loader does)
    ggml_context * ctx_w = ggml_init(ip);
    ggml_tensor * as_up   = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, n_embd, n_ff,   n_expert);
    ggml_tensor * as_gate = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, n_embd, n_ff,   n_expert);
    ggml_tensor * as_down = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, n_ff,   n_embd, n_expert);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, cpu);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fill_f32(as_up, 1); fill_f32(as_gate, 2); fill_f32(as_down, 3);

    // small dense weights on the activation device (norm weight, router): the ops using them are what anchor the
    // layer to that device, exactly like ffn_norm / ffn_gate_inp do in llama.cpp
    ggml_context * ctx_d = ggml_init(ip);
    ggml_tensor * w_norm   = ggml_new_tensor_1d(ctx_d, GGML_TYPE_F32, n_embd);
    ggml_tensor * w_router = ggml_new_tensor_2d(ctx_d, GGML_TYPE_F32, n_embd, n_expert);
    ggml_backend_buffer_t buf_d = ggml_backend_alloc_ctx_tensors(ctx_d, c.act_backend);
    ggml_backend_buffer_set_usage(buf_d, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fill_f32(w_norm, 4); fill_f32(w_router, 6);

    // layer input on the activation device
    ggml_context * ctx_a = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx_a, GGML_TYPE_F32, n_embd, c.n_tokens);
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, c.act_backend);
    fill_f32(x, 5);

    // graph: the same shape as llm_graph_context::build_moe_ffn
    // (norm -> router -> top-k ids; norm -> reshape -> up/gate -> swiglu -> down)
    // the activations and the ids must be intermediates (not pre-allocated): a pre-allocated source on another CUDA
    // device makes CUDA's supports_op reject the op, which would hide what this test is about
    ggml_init_params gp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_g = ggml_init(gp);
    ggml_tensor * cur  = ggml_mul(ctx_g, x, w_norm);
    ggml_tensor * probs = ggml_soft_max(ctx_g, ggml_mul_mat(ctx_g, w_router, cur));  // [n_expert, n_tokens]
    ggml_tensor * ids  = ggml_argsort_top_k(ctx_g, probs, n_expert_used);            // [n_expert_used, n_tokens]
    cur = ggml_reshape_3d(ctx_g, cur, n_embd, 1, c.n_tokens);
    ggml_tensor * up   = ggml_mul_mat_id(ctx_g, as_up,   cur, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx_g, as_gate, cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx_g, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx_g, as_down, act, ids);
    ggml_set_output(down);
    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, down);

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        const_cast<ggml_backend_t *>(c.backends.data()), nullptr, (int) c.backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, /*op_offload =*/ true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));

    int rc = 0;
    ggml_tensor * checked[3] = { up, gate, down };
    for (ggml_tensor * t : checked) {
        ggml_backend_t got = ggml_backend_sched_get_tensor_backend(sched, t);
        ggml_backend_t want = c.backends[c.expect];
        printf("  %-28s %-6s -> %-6s (expected %s)%s\n", c.desc, ggml_get_name(t)[0] ? ggml_get_name(t) : "?",
            got ? ggml_backend_name(got) : "NULL", ggml_backend_name(want), got == want ? "" : "  MISMATCH");
        if (got != want) {
            rc = 1;
        }
    }

    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched);
    out.resize(ggml_nelements(down));
    ggml_backend_tensor_get(down, out.data(), 0, ggml_nbytes(down));

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_d);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_g); ggml_free(ctx_a); ggml_free(ctx_d); ggml_free(ctx_w);
    return rc;
}

// normalized mean squared error, the metric test-backend-ops uses; the CUDA f32 mul_mat_id path runs on TF32 MMA,
// so bitwise agreement with the CPU is not expected
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        err += (double) (a[i] - b[i]) * (a[i] - b[i]);
        ref += (double) b[i] * b[i];
    }
    return err / std::max(ref, 1e-30);
}

int main() {
    // the CUDA backend reads the offload threshold from the environment; the cases below assume the default (32)
    unsetenv("GGML_OP_OFFLOAD_MIN_BATCH");

    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);
    ggml_backend_cpu_set_n_threads(cpu, 4);

    auto init_dev = [](const char * name) -> ggml_backend_t {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
        return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    };
    ggml_backend_t gpu0 = init_dev("CUDA0");
    ggml_backend_t gpu1 = init_dev("CUDA1");

    if (!gpu0) {
        printf("no CUDA0 device, skipping\n");
        return 0;
    }

    std::vector<moe_case> cases;
    cases.push_back({ { gpu0, cpu }, gpu0, 64, 0, "1gpu  act=CUDA0 nt=64" });
    cases.push_back({ { gpu0, cpu }, gpu0,  8, 1, "1gpu  act=CUDA0 nt=8" });
    if (gpu1) {
        cases.push_back({ { gpu0, gpu1, cpu }, gpu1, 64, 1, "2gpu  act=CUDA1 nt=64" }); // the case this test is about
        cases.push_back({ { gpu0, gpu1, cpu }, gpu0, 64, 0, "2gpu  act=CUDA0 nt=64" });
        cases.push_back({ { gpu1, gpu0, cpu }, gpu0, 64, 1, "2gpu' act=CUDA0 nt=64" }); // independent of device order
        cases.push_back({ { gpu0, gpu1, cpu }, gpu1,  8, 2, "2gpu  act=CUDA1 nt=8" });  // below the threshold: CPU
    } else {
        printf("no CUDA1 device, only running the single-GPU cases\n");
    }

    int n_fail = 0;
    for (const auto & c : cases) {
        std::vector<float> out, ref;
        int rc = run_case(c, out);
        moe_case ref_case = { { cpu }, cpu, c.n_tokens, 0, "cpu reference" };
        run_case(ref_case, ref);
        const double err = nmse(out, ref);
        const bool ok_val = err <= 5e-4; // same threshold as test-backend-ops for MUL_MAT_ID
        printf("  %-28s placement %s, nmse vs CPU %.2e %s\n", c.desc, rc == 0 ? "OK" : "WRONG", err, ok_val ? "OK" : "TOO LARGE");
        n_fail += rc + (ok_val ? 0 : 1);
    }

    if (gpu1) { ggml_backend_free(gpu1); }
    ggml_backend_free(gpu0);
    ggml_backend_free(cpu);

    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
