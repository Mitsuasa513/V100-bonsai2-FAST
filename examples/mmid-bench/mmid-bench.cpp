#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
// MUL_MAT_ID micro-bench at DeepSeek-V4.1-Flash expert shape: 384 experts, [5120 x 2304] MXFP4, top-6.
//
// not part of the cmake build; build it against a CPU-only build tree with
//   g++ -O2 -std=c++17 -I ggml/include examples/mmid-bench/mmid-bench.cpp -o mmid-bench \
//       -L build-cpu/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build-cpu/bin
//
// usage: mmid-bench [n_tok] [threads] [iters] [--dump FILE | --check FILE] [--rand-e] [--repack] [--no-scalar]
//   --dump FILE   write the output tensor of a fixed-seed run to FILE (reference)
//   --check FILE  compare the output tensor of the same fixed-seed run against FILE
//   --rand-e      randomize the e8m0 block scales (incl. the e<2 denormal codes) instead of e=127 everywhere
//   --repack      allocate the weights in the CPU extra (repack) buffer type
//   --no-scalar   skip the slow scalar reference check of sampled (row, slot, token) outputs
//   --nodes N     N independent MUL_MAT_ID nodes per graph (amortizes the per-graph overhead; ms/op is per node)
//   --use-ref     run the CPU backend's reference path (no mul_mat_id multi-column kernel, no fusion): use it to
//                 produce the --dump reference from the very same binary
//   --fixed-ids   keep the same expert ids every iteration (the per-thread slice then stays in its private L2:
//                 separates the instruction cost from the DRAM stream)
static const int8_t kvalues_mxfp4_ref[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static double e8m0_half_ref(uint8_t e) { // = ggml_e8m0_to_fp32_half: 2^(e-128), denormal patterns for e < 2
    uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
struct sample { int t, u, r, e; std::vector<uint8_t> row; };
int main(int argc, char ** argv) {
    const int64_t n_embd = 5120, n_ff = 2304, n_exp = 384, n_used = 6;
    int n_tok = 1, nth = 64, iters = 100, npos = 0;
    std::string dump, check; bool repack = false, scalar = true, rand_e = false, fixed = false, use_ref = false; int nodes = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--dump"  && i + 1 < argc) dump  = argv[++i];
        else if (a == "--check" && i + 1 < argc) check = argv[++i];
        else if (a == "--repack")    repack = true;
        else if (a == "--rand-e")    rand_e = true;
        else if (a == "--no-scalar") scalar = false;
        else if (a == "--fixed-ids")  fixed = true;
        else if (a == "--use-ref")    use_ref = true;
        else if (a == "--nodes" && i + 1 < argc) nodes = atoi(argv[++i]);
        else { int v = atoi(argv[i]); if (npos == 0) n_tok = v; else if (npos == 1) nth = v; else iters = v; npos++; }
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    if (use_ref) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
        auto fn = (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
        if (!fn) { fprintf(stderr, "no ggml_backend_cpu_set_use_ref\n"); return 1; }
        fn(be, true);
        printf("using the CPU reference path\n");
    }
    ggml_backend_buffer_type_t w_buft = ggml_backend_get_default_buffer_type(be);
    if (repack) {
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        ggml_backend_buffer_type_t * extra = get_extra ? get_extra(dev) : nullptr;
        if (!extra || !*extra) { fprintf(stderr, "no extra buffer types\n"); return 1; }
        w_buft = *extra;
        printf("weights in buffer type: %s\n", ggml_backend_buft_name(w_buft));
    }
    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    ggml_init_params ipc = { ggml_tensor_overhead()*(8 + nodes), nullptr, true };
    ggml_context * ctx  = ggml_init(ipc);
    ggml_tensor * w   = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_exp);  // gate/up shape
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, n_tok);
    std::vector<ggml_tensor *> idsv(nodes);
    for (int k = 0; k < nodes; k++) idsv[k] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * ids = idsv[0];
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, w_buft);
    ggml_backend_buffer_t buf  = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!wbuf || !buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    // fixed-seed ids for the dump/check/scalar run (independent of iters), and the sampled outputs to verify
    std::vector<int32_t> idv(n_used*n_tok), idv_chk(n_used*n_tok);
    auto draw_ids = [&](std::mt19937 & r, std::vector<int32_t> & v) { for (int t = 0; t < n_tok; t++) { std::vector<int> p(n_exp); for (int i=0;i<n_exp;i++) p[i]=i; std::shuffle(p.begin(), p.end(), r); for (int u=0;u<n_used;u++) v[t*n_used+u]=p[u]; } };
    { std::mt19937 r(1234); draw_ids(r, idv_chk); }
    std::vector<sample> samples;
    {
        auto add = [&](int t, int u, int r) { samples.push_back({t, u, r, idv_chk[t*n_used+u], std::vector<uint8_t>(n_embd/32*17)}); };
        add(0, 0, 0); add(n_tok-1, n_used-1, n_ff-1); add(0, n_used-1, 0); add(n_tok-1, 0, n_ff-1);
        for (int k = 0; k < 60; k++) add((k*31 + 5) % n_tok, (k*7 + 1) % n_used, (k*997 + 13) % n_ff);
    }
    const size_t nb1 = n_embd/32*17, nb2 = nb1*n_ff;
    // fill weights: e8m0 exponent = 127 (scale 1.0) unless --rand-e, random nibbles; capture the sampled rows
    {
        const size_t nb = ggml_nbytes(w); std::vector<uint8_t> chunk(17*4096);
        std::vector<uint8_t> whole(repack ? nb : 0); // the repack buffer only accepts whole-tensor set_tensor
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t off = 0; off < nb; off += chunk.size()) {
            size_t n = std::min(chunk.size(), nb - off);
            for (size_t i = 0; i < n; i += 17) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                chunk[i] = 127;
                if (rand_e) { uint32_t r = (uint32_t)(s >> 40); chunk[i] = (r & 0xff) < 2 ? (r & 1) : (uint8_t)(118 + ((r >> 8) & 0xf)); }
                for (int j = 1; j < 17 && i+j < n; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; chunk[i+j] = (uint8_t)s; }
            }
            for (auto & sm : samples) {
                size_t ro = (size_t)sm.e*nb2 + (size_t)sm.r*nb1;
                size_t lo = std::max(ro, off), hi = std::min(ro + nb1, off + n);
                if (lo < hi) memcpy(sm.row.data() + (lo - ro), chunk.data() + (lo - off), hi - lo);
            }
            if (repack) memcpy(whole.data() + off, chunk.data(), n); else ggml_backend_tensor_set(w, chunk.data(), off, n);
        }
        if (repack) ggml_backend_tensor_set(w, whole.data(), 0, nb);
    }
    std::vector<float> xf(n_embd*n_used*n_tok); for (auto & v : xf) v = (rand()/(float)RAND_MAX) - 0.5f;
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size()*4);
    ggml_init_params gp = { ggml_tensor_overhead()*(16 + 2*nodes) + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * out = nullptr;
    for (int k = 0; k < nodes; k++) { ggml_tensor * o = ggml_mul_mat_id(gctx, w, x, idsv[k]); ggml_build_forward_expand(gf, o); if (k == 0) out = o; }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    std::mt19937 rng(42);
    auto set_ids = [&](){ for (int k = 0; k < nodes; k++) { if (!fixed) draw_ids(rng, idv); else idv = idv_chk; ggml_backend_tensor_set(idsv[k], idv.data(), 0, idv.size()*4); } };
    for (int i = 0; i < 3; i++) { set_ids(); ggml_backend_graph_compute(be, gf); }
    double tot = 0; size_t bytes_tot = 0; std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        set_ids();
        std::vector<char> used(n_exp, 0); for (auto v : idv) used[v] = 1; size_t nd = std::count(used.begin(), used.end(), 1);
        auto t0 = std::chrono::steady_clock::now(); ggml_backend_graph_compute(be, gf); auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1-t0).count() / nodes; tot += dt; ts.push_back(dt); bytes_tot += nd * (size_t)w->nb[2];
    }
    std::sort(ts.begin(), ts.end());
    double ms = tot/iters*1e3; double gbs = bytes_tot/tot/1e9; double gflops = 2.0*n_embd*n_ff*n_used*n_tok*iters/tot/1e9;
    printf("n_tok=%d threads=%d : %.3f ms/op  %.1f GB/s of expert bytes (%.1f MB/op)  %.0f GFLOP/s  | min %.3f  p10 %.3f  median %.3f  p90 %.3f  max %.3f ms\n",
        n_tok, nth, ms, gbs, (double)bytes_tot/iters/1e6, gflops, ts[0]*1e3, ts[ts.size()/10]*1e3, ts[ts.size()/2]*1e3, ts[ts.size()*9/10]*1e3, ts.back()*1e3);
    // fixed-seed run for dump / check / scalar reference
    ggml_backend_tensor_set(ids, idv_chk.data(), 0, idv_chk.size()*4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> o(ggml_nelements(out)); ggml_backend_tensor_get(out, o.data(), 0, o.size()*4);
    int rc = 0;
    for (float v : o) if (!std::isfinite(v)) { printf("output has non-finite values: FAIL\n"); rc = 1; break; }
    if (!dump.empty()) {
        FILE * f = fopen(dump.c_str(), "wb"); if (!f) { perror("dump"); return 1; }
        int32_t hdr[3] = { n_tok, (int32_t) n_used, (int32_t) n_ff }; fwrite(hdr, 4, 3, f); fwrite(o.data(), 4, o.size(), f); fclose(f);
        printf("dumped %zu outputs to %s\n", o.size(), dump.c_str());
    }
    if (!check.empty()) {
        FILE * f = fopen(check.c_str(), "rb"); if (!f) { perror("check"); return 1; }
        int32_t hdr[3]; std::vector<float> ref(o.size());
        if (fread(hdr, 4, 3, f) != 3 || hdr[0] != n_tok || hdr[1] != n_used || hdr[2] != n_ff || fread(ref.data(), 4, ref.size(), f) != ref.size()) { printf("check: shape mismatch (file n_tok=%d)\n", hdr[0]); return 1; }
        fclose(f);
        double max_abs = 0, max_rel = 0, max_ref = 0, max_rel_sig = 0; size_t i_abs = 0;
        for (size_t i = 0; i < o.size(); i++) max_ref = std::max(max_ref, (double) fabsf(ref[i]));
        for (size_t i = 0; i < o.size(); i++) {
            double d = fabs((double) o[i] - ref[i]); if (d > max_abs) { max_abs = d; i_abs = i; }
            double rel = d / std::max(std::max(fabs((double) ref[i]), fabs((double) o[i])), 1e-30);
            max_rel = std::max(max_rel, rel);
            if (fabs(ref[i]) > 1e-2 * max_ref) max_rel_sig = std::max(max_rel_sig, rel);
        }
        bool ok = max_abs <= 1e-4 * max_ref && max_rel_sig <= 1e-4;
        printf("check vs %s: max abs diff %.3e (at %zu: ref %.6g got %.6g, max |ref| %.4g), max rel diff %.3e (all), %.3e (|ref| > 1e-2 max) : %s\n",
            check.c_str(), max_abs, i_abs, ref[i_abs], o[i_abs], max_ref, max_rel, max_rel_sig, ok ? "PASS" : "FAIL");
        if (!ok) rc = 1;
    }
    if (scalar) {
        // slow scalar reference: dequantize the MXFP4 row (e8m0 scale x kvalues table) and dot it with the q8_0-quantized
        // activation exactly as the kernel does (src1 is quantized to vec_dot_type = Q8_0 with the CPU from_float)
        const ggml_type_traits_cpu * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        std::vector<uint8_t> yq(n_embd/32*34);
        double worst = 0; int nfail = 0;
        for (auto & sm : samples) {
            q8->from_float(xf.data() + ((size_t)sm.t*n_used + sm.u)*n_embd, yq.data(), n_embd);
            double acc = 0, absacc = 0;
            for (int b = 0; b < n_embd/32; b++) {
                const uint8_t * xb = sm.row.data() + b*17; const uint8_t * yb = yq.data() + b*34;
                uint16_t dh; memcpy(&dh, yb, 2); const int8_t * yqs = (const int8_t *) yb + 2;
                double d = e8m0_half_ref(xb[0]) * (double) ggml_fp16_to_fp32(dh);
                long sumi = 0;
                for (int j = 0; j < 16; j++) { sumi += yqs[j] * kvalues_mxfp4_ref[xb[1+j] & 0xf]; sumi += yqs[j+16] * kvalues_mxfp4_ref[xb[1+j] >> 4]; }
                acc += d * sumi; absacc += fabs(d * sumi);
            }
            float got = o[((size_t)sm.t*n_used + sm.u)*n_ff + sm.r];
            double err = fabs(got - acc) / std::max(absacc, 1e-30);
            worst = std::max(worst, err);
            if (err > 1e-5) { nfail++; if (nfail <= 5) printf("  scalar mismatch t=%d u=%d r=%d e=%d: ref %.8g got %.8g (sum|terms| %.4g)\n", sm.t, sm.u, sm.r, sm.e, acc, got, absacc); }
        }
        printf("scalar reference: %zu samples, max |diff| / sum|terms| = %.3e : %s\n", samples.size(), worst, nfail ? "FAIL" : "PASS");
        if (nfail) rc = 1;
    }
    return rc;
}
