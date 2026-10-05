#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-dsv41.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    if (hparams.n_layer_nextn > 0) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    // DeepSeek-V4.1 (arch deepseek4, discriminated by the presence of engram layers). Reads the
    // extra V4.1 metadata -- engram config and the CSA2/CED layer topology. All optional/guarded so
    // a V4-Flash-0731 checkpoint (which ships none of these) is completely unaffected.
    {
        std::vector<uint32_t> engram_layers;
        if (ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, engram_layers, false) && !engram_layers.empty()) {
            hparams.is_dsv41 = true;
            ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
            ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_head_size);
            ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram);
            for (uint32_t l : engram_layers) {
                if (l < LLAMA_MAX_LAYERS) { hparams.is_engram_impl[l] = 1; }
            }

            std::vector<uint32_t> layers;
            if (ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS, layers, false)) {
                for (uint32_t l : layers) { if (l < LLAMA_MAX_LAYERS) { hparams.is_kv_source_impl[l] = 1; } }
            }
            layers.clear();
            if (ml.get_arr(LLM_KV_ATTENTION_INDEX_SOURCE_LAYERS, layers, false)) {
                for (uint32_t l : layers) { if (l < LLAMA_MAX_LAYERS) { hparams.is_index_source_impl[l] = 1; } }
            }
            ml.get_key(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER, hparams.candidate_source_layer, false);
            ml.get_key(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,   hparams.candidate_block_size,   false);
            ml.get_key(LLM_KV_ATTENTION_CANDIDATE_TOP_K_BLOCKS, hparams.candidate_top_k_blocks, false);
        }
    }

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    // tokens of an image span attend bidirectionally to the whole span, the window only applies to older tokens
    // ref: get_window_topk_idxs_visible in the reference impl
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_FULL;
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int hc_head_flags = hparams.is_dsv41 ? TENSOR_NOT_REQUIRED : 0;  // V4.1 has no output_hc_*
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, hc_head_flags);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, hc_head_flags);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, hc_head_flags);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        // for wo_a, the shape in the file is (n_head * n_embd_head / o_groups, o_lora_rank*o_groups)
        // so we reshape here, to avoid reshaping the tensor in the graph
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (hparams.is_dsv41) {
            // V4.1 CSA2: compressor only on kv_source layers, indexer on index_source layers.
            const int64_t idx_head = hparams.indexer_head_size;
            if (i < n_layer && hparams.is_kv_source_impl[i]) {
                layer.attn_comp_wkv  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,  "weight", i), {n_embd, n_embd_head}, flags);
                layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", i), {n_embd_head}, flags);
                if (ratio > 1) {  // ratio-2 encoder groups pool 2 tokens with a softmax gate
                    layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, flags);
                }
            }
            if (i < n_layer && hparams.is_index_source_impl[i]) {
                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * idx_head}, flags);
                if (hparams.is_kv_source_impl[i]) {  // CSA2 Full mode owns its index-K (Reindex layers reuse it)
                    layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, idx_head}, flags);
                    layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {idx_head}, flags);
                }
            }
            if (i < n_layer && hparams.is_engram_impl[i]) {
                const int64_t hc          = hc_mult;
                const int64_t ehd         = hparams.engram_head_size;
                const int64_t n_hash_cols = (int64_t)(hparams.engram_max_ngram - 1) * hparams.engram_n_head;
                // engram table rows are per-layer and absent from metadata: read them from the tensor
                const std::string embd_name = "blk." + std::to_string(i) + ".engram_embd.weight";
                const auto * ew = ml.get_weight(embd_name.c_str());
                const int64_t rows = ew ? ew->tensor->ne[1] : 0;
                layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {ehd, rows}, flags | TENSOR_READ_LAZY);   // 104 GB tables: rows on demand from the mmap
                layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_hash_cols * ehd, n_embd * (hc + 1)}, flags);
                layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc}, flags);
                layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc}, flags);
            }
        } else if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, flags | TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.is_dsv41) {
        GGML_ASSERT(params.gtype != LLM_GRAPH_TYPE_DECODER_MTP && "DeepSeek-V4.1 DSpark MTP not implemented yet");
        return std::make_unique<graph_v41>(*this, params);
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;

// mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * dsv4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

void llama_model_deepseek4::graph::build_hc_mixes(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** pre,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    *pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    *pre = dsv4_hc_affine(ctx0, *pre, scale_pre, base_pre);
    *pre = ggml_sigmoid(ctx0, *pre);
    *pre = ggml_scale_bias(ctx0, *pre, 1.0f, hparams.dsv4_hc_eps);
    cb(*pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    ggml_tensor * pre = nullptr;
    build_hc_mixes(x, hc_fn, hc_scale, hc_base, &pre, post, comb, il);
    return build_hc_pre(x, pre, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_pre(x, pre, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    indexer_q = ggml_rope_set_offset(indexer_q, n_embd_indexer_head_nope);
    cb(indexer_q, "lid_q_rope", il);

    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    cb(kq_mask, "csa_lid_kq_mask", il);

    const int64_t n_kv_max = std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0];
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_dsv4->get_hca().k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, inp_dsv4->get_hca().k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_hca(), hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, inp_dsv4->get_hca().state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_dsv4->get_hca().state_persist_src_idxs);

        hca_state_kv = inp_dsv4->mctx->get_hca_state()->cpy_kv(ctx0,
                hca_persist_kv, inp_dsv4->get_hca().state_persist_dst_idxs, il);
        hca_state_score = inp_dsv4->mctx->get_hca_state()->cpy_score(ctx0,
                hca_persist_score, inp_dsv4->get_hca().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, kv,
                nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl is input is from mtmd
        const bool is_media = ubatch.embd != nullptr;
        if (is_media) {
            if (layer.ffn_exp_probs_b_vl) {
                exp_probs_b = layer.ffn_exp_probs_b_vl;
            }
        } else if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK4 MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    GGML_ASSERT(ubatch.token && "DEEPSEEK4 MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "DEEPSEEK4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT((uint32_t) il >= hparams.dsv4_hash_layer_count && "DEEPSEEK4 MTP does not support hash-routed MTP blocks");
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(),
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    inpL = build_cvec(inpL, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "mtp_hc_head", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK4 MTP missing shared head norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DEEPSEEK4 MTP missing LM head");
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}


// ---------------------------------------------------------------------------------------------
// DeepSeek-V4.1 graph (first pass)
// ---------------------------------------------------------------------------------------------

// Engram hash ids per engram layer: I32 [n_hash_cols, n_tokens]. For the M4 diff they come precomputed
// from the reference hasher via DSV41_ENGRAM_HASHES=<raw i32 laid out [n_engram_layers][n_pos][n_hash_cols]>;
// the in-runtime hasher (compressed token map + xor/mod hash, the multipliers/primes precomputed into
// the GGUF) is TODO.
// Engram hasher constants (the reference NgramHashState buffers), exported by
// oracle/check_engram_consts.py into DSV41_ENGRAM_CONSTS=<dir>: token_map.i32 [n_vocab] (token -> compressed
// id), multipliers.i64 [n_layers][max_ngram], primes.i64 [n_layers][max_ngram-1][n_heads],
// offsets.i64 [n_layers][(max_ngram-1)*n_heads], meta.txt. TODO: carry these in the GGUF instead.
struct dsv41_engram_consts {
    bool loaded = false;
    int32_t n_layers = 0, max_ngram = 0, n_heads = 0, pad_id = 0;
    std::vector<int32_t> token_map;
    std::vector<int64_t> multipliers, primes, offsets;

    static const dsv41_engram_consts & get() {
        static dsv41_engram_consts c;
        if (c.loaded) { return c; }
        const char * dir = getenv("DSV41_ENGRAM_CONSTS");
        if (!dir) { return c; }
        auto read_vec = [&](const char * name, auto & vec) {
            std::string path = std::string(dir) + "/" + name;
            FILE * f = std::fopen(path.c_str(), "rb");
            GGML_ASSERT(f && "engram consts: missing file");
            std::fseek(f, 0, SEEK_END); const size_t n = (size_t) std::ftell(f) / sizeof(vec[0]); std::fseek(f, 0, SEEK_SET);
            vec.resize(n);
            const size_t got = std::fread(vec.data(), sizeof(vec[0]), n, f);
            std::fclose(f);
            GGML_ASSERT(got == n);
        };
        read_vec("token_map.i32",   c.token_map);
        read_vec("multipliers.i64", c.multipliers);
        read_vec("primes.i64",      c.primes);
        read_vec("offsets.i64",     c.offsets);
        {
            std::string path = std::string(dir) + "/meta.txt";
            FILE * f = std::fopen(path.c_str(), "r");
            GGML_ASSERT(f && "engram consts: missing meta.txt");
            char key[64]; long long val;
            while (std::fscanf(f, "%63s %lld", key, &val) == 2) {
                if      (!strcmp(key, "n_layers"))  c.n_layers  = (int32_t) val;
                else if (!strcmp(key, "max_ngram")) c.max_ngram = (int32_t) val;
                else if (!strcmp(key, "n_heads"))   c.n_heads   = (int32_t) val;
                else if (!strcmp(key, "pad_id"))    c.pad_id    = (int32_t) val;
                else { int ch; while ((ch = std::fgetc(f)) != EOF && ch != '\n') {} }   // skip the rest of the line (layer_ids ...)
            }
            std::fclose(f);
        }
        GGML_ASSERT(c.n_layers > 0 && c.max_ngram > 1 && c.n_heads > 0);
        GGML_ASSERT((int64_t) c.multipliers.size() == (int64_t) c.n_layers * c.max_ngram);
        GGML_ASSERT((int64_t) c.primes.size()      == (int64_t) c.n_layers * (c.max_ngram - 1) * c.n_heads);
        GGML_ASSERT((int64_t) c.offsets.size()     == (int64_t) c.n_layers * (c.max_ngram - 1) * c.n_heads);
        c.loaded = true;
        return c;
    }
};

// Engram row prefetch.
//
// The engram tables are TENSOR_READ_LAZY: a 194 GiB window of the GGUF mmap carrying
// POSIX_MADV_RANDOM, so the kernel does no readahead and every row the gather wants that is not
// already resident becomes a synchronous 4 KiB page fault taken *inside* GET_ROWS, one at a time.
// Measured on astra (KC3000 NVMe, 2 engram layers x 24 hash columns = 48 lookups/token): a cold
// 1024-token ubatch spends ~3.8 s at ~10.2k IOPS with aqu-sz ~0.85, rareq-sz 4.00 KiB and
// r_await ~0.087 ms while the CPU sits at 0.4% -- up to 3.2 ms/token of pure, fully serialised
// fault latency, ~40% of a cold prefill. The row ids for the entire ubatch are known here, before
// the graph runs, so fault them in parallel and hand the device a deep queue instead of QD~1.
// Rows already resident cost a ~10 ns touch, so this stays a no-op on warm content.
//
//   DSV41_ENGRAM_PREFETCH          0 = off, 1 = parallel touch (default), 2 = MADV_WILLNEED,
//                                  3 = MADV_WILLNEED then parallel touch
//   DSV41_ENGRAM_PREFETCH_THREADS  worker count for the touch modes (default 64)
//   DSV41_ENGRAM_PREFETCH_SMALL    page sets up to this size use MADV_WILLNEED + serial touch, no threads (512)
//
// Read-only by construction: it only reads bytes the gather is about to read anyway, so it cannot
// change any output.
static int dsv41_engram_prefetch_mode() {
    static const int mode = [] {
        const char * e = getenv("DSV41_ENGRAM_PREFETCH");
        return e ? atoi(e) : 1;
    }();
    return mode;
}

// below this many distinct pages the prefetch uses MADV_WILLNEED + a serial touch instead of a thread pool
static size_t dsv41_engram_prefetch_small() {
    static const size_t n = [] {
        const char * e = getenv("DSV41_ENGRAM_PREFETCH_SMALL");
        return (size_t) (e ? atoi(e) : 512);
    }();
    return n;
}

static int dsv41_engram_prefetch_nthread() {
    static const int n = [] {
        const char * e = getenv("DSV41_ENGRAM_PREFETCH_THREADS");
        const int v = e ? atoi(e) : 64;
        return v < 1 ? 1 : v;
    }();
    return n;
}

static void dsv41_engram_prefetch(const std::vector<const ggml_tensor *>  & embd,
                                  const std::vector<std::vector<int32_t>> & ids) {
#if defined(_WIN32)
    GGML_UNUSED(embd);
    GGML_UNUSED(ids);
#else
    const int mode = dsv41_engram_prefetch_mode();
    if (mode == 0 || embd.empty()) {
        return;
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);

    // The distinct pages the gather will fault. A row is 272 B (256 x Q8_0) so a page holds ~15 of
    // them; dedup keeps the touch count at the number of real faults rather than 48/token.
    std::vector<uintptr_t> pages;
    for (size_t li = 0; li < embd.size() && li < ids.size(); ++li) {
        const ggml_tensor * t = embd[li];
        if (!t || !t->data) {
            continue;                     // not a lazy mmap (fully resident tensor): nothing to do
        }
        const size_t  stride = t->nb[1];
        const size_t  rsz    = ggml_row_size(t->type, t->ne[0]);
        const int64_t nrow   = t->ne[1];

        pages.reserve(pages.size() + ids[li].size());
        for (const int32_t id : ids[li]) {
            if (id < 0 || (int64_t) id >= nrow) {
                continue;                 // hashes are bounded by construction; stay safe anyway
            }
            const uintptr_t a  = (uintptr_t) t->data + (size_t) id * stride;
            const uintptr_t p0 = a & ~(uintptr_t) (page - 1);
            const uintptr_t p1 = (a + rsz - 1) & ~(uintptr_t) (page - 1);
            pages.push_back(p0);
            if (p1 != p0) {
                pages.push_back(p1);      // a row straddling a page boundary needs both
            }
        }
    }
    if (pages.empty()) {
        return;
    }

    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    if (mode == 2 || mode == 3) {
        // one advice call per run of consecutive pages
        for (size_t i = 0; i < pages.size(); ) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j-1] + page) {
                ++j;
            }
            posix_madvise((void *) pages[i], (pages[j-1] - pages[i]) + page, POSIX_MADV_WILLNEED);
            i = j;
        }
        if (mode == 2) {
            return;
        }
    }

    // Small sets (a decode token is ~48 rows -> <= 48 pages) must not pay a thread-pool spawn: creating and joining
    // 48-64 std::threads costs 2-3 ms on the Altra (measured 2026-09-13; it showed up as -6% decode on the first
    // production build). Hand the block layer the whole set at once with MADV_WILLNEED (async readahead) and
    // touch serially: the reads overlap in the device queue, no threads. Warm pages make this a ~50 us no-op.
    const size_t small = dsv41_engram_prefetch_small();
    if (mode == 1 && pages.size() <= small) {
        for (size_t i = 0; i < pages.size(); ) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j-1] + page) {
                ++j;
            }
            posix_madvise((void *) pages[i], (pages[j-1] - pages[i]) + page, POSIX_MADV_WILLNEED);
            i = j;
        }
        uint8_t s = 0;
        for (const uintptr_t pg : pages) {
            s ^= *(const volatile uint8_t *) pg;
        }
        asm volatile("" :: "r"(s));
        return;
    }

    // Blocking faults, but many at once: the point is queue depth, not per-fault latency.
    const size_t n_thread = std::min<size_t>((size_t) dsv41_engram_prefetch_nthread(), pages.size());
    auto touch = [&pages](size_t first, size_t step) {
        uint8_t s = 0;
        for (size_t i = first; i < pages.size(); i += step) {
            s ^= *(const volatile uint8_t *) pages[i];
        }
        // keep the loads from being optimised away without writing anything observable
        asm volatile("" :: "r"(s));
    };

    if (n_thread <= 1) {
        touch(0, 1);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(n_thread - 1);
    for (size_t k = 1; k < n_thread; ++k) {
        pool.emplace_back(touch, k, n_thread);
    }
    touch(0, n_thread);
    for (auto & th : pool) {
        th.join();
    }
#endif
}

class llm_graph_input_dsv41_engram : public llm_graph_input_i {
public:
    // A position with no token id (an image span: llama.cpp delivers those as an embedding-only
    // ubatch) is a "dead" token for the engram. The reference caches it as DEAD, look-back stops
    // there so no n-gram ever spans an image, and the engram gate is forced to 0 so the position
    // passes through untouched. ref: NgramHashState / Engram.forward in inference/{engram,model}.py
    static constexpr int32_t DEAD = -1;

    llm_graph_input_dsv41_engram(std::vector<int32_t> layer_ids, int64_t n_hash_cols) :
        layer_ids(std::move(layer_ids)), n_hash_cols(n_hash_cols) {}

    // In-runtime hasher: the reference NgramHashState. Each position is hashed with the compressed ids
    // of itself and the max_ngram-1 tokens before it (pad before the sequence start); the running XOR of
    // id*multiplier after step i is the (i+1)-gram hash, each landing in its own prime-sized bucket range.
    // The per-sequence compressed-id history persists across ubatches (decode) in a process-wide table,
    // reset when a sequence restarts at position 0. TODO: move the history into the context state.
    void set_input_hashed(const llama_ubatch * ubatch) {
        const auto & c = dsv41_engram_consts::get();
        GGML_ASSERT(c.loaded && "set DSV41_ENGRAM_CONSTS=<dir> (or DSV41_ENGRAM_HASHES for a precomputed table)");
        // no token ids => an image span; every position in this ubatch is DEAD (see above)
        const bool is_image = ubatch->token == nullptr;
        static std::map<int32_t, std::vector<int32_t>> hist;   // seq_id -> compressed id per position
        const int64_t n_tokens = ubatch->n_tokens;
        const int32_t G = c.max_ngram, H = c.n_heads;
        GGML_ASSERT((int64_t) (G - 1) * H == n_hash_cols);
        std::vector<std::vector<int32_t>> buf(layer_ids.size(), std::vector<int32_t>((size_t) n_hash_cols * n_tokens));
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int32_t seq = ubatch->seq_id[t][0];
            const int64_t pos = ubatch->pos[t];
            auto & h = hist[seq];
            if (pos == 0) { h.clear(); }
            if ((int64_t) h.size() <= pos) { h.resize(pos + 1, c.pad_id); }
            if (is_image) {
                h[pos] = DEAD;
            } else {
                const llama_token tok = ubatch->token[t];
                GGML_ASSERT(tok >= 0 && (size_t) tok < c.token_map.size());
                h[pos] = c.token_map[tok];
            }
            // once blocked, stay blocked: the sequence start and any dead token both end the look-back
            std::vector<int64_t> toks(G);
            bool blocked = false;
            for (int32_t sh = 0; sh < G; ++sh) {
                const int64_t p = pos - sh;
                blocked = blocked || p < 0 || h[p] == DEAD;
                toks[sh] = blocked ? c.pad_id : h[p];
            }
            for (size_t li = 0; li < layer_ids.size(); ++li) {
                const int64_t * mult = &c.multipliers[li * G];
                int64_t rolling = toks[0] * mult[0];
                for (int32_t i = 1; i < G; ++i) {
                    rolling ^= toks[i] * mult[i];
                    for (int32_t hh = 0; hh < H; ++hh) {
                        const int64_t prime  = c.primes [(li * (G - 1) + (i - 1)) * H + hh];
                        const int64_t offset = c.offsets[ li * (G - 1) * H + (i - 1) * H + hh];
                        buf[li][(size_t) t * n_hash_cols + (i - 1) * H + hh] = (int32_t) (rolling % prime + offset);
                    }
                }
            }
        }
        dsv41_engram_prefetch(embd, buf);
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            ggml_backend_tensor_set(hashes[li], buf[li].data(), 0, buf[li].size() * sizeof(int32_t));
        }
    }

    void set_input(const llama_ubatch * ubatch) override {
        if (text_mask) {
            // uniform per ubatch: llama.cpp never mixes token and embedding rows in one ubatch
            const std::vector<float> m((size_t) ubatch->n_tokens, ubatch->token ? 1.0f : 0.0f);
            ggml_backend_tensor_set(text_mask, m.data(), 0, m.size() * sizeof(float));
        }
        const char * path = getenv("DSV41_ENGRAM_HASHES");
        if (!path) {
            set_input_hashed(ubatch);
            return;
        }
        if (table.empty()) {
            FILE * f = std::fopen(path, "rb");
            GGML_ASSERT(f && "cannot open DSV41_ENGRAM_HASHES");
            std::fseek(f, 0, SEEK_END);
            const size_t n = (size_t) std::ftell(f) / sizeof(int32_t);
            std::fseek(f, 0, SEEK_SET);
            table.resize(n);
            const size_t got = std::fread(table.data(), sizeof(int32_t), n, f);
            std::fclose(f);
            GGML_ASSERT(got == n);
            n_pos_file = n / (layer_ids.size() * (size_t) n_hash_cols);
        }
        const int64_t n_tokens = ubatch->n_tokens;
        std::vector<std::vector<int32_t>> buf(layer_ids.size(), std::vector<int32_t>((size_t) n_hash_cols * n_tokens));
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            for (int64_t t = 0; t < n_tokens; ++t) {
                const size_t pos = (size_t) ubatch->pos[t];
                GGML_ASSERT(pos < n_pos_file && "DSV41_ENGRAM_HASHES has no row for this position");
                for (int64_t c = 0; c < n_hash_cols; ++c) {
                    buf[li][(size_t) t*n_hash_cols + c] = table[(li*n_pos_file + pos)*(size_t) n_hash_cols + (size_t) c];
                }
            }
        }
        dsv41_engram_prefetch(embd, buf);
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            ggml_backend_tensor_set(hashes[li], buf[li].data(), 0, buf[li].size()*sizeof(int32_t));
        }
    }

    // The engram graph topology depends only on the (constant) layer list and hash width: the hashes
    // themselves are re-uploaded by set_input on every ubatch. Allowing reuse here is what lets decode
    // skip the rebuild -- and, cross-node, lets the RPC backend send GRAPH_RECOMPUTE instead of
    // re-serializing the whole remote subgraph for every token.
    bool can_reuse(const llm_graph_params & params) override {
        const int64_t n_tokens = params.ubatch.n_tokens;
        bool res = true;
        for (const auto * t : hashes) {
            res &= t->ne[0] == n_hash_cols;
            res &= t->ne[1] == n_tokens;
        }
        if (text_mask) {
            res &= text_mask->ne[1] == n_tokens;
        }
        return res;
    }

    std::vector<int32_t> layer_ids;                 // engram layers in order (index = layer_hash_index)
    int64_t n_hash_cols;
    std::vector<ggml_tensor *> hashes;              // I32 [n_hash_cols, n_tokens] per engram layer
    std::vector<const ggml_tensor *> embd;          // engram_embd per engram layer, for the row prefetch
    ggml_tensor * text_mask = nullptr;              // F32 [1, n_tokens]: 1 for text, 0 for image spans
    std::vector<int32_t> table;
    size_t n_pos_file = 0;
};

ggml_tensor * llama_model_deepseek4::graph_v41::build_engram(
        const llama_model & model,
        ggml_tensor * hashes,
        ggml_tensor * text_mask,
        ggml_tensor * x,
        int il) const {
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.engram_embd && layer.engram_wkv && layer.engram_q && layer.engram_k);

    const int64_t hc   = hparams.dsv4_hc_mult;
    const int64_t nt   = x->ne[2];
    const int64_t ehd  = hparams.engram_head_size;      // 256
    const int64_t ncol = hashes->ne[0];                 // (max_ngram_size-1)*n_heads = 24
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == hc && hashes->ne[1] == nt);

    // The per-stream views of x taken below are strided: ne = [n_embd, nt] with nb1 = x->nb[2], one n_embd row per
    // hc*n_embd token stride. ggml_nbytes of such a view spans from its offset to the end of its last row, i.e. very
    // nearly the whole tensor, and that is what ggml_backend_sched copies for a split input. When x is produced on
    // another device the hc views therefore each drag almost the entire residual across the link: measured on the
    // RPC0 (GB10) -> CUDA0 boundary, 4 x 79 MiB per 1024-token ubatch of an 80 MiB tensor, ~316 KiB/token where
    // 80 KiB/token is needed.
    //
    // Take one contiguous copy of the residual at the device boundary and pin it to this layer's own device (see
    // llama_context::graph_get_cb, "engram_inp"): the boundary then carries the base once, contiguously, and all hc
    // stream views are local to the consumer. ggml_cont is a plain copy - same values, same order, bit-exact.
    if (il > 0 && model.dev_layer(il) != model.dev_layer(il - 1)) {
        x = ggml_cont(ctx0, x);
        cb(x, "engram_inp", il);
    }

    // gather the n-gram rows, then one wkv projection over their concatenation -> key per copy + value
    ggml_tensor * ids  = ggml_reshape_1d(ctx0, hashes, ncol*nt);
    ggml_tensor * rows = ggml_get_rows(ctx0, layer.engram_embd, ids);           // [ehd, ncol*nt]
    rows = ggml_reshape_2d(ctx0, rows, ncol*ehd, nt);                           // [ncol*ehd, nt]
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.engram_wkv, rows);              // [n_embd*(hc+1), nt]
    cb(kv, "engram_kv", il);

    const size_t es = ggml_element_size(kv);
    ggml_tensor * key = ggml_view_3d(ctx0, kv, n_embd, hc, nt, n_embd*es, kv->nb[1], 0);   // [n_embd, hc, nt]
    ggml_tensor * val = ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc*n_embd*es);       // [n_embd, nt]

    // q/k ship as BF16; the CPU backend has no F32 x BF16 mul, and the reference forms the product in f32
    ggml_tensor * w = ggml_mul(ctx0,
            ggml_cast(ctx0, layer.engram_q, GGML_TYPE_F32),
            ggml_cast(ctx0, layer.engram_k, GGML_TYPE_F32));                            // [n_embd, hc] f32
    const float eps = hparams.f_norm_rms_eps;

    ggml_tensor * out = nullptr;
    for (int64_t s = 0; s < hc; ++s) {
        ggml_tensor * xs = ggml_view_2d(ctx0, x,   n_embd, nt, x->nb[2],   s*x->nb[1]);
        ggml_tensor * ks = ggml_view_2d(ctx0, key, n_embd, nt, key->nb[2], s*key->nb[1]);
        ggml_tensor * ws = ggml_view_1d(ctx0, w,   n_embd, s*w->nb[1]);

        // gate = sigmoid(signed_sqrt(<rms_norm(x), w * rms_norm(key)> / sqrt(dim))), per (token, copy)
        ggml_tensor * xn  = ggml_rms_norm(ctx0, ggml_cont(ctx0, xs), eps);
        ggml_tensor * kn  = ggml_rms_norm(ctx0, ggml_cont(ctx0, ks), eps);
        ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, xn, ws), kn);
        dot = ggml_sum_rows(ctx0, dot);                                           // [1, nt]
        dot = ggml_scale(ctx0, dot, 1.0f/sqrtf((float) n_embd));
        ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, dot), mag)); // [1, nt]
        if (text_mask) {
            // image-span positions take no engram contribution (reference: gate.masked_fill(~mask, 0))
            gate = ggml_mul(ctx0, gate, text_mask);
        }

        ggml_tensor * ys = ggml_add(ctx0, xs, ggml_mul(ctx0, val, gate));        // x + gate * value
        ys = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ys), n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, ys, 1) : ys;
    }
    cb(out, "engram_out", il);
    return out;
}

// CSA2/CED compressed-position inputs for a prefill ubatch, one entry per compress ratio present:
// Graph inputs for the V4.1 memory (llama_kv_cache_dsv41): the window cache's write indices and
// mask (every layer), and for the compressed cache the write indices plus the two visibility masks
// over its cells:
//   r1 (ratio-1 sources): plain causal -- pos_j <= p, one compressed K per token;
//   r2 (ratio-2 sources): a 2-token group [2m, 2m+1] lives at the cell of token 2m+1 and is visible
//      once the query has passed it (2m+1 <= p), i.e. causal AND the cell position is odd.
// Ratio-2 pooling reads the even partner's raw projections (parked in the partner's cell) through
// cell_prev (I32, for get_rows) and selects per token with odd_f; the group RoPE position is pos_grp.
// Every tensor is [n_tokens]-shaped so the graph shape does not depend on the token parity split.
// Mask shapes mirror the window mask so the two concatenate: [n_kv, n_tokens/n_stream, 1, n_stream].
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_cparams & cparams, const llama_kv_cache_dsv41_context * mctx) :
        cparams(cparams), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto * ctx_win  = mctx->get_win();
        const auto * ctx_comp = mctx->get_comp();

        ctx_win->set_input_k_idxs(k_idxs_win, ubatch);
        ctx_win->set_input_kq_mask(kq_mask_win, ubatch, cparams.causal_attn);

        ctx_comp->set_input_k_idxs(k_idxs_comp, ubatch);
        mctx->get_idx()->set_input_k_idxs(k_idxs_idx, ubatch);
        ctx_comp->set_input_kq_mask(kq_mask_r1, ubatch, cparams.causal_attn);

        const int64_t n_tokens = ubatch->n_tokens;
        GGML_ASSERT(kq_mask_r1->ne[3] == 1 && "V4.1: single stream");
        GGML_ASSERT(ggml_backend_buffer_is_host(k_idxs_comp->buffer));
        const int64_t * cell = (const int64_t *) k_idxs_comp->data;

        std::vector<int32_t> c_prev((size_t) n_tokens), p_grp((size_t) n_tokens);
        std::vector<float>   odd((size_t) n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            const llama_pos    pos = ubatch->pos[t];
            const llama_seq_id seq = ubatch->seq_id[t][0];
            const bool is_odd = (pos % 2) == 1;
            odd[t] = is_odd ? 1.0f : 0.0f;
            // even tokens (and pos 0) pool with themselves -- the select discards the result anyway
            int64_t prev = cell[t];
            int32_t pg   = pos;
            if (is_odd) {
                pg = pos - 1;
                if (t > 0 && ubatch->pos[t-1] == pos - 1 && ubatch->seq_id[t-1][0] == seq) {
                    prev = cell[t-1];
                } else {
                    // partner written by an earlier ubatch: find its cell by position
                    prev = -1;
                    const auto & cells = mctx->get_comp_cache()->get_cells(seq);
                    for (uint32_t j = 0; j < cells.size(); ++j) {
                        if (!cells.is_empty(j) && cells.seq_has(j, seq) && cells.pos_get(j) == pos - 1) {
                            prev = (int64_t) j;
                            break;
                        }
                    }
                    GGML_ASSERT(prev >= 0 && "V4.1: the even partner of an odd token is not in the compressed cache");
                }
            }
            c_prev[t] = (int32_t) prev;
            p_grp [t] = pg;
        }
        ggml_backend_tensor_set(cell_prev, c_prev.data(), 0, c_prev.size()*sizeof(int32_t));
        ggml_backend_tensor_set(pos_grp,   p_grp .data(), 0, p_grp .size()*sizeof(int32_t));
        ggml_backend_tensor_set(odd_f,     odd   .data(), 0, odd   .size()*sizeof(float));

        // r2 mask: causal over the odd-position cells
        {
            const int64_t n_kv = kq_mask_r2->ne[0];
            std::vector<float> m((size_t) n_kv*n_tokens, -INFINITY);
            for (int64_t t = 0; t < n_tokens; ++t) {
                const llama_seq_id seq = ubatch->seq_id[t][0];
                const llama_pos    p   = ubatch->pos[t];
                const auto & cells = mctx->get_comp_cache()->get_cells(seq);
                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) { continue; }
                    const llama_pos pj = cells.pos_get(j);
                    if (pj <= p && (pj % 2) == 1) { m[(size_t) t*n_kv + j] = 0.0f; }
                }
            }
            if (kq_mask_r2->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> h(m.size());
                for (size_t i = 0; i < m.size(); ++i) { h[i] = ggml_fp32_to_fp16(m[i]); }
                ggml_backend_tensor_set(kq_mask_r2, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
            } else {
                ggml_backend_tensor_set(kq_mask_r2, m.data(), 0, m.size()*sizeof(float));
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_new = static_cast<const llama_kv_cache_dsv41_context *>(params.mctx);
        this->mctx = mctx_new;
        const int64_t n_tokens = params.ubatch.n_tokens;
        bool res = true;
        res &= k_idxs_win ->ne[0] == n_tokens;
        res &= k_idxs_comp->ne[0] == n_tokens;
        res &= k_idxs_idx ->ne[0] == n_tokens;
        res &= kq_mask_win->ne[0] == (int64_t) mctx_new->get_win ()->get_n_kv();
        res &= kq_mask_r1 ->ne[0] == (int64_t) mctx_new->get_comp()->get_n_kv();
        res &= kq_mask_win->ne[1] == n_tokens && kq_mask_win->ne[3] == 1;
        res &= kq_mask_r1 ->ne[1] == n_tokens && kq_mask_r1 ->ne[3] == 1;
        return res;
    }

    ggml_tensor * k_idxs_win  = nullptr; // I64 [n_tokens]
    ggml_tensor * kq_mask_win = nullptr; // F32/F16 [n_kv_win, n_tokens, 1, 1]

    ggml_tensor * k_idxs_comp = nullptr; // I64 [n_tokens]  every token's compressed cell
    ggml_tensor * k_idxs_idx  = nullptr; // I64 [n_tokens]  same cell, in the indexer-key cache
    ggml_tensor * kq_mask_r1  = nullptr; // F32/F16 [n_kv_comp, n_tokens, 1, 1]
    ggml_tensor * kq_mask_r2  = nullptr; // F32/F16 [n_kv_comp, n_tokens, 1, 1]

    ggml_tensor * cell_prev   = nullptr; // I32 [n_tokens]  cell of the even partner (own cell for even tokens)
    ggml_tensor * pos_grp     = nullptr; // I32 [n_tokens]  group RoPE position (pos - 1 for odd tokens)
    ggml_tensor * odd_f       = nullptr; // F32 [1, n_tokens] 1.0 for odd tokens

    const llama_cparams cparams;
    const llama_kv_cache_dsv41_context * mctx;
};

llm_graph_input_dsv41 * llama_model_deepseek4::graph_v41::build_inp_dsv41() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv41_context *>(mctx);
    const auto * ctx_win  = mctx_cur->get_win();
    const auto * ctx_comp = mctx_cur->get_comp();

    // the indexer-key cache mirrors the compressed cache's cells, so the two must agree on n_kv
    GGML_ASSERT(mctx_cur->get_idx()->get_n_kv() == ctx_comp->get_n_kv());

    auto inp = std::make_unique<llm_graph_input_dsv41>(cparams, mctx_cur);

    // flash attention requires an f16 mask
    const ggml_type mtype = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    auto mask = [&](uint32_t n_kv) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx0, mtype, n_kv, n_tokens, 1, 1);
        ggml_set_input(t);
        return t;
    };
    auto vec = [&](ggml_type type, int64_t ne0, int64_t ne1 = 0) {
        ggml_tensor * t = ne1 ? ggml_new_tensor_2d(ctx0, type, ne0, ne1) : ggml_new_tensor_1d(ctx0, type, ne0);
        ggml_set_input(t);
        return t;
    };

    inp->k_idxs_win  = ctx_win->build_input_k_idxs(ctx0, ubatch);
    inp->kq_mask_win = mask(ctx_win->get_n_kv());

    inp->k_idxs_comp = ctx_comp->build_input_k_idxs(ctx0, ubatch);
    inp->k_idxs_idx  = mctx_cur->get_idx()->build_input_k_idxs(ctx0, ubatch);
    inp->kq_mask_r1  = mask(ctx_comp->get_n_kv());
    inp->kq_mask_r2  = mask(ctx_comp->get_n_kv());

    inp->cell_prev = vec(GGML_TYPE_I32, n_tokens);
    inp->pos_grp   = vec(GGML_TYPE_I32, n_tokens);
    inp->odd_f     = vec(GGML_TYPE_F32, 1, n_tokens);

    return (llm_graph_input_dsv41 *) res->add_input(std::move(inp));
}

// CSA2 indexer: score this layer's queries against the indexer keys its CED source layer published,
// and return the indices of the best hparams.indexer_top_k compressed positions. Mirrors the DSV4
// lightning indexer (build_lid_top_k) except that the keys live in the V4.1 indexer-key cache, whose
// cells coincide with the compressed cache's, and the visibility mask is the compressed one.
ggml_tensor * llama_model_deepseek4::graph_v41::build_indexer_top_k_v41(
        const llama_model & model,
        llm_graph_input_dsv41 * inp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_rope = hparams.n_rot();
    const int64_t n_idx_nope = n_idx_dim - n_idx_rope;
    const int64_t nt         = cur->ne[1];

    const int src = ced_src[il];
    GGML_ASSERT(src >= 0 && "CSA2 index layer before its kv_source");
    GGML_ASSERT(n_idx_dim >= n_idx_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_nope);
    cb(idx_q, "csa2_idx_q", il);

    // one weight per head, scaled to the reference's softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_dim*n_idx_head)));
    cb(idx_w, "csa2_idx_w", il);

    ggml_tensor * mask  = hparams.dsv4_compress_ratios[src] == 1 ? inp->kq_mask_r1 : inp->kq_mask_r2;
    ggml_tensor * idx_k = inp->mctx->get_idx()->get_k(ctx0, src);

    const int64_t n_comp = mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "csa2_idx_k_all", il);

    const int64_t n_stream = idx_k->ne[3];
    idx_q = ggml_view_4d(ctx0, idx_q,
            idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
    idx_w = ggml_view_4d(ctx0, idx_w,
            idx_w->ne[0], idx_w->ne[1]/n_stream, idx_w->ne[2], n_stream,
            idx_w->nb[1], idx_w->nb[2]/n_stream, idx_w->nb[3]/n_stream, 0);

    // The fused lightning indexer (the same op the DSV4 build_lid_top_k uses) computes
    // relu(q.k) * w summed over heads, plus the mask, without ever materialising the
    // [n_comp, nt, n_head] f32 intermediates of the unfused chain below -- those are what made
    // the compute reserve scale as context x ubatch. DSV41_UNFUSED_LID=1 forces the unfused
    // chain so the two can be A/B'd inside one build.
    static const bool unfused_lid = getenv("DSV41_UNFUSED_LID") != nullptr;

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid && !unfused_lid) {
        // the op needs an f16 mask; the V4.1 compressed mask is f16 only under flash attention.
        // It holds nothing but 0 and -inf, so narrowing it is exact.
        if (mask->type != GGML_TYPE_F16) {
            mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        }
        score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, mask);
        cb(score, "csa2_idx_score", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
        idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        score = ggml_mul_mat(ctx0, idx_k, idx_q);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, idx_w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        // the mask is F16 under flash attention and the score is F32; the mask only ever holds 0 or -inf,
        // so widening it is exact.
        if (mask->type != score->type) {
            mask = ggml_cast(ctx0, mask, score->type);
        }
        score = ggml_add(ctx0, score, mask);
        cb(score, "csa2_idx_score", il);
    }

    const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "csa2_idx_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph_v41::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv41 * inp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    // compress layers rope with the compress base + YaRN; pure window layers use the base theta, no YaRN
    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    // V4.1: no per-head RMS norm on q (0731's graph applies one here); rope on the tail dims only
    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const float   kq_scale = 1.0f/sqrtf(float(n_embd_head));
    const int64_t ratio    = hparams.dsv4_compress_ratios[il];

    const auto * ctx_win  = inp->mctx->get_win();
    const auto * ctx_comp = inp->mctx->get_comp();
    const auto * ctx_idx  = inp->mctx->get_idx();

    auto rope = [&](ggml_tensor * t, ggml_tensor * pos) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        return ggml_rope_set_offset(t, n_embd_head_nope);
    };

    // window K through the window cache exactly as build_attn does it: K = V = the single shared kv head
    GGML_ASSERT(hparams.is_swa(il));
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);
    ggml_build_forward_expand(gf, ctx_win->cpy_k(ctx0, kv, inp->k_idxs_win, il));
    ggml_tensor * k_all   = ctx_win->get_k(ctx0, il);                                              // [n_embd_head, 1, n_kv_win, 1]
    ggml_tensor * kq_mask = inp->kq_mask_win;
    GGML_ASSERT(k_all->ne[3] == 1 && "V4.1: single stream");

    if (ratio != 0) {
        // kv_source layers compress their own KV for their CED group into the compressed cache: the
        // reference Compressor -- ratio 1 = plain wkv + norm; ratio 2 = softmax(wgate)-weighted pool of
        // each group of 2 tokens, norm -- then RoPE at the group position. The normed, pre-RoPE latent
        // (comp_pre) is what the indexer's key projection consumes.
        if (hparams.is_kv_source_impl[il]) {
            ggml_tensor * comp = nullptr;
            ggml_tensor * comp_pre = nullptr;   // normed compressed latent, before RoPE  [n_embd_head, nt]
            ggml_tensor * comp_pos = nullptr;   // the positions comp was roped at
            if (ratio == 1) {
                comp = build_lora_mm(layer.attn_comp_wkv, cur);                                    // [n_embd_head, nt]
                comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                comp_pre = comp;
                comp_pos = inp_pos;
                comp = rope(ggml_reshape_3d(ctx0, comp, n_embd_head, 1, nt), inp_pos);
            } else {
                GGML_ASSERT(ratio == 2);
                ggml_tensor * ckv = build_lora_mm(layer.attn_comp_wkv,   cur);                     // [n_embd_head, nt]
                ggml_tensor * csc = build_lora_mm(layer.attn_comp_wgate, cur);
                cb(ckv, "ced_raw_kv", il);
                cb(csc, "ced_raw_gate", il);

                // every token parks its raw projections in its own cell first (K = wkv, V = wgate) --
                // an unpaired even token keeps them there until its partner arrives
                ggml_build_forward_expand(gf, ctx_comp->cpy_k(ctx0, ggml_reshape_3d(ctx0, ckv, n_embd_head, 1, nt), inp->k_idxs_comp, il));
                ggml_build_forward_expand(gf, ctx_comp->cpy_v(ctx0, ggml_reshape_3d(ctx0, csc, n_embd_head, 1, nt), inp->k_idxs_comp, il));

                // the partner's parked projections (for even tokens: its own -- discarded below)
                ggml_tensor * k_view = ctx_comp->get_k(ctx0, il);
                ggml_tensor * v_view = ctx_comp->get_v(ctx0, il);
                k_view = ggml_view_2d(ctx0, k_view, n_embd_head, k_view->ne[2], k_view->nb[2], 0);
                v_view = ggml_view_2d(ctx0, v_view, n_embd_head, v_view->ne[2], v_view->nb[2], 0);
                ggml_tensor * pkv = ggml_get_rows(ctx0, k_view, inp->cell_prev);                   // [n_embd_head, nt] f32
                ggml_tensor * psc = ggml_get_rows(ctx0, v_view, inp->cell_prev);

                ggml_tensor * values = ggml_concat(ctx0, ggml_reshape_3d(ctx0, pkv, n_embd_head, 1, nt),
                                                         ggml_reshape_3d(ctx0, ckv, n_embd_head, 1, nt), 1); // [n_embd_head, 2, nt]
                ggml_tensor * scores = ggml_concat(ctx0, ggml_reshape_3d(ctx0, psc, n_embd_head, 1, nt),
                                                         ggml_reshape_3d(ctx0, csc, n_embd_head, 1, nt), 1);
                values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));                  // [2, n_embd_head, nt]
                scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));
                ggml_tensor * w = ggml_soft_max(ctx0, scores);                                     // over the group members
                comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, values, w));                             // [1, n_embd_head, nt]
                comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));                      // [n_embd_head, 1, nt]
                comp = ggml_reshape_2d(ctx0, comp, n_embd_head, nt);
                comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                comp_pre = comp;
                comp_pos = inp->pos_grp;
                comp = rope(ggml_reshape_3d(ctx0, comp, n_embd_head, 1, nt), inp->pos_grp);

                // odd tokens store the group K at their cell; even tokens keep their parked raw wkv
                ggml_tensor * odd  = ggml_reshape_3d(ctx0, inp->odd_f, 1, 1, nt);
                ggml_tensor * even = ggml_scale_bias(ctx0, odd, -1.0f, 1.0f);
                ggml_tensor * raw  = ggml_reshape_3d(ctx0, ckv, n_embd_head, 1, nt);
                comp = ggml_add(ctx0, ggml_mul(ctx0, comp, odd), ggml_mul(ctx0, raw, even));
            }
            cb(comp, "ced_k", il);
            ggml_build_forward_expand(gf, ctx_comp->cpy_k(ctx0, comp, inp->k_idxs_comp, il));

            // A CSA2-Full layer also publishes the indexer key for the same cell, so the layers in its
            // group can score against it. Roped at the same positions as the compressed K it describes,
            // so the indexer's q.k is in one rotational frame (no folded-rotation trick needed here).
            if (layer.indexer_attn_k) {
                const int64_t idx_head = hparams.indexer_head_size;
                ggml_tensor * kidx = build_lora_mm(layer.indexer_attn_k, comp_pre);                // [idx_head, nt]
                kidx = build_norm(kidx, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                kidx = ggml_reshape_3d(ctx0, kidx, idx_head, 1, nt);
                kidx = ggml_rope_ext(ctx0, kidx, comp_pos, nullptr, n_embd_head_rope, rope_type,
                        n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                        beta_fast_l, beta_slow_l);
                kidx = ggml_rope_set_offset(kidx, idx_head - n_embd_head_rope);
                cb(kidx, "csa2_idx_k", il);
                ggml_build_forward_expand(gf, ctx_idx->cpy_k(ctx0, kidx, inp->k_idxs_idx, il));
            }
        }

        // ONE softmax over [window ; compressed] with the sinks (the reference sparse_attn over
        // cat([kv, compress_kv])); the compressed K comes from the CED source layer's cache
        const int src = ced_src[il];
        GGML_ASSERT(src >= 0 && "CED consumer layer before its kv_source");
        ggml_tensor * k_comp = ctx_comp->get_k(ctx0, src);                                         // [n_embd_head, 1, n_kv_comp, 1]
        ggml_tensor * m_comp = hparams.dsv4_compress_ratios[src] == 1 ? inp->kq_mask_r1 : inp->kq_mask_r2;

        // An index_source layer picks the best indexer_top_k compressed positions; the layers after it
        // in the same group reuse that selection (the reference Reindex behaviour). Below the top-k the
        // selection covers everything visible and this is a no-op.
        // DSV41_NO_IDX_TOPK=1 reverts to the pre-M9 behaviour (attend every visible compressed
        // position) so the top-k can be A/B'd against it inside one build.
        static const bool no_idx_topk = getenv("DSV41_NO_IDX_TOPK") != nullptr;
        if (!no_idx_topk && hparams.is_index_source_impl[il]) {
            csa2_top_k = build_indexer_top_k_v41(model, inp, qr, cur, inp_pos, il);
        }
        if (csa2_top_k) {
            m_comp = build_top_k_mask(m_comp, csa2_top_k, "csa2_kq_mask_top_k", il);
        }

        if (k_comp->type != k_all->type) { k_comp = ggml_cast(ctx0, k_comp, k_all->type); }
        k_all   = ggml_concat(ctx0, k_all, k_comp, 2);
        kq_mask = ggml_concat(ctx0, kq_mask, m_comp, 0);
        cb(k_all,   "csa2_k_all",   il);
        cb(kq_mask, "csa2_kq_mask", il);
    }

    // With a top-k selection the mask leaves at most (sliding window + indexer_top_k) positions
    // unmasked per query, however long the compressed stream is. Handing that bound to
    // ggml_flash_attn_ext lets the kernel stop scanning there instead of walking all of k_all --
    // 640 positions instead of 33 792 at a 32K context. 0 means "no bound" (the pure-window layers,
    // and the DSV41_NO_IDX_TOPK path, where every visible position really can be unmasked).
    // DSV41_NO_FA_KV_MAX=1 disables just the bound (keeping the top-k) so the two can be compared
    // bit-for-bit on the CUDA path. The CPU backend ignores n_kv_max entirely, so a CPU run cannot
    // validate it -- an under-sized bound would silently drop unmasked positions.
    static const bool no_fa_kv_max = getenv("DSV41_NO_FA_KV_MAX") != nullptr;
    const int64_t n_kv_max = (csa2_top_k && !no_fa_kv_max)
        ? std::min<int64_t>(inp->kq_mask_win->ne[0], hparams.n_swa) + csa2_top_k->ne[0]
        : 0;
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
    cb(out, ratio != 0 ? "attn_csa2" : "attn_window", il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    // block-diagonal wo_a over the head groups, then wo_b
    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek4::graph_v41::graph_v41(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.is_dsv41);
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv41 * inp_attn = build_inp_dsv41();

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Single-Pass mHC: each sublayer applies the pre-mix the PREVIOUS sublayer produced. The chain
    // starts with the one-hot mix selecting stream 0 (the reference make_identity_pre_mix).
    ggml_tensor * ones  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f);
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f);
    ggml_tensor * pre_mix = ggml_concat(ctx0, ones, zeros, 0); // [hc, n_tokens]
    cb(pre_mix, "hc_pre_mix_init", -1);

    // Engram hash-id inputs, one per engram layer
    llm_graph_input_dsv41_engram * inp_engram = nullptr;
    std::vector<int32_t> engram_layers;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.is_engram_impl[il]) { engram_layers.push_back(il); }
    }
    if (!engram_layers.empty()) {
        const int64_t n_hash_cols = (int64_t) (hparams.engram_max_ngram - 1) * hparams.engram_n_head;
        auto inp = std::make_unique<llm_graph_input_dsv41_engram>(engram_layers, n_hash_cols);
        for (size_t li = 0; li < engram_layers.size(); ++li) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_hash_cols, n_tokens);
            ggml_set_input(t);
            inp->hashes.push_back(t);
            inp->embd.push_back(model.layers[engram_layers[li]].engram_embd);
        }
        // 1 for text, 0 for an image span: shuts the engram gate so those positions pass through
        inp->text_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_input(inp->text_mask);
        ggml_set_name(inp->text_mask, "engram_text_mask");
        inp_engram = (llm_graph_input_dsv41_engram *) res->add_input(std::move(inp));
    }

    // CED: the kv_source layer whose compressed K each layer attends (nearest at or before il)
    ced_src.assign(n_layer, -1);
    for (int il = 0, src = -1; il < n_layer; ++il) {
        if (hparams.is_kv_source_impl[il]) { src = il; }
        ced_src[il] = src;
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        if (hparams.is_engram_impl[il]) {
            const size_t li = std::find(engram_layers.begin(), engram_layers.end(), il) - engram_layers.begin();
            inpL = build_engram(model, inp_engram->hashes[li], inp_engram->text_mask, inpL, il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr, * attn_post = nullptr, * attn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &attn_pre, &attn_post, &attn_comb, il);

        cb(pre_mix, "hc_pre_mix", il);                 // the gates themselves, for the oracle diff
        cur = build_hc_pre(inpL, pre_mix, il);          // the previous sublayer's mix
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(model, inp_attn, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, attn_post, attn_comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        ggml_tensor * ffn_pre = nullptr, * ffn_post = nullptr, * ffn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &ffn_pre, &ffn_post, &ffn_comb, il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_post);
        ggml_build_forward_expand(gf, ffn_comb);

        cur = build_hc_pre(inpL, attn_pre, il);         // this layer's attention mix feeds its FFN
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_post, ffn_comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);

        pre_mix = ffn_pre;                              // threaded to the next layer's attention
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }

    if (inp_out_ids) {
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);
    }

    // V4.1 output collapse: the threaded pre-mix (the last FFN's), no output_hc_* head
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
