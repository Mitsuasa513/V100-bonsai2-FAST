#include "llama-kv-cache-dsv41.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_dsv41
//

llama_kv_cache_dsv41::llama_kv_cache_dsv41(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad) : unified(unified), hparams_idx(model.hparams) {
    const auto & hparams = model.hparams;

    const layer_filter_cb filter_win  = [](int32_t il) { GGML_UNUSED(il); return true; };
    const layer_filter_cb filter_comp = [&](int32_t il) { return il < (int32_t) hparams.n_layer() && hparams.is_kv_source_impl[il] != 0; };
    // only a layer that is BOTH index- and kv-source owns indexer_attn_k ("CSA2 Full"); the Reindex
    // layers score their own queries against the keys their kv_source published, so the key cache
    // mirrors kv_comp exactly -- same layers, same cells, only narrower.
    const layer_filter_cb filter_idx  = [&](int32_t il) { return il < (int32_t) hparams.n_layer() && hparams.is_kv_source_impl[il] != 0 && hparams.is_index_source_impl[il] != 0; };

    // the window cache is padded to 256 for performance, as in llama_kv_cache_iswa
    uint32_t size_win = GGML_PAD(std::min(kv_size, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
    if (swa_full) {
        size_win = kv_size;
    }

    LLAMA_LOG_INFO("%s: creating window KV cache (all layers), size = %u cells\n", __func__, size_win);
    kv_win = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_win, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, nullptr, filter_win, nullptr, nullptr, "win");

    // K / V both used (V parks the ratio-2 gate scores), never transposed
    LLAMA_LOG_INFO("%s: creating compressed KV cache (kv_source layers), size = %u cells\n", __func__, kv_size);
    kv_comp = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            /*v_trans*/ false, offload, unified, kv_size, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_comp, nullptr, nullptr, "comp");

    // one indexer key per compressed cell on the index_source layers: one head, indexer_head_size
    // wide, K only. Same kv_size as the compressed cache so the cell indices coincide.
    std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
    hparams_idx.n_embd_head_k_full = hparams.indexer_head_size;
    hparams_idx.n_embd_head_v_full = hparams.indexer_head_size;
    hparams_idx.n_embd_head_k_swa  = hparams.indexer_head_size;
    hparams_idx.n_embd_head_v_swa  = hparams.indexer_head_size;
    // K-only storage: llama_kv_cache keys this off is_mla()
    hparams_idx.n_embd_head_k_mla_impl = hparams.indexer_head_size;
    hparams_idx.n_embd_head_v_mla_impl = hparams.indexer_head_size;

    LLAMA_LOG_INFO("%s: creating indexer-key KV cache (CSA2-Full layers), size = %u cells\n", __func__, kv_size);
    kv_idx = std::make_unique<llama_kv_cache>(
            model, hparams_idx, type_k, type_v,
            /*v_trans*/ false, offload, unified, kv_size, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_idx, nullptr, nullptr, "idx");
}

void llama_kv_cache_dsv41::clear(bool data) {
    kv_win ->clear(data);
    kv_comp->clear(data);
    kv_idx ->clear(data);
}

bool llama_kv_cache_dsv41::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;
    res = res & kv_win ->seq_rm(seq_id, p0, p1);
    res = res & kv_comp->seq_rm(seq_id, p0, p1);
    res = res & kv_idx ->seq_rm(seq_id, p0, p1);
    return res;
}

void llama_kv_cache_dsv41::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_win ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_comp->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_idx ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_dsv41::seq_keep(llama_seq_id seq_id) {
    kv_win ->seq_keep(seq_id);
    kv_comp->seq_keep(seq_id);
    kv_idx ->seq_keep(seq_id);
}

void llama_kv_cache_dsv41::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_win ->seq_add(seq_id, p0, p1, shift);
    kv_comp->seq_add(seq_id, p0, p1, shift);
    kv_idx ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsv41::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_win ->seq_div(seq_id, p0, p1, d);
    kv_comp->seq_div(seq_id, p0, p1, d);
    kv_idx ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsv41::seq_pos_min(llama_seq_id seq_id) const {
    return kv_comp->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_dsv41::seq_pos_max(llama_seq_id seq_id) const {
    return kv_comp->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv41::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_win->memory_breakdown();
    for (const auto & buft_size : kv_idx->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_comp->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_dsv41::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        auto sinfos_win = kv_win->prepare(ubatches);
        if (sinfos_win.empty()) {
            break;
        }

        auto sinfos_comp = kv_comp->prepare(ubatches);
        if (sinfos_comp.empty()) {
            break;
        }

        auto sinfos_idx = kv_idx->prepare(ubatches);
        if (sinfos_idx.empty()) {
            break;
        }

        assert(sinfos_win.size() == sinfos_comp.size());
        assert(sinfos_idx.size() == sinfos_comp.size());

        return std::make_unique<llama_kv_cache_dsv41_context>(
                this, std::move(sinfos_win), std::move(sinfos_comp), std::move(sinfos_idx), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        auto sinfos_win = kv_win->prepare(ubatches);
        if (sinfos_win.empty()) {
            break;
        }

        auto sinfos_comp = kv_comp->prepare(ubatches);
        if (sinfos_comp.empty()) {
            break;
        }

        auto sinfos_idx = kv_idx->prepare(ubatches);
        if (sinfos_idx.empty()) {
            break;
        }

        assert(sinfos_win.size() == sinfos_comp.size());
        assert(sinfos_idx.size() == sinfos_comp.size());

        return std::make_unique<llama_kv_cache_dsv41_context>(
                this, std::move(sinfos_win), std::move(sinfos_comp), std::move(sinfos_idx), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_dsv41_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv41::init_full() {
    return std::make_unique<llama_kv_cache_dsv41_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv41::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv41_context>(this, lctx, optimize);
}

bool llama_kv_cache_dsv41::get_can_shift() const {
    // the compressed K is RoPE'd at group positions: shifting is not supported yet
    return false;
}

void llama_kv_cache_dsv41::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_comp->state_write(io, seq_id, flags);
        kv_idx ->state_write(io, seq_id, flags);
    }
    kv_win->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsv41::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_comp->state_read(io, seq_id, flags);
        kv_idx ->state_read(io, seq_id, flags);
    }
    kv_win->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_dsv41::get_win() const {
    return kv_win.get();
}

llama_kv_cache * llama_kv_cache_dsv41::get_comp() const {
    return kv_comp.get();
}

llama_kv_cache * llama_kv_cache_dsv41::get_idx() const {
    return kv_idx.get();
}

//
// llama_kv_cache_dsv41_context
//

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(
        llama_kv_cache_dsv41 * kv) :
    ctx_win (kv->get_win ()->init_full()),
    ctx_comp(kv->get_comp()->init_full()),
    ctx_idx (kv->get_idx ()->init_full()),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_win->get_status(), ctx_comp->get_status()), ctx_idx->get_status())),
    kv_comp(kv->get_comp()) {
}

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(
        llama_kv_cache_dsv41 * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_win (kv->get_win ()->init_update(lctx, optimize)),
    ctx_comp(kv->get_comp()->init_update(lctx, optimize)),
    ctx_idx (kv->get_idx ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_win->get_status(), ctx_comp->get_status()), ctx_idx->get_status())),
    kv_comp(kv->get_comp()) {
}

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(
        llama_kv_cache_dsv41 * kv,
        slot_info_vec_t sinfos_win,
        slot_info_vec_t sinfos_comp,
        slot_info_vec_t sinfos_idx,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    ctx_win (new llama_kv_cache_context(kv->get_win (), std::move(sinfos_win),  this->ubatches)),
    ctx_comp(new llama_kv_cache_context(kv->get_comp(), std::move(sinfos_comp), this->ubatches)),
    ctx_idx (new llama_kv_cache_context(kv->get_idx (), std::move(sinfos_idx),  this->ubatches)),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_win->get_status(), ctx_comp->get_status()), ctx_idx->get_status())),
    kv_comp(kv->get_comp()) {
}

llama_kv_cache_dsv41_context::~llama_kv_cache_dsv41_context() = default;

bool llama_kv_cache_dsv41_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_win ->next();
    ctx_comp->next();
    ctx_idx ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv41_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_win ->apply();
    res = res & ctx_comp->apply();
    res = res & ctx_idx ->apply();

    return res;
}

llama_memory_status llama_kv_cache_dsv41_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv41_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_dsv41_context::get_win() const {
    return static_cast<const llama_kv_cache_context *>(ctx_win.get());
}

const llama_kv_cache_context * llama_kv_cache_dsv41_context::get_comp() const {
    return static_cast<const llama_kv_cache_context *>(ctx_comp.get());
}

const llama_kv_cache_context * llama_kv_cache_dsv41_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}
