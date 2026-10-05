#pragma once

#include "llama-kv-cache.h"

#include <vector>

// DeepSeek-V4.1 memory: a sliding-window K cache over EVERY layer (window attention) plus a full,
// token-indexed K/V cache over the kv_source layers holding the CED compressed K.
//   ratio-1 source layers store one compressed K per token (a plain causal cache);
//   ratio-2 source layers store each 2-token group's K at the cell of the group's LAST token, so the
//   plain causal mask plus an odd-position filter yields the reference visibility rule; an unpaired
//   even token parks its raw wkv / wgate projections in its own cell's K / V until its partner
//   arrives -- the pending half-group survives across ubatches with no extra state.
// Plus an indexer-key cache over the CSA2-Full layers (index_source AND kv_source): each publishes one
// indexer key (indexer_head_size wide, K only) per compressed cell, so the CSA2 indexer can score
// this layer's queries against them and keep only indexer_top_k compressed positions. It shares the
// compressed cache's cell layout exactly -- same kv_size, same k_idxs -- so cell i means the same
// compressed position in both, for either ratio.
// The layer sets overlap, which llama_kv_cache_iswa cannot express, hence this composition.
class llama_kv_cache_dsv41 : public llama_memory_i {
public:
    llama_kv_cache_dsv41(
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
                     uint32_t   n_pad);

    ~llama_kv_cache_dsv41() = default;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_win () const;
    llama_kv_cache * get_comp() const;
    llama_kv_cache * get_idx () const;

private:
    const bool unified;

    // the indexer cache is narrower than the model's attention (indexer_head_size, one head, K only),
    // so it needs its own hparams; llama_kv_cache keeps a reference to what it is handed.
    llama_hparams hparams_idx;

    std::unique_ptr<llama_kv_cache> kv_win;
    std::unique_ptr<llama_kv_cache> kv_comp;
    std::unique_ptr<llama_kv_cache> kv_idx;
};

class llama_kv_cache_dsv41_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv41_context(llama_memory_status status);
    llama_kv_cache_dsv41_context(llama_kv_cache_dsv41 * kv);
    llama_kv_cache_dsv41_context(llama_kv_cache_dsv41 * kv, llama_context * lctx, bool optimize);
    llama_kv_cache_dsv41_context(
            llama_kv_cache_dsv41 * kv,
            slot_info_vec_t sinfos_win,
            slot_info_vec_t sinfos_comp,
            slot_info_vec_t sinfos_idx,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_dsv41_context();

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_context * get_win () const;
    const llama_kv_cache_context * get_comp() const;
    const llama_kv_cache_context * get_idx () const;

    // the compressed cache itself, for cell -> position lookups when building the group masks
    const llama_kv_cache * get_comp_cache() const { return kv_comp; }

private:
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_win;
    const llama_memory_context_ptr ctx_comp;
    const llama_memory_context_ptr ctx_idx;

    const llama_memory_status status;

    const llama_kv_cache * kv_comp = nullptr;
};
