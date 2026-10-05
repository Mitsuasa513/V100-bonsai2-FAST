// llama-dsv41-dump: run ONE forward on a prompt and write whole tensors whose names match a
// regex to a directory as raw f32/i32 (+ a JSON-lines manifest with the ggml ne). The M4 tool for
// diffing the llama.cpp DeepSeek-V4.1 graph against the CPU reference oracle layer by layer.
//
//   DSV41_DUMP_DIR=/path/out DSV41_DUMP_FILTER='^(hc_init|l_last-[0-9]+|hc_out|result_norm|result_output)$' [DSV41_DUMP_ALL_LOGITS=1]
//   llama-dsv41-dump -m model.gguf -p "prompt" [usual llama args]
//
// Env-configured so the common arg parser is untouched.
//   DSV41_DUMP_ALL_LOGITS=1  request logits for every token: result_output becomes [n_vocab, n_tokens_ub]
//                            instead of the last token only.
//   DSV41_FORCE_BOS=1        prepend BOS even if the GGUF tokenizer metadata says not to.
//   DSV41_DUMP_UB_MARKER     regex of a once-per-graph node NAME marking a ubatch start (default: structural
//                            match of the token-embedding GET_ROWS node, see below).
//
// Files: one per (tensor, ubatch): <name>-ub<k>.f32 (f32/f16/bf16 sources, converted to f32) or
// <name>-ub<k>.i32 (i32 sources, e.g. ffn_moe_topk-<il>). When the prompt fits in a single ubatch,
// <name>.f32/.i32 is hard-linked to the -ub0 file so the older single-file scripts keep working.
// manifest.jsonl records per entry: name, file, type (f32|i32 on disk), src_type, op, ne, ub, n_ub,
// tok0, tok1 (token range of the ubatch, [tok0, tok1)), and alias when a hard link was made.
// Non-contiguous tensors (the ffn_moe_topk view of the argsort) are gathered element-wise by stride.
// Ubatch boundaries are detected on the per-graph token-embedding node (GET_ROWS of token_embd.weight by the
// inp_tokens leaf; it is unnamed in the graph, so it is matched structurally; ne[1] = the ubatch's token
// count). DSV41_DUMP_UB_MARKER=<regex> replaces that with a name match. If no marker ever appears, a repeated
// matched name is used as a fallback (cb names such as "norm-0" can recur inside one graph, so only that).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

struct dump_state {
    std::regex    filter;
    std::regex    marker;          // per-graph node that starts a ubatch (name regex; empty = structural)
    bool          marker_by_name = false;
    bool          marker_seen = false;
    bool          fallback_warned = false;
    int           ub_ntok   = 0;   // tokens in the current ubatch (from the marker's ne[1])
    int           tok0      = 0;   // first token of the current ubatch
    int           next_tok0 = 0;
    std::string   outdir;
    std::ofstream manifest;
    int           n_written = 0;
    int           n_skipped = 0;
    // ubatch bookkeeping: every cb name occurs once per graph, so a repeated name = next ubatch
    int           ub        = 0;   // current ubatch index
    int           n_ub      = 1;   // expected number of ubatches (single sequence, simple split)
    int           n_ubatch  = 0;
    int           n_tokens  = 0;
    std::set<std::string>    seen;     // names already written in the current ubatch
    std::vector<std::string> aliases;  // <name>.<ext> hard links made for the single-ubatch case
};

static std::string sanitize(std::string s) {
    for (auto & c : s) { if (c == '/' || c == ' ') c = '_'; }
    return s;
}

// gather the elements of a (possibly non-contiguous) tensor into a contiguous ne0-fastest byte buffer
static std::vector<uint8_t> gather_elements(const ggml_tensor * t) {
    const size_t ts = ggml_type_size(t->type);
    GGML_ASSERT(ggml_blck_size(t->type) == 1);
    std::vector<uint8_t> raw(ggml_nbytes(t));           // for a view: the byte span up to the last element
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    if (ggml_is_contiguous(t)) {
        return raw;
    }
    std::vector<uint8_t> out((size_t) ggml_nelements(t) * ts);
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
                    std::memcpy(out.data() + o, raw.data() + off, ts);
                    o += ts;
                }
            }
        }
    }
    return out;
}

static void dsv41_next_ubatch(dump_state * st) {
    st->ub++;
    st->seen.clear();
    if (st->ub >= st->n_ub) {
        LOG_WRN("dsv41-dump: ubatch %d seen but only %d expected -- removing single-ubatch aliases\n", st->ub, st->n_ub);
        for (const auto & p : st->aliases) { ::unlink(p.c_str()); }
        st->aliases.clear();
        st->n_ub = st->ub + 1;
    }
}

static bool dsv41_is_marker(const dump_state * st, const ggml_tensor * t) {
    if (st->marker_by_name) {
        return std::regex_match(t->name, st->marker);
    }
    // the token-embedding lookup: once per graph, ne[1] = ubatch tokens (text path of build_inp_embd)
    return t->op == GGML_OP_GET_ROWS && t->src[0] && t->src[1] &&
           std::strcmp(t->src[0]->name, "token_embd.weight") == 0 && std::strcmp(t->src[1]->name, "inp_tokens") == 0;
}

static bool dsv41_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool match = std::regex_match(t->name, st->filter);
    if (ask) {
        if (dsv41_is_marker(st, t)) {                    // one per graph = one per ubatch
            if (st->marker_seen) {
                dsv41_next_ubatch(st);
            }
            st->marker_seen = true;
            st->ub_ntok = (int) t->ne[1];
            st->tok0 = st->next_tok0;
            st->next_tok0 += st->ub_ntok;
            LOG_INF("dsv41-dump: ubatch %d starts at token %d (%d tokens; marker %s ne={%lld,%lld})\n",
                    st->ub, st->tok0, st->ub_ntok, t->name, (long long) t->ne[0], (long long) t->ne[1]);
        }
        return match;              // only ask the scheduler to materialise what we will write
    }
    if (!match) {
        return true;
    }

    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> outbuf;
    const char * ext = nullptr;
    switch (t->type) {
        case GGML_TYPE_F32:  ext = "f32"; outbuf = gather_elements(t); break;
        case GGML_TYPE_I32:  ext = "i32"; outbuf = gather_elements(t); break;
        case GGML_TYPE_F16: {
            ext = "f32";
            const std::vector<uint8_t> el = gather_elements(t);
            outbuf.resize((size_t) n * sizeof(float));
            float * d = (float *) outbuf.data();
            const ggml_fp16_t * s = (const ggml_fp16_t *) el.data();
            for (int64_t i = 0; i < n; ++i) { d[i] = ggml_fp16_to_fp32(s[i]); }
        } break;
        case GGML_TYPE_BF16: {
            ext = "f32";
            const std::vector<uint8_t> el = gather_elements(t);
            outbuf.resize((size_t) n * sizeof(float));
            float * d = (float *) outbuf.data();
            const ggml_bf16_t * s = (const ggml_bf16_t *) el.data();
            for (int64_t i = 0; i < n; ++i) { d[i] = ggml_bf16_to_fp32(s[i]); }
        } break;
        default:
            LOG_WRN("dsv41-dump: %s has type %s -- not dumped\n", t->name, ggml_type_name(t->type));
            st->n_skipped++;
            return true;
    }

    const std::string name(t->name);
    if (!st->marker_seen && st->seen.count(name)) {   // fallback: same matched name again = a new ubatch
        if (!st->fallback_warned) {
            LOG_WRN("dsv41-dump: ubatch marker never seen; falling back to repeated-name detection (unreliable if a cb name recurs within one graph)\n");
            st->fallback_warned = true;
        }
        dsv41_next_ubatch(st);
    }
    st->seen.insert(name);

    const std::string fname = sanitize(name) + "-ub" + std::to_string(st->ub) + "." + ext;
    const std::string path  = st->outdir + "/" + fname;
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("dsv41-dump: cannot open %s\n", path.c_str());
        return true;
    }
    std::fwrite(outbuf.data(), 1, outbuf.size(), f);
    std::fclose(f);

    std::string alias;
    if (st->n_ub == 1 && st->ub == 0) {
        alias = sanitize(name) + "." + ext;
        const std::string apath = st->outdir + "/" + alias;
        ::unlink(apath.c_str());
        if (::link(path.c_str(), apath.c_str()) != 0) {
            LOG_WRN("dsv41-dump: hard link %s -> %s failed; copying\n", alias.c_str(), fname.c_str());
            FILE * g = std::fopen(apath.c_str(), "wb");
            if (g) { std::fwrite(outbuf.data(), 1, outbuf.size(), g); std::fclose(g); }
        }
        st->aliases.push_back(apath);
    }

    const int tok0 = st->marker_seen ? st->tok0 : st->ub * st->n_ubatch;
    const int tok1 = st->marker_seen ? st->tok0 + st->ub_ntok : std::min(tok0 + st->n_ubatch, st->n_tokens);
    st->manifest << "{\"name\":\"" << name << "\",\"file\":\"" << fname << "\",\"type\":\"" << ext
                 << "\",\"src_type\":\"" << ggml_type_name(t->type) << "\",\"op\":\"" << ggml_op_desc(t)
                 << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]"
                 << ",\"contiguous\":" << (ggml_is_contiguous(t) ? "true" : "false")
                 << ",\"ub\":" << st->ub << ",\"n_ub\":" << st->n_ub << ",\"tok0\":" << tok0 << ",\"tok1\":" << tok1;
    if (!alias.empty()) { st->manifest << ",\"alias\":\"" << alias << "\""; }
    st->manifest << "}\n";
    st->manifest.flush();
    st->n_written++;
    LOG_INF("dsv41-dump: wrote %-28s %-4s ne={%lld,%lld,%lld,%lld} ub=%d%s\n", fname.c_str(), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], st->ub,
            ggml_is_contiguous(t) ? "" : " (strided gather)");
    return true;
}

static bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && v[0] && !(v[0] == '0' && v[1] == 0);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    dump_state st;
    const char * dir = std::getenv("DSV41_DUMP_DIR");
    const char * flt = std::getenv("DSV41_DUMP_FILTER");
    const char * mrk = std::getenv("DSV41_DUMP_UB_MARKER");
    st.outdir = dir ? dir : "dsv41-dump-out";
    st.filter = std::regex(flt ? flt : "^(hc_init|hc_pre_mix_init|l_last-[0-9]+|hc_out|result_norm|result_output)$");
    if (mrk) { st.marker = std::regex(mrk); st.marker_by_name = true; }
    const bool all_logits = env_flag("DSV41_DUMP_ALL_LOGITS");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = dsv41_dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }
    // the reference encoder always prepends BOS; the GGUF tokenizer metadata may say otherwise
    if (env_flag("DSV41_FORCE_BOS") && tokens.front() != llama_vocab_bos(vocab)) {
        tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
    }
    if (tokens.size() > llama_n_batch(ctx)) {
        LOG_ERR("dsv41-dump: %zu tokens exceed n_batch=%u (the tool issues ONE llama_decode); raise -b\n", tokens.size(), llama_n_batch(ctx));
        return 1;
    }

    st.n_tokens = (int) tokens.size();
    st.n_ubatch = (int) llama_n_ubatch(ctx);
    st.n_ub     = (st.n_tokens + st.n_ubatch - 1) / st.n_ubatch;

    std::string mkdir_cmd = "mkdir -p '" + st.outdir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) { LOG_WRN("mkdir failed for %s\n", st.outdir.c_str()); }
    st.manifest.open(st.outdir + "/manifest.jsonl");
    {
        std::ofstream tf(st.outdir + "/tokens.txt");
        for (auto tok : tokens) { tf << tok << "\n"; }
    }
    LOG_INF("dsv41-dump: %zu tokens (add_bos=%d), n_ubatch=%d -> %d ubatch(es), all_logits=%d, dumping to %s with filter %s\n",
            tokens.size(), (int) add_bos, st.n_ubatch, st.n_ub, (int) all_logits, st.outdir.c_str(), flt ? flt : "(default)");

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, all_logits || i + 1 == tokens.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc) {
        LOG_ERR("decode failed (%d)\n", rc);
        return 1;
    }
    if (st.ub + 1 != st.n_ub) {
        LOG_WRN("dsv41-dump: expected %d ubatches, saw %d\n", st.n_ub, st.ub + 1);
    }
    if (st.marker_seen && st.next_tok0 != st.n_tokens) {
        LOG_WRN("dsv41-dump: ubatch marker token count %d != %d prompt tokens\n", st.next_tok0, st.n_tokens);
    }

    // logits of the last token (sanity: the greedy next token)
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) { if (logits[i] > logits[best]) best = i; }
    LOG_INF("dsv41-dump: wrote %d tensors, skipped %d; greedy next token id = %d (%s)\n",
            st.n_written, st.n_skipped, best, common_token_to_piece(ctx, best).c_str());
    {
        std::ofstream lf(st.outdir + "/next_token.txt"); lf << best << "\n";
        std::ofstream jf(st.outdir + "/dump_info.json");
        jf << "{\"n_tokens\":" << st.n_tokens << ",\"n_ubatch\":" << st.n_ubatch << ",\"n_ub\":" << st.ub + 1
           << ",\"all_logits\":" << (all_logits ? "true" : "false") << ",\"n_written\":" << st.n_written
           << ",\"n_skipped\":" << st.n_skipped << ",\"next_token\":" << best << "}\n";
    }

    llama_backend_free();
    return 0;
}
