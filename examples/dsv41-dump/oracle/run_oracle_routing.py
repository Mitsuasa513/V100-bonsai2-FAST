#!/usr/bin/env python
"""Routing-diagnostic oracle: run_oracle.py (dgx 2026-09-11 version) + per-layer MoE router capture.

Same forward as run_oracle.py (reference model.py on CPU, weights streamed from OUR GGUF), but every
layer's Gate additionally records, per token:
  route{il}_topk        [T, 6]   i32  selected expert ids, in the reference's topk order (desc. selection score)
  route{il}_sel_topk    [T, 6]   f32  selection scores (score + bias) of the selected experts
  route{il}_score_topk  [T, 6]   f32  unbiased scores of the selected experts (what the weights come from)
  route{il}_weights     [T, 6]   f32  final routing weights (normalised, * route_scale)
  route{il}_7th         [T]      i32  the 7th candidate (highest selection score NOT selected)
  route{il}_gap         [T]      f32  selection score of the 6th selected minus the 7th candidate
  route{il}_sel         [T, 384] f32  full selection-score matrix (score + bias)
  route{il}_scores      [T, 384] f32  full unbiased score matrix (sqrt(softplus(logit / gate_temp)))
Hidden states are saved only for --save-layers (default 0,1,2,14,20,39) to keep the npz small; logits for
all positions are always saved so the run can be cross-checked against the full oracle npz.

Memory: experts are unloaded right after their forward (not just per layer), so the peak is the dense
params (25.6 GiB f32) + one expert + activations. Paths: reference code + tokenizer from
~/ml-local/models/DeepSeek-V4.1-Flash-ref (astra and dgx both), GGUF from ~/ml-local/models/DeepSeek-V4.1-Flash-GGUF.
"""
import argparse, dataclasses, json, os, re, sys, time, types
import numpy as np

HOME = os.path.expanduser("~")
REF = f"{HOME}/ml-local/models/DeepSeek-V4.1-Flash-ref"; INF = f"{REF}/inference"
HF = f"{HOME}/ml-local/models/DeepSeek-V4.1-Flash-HF"
if not os.path.isdir(HF):
    HF = REF   # astra: the tokenizer files live next to the reference code
GGUF = f"{HOME}/ml-local/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-Q8_0.gguf"
for p in (INF, os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)
_conv_gguf = f"{HOME}/ml-local/llama.cpp-v41-convert/gguf-py"
if os.path.isdir(_conv_gguf):
    sys.path.insert(0, _conv_gguf)     # else: the venv's installed gguf (same constants on astra)

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--out", default=f"{HOME}/ml-local/tools/v41-port/oracle/oracle_routing_out.npz")
ap.add_argument("--max-layers", type=int, default=None, help="stop after N layers (debug)")
ap.add_argument("--prompt-file", default=None, help="read the prompt from a file instead of --prompt")
ap.add_argument("--max-seq-len", type=int, default=512, help="reference max_seq_len; must exceed the prompt for the indexer top-k to engage")
ap.add_argument("--dtype", choices=["bf16","f32"], default="bf16", help="f32 = true-f32 reference (same dequantized weights as llama.cpp -> razor-sharp diff)")
ap.add_argument("--threads", type=int, default=64)
ap.add_argument("--save-layers", default="0,1,2,14,20,39", help="comma list of layers whose hidden state is saved ('all' = every layer)")
cli = ap.parse_args()

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def rss_gib():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except Exception:  # noqa
        pass
    return float("nan")

# transformers must import BEFORE the fake PIL (it tolerates a missing PIL, not a stub)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(HF)
import fake_pil  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import cpu_kernels  # noqa: E402
sys.modules["kernel"] = cpu_kernels
import model as m  # noqa: E402
import gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import MODEL_TENSOR as MT, TENSOR_NAMES  # noqa: E402

torch.set_num_threads(cli.threads)
DT = torch.float32 if cli.dtype == "f32" else torch.bfloat16
torch.set_default_dtype(DT)
log(f"torch {torch.__version__} threads={torch.get_num_threads()} dtype={cli.dtype} gguf={os.path.dirname(gguf.__file__)}")

# ---------------- model args (real config; text-only; no DSpark MTP) ----------------
cfg = json.load(open(f"{INF}/config.json"))
fields = {f.name for f in dataclasses.fields(m.ModelArgs)}
kw = {k: v for k, v in cfg.items() if k in fields}
kw.update(dtype="bf16", expert_dtype=None, max_batch_size=1, max_seq_len=cli.max_seq_len,
          dspark_block_size=0, vision_n_layers=0)
args = m.ModelArgs(**kw)
log(f"ModelArgs: layers={args.n_layers} dim={args.dim} experts={args.n_routed_experts}/{args.n_activated_experts} "
    f"hc={args.hc_mult} engram={args.engram_layer_ids} kv_src={args.kv_source_layers} idx_src={args.index_source_layers} "
    f"score_func={args.score_func} gate_temp={args.gate_temp} route_scale={args.route_scale} norm_topk={args.norm_topk_prob}")

# ---------------- meta-ify the memory hogs before construction ----------------
_orig_expert_init = m.Expert.__init__
def _meta_expert_init(self, *a, **k):
    with torch.device("meta"):
        _orig_expert_init(self, *a, **k)
m.Expert.__init__ = _meta_expert_init
_orig_eng_init = m.ParallelEngramEmbedding.__init__
def _meta_eng_init(self, *a, **k):
    with torch.device("meta"):
        _orig_eng_init(self, *a, **k)
m.ParallelEngramEmbedding.__init__ = _meta_eng_init

# ---------------- router capture: wrap Gate.forward ----------------
ROUTE = {}   # layer id -> dict of numpy arrays
_orig_gate_fwd = m.Gate.forward
def _gate_fwd_capture(self, x, image_mask=None):
    weights, indices = _orig_gate_fwd(self, x, image_mask)
    bid = getattr(self, "_bid", None)
    if bid is None:
        return weights, indices
    with torch.no_grad():
        # recompute the selection scores exactly as Gate.forward does (kept in sync with the reference)
        scores = m.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        sel = scores + bias
        k = self.topk
        top = sel.topk(k + 1, dim=-1)                       # values sorted descending
        if not torch.equal(top.indices[:, :k], indices):
            n_bad = int((top.indices[:, :k] != indices).any(dim=-1).sum())
            log(f"  WARNING layer {bid}: recomputed top-{k} differs from the reference's on {n_bad} tokens (ties?)")
        ROUTE[bid] = {
            "topk":       indices.detach().cpu().numpy().astype(np.int32),
            "sel_topk":   sel.gather(1, indices).cpu().numpy().astype(np.float32),
            "score_topk": scores.gather(1, indices).cpu().numpy().astype(np.float32),
            "weights":    weights.detach().float().cpu().numpy().astype(np.float32),
            "7th":        top.indices[:, k].cpu().numpy().astype(np.int32),
            "gap":        (top.values[:, k - 1] - top.values[:, k]).cpu().numpy().astype(np.float32),
            "sel":        sel.cpu().numpy().astype(np.float32),
            "scores":     scores.cpu().numpy().astype(np.float32),
        }
    return weights, indices
m.Gate.forward = _gate_fwd_capture

t0 = time.time()
net = m.Transformer(args, tok)
net.eval()
if DT == torch.float32:
    # the reference hardcodes bf16 Linear weights; lift every real bf16 param to f32 for a true-f32 run
    with torch.no_grad():
        for _p in net.parameters():
            if not _p.is_meta and _p.dtype == torch.bfloat16:
                _p.data = _p.data.float()
        for _n, _b in net.named_buffers():
            if not _b.is_meta and _b.dtype == torch.bfloat16:
                _b.data = _b.data.float()
for bid, layer in enumerate(net.layers):
    layer.ffn.gate._bid = bid
log(f"model built on CPU (experts+engram tables on meta) in {time.time()-t0:.1f}s  RSS={rss_gib():.1f} GiB")

# ---------------- GGUF ----------------
r = GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}
log(f"GGUF: {len(T)} tensors, arch={r.fields['general.architecture'].contents()}")

def deq(t, rows=None):
    """Dequantize a GGUF tensor (or a leading-dim row subset) to float32 in its logical numpy shape."""
    data = t.data if rows is None else t.data[rows]
    return gguf.dequantize(np.ascontiguousarray(data), t.tensor_type)

# reference param suffix -> (MODEL_TENSOR, gguf suffix): the converter's own table, inverted
SUF = {
    "embed.weight": (MT.TOKEN_EMBD, ".weight"), "norm.weight": (MT.OUTPUT_NORM, ".weight"), "head.weight": (MT.OUTPUT, ".weight"),
    "hc_attn_fn": (MT.HC_ATTN_FN, ".weight"), "hc_attn_base": (MT.HC_ATTN_BASE, ".weight"), "hc_attn_scale": (MT.HC_ATTN_SCALE, ".weight"),
    "hc_ffn_fn": (MT.HC_FFN_FN, ".weight"), "hc_ffn_base": (MT.HC_FFN_BASE, ".weight"), "hc_ffn_scale": (MT.HC_FFN_SCALE, ".weight"),
    "attn.attn_sink": (MT.ATTN_SINKS, ".weight"),
    "attn.wq_a.weight": (MT.ATTN_Q_A, ".weight"), "attn.wq_b.weight": (MT.ATTN_Q_B, ".weight"), "attn.q_norm.weight": (MT.ATTN_Q_A_NORM, ".weight"),
    "attn.wkv.weight": (MT.ATTN_KV, ".weight"), "attn.kv_norm.weight": (MT.ATTN_KV_NORM, ".weight"),
    "attn.wo_a.weight": (MT.ATTN_OUT_A, ".weight"), "attn.wo_b.weight": (MT.ATTN_OUT_B, ".weight"),
    "attn.compressor.wkv.weight": (MT.ATTN_COMPRESSOR_WKV, ".weight"), "attn.compressor.wgate.weight": (MT.ATTN_COMPRESSOR_WGATE, ".weight"),
    "attn.compressor.norm.weight": (MT.ATTN_COMPRESSOR_NORM, ".weight"),
    "attn.indexer.wq_b.weight": (MT.INDEXER_ATTN_Q_B, ".weight"), "attn.indexer.weights_proj.weight": (MT.INDEXER_PROJ, ".weight"),
    "attn.indexer.k_norm.weight": (MT.INDEXER_K_NORM, ".weight"), "attn.indexer.wk.weight": (MT.INDEXER_ATTN_K, ".weight"),
    "attn_norm.weight": (MT.ATTN_NORM, ".weight"), "ffn_norm.weight": (MT.FFN_NORM, ".weight"),
    "ffn.gate.weight": (MT.FFN_GATE_INP, ".weight"), "ffn.gate.bias": (MT.FFN_EXP_PROBS_B, ".bias"),
    "engram.k_weight": (MT.ENGRAM_K, ".weight"), "engram.q_weight": (MT.ENGRAM_Q, ".weight"), "engram.wkv.weight": (MT.ENGRAM_WKV, ".weight"),
}
def gguf_name(pname):
    mo = re.match(r"layers\.(\d+)\.(.+)$", pname)
    bid, suf = (int(mo.group(1)), mo.group(2)) if mo else (None, pname)
    mt, ext = SUF[suf]
    return TENSOR_NAMES[mt].format(bid=bid) + ext

# ---------------- load dense params (everything real) ----------------
t0 = time.time(); n_loaded = 0; n_bytes = 0
with torch.no_grad():
    for pname, p in net.named_parameters():
        if p.is_meta:
            continue  # experts + engram tables: lazy
        gn = gguf_name(pname)
        t = T[gn]
        arr = deq(t)
        assert tuple(arr.shape) == tuple(p.shape), f"shape mismatch {pname} <- {gn}: gguf {arr.shape} vs param {tuple(p.shape)}"
        p.copy_(torch.from_numpy(arr).to(p.dtype))
        n_loaded += 1; n_bytes += p.numel() * p.element_size()
        del arr
log(f"dense params loaded: {n_loaded} tensors, {n_bytes/2**30:.1f} GiB, in {time.time()-t0:.1f}s  RSS={rss_gib():.1f} GiB")

# ---------------- lazy experts (routed + shared), freed right after each forward ----------------
EXP = {"w1": "ffn_gate_exps", "w2": "ffn_down_exps", "w3": "ffn_up_exps"}
SHX = {"w1": "ffn_gate_shexp", "w2": "ffn_down_shexp", "w3": "ffn_up_shexp"}
stats = {"experts_loaded": 0}
def load_expert(exp, bid, eid):
    for w in ("w1", "w2", "w3"):
        lin = getattr(exp, w)
        if not lin.weight.is_meta:
            continue
        if eid is None:
            t = T[f"blk.{bid}.{SHX[w]}.weight"]; arr = deq(t)
        else:
            t = T[f"blk.{bid}.{EXP[w]}.weight"]; arr = deq(t, rows=eid)
        assert tuple(arr.shape) == tuple(lin.weight.shape), (bid, eid, w, arr.shape, tuple(lin.weight.shape))
        lin.weight = torch.nn.Parameter(torch.from_numpy(arr).to(DT), requires_grad=False)
    stats["experts_loaded"] += 1
def unload_expert(exp):
    for w in ("w1", "w2", "w3"):
        lin = getattr(exp, w)
        if not lin.weight.is_meta:
            lin.weight = torch.nn.Parameter(torch.empty(lin.weight.shape, device="meta", dtype=lin.weight.dtype), requires_grad=False)
for bid, layer in enumerate(net.layers):
    moe = layer.ffn
    for eid, exp in enumerate(moe.experts):
        exp.register_forward_pre_hook(lambda mod, inp, bid=bid, eid=eid: load_expert(mod, bid, eid))
        exp.register_forward_hook(lambda mod, inp, out: unload_expert(mod))
    moe.shared_experts.register_forward_pre_hook(lambda mod, inp, bid=bid: load_expert(mod, bid, None))
    moe.shared_experts.register_forward_hook(lambda mod, inp, out: unload_expert(mod))

# ---------------- engram tables: gather rows straight from the GGUF ----------------
def make_engram_fwd(t):
    def fwd(self, indices):
        idx = indices.reshape(-1).cpu().numpy().astype(np.int64)
        rows = deq(t, rows=idx)                                 # [n, 256] f32
        return torch.from_numpy(rows).to(DT).view(*indices.shape, rows.shape[-1])
    return fwd
for bid, layer in enumerate(net.layers):
    if layer.engram is not None:
        t = T[f"blk.{bid}.engram_embd.weight"]
        layer.engram.embed.forward = types.MethodType(make_engram_fwd(t), layer.engram.embed)
        log(f"engram layer {bid}: rows gathered from {t.name} ({t.tensor_type.name}, rows={t.shape[1]})")

# ---------------- prompt ----------------
_ptext = open(cli.prompt_file).read() if cli.prompt_file else cli.prompt
ids = [tok.bos_token_id] + tok(_ptext, add_special_tokens=False)["input_ids"]
input_ids = torch.tensor([ids], dtype=torch.long)
log(f"prompt {'file ' + cli.prompt_file if cli.prompt_file else cli.prompt!r} -> {len(ids)} tokens (BOS prepended)")

save_layers = None if cli.save_layers == "all" else {int(s) for s in cli.save_layers.split(",") if s}

# ---------------- replicate Transformer.forward with per-layer capture ----------------
out = {}
with torch.inference_mode():
    hashes = net.engram_hash(input_ids, 0, None)              # [1, L, n_engram_layers, n_hash_cols]
    out["engram_hashes"] = hashes.numpy()
    h = net.embed(input_ids)                                   # [1, L, dim]
    out["embed"] = h.float().numpy()
    h = h.unsqueeze(2).repeat(1, 1, net.hc_mult, 1)            # [1, L, hc, dim]
    pre_mix = m.make_identity_pre_mix(h, net.hc_mult)
    n_layers = len(net.layers) if cli.max_layers is None else min(cli.max_layers, len(net.layers))
    for i in range(n_layers):
        layer = net.layers[i]; t0 = time.time(); e0 = stats["experts_loaded"]
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
            if save_layers is None or i in save_layers:
                out[f"layer{i}_post_engram"] = h.float().numpy()
        h, pre_mix = layer(h, 0, pre_mix, None)
        if save_layers is None or i in save_layers:
            out[f"layer{i}"] = h.float().numpy(); out[f"layer{i}_pre_mix"] = pre_mix.float().numpy()
        for exp in layer.ffn.experts:
            unload_expert(exp)
        unload_expert(layer.ffn.shared_experts)
        rt = ROUTE.pop(i, None)
        if rt is None:
            log(f"  WARNING: no routing capture for layer {i}")
        else:
            for k_, v_ in rt.items():
                out[f"route{i}_{k_}"] = v_
        a = h.float().numpy()
        log(f"layer {i:2d} done {time.time()-t0:6.1f}s  experts={stats['experts_loaded']-e0:3d}  "
            f"|h| mean={np.abs(a).mean():.4f} max={np.abs(a).max():.3f} finite={np.isfinite(a).all()}  "
            f"gap med={np.median(rt['gap']) if rt else float('nan'):.4f}  RSS={rss_gib():.1f} GiB")
        del a
    h = net.layers[n_layers - 1].hc_pre(h, pre_mix)            # final collapse with the threaded pre-mix (no output_hc)
    out["final_collapsed"] = h.float().numpy()
    hn = net.norm(h); out["final_normed"] = hn.float().numpy()
    logits = net.head(hn, full_logits=True)                    # [1, L, vocab] f32
    out["logits"] = logits.float().numpy()

np.savez(cli.out, input_ids=np.array(ids), prompt=np.array(_ptext), **out)
log(f"saved {cli.out}  RSS peak see VmHWM below")
try:
    log([l.strip() for l in open("/proc/self/status") if l.startswith(("VmHWM", "VmRSS"))])
except Exception:  # noqa
    pass
lg = out["logits"][0]
for pos in (0, 1, lg.shape[0] - 1):
    top = np.argsort(-lg[pos])[:5]
    log(f"pos {pos} ({tok.decode([ids[pos]])!r}) -> top5: " + ", ".join(f"{tok.decode([int(t)])!r}:{lg[pos,t]:.2f}" for t in top))
log(f"NEXT TOKEN (greedy) after prompt: {tok.decode([int(np.argmax(lg[-1]))])!r} (id {int(np.argmax(lg[-1]))})")
log("ORACLE DONE")
