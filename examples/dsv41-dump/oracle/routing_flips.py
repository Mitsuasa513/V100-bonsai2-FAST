#!/usr/bin/env python
"""Routing diagnostic: compare llama.cpp's per-layer MoE top-k (ffn_moe_topk-<il>, [6, T] i32 dump) with the
reference oracle's (route<il>_topk from run_oracle_routing.py), per layer:
  - flip rate: fraction of tokens whose 6-expert SET differs (BOS excluded)
  - set-difference size distribution (1..6 experts swapped)
  - the oracle's selection-score gap (6th selected - 7th candidate) at flipped tokens vs all tokens
  - the swap margin at flips: oracle sel score of the lowest oracle-only expert minus the highest ours-only expert
    (how near a tie the swapped pair was, in the oracle's own scores)
  - if ffn_moe_probs_biased-<il> is dumped: numeric error of our selection scores vs the oracle's
"""
<<<<<<< HEAD
import argparse, json, os, sys
=======
import argparse, json, os, re, sys
>>>>>>> v41-dump-logits
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--oracle", required=True, help="npz from run_oracle_routing.py")
ap.add_argument("--out", default=None, help="npz with per-layer per-token flip arrays")
a = ap.parse_args()

z = np.load(a.oracle)
ids = z["input_ids"]; L = len(ids)
man = {}
for line in open(os.path.join(a.dump_dir, "manifest.jsonl")):
    e = json.loads(line)
    if e.get("n_ub", 1) != 1:
        sys.exit("multi-ubatch dumps not supported here")
    man[e["name"]] = e
toks = [int(x) for x in open(os.path.join(a.dump_dir, "tokens.txt")).read().split()]
assert toks == ids.tolist(), "token mismatch"

def load(name, dtype):
    e = man[name]
    arr = np.fromfile(os.path.join(a.dump_dir, e["file"]), dtype=dtype)
    return arr.reshape([n for n in reversed(e["ne"]) if n != 1] or [1])

<<<<<<< HEAD
layers = sorted(int(k[len("route"):-len("_topk")]) for k in z.files if k.startswith("route") and k.endswith("_topk"))
=======
layers = sorted(int(mo.group(1)) for mo in (re.fullmatch(r"route(\d+)_topk", k) for k in z.files) if mo)
>>>>>>> v41-dump-logits
q = lambda v, p: np.percentile(v, p) if len(v) else float("nan")
rows = []; save = {}
all_gap_flip = []; all_gap = []; all_margin = []
print(f"L={L} (BOS excluded below); layers with routing data: {len(layers)}")
print("\n| layer | flip rate | n flipped | swap 1 | swap 2 | swap 3+ | gap med (all) | gap med (flips) | gap p90 (flips) | swap margin med | margin max | sel-score max abs err | sel err med |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for il in layers:
    ours = load(f"ffn_moe_topk-{il}", np.int32)        # [T, 6]
    ref = z[f"route{il}_topk"]                          # [T, 6]
    assert ours.shape == ref.shape == (L, 6), (ours.shape, ref.shape)
    sel = z[f"route{il}_sel"]                           # [T, 384] oracle selection scores
    gap = z[f"route{il}_gap"]                           # [T]
    ndiff = np.zeros(L, np.int32); margin = np.full(L, np.nan, np.float32)
    for t in range(L):
        so, sr = set(ours[t].tolist()), set(ref[t].tolist())
        d_ref_only = sr - so; d_ours_only = so - sr
        ndiff[t] = len(d_ref_only)
        if d_ref_only:
            # nearest-tie measure: lowest oracle score among the experts we dropped minus highest among the ones we added
            margin[t] = sel[t, list(d_ref_only)].min() - sel[t, list(d_ours_only)].max()
    m = np.arange(L) > 0
    flipped = m & (ndiff > 0)
    nf = int(flipped.sum())
    err_max = err_med = float("nan")
    if f"ffn_moe_probs_biased-{il}" in man:
        ob = load(f"ffn_moe_probs_biased-{il}", np.float32)   # [T, 384]
        err = np.abs(ob[m].astype(np.float64) - sel[m].astype(np.float64))
        err_max = err.max(); err_med = np.median(err)
    rows.append((il, 100 * nf / m.sum(), nf, int((ndiff[m] == 1).sum()), int((ndiff[m] == 2).sum()), int((ndiff[m] >= 3).sum()),
                 np.median(gap[m]), q(gap[flipped], 50), q(gap[flipped], 90), q(margin[flipped], 50),
                 (np.nanmax(margin[flipped]) if nf else float("nan")), err_max, err_med))
    print("| %d | %.2f%% | %d | %d | %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.2e | %.2e |" % rows[-1])
    all_gap_flip.append(gap[flipped]); all_gap.append(gap[m]); all_margin.append(margin[flipped])
    save[f"ndiff{il}"] = ndiff; save[f"margin{il}"] = margin; save[f"gap{il}"] = gap

gf = np.concatenate(all_gap_flip); ga = np.concatenate(all_gap); mg = np.concatenate(all_margin)
print(f"\nall layers: tokens*layers={len(ga)}, flipped={len(gf)} ({100*len(gf)/len(ga):.2f}%)")
print("\n| gap (oracle 6th-7th selection score) | p10 | p50 | p90 | p99 | max |")
print("|---|---:|---:|---:|---:|---:|")
print("| all tokens | %.4f | %.4f | %.4f | %.4f | %.4f |" % tuple(q(ga, p) for p in (10, 50, 90, 99, 100)))
print("| flipped tokens | %.4f | %.4f | %.4f | %.4f | %.4f |" % tuple(q(gf, p) for p in (10, 50, 90, 99, 100)))
print("| swap margin at flips | %.4f | %.4f | %.4f | %.4f | %.4f |" % tuple(q(mg, p) for p in (10, 50, 90, 99, 100)))
for thr in (0.001, 0.005, 0.01, 0.02, 0.05):
    print(f"gap <= {thr}: all {100*np.mean(ga <= thr):.2f}%  flipped {100*np.mean(gf <= thr):.2f}%   | swap margin <= {thr}: {100*np.mean(mg <= thr):.2f}%")
per_tok = np.stack([save[f"ndiff{il}"] > 0 for il in layers], 1)[1:]   # [L-1, n_layers]
print(f"\ntokens (BOS excl.) flipped in >=1 layer: {100*per_tok.any(1).mean():.1f}%; mean layers flipped per token: {per_tok.sum(1).mean():.2f}; max: {per_tok.sum(1).max()}")
if a.out:
    np.savez(a.out, layers=np.array(layers), **save)
    print("saved", a.out)
