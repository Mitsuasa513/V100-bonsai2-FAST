#!/usr/bin/env python
"""Per-layer median per-token relative L2 of a llama-dsv41-dump vs the CPU oracle npz.
Verifies pairing first (last-token argmax of result_output vs oracle logits[-1])."""
import argparse, json, os, re, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--dump-dir", required=True); ap.add_argument("--oracle", required=True)
ap.add_argument("--layers", default="0,1,2,14,20,39")
ap.add_argument("--skip-bos", type=int, default=1)
a = ap.parse_args()
z = np.load(a.oracle)
ids = z["input_ids"]; L = len(ids)
man = {}
for line in open(os.path.join(a.dump_dir, "manifest.jsonl")):
    e = json.loads(line); man[e["name"]] = e
def load(e, dtype=np.float32):
    arr = np.fromfile(os.path.join(a.dump_dir, e["file"]), dtype=dtype)
    return arr.reshape([n for n in reversed(e["ne"])])
toks = [int(x) for x in open(os.path.join(a.dump_dir, "tokens.txt")).read().split()]
print(f"oracle {os.path.basename(a.oracle)}: L={L}; dump {a.dump_dir}: {len(toks)} tokens; ids match={toks == ids.tolist()}")
# pairing check on the last logit
lg_or = z["logits"][0]                      # [L, V]
ro = load(man["result_output"]).reshape(-1, lg_or.shape[-1])
x = ro[-1].astype(np.float64); y = lg_or[-1].astype(np.float64)
def logsm(v): v = v - v.max(); return v - np.log(np.exp(v).sum())
lx, ly = logsm(x), logsm(y); kld = float(np.sum(np.exp(ly) * (ly - lx)))
print(f"last-token argmax: dump={int(x.argmax())} oracle={int(y.argmax())} -> {'ALIGNED' if x.argmax()==y.argmax() else 'MISMATCH'}; KLD(oracle||ours)={kld:.4f}")
del lg_or
for il in [int(s) for s in a.layers.split(",")]:
    e = man[f"l_last-{il}"]
    xx = load(e).reshape(-1, 4*5120).astype(np.float64)
    yy = z[f"layer{il}"].reshape(-1, 4*5120).astype(np.float64)
    assert xx.shape == yy.shape, (xx.shape, yy.shape)
    s = a.skip_bos
    rel = np.linalg.norm(xx[s:] - yy[s:], axis=1) / np.linalg.norm(yy[s:], axis=1)
    print(f"layer {il:2d}: median rel L2 = {100*np.median(rel):.3f}%  mean={100*rel.mean():.3f}%  max={100*rel.max():.3f}%  (n={len(rel)})", flush=True)
