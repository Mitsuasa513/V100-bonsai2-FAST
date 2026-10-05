#!/usr/bin/env python
"""All-token KLD(oracle || ours) for a llama-dsv41-dump made with DSV41_DUMP_ALL_LOGITS=1 (single ubatch,
result_output = [n_vocab, n_tokens]) against the f32 CPU-oracle npz (logits [1, L, n_vocab]).

Per token (BOS = position 0 excluded from the summaries): KLD with float64 log-softmax, top-1 agreement,
the oracle's p(top-1), and our p of the oracle's top-1. Saves the per-token arrays as an npz and prints
markdown summary tables. Refuses to run on a crossed oracle/dump pair (last-token argmax must agree).
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--oracle", required=True)
ap.add_argument("--out", required=True, help="npz with the per-token arrays")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--tokenizer", default=None, help="dir with tokenizer.json, to print token pieces (optional)")
ap.add_argument("--worst", type=int, default=10)
a = ap.parse_args()

z = np.load(a.oracle)
ids = z["input_ids"].astype(np.int64); L = len(ids)
man = [json.loads(l) for l in open(os.path.join(a.dump_dir, "manifest.jsonl"))]
ro = [e for e in man if e["name"] == "result_output"]
if len(ro) != 1 or ro[0]["ne"][1] != L:
    sys.exit(f"need exactly one result_output entry with ne[1]=={L} (all-token logits, single ubatch); got {ro}")
V = ro[0]["ne"][0]
ours = np.memmap(os.path.join(a.dump_dir, ro[0]["file"]), dtype=np.float32, mode="r", shape=(L, V))
toks = [int(x) for x in open(os.path.join(a.dump_dir, "tokens.txt")).read().split()]
if toks != ids.tolist():
    sys.exit("token id mismatch between dump and oracle")
orc = z["logits"][0]
assert orc.shape == (L, V), orc.shape

def logsoftmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

# pairing check first (the two prompts share 1200 positions; only the last logit tells them apart)
lo = int(np.argmax(orc[-1])); lu = int(np.argmax(ours[-1]))
if lo != lu:
    sys.exit(f"PAIRING MISMATCH: last-token argmax oracle={lo} ours={lu} -- wrong oracle/dump pair")

kld = np.zeros(L); am_o = np.zeros(L, np.int64); am_u = np.zeros(L, np.int64)
p1_o = np.zeros(L); p_u_of_o1 = np.zeros(L); p_o_of_u1 = np.zeros(L); ent_o = np.zeros(L)
for c in range(0, L, a.chunk):
    x = np.asarray(ours[c:c+a.chunk], dtype=np.float64); y = orc[c:c+a.chunk].astype(np.float64)
    lx = logsoftmax(x); ly = logsoftmax(y); py = np.exp(ly)
    n = x.shape[0]; r = np.arange(n)
    kld[c:c+n] = (py * (ly - lx)).sum(axis=1)
    ent_o[c:c+n] = -(py * ly).sum(axis=1)
    am_o[c:c+n] = ly.argmax(axis=1); am_u[c:c+n] = lx.argmax(axis=1)
    p1_o[c:c+n] = py[r, am_o[c:c+n]]
    p_u_of_o1[c:c+n] = np.exp(lx[r, am_o[c:c+n]])
    p_o_of_u1[c:c+n] = py[r, am_u[c:c+n]]

pos = np.arange(L)
np.savez(a.out, pos=pos, token_id=ids, kld=kld, argmax_oracle=am_o, argmax_ours=am_u,
         p_top1_oracle=p1_o, p_ours_of_oracle_top1=p_u_of_o1, p_oracle_of_ours_top1=p_o_of_u1,
         entropy_oracle=ent_o, agree=(am_o == am_u))
print(f"saved {a.out}  (L={L}, V={V}; last-token argmax {lo} both sides -> pairing OK)")

dec = None
if a.tokenizer:
    try:
        from transformers import AutoTokenizer
        _t = AutoTokenizer.from_pretrained(a.tokenizer)
        dec = lambda i: repr(_t.decode([int(i)]))
    except Exception as e:  # noqa
        print("tokenizer unavailable:", e)

def summary(mask, label):
    k = kld[mask]; ag = (am_o == am_u)[mask]
    return (f"| {label} | {mask.sum()} | {k.mean():.5f} | {np.median(k):.5f} | {np.percentile(k, 90):.5f} | "
            f"{np.percentile(k, 99):.5f} | {k.max():.5f} | {100*ag.mean():.2f}% |")
m_all = pos > 0
m_conf = m_all & (p1_o > 0.5)
print("\n| subset | n | mean KLD | median | p90 | p99 | max | top-1 agree |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
print(summary(m_all, "all tokens, BOS excluded"))
print(summary(m_conf, "oracle p_top1 > 0.5"))
print(f"\nmean oracle p_top1 = {p1_o[m_all].mean():.4f}; tokens with p_top1>0.5: {m_conf.sum()}; "
      f"disagreements: {int((~(am_o == am_u))[m_all].sum())}")

worst = np.argsort(-np.where(m_all, kld, -1))[:a.worst]
print(f"\n| rank | pos | input token id | KLD | oracle top-1 (p) | ours top-1 (our p of oracle top-1) | agree |")
print("|---:|---:|---:|---:|---|---|---|")
for r_, p in enumerate(worst, 1):
    s_o = f"{am_o[p]}" + (f" {dec(am_o[p])}" if dec else "") + f" ({p1_o[p]:.3f})"
    s_u = f"{am_u[p]}" + (f" {dec(am_u[p])}" if dec else "") + f" ({p_u_of_o1[p]:.3f})"
    s_in = f"{ids[p]}" + (f" {dec(ids[p])}" if dec else "")
    print(f"| {r_} | {p} | {s_in} | {kld[p]:.4f} | {s_o} | {s_u} | {'yes' if am_o[p]==am_u[p] else 'NO'} |")
