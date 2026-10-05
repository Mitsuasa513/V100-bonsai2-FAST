# Patches

There are two, and they have **different bases**. Use the right one for what you are doing.

| file | size | base | use it when |
|---|---|---|---|
| `vs-prism-fork-cuda.patch` | 41 files, ~540 KB | the **raw PrismML fork** `ggml/src/ggml-cuda/` | you are building a **fork branch** (fork → checkout → apply → commit) |
| `state-path-fusions.patch` | 17 files, ~190 KB | the **merged working tree** this project was developed on (fork + an earlier round of V100 tuning) | you already have that tree and want only this project's delta |

The earlier round of V100 tuning (Volta MMQ config tables, flash-attention tweaks, norm kernels, …)
is *not* part of `state-path-fusions.patch` — it is already in the base tree. If you start from the
raw fork you need `vs-prism-fork-cuda.patch`, which contains both rounds:

```
ggml/src/ggml-cuda/   (36 modified + 6 added files)
  added: mmq-config-volta.cuh, mmq-config-pascal-dp4a.cuh, mmq-config-pascal-older.cuh,
         fattn-swizzle.cuh, moe-weighted-reduction.cu/.cuh
```

(A stray editor backup, `fattn-tile.cuh.bak2`, exists in the working tree and is deliberately
**not** included.)

## Verified bases

`git apply --check -p1` of `vs-prism-fork-cuda.patch`, with the base files fetched straight from
GitHub at each revision:

| fork revision | date | result |
|---|---|---|
| `prism-b10743-adfffbe` (= `adfffbe41b`) | 2026-09-25, the tip when this snapshot was taken | **clean** |
| `prism-b10709-9a9394a`, `prism-b10687-5d80cff` | 2026-09-21 | clean |
| `prism-b10754-2459f68` | 2026-10-02 (newest at the time of writing) | **9 files conflict** |

The conflicts on the newest revision are exactly the files upstream also changed in the meantime
(`ggml-cuda.cu`, `fwht.cu/.cuh`, `mmvq.cu/.cuh`, `vecdotq.cuh`, …): their `2459f68b5` "cuda: fused
FWHT quantizer for 64-wide warps" overlaps with §24 of the report, so drop that part of the patch
when rebasing onto it.

## Building a fork branch (recommended)

```bash
# 1. Fork PrismML-Eng/llama.cpp on GitHub, then:
git clone https://github.com/<you>/llama.cpp && cd llama.cpp
git checkout -b v100-cuda prism-b10743-adfffbe     # the verified clean base
git -c core.autocrlf=false apply -p1 /path/to/vs-prism-fork-cuda.patch
git add -A
git commit -m "V100: state-path fusions + Volta MMQ/FA tuning for ternary Bonsai 2"
git push -u origin v100-cuda
```

`core.autocrlf=false` matters on Windows: with line-ending conversion the patch hashes change and
you silently build something else.

## This project's delta (`state-path-fusions.patch`)

Applied to the merged tree it was generated from, it was verified end-to-end: `git apply --check -p1`
clean, then `git apply -p1` and SHA256 of all 17 files identical to that tree (17/17) — a fresh
`git clone` of this repository reproduces the result.

All fusions are on by default and every one has a runtime switch, so the safest way to adopt the
patch is to benchmark it against its own reference arm (`../tools/run_acceptance.ps1`):

```
GGML_CUDA_GDN_ROWS_READ=0     # reverts the whole state path (A/B reference)
GGML_CUDA_CONV_STATE_FUSION=0 # only the conv gather/write-back part
GGML_CUDA_MMVF_PAIR=0         # only the ssm_alpha/ssm_beta pair launch
GGML_CUDA_GLU_PERMUTE=0       # only the de-interleaving GLU store
GGML_CUDA_CONV_L2=0           # only the L2-norm fold
GGML_CUDA_FWHT_Q8_1=0         # only the FWHT -> q8_1 pre-quantisation
GGML_V100_*                   # instruments (launch count, op hist/time, logits dump, node dump)
GGML_CUDA_PQ2_0_REPACK=1      # the measured-negative aligned repack experiment
```
