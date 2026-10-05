# Bonsai 2 27B (ternary) on a Tesla V100 — CUDA decode patch + MTP speculative decoding

Companion repository for squeezing more decode throughput out of
[PrismML's Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (ternary /
2-bit hybrid-attention LLM) running on a **Tesla V100-SXM2 32 GB (sm_70)** with the
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (fork snapshot 2026-09-26,
mainline build `b11004`).

Everything here was measured on that one machine; every optimization is gated by a **bit-exact
logits comparison**, not by "the text looks the same".

## Headline results

Decode (`-ngl 99 -fa 1`, greedy, single stream, context depth d):

| configuration | d=0 | 32K | 90K |
|---|---:|---:|---:|
| handover baseline (was already P1-tuned) | 58.89 | — | 38.31 |
| + this patch (state-path fusions, bit-exact) | 64.95 | ~66 | 41.21 |
| + MTP speculative decoding (grafted head, n-max 4 / 3) | **86.7** | **68.1** | 44.5 |

Relative to the *patched* build the MTP gain is **+41% at d=0**, **+33% at 32K**, **+18% at 90K**
(and **−10%** on a 100K-token chat/summarize workload, see the caveat below).

Prefill is unchanged by the patch (~818 t/s pp512 at d=0, ~505 t/s at 90K).

## What is in here

| path | content |
|---|---|
| `patch/vs-prism-fork-cuda.patch` | **use this one for a fork branch**: the whole CUDA work (Volta MMQ/FA tuning + this project, 41 files) relative to the raw PrismML fork; verified to apply cleanly on `prism-b10743-adfffbe` (2026-09-25) |
| `patch/state-path-fusions.patch` | only this project's delta (17 files, 190 KB), relative to the merged tree it was developed on |
| `docs/report.zh-CN.md` | full 2000-line engineering log (Chinese): every experiment, A/B, ncu profile, SASS dump, dead end |
| `docs/results.md` | the measured tables in English (fusions, launch counts, op-time breakdown, ncu, MTP sweeps) |
| `mtp/` | recipe + tool to graft an MTP head into the stock GGUF (no model weights included) |
| `tools/` | the acceptance suite, logits-dump A/B scripts, benchmarks, node-dump parser, ncu helper, build helpers |
| `tools/server/` | ready-to-run `llama-server` launchers (262144 context, q8_0 KV, LAN, MTP on/off) |

## The seven fusions in the patch

All of them are bit-identical to the unfused path (verified with per-token logits dumps) and all
are gated by an environment variable so they can be A/B-ed at runtime:

| # | what | switch | gain |
|---|---|---|---|
| 1 | GDN state read by row index (skip the per-layer state gather) | `GGML_CUDA_GDN_ROWS_READ` | ~+2.9% |
| 2 | conv-state gather + concat fused, write-back folded into the conv consumer kernel | `GGML_CUDA_CONV_STATE_FUSION` | ~+1.9% |
| 3 | `[ADD, RMS_NORM, MUL]` fused on Volta | `GGML_CUDA_ADD_RMS_NORM` | +0.4…0.8% |
| 4 | `ssm_alpha` / `ssm_beta` matvec pair in one launch | `GGML_CUDA_MMVF_PAIR` | +0.6…1.8% |
| 5 | de-interleaving GLU store (write moved to the CONT node + `add_alloc_dep`) | `GGML_CUDA_GLU_PERMUTE` | ~+0.5…0.9% |
| 6 | L2-norm folded into the pair launch | `GGML_CUDA_CONV_L2` | ~+0.5% |
| 7 | FWHT → q8_1 pre-quantisation (the transform writes the matvec's activation) | `GGML_CUDA_FWHT_Q8_1` | 0 at d=0, +0.7% at 90K |

Instruments added along the way: `GGML_V100_LAUNCH_COUNT`, `GGML_V100_OP_HIST`, `GGML_V100_OP_TIME`
(now also times fusion arms), `GGML_V100_DUMP_LAST`, `GGML_V100_DUMP_NODES`, `GGML_V100_ARM_DUMP`.

## Negative results worth knowing (all documented in the report)

* **Kernel count is not the lever.** Folding 257 q8_1 quantiser launches per token into the FWHT
  kernel (−18% launches) changed wall time by **0%**: the work simply moved into the producer
  (MUL_MAT −0.46 ms, FWHT arm +0.38 ms).
* **The PQ2_0 matvec is not limited by its 16-bit weight loads.** Repacking the 34-byte
  `block_pq2_0` into an 8-byte-aligned layout made the SASS strictly better (134 → 127
  instructions per 64 weights, weight loads 11×`LDG.E.U16` → 2×`LDG.E.64`) and the result
  bit-identical, yet it measured **−1.6%**. ncu: DRAM 61–75%, issue 42–64%, i.e. the kernel is
  memory-latency bound, not load-width bound. (`GGML_CUDA_PQ2_0_REPACK=1` re-enables the experiment.)
* **MTP is a loss at very long context** (~100K): the draft head attends over the whole KV itself,
  and draft acceptance drops from ~75% (short prompts) to ~36%. Use MTP off (or n-max 1) for long
  document work.

## Quick start

### 1. Apply a patch and build

Two bases exist — read `patch/README.md` first. For a fork branch:

```bash
# 1. Fork PrismML-Eng/llama.cpp on GitHub, then clone your fork
git clone https://github.com/<you>/llama.cpp && cd llama.cpp
git checkout -b v100-cuda prism-b10743-adfffbe     # verified clean base (2026-09-25)
git -c core.autocrlf=false apply -p1 /path/to/vs-prism-fork-cuda.patch
cmake -B build-v100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF
cmake --build build-v100 -j 16 --target llama-cli llama-bench llama-server
```

If you want *this project's* 17-file delta instead, apply `patch/state-path-fusions.patch` on top of
the merged tree it was generated from (that tree already contains the earlier Volta tuning). On the
fork's newest revision (`prism-b10754-2459f68`, 2026-10-02) nine files of the full patch conflict,
because upstream added an overlapping FWHT-q8_1 fusion — rebase those by hand.

On Windows the helper `tools/build_incremental.ps1` (vcvars64 + ninja) and `tools/cc_check.ps1`
(single-TU syntax check) are what was used here; both take their paths from environment variables
(`LLAMA_TREE`, `VCVARS64`, …), see `tools/README.md`.

### 2. Verify (bit-exactness, no eyeballing)

```powershell
$env:LLAMA_BIN   = "C:\path\to\llama.cpp-bonsai\build-v100\bin"
$env:BONSAI_MODEL= "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf"
$env:LLAMA_WORK  = "C:\path\to\work"
powershell -File tools\run_acceptance.ps1      # 18 checks: logits bit-exactness + text + PPL + perf
```

The gate is a **per-token logits dump** (`GGML_V100_DUMP_LAST`) compared bit-for-bit across
configurations — text comparison alone is not enough (it missed a real race condition during this
work, see report §16).

### 3. MTP speculative decoding (the big win)

The stock GGUF has **no MTP head**, so `--spec-type draft-mtp` refuses to start. `mtp/graft_mtp.py`
appends the 15 `blk.64.*` tensors of a Qwen3.8-27B MTP block (0.35 GB range-downloaded from
`unsloth/Qwen3.8-27B-GGUF`, `UD-Q2_K_XL`) to your own PQ2_0 file and sets
`qwen35.block_count=65`, `qwen35.nextn_predict_layers=1`. The trunk stays bit-identical.

```bash
python mtp/graft_mtp.py Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf
llama-server -m Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf -ngl 99 -fa on -c 32768 ^
             --spec-type draft-mtp --spec-draft-n-max 3 --jinja
```

`tools/server/` has two ready launchers (262144 context, q8_0 KV, LAN): one with MTP (chat) and one
without (long documents).

## Hardware / software used

Tesla V100-SXM2 32 GB (sm_70), CUDA 12.6, MSVC 14.44, Windows 10, llama.cpp fork snapshot
2026-09-26 (`b11004`). The machine was shared with another CUDA job, so benchmark jitter is
±1.4% per run and every A/B here is interleaved multi-round with best-of/median, never a single run.

## Credits / license

* [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT.
* [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) — MIT; the patch applies on top
  of it, and this work would not exist without the fork's low-bit formats and kernel work.
* MTP: authored/verified upstream by PrismML PR
  [#205](https://github.com/PrismML-Eng/llama.cpp/pull/205); the graft script is from
  [`decent-jawfish/bonsai-2-27b-mtp`](https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp)
  (Apache-2.0), MTP tensors come from
  [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). See `NOTICE`.
* This repository (patch, docs, tools): MIT, see `LICENSE`. No model weights are included.
