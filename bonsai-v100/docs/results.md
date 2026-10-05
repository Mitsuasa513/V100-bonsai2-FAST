# Measured results (English summary)

All numbers: Tesla V100-SXM2 32 GB, `-ngl 99 -fa 1`, greedy (`--temp 0`), single stream, unless
noted. The machine was shared with another CUDA job, so every A/B is interleaved multi-round with
best-of / median reported.

## 1. Decode throughput (llama-bench, tg128 unless noted)

| configuration | d=0 | 32K | 90K | pp512 (d=0) | pp512 (90K) |
|---|---:|---:|---:|---:|---:|
| handover baseline | 58.89 | — | 38.31 | 787.65 | 488.92 |
| reference arm of the acceptance run (`GGML_CUDA_GDN_ROWS_READ=0`) | 63.03 ± 0.19 | — | 39.99 | 814.75 | 504.47 |
| **patched build, all fusions on** | **66.47 ± 0.10** | — | **41.21 ± 0.07** | 818.47 | 505.90 |

Acceptance: `SUMMARY: 18/18 checks passed` (correctness 15 + perf 3).

## 2. Per-fusion A/B (same build, switch off vs on)

| fusion | switch | effect |
|---|---|---|
| GDN state read by row index | `GGML_CUDA_GDN_ROWS_READ` | +2.9% |
| conv state gather+concat (+ write-back folded) | `GGML_CUDA_CONV_STATE_FUSION` | +1.9% |
| `[ADD, RMS_NORM, MUL]` | `GGML_CUDA_ADD_RMS_NORM` | +0.4…0.8% |
| `ssm_alpha`/`ssm_beta` pair launch | `GGML_CUDA_MMVF_PAIR` | +0.6…1.8% (acceptance run: +3.0%) |
| de-interleaving GLU store | `GGML_CUDA_GLU_PERMUTE` | +0.5…0.9% |
| L2-norm folded into the pair launch | `GGML_CUDA_CONV_L2` | ~+0.5% |
| FWHT → q8_1 pre-quantisation | `GGML_CUDA_FWHT_Q8_1` | 0.0% at d=0 / +0.71% at 90K (see §4) |

## 3. Launch accounting and kernel time (steady-state decode token, `GGML_V100_LAUNCH_COUNT=1`)

| configuration | kernel launches / token |
|---|---:|
| patched build | **1174** |
| `GGML_CUDA_FWHT_Q8_1=0` | 1431 |
| `GGML_CUDA_DISABLE_FUSION=1` | 2217 |
| `GGML_CUDA_GDN_ROWS_READ=0` | 1575 |

Folding 257 quantiser launches into the FWHT kernel (−18% launches) changed wall time by **0%**:

| bucket | fused | unfused |
|---|---:|---:|
| `MUL_MAT` (dispatched MMVQ) | 6.220 ms | 6.677 ms |
| FWHT arm (`[MUL,RESHAPE,MUL_MAT]`) | 1.784 ms | 1.401 ms |
| node loop total | 16.27 ms | 16.27 ms |

Per-arm cost of one decode token (`GGML_V100_OP_TIME=1`, 6.8 GB of weights per token):

| bucket | ms | x | per arm |
|---|---:|---:|---:|
| `MUL_MAT` (dispatched) | 6.40 | 242 | 26.5 µs |
| `arm:MUL_MAT` (MatMul+GLU / MatMul+ADD epilogues, real weight reads) | 5.36 | 120 | 44.7 µs |
| FWHT arm | 1.83 | 257 | 7.1 µs |
| `RMS_NORM` arm | 1.12 | 161 | 6.9 µs |
| `ADD` + `SSM_CONV` arms | 0.69 | 96 | |
| FA / ROPE / SET_ROWS / GLU / CONT | 0.77 | 120 | |

## 4. ncu profile of `mul_mat_vec_q<142,…>` (PQ2_0), 260 launches ≈ one token

| grid | inv | Mbyte | µs | GB/s | DRAM % | SM/issue % |
|---|---:|---:|---:|---:|---:|---:|
| 124160 (output projection) | 1 | 365 | 567 | 644 | 71.8 | 63.7 |
| 8704 (ffn, GLU epilogue arm) | 29 | 1584 | 2369 | 669 | 74.5 | 50.7 |
| 8704 | 35 | 1039 | 1738 | 598 | 67.7 | 55.5 |
| 5120 | 35 | 674 | 1169 | 577 | 64.5 | 50.5 |
| 3072 | 35 | 460 | 839 | 549 | 60.9 | 44.0 |
| 2560 | 92 | 1970 | 3264 | 604 | 66 | 44 |
| 6144 | 11 | 245 | 416 | 590 | 65.5 | 52.7 |
| 512 (attn_k/attn_v) | 22 | 72 | 206 | **350** | **38.4** | **17.7** |
| **total** | 260 | **6410** | **10566** | **607** | ~67 | ~50 |

Weight bytes per token: **6.80 GB** (401 quantised `MUL_MAT`, 6 matrices × 64 layers + output).
With the matvec buckets at 11.76 ms that is 578 GB/s of the ~700–850 GB/s a V100 can actually
stream.

SASS of the 64-element inner loop (before/after the repack experiment):

| variant | instructions | weight loads | regs |
|---|---:|---|---:|
| packed (`block_pq2_0`, 34 B) | 134 | 11 × `LDG.E.U16` | 47 |
| repacked (8-byte aligned qs) | 127 | 2 × `LDG.E.64` | 48 |

…and the repacked variant is **1.6% slower** in wall time → the kernel is latency/DRAM bound.

## 5. MTP speculative decoding (llama-cli, greedy, same prompt)

Acceptance = drafted/accepted from `--verbose`; the grafted trunk is bit-identical to the stock file
and the greedy output is byte-identical with and without MTP at every depth below.

| depth | no MTP | n-max 1 | n-max 2 | n-max 3 | n-max 4 | n-max 6 |
|---|---:|---:|---:|---:|---:|---:|
| d=0 (512 tok, 4 interleaved rounds) | 60.90 | 74.60 (+22.5%) | 72.30 (+18.7%) | 83.90 (+37.1%) | **86.70 (+40.7%)** | 82.40 (+35.0%) |
| 32K (128 tok, 3 rounds) | 51.2 | 65.6 (+28.1%) | 67.5 (+31.8%) | **68.1 (+33.0%)** | 65.0 (+25.7%) | — |
| 96K (128 tok, 2 rounds) | 37.6 | **44.5 (+18.4%)** | 44.4 (+18.1%) | 44.2 (+17.6%) | 37.0 (−2.9%) | 34.8 (−7.4%) |

Acceptance measured at d=0 / n-max 4: **75%**. There is a cliff between n-max 3 and 4 at 96K
(verify batch of 4 vs 5 tokens).

## 6. MTP in llama-server (262144 context, q8_0 KV, single slot)

| measurement | value |
|---|---:|
| VRAM at 262144 + q8_0 KV | 19.3 GB |
| prefill | 101003 tok / 175 s = **577 t/s** (⇒ ~7.5 min to fill 262K) |
| short chat request | 71.5 t/s, acceptance 64% |
| 100K-token chat + summarize, MTP n-max 3 | 31.6 t/s (acceptance 36%) |
| same, **no MTP** | **35.3 t/s** |
| same, MTP n-max 1 | 31.6 t/s (acceptance 59%) |

Conclusion: keep MTP on for chat / short and medium prompts, turn it off for long-document work.

## 7. Reproducing

See `../README.md` (build + acceptance) and `../tools/README.md` (what each script does and which
environment variables it reads). The full experiment log with every dead end is in
`report.zh-CN.md`.
