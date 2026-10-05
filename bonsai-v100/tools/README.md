# Tools

These are the exact scripts used for the measurements in `../docs/`. They are PowerShell (Windows,
MSVC + CUDA) because that is what the target machine runs; they are templates — every path is read
from an environment variable with a neutral placeholder as fallback.

## Environment variables

| variable | meaning |
|---|---|
| `LLAMA_BIN` | `…\llama.cpp-bonsai\build-v100\bin` (contains `llama-cli.exe`, `llama-bench.exe`, `llama-server.exe`, `ggml-cuda.dll`) |
| `LLAMA_TREE` | the llama.cpp source tree root |
| `LLAMA_WORK` | scratch/output directory for the logits dumps |
| `LLAMA_WORK_MTP` | scratch directory for the MTP runs |
| `LLAMA_PROMPT_32K` | a ~32K-token prompt file used by the long-context tests |
| `LLAMA_PRISTINE` | pristine fork checkout, used by `regen_patch.ps1` |
| `BONSAI_MODEL` | the GGUF under test (PQ2_0, or the MTP-grafted copy) |
| `VCVARS64` | `vcvars64.bat` of your Visual Studio install |

```powershell
$env:LLAMA_BIN    = "C:\dev\llama.cpp-bonsai\build-v100\bin"
$env:BONSAI_MODEL = "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf"
$env:LLAMA_WORK   = "C:\dev\work"
```

## The scripts

| script | what it does |
|---|---|
| `run_acceptance.ps1` | **the one-shot gate**: build freshness, logits bit-exactness (short prompt, 5 configs × both graph modes), 32K prompt, matvec-pair check, text compare, perplexity, multi-sequence server test, and the perf A/B. Prints `SUMMARY: n/18 checks passed`. |
| `dump_matrix.ps1`, `dump_gon.ps1`, `dump_pair.ps1`, `dump_glu.ps1`, `dump_l2.ps1`, `dump_fwhtq8.ps1`, `dump_pq2repack.ps1` | per-feature bit-exactness: run the same greedy prompt under two configurations and compare the dumped logits token by token. |
| `multiseq2.ps1` | two-sequence `llama-server` test used by the acceptance suite. |
| `ab_matrix.ps1`, `ab_matrix2.ps1`, `ab_fwhtq8.ps1`, `ab_pq2repack.ps1` | interleaved A/B throughput drivers (alternate the arms so machine drift hits both equally, then report best/median). |
| `ab_mtp.ps1`, `bench_mtp.ps1`, `bench_mtp_depth.ps1` | MTP: three-gate validation, n-max sweep, and the 32K/96K depth sweep. |
| `parse_nodes.py` | parses `GGML_V100_DUMP_NODES` output and answers structural questions (`summary`, `ops <ne0>`, `node <i>`, `consumers <i>`) — this is how the fused patterns were identified. |
| `ncu_mmvq_roof.bat` | Nsight Compute run that profiles ~1 token of `mul_mat_vec_q` and answers “bandwidth-bound or not”. Needs an **elevated** shell (GPU performance counters). |
| `build_incremental.ps1` | vcvars64 + `ninja` incremental build of `llama-cli`/`llama-bench`/`llama-perplexity`/`llama-server`, logging to `build-log.txt`. |
| `cc_check.ps1` | syntax-check a single CUDA translation unit by replaying its `compile_commands.json` command (~15–25 s instead of a full build). |
| `regen_patch.ps1` | regenerates `patch/state-path-fusions.patch` from a pristine tree and the working tree, staged so that `git apply -p1` works. |
| `server/` | ready-to-run `llama-server` launchers: 262144 context, q8_0 KV, LAN (`--host 0.0.0.0 --port 8080`), one with MTP and one without, plus `stop-server.cmd`. |

## Notes that saved a lot of time

* The acceptance gate is a **logits dump**, not the generated text. Text comparison missed a real
  read-after-write race during this work (report §16); a `bitDiff=` line caught it immediately.
* `llama-bench` cannot measure speculative decoding — MTP throughput comes from `llama-cli` /
  `llama-server` timings (`Generation: x t/s` on stdout, acceptance via `--verbose`).
* With MTP, `GGML_CUDA_DISABLE_GRAPHS=1` is ~9% faster than the default (per-step verify batches
  have different shapes, which the CUDA-graph cache handles poorly); without MTP the two are equal.
* Benchmarking on a shared GPU: never trust a single run. The A/B drivers here alternate the arms
  and report best-of/median, and `±` values above ~2% mean “run it again”.
