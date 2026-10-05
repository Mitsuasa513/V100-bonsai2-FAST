---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
tags:
- gguf
- ternary
- 2-bit
- mtp
- speculative-decoding
- llama-cpp
- qwen35
---

# Bonsai 2 27B + MTP (GGUF)

`prism-ml/Ternary-Bonsai-2-27B-gguf` (PQ2_0) with the **multi-token-prediction
head from Qwen3.8-27B grafted in**, so it can drive `--spec-type draft-mtp`
speculative decoding.

**~44% faster decode at the same quality** — the ternary weights are untouched,
and speculative decoding verifies every draft against the target model, so the
output distribution is unchanged.

| | tok/s | acceptance |
|---|---:|---:|
| Bonsai 2 27B PQ2_0, no MTP | 41.9 | — |
| **this model, `--spec-draft-n-max 2`** | **60.5** | **0.60** |
| this model, `--spec-draft-n-max 3` | 58.9 | 0.49 |
| this model, `--spec-draft-n-max 4` | 55.4 | 0.42 |

Measured on one NVIDIA RTX PRO 6000 Blackwell **MIG 1g.24gb slice** (a quarter
GPU, 24 GB), driver 580.126.20, `--parallel 1`, 400 generated tokens, 5 prompts,
median. On a full GPU the absolute numbers will be higher; the ratio is the
durable part.

---

## ⚠️ Requires a patched llama.cpp

**With stock binaries this model loads and runs normally, but `--spec-type
draft-mtp` fails at startup:**

```
llama_verify_hadamard_graph: latent lookup 'mtp_tok_embd-64' consumed by op=RMS_NORM
llama_init_from_model: failed to initialize the context:
    Hadamard-latent table 'token_embd.weight' is read without the inverse transform
```

This is not a defect in the graft. PrismML's ternary models store
`token_embd.weight` in a rotated ("Hadamard-latent") domain, and every read must
apply the inverse transform first. The main model path in
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) does this; the
MTP draft graph, which is upstream llama.cpp code predating the folding, has its
own embedding lookup and skips it.

`0001-qwen35-mtp-hadamard-inverse.patch` in this repo fixes it in ~8 lines against
tag `prism-b10685-7dffb15`. Build:

```sh
curl -fsSL -o src.tgz https://github.com/PrismML-Eng/llama.cpp/archive/refs/tags/prism-b10685-7dffb15.tar.gz
mkdir src && tar -xzf src.tgz -C src --strip-components=1
cd src && patch -p1 < ../0001-qwen35-mtp-hadamard-inverse.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<your arch> -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j
```

Single CUDA arch, server target only: ~150 s on 16 cores.

## Run

```sh
llama-server -m Bonsai-2-27B-PQ2_0-MTP.gguf \
  -ngl 99 -fa on -c 32768 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Sampling, per the base model card — thinking mode: `temperature=1.0, top_p=0.95,
top_k=20, min_p=0.0`; instruct: `temperature=0.7, top_p=0.80, top_k=20,
presence_penalty=1.5`.

`--spec-draft-n-max 2` was best on a 24 GB slice. Sweep it for your card; larger
values draft deeper but acceptance falls.

### Concurrency

Unusually for speculative decoding, the gain here does **not** vanish under load —
the MTP head is one small block, so drafting is cheap against a
bandwidth-bound ternary trunk:

| concurrent | vanilla agg tok/s | this model agg tok/s | |
|---:|---:|---:|---|
| 1 | 37.6 | **57.8** | +54% |
| 2 | 60.4 | 56.1 | **-7%** |
| 3 | — | 99.2 | |
| 4 | 88.4 | **115.5** | +31% |
| 8 | 85.1 | **111.9** | +32% |

(both `-c 65536 -np 4`.)

**There is a reproducible dip at exactly 2 concurrent streams**, where per-request
throughput falls to 29 tok/s — below the 37 tok/s seen at 3 concurrent — and MTP
ends up ~7% *behind* vanilla. Confirmed over four runs. Everywhere else MTP leads
by 31-54%. If your workload sits at exactly 2 concurrent requests, benchmark before
adopting.

The MTP draft context also costs VRAM, so on a 24 GB card you can allocate fewer
slots than vanilla: vanilla reaches `-np 32` and peaks at ~186 agg tok/s, while
this model OOMs past `-np 16` and peaks at ~175. Vanilla wins the absolute
aggregate ceiling by ~7%; MTP wins everywhere below it.

## How it was made

15 `blk.64.*` tensors — a full transformer block plus the `nextn` projections —
were copied **verbatim and unrotated** from
`unsloth/Qwen3.8-27B-GGUF` `UD-Q2_K_XL` into the Bonsai GGUF, with
`qwen35.block_count` set to 65 and `qwen35.nextn_predict_layers` to 1.

No rotation was applied to them, and they are deliberately **not** listed in
`prism.hadamard.weight_names`. The justification: Bonsai's RMSNorm weights match
stock Qwen3.8 elementwise (cosine > 0.99996 on `blk.0.attn_norm`,
`blk.0.post_attention_norm`, `output_norm`). An elementwise scale cannot commute
with a rotation, so the residual stream is in the **original** basis and
`prism.hadamard.*` is an online per-matmul quantisation aid, not a change of
basis. A stock MTP block therefore receives exactly the activations it expects.

Only the embedding table is genuinely latent, which is why the one patch is needed.

Base tensors are bit-identical to the source GGUF; only metadata changed and 15
tensors were appended. Merge tool: `graft_mtp.py` in this repo.

### Acceptance

0.60 here versus 0.70 for the same head on stock Qwen3.8 — the expected cost of a
head trained on FP16 hidden states drafting for a ternary trunk. Well above
break-even, hence the speedup.

## Provenance and attribution

Not affiliated with, or endorsed by, PrismML or the Qwen team. All components are
Apache-2.0, as is this derivative.

- Language model: [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0)
- MTP head: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q2_K_XL` (Apache-2.0), quantised from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- Runtime: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) `prism-b10685-7dffb15` (MIT)
- MTP recipe and the observation that the head ships inside the GGUF: [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)

## Limitations

- **Needs the patch** for MTP. Without it, works as a normal Bonsai 2 27B.
- Quality was **not** re-benchmarked. The claim that quality is unchanged rests on
  speculative decoding being verified-lossless and the base weights being
  bit-identical — not on a benchmark run.
- The MTP block is Q6_K/Q8_0, not ternary, adding ~0.35 GB over the base model.
- Tested only on CUDA sm_120. Metal and CPU paths are untested.
- The vision `mmproj` is not included; text-only.
