# MTP speculative decoding for Bonsai 2 27B

The stock `Ternary-Bonsai-2-27B-*.gguf` files **contain no MTP head**, so the fork's
`--spec-type draft-mtp` refuses to start:

```
W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
E common_speculative_init_result: failed to create MTP context
```

(`src/llama-context.cpp`: `ctx_type == MTP && hparams.n_layer_nextn == 0` → nullptr.)

PrismML fixed the related graph bug in PR
[#205](https://github.com/PrismML-Eng/llama.cpp/pull/205) (merged 2026-09-21, "qwen35: apply the
Hadamard inverse to the MTP token-embedding lookup") — that fix is required, and it is present in
the fork snapshot this repo is built on. What is still missing is the **head itself**, which ships
separately as a 15-tensor Transformer block.

## Graft the head into your own file (0.35 GB download)

`graft_mtp.py` (from [`decent-jawfish/bonsai-2-27b-mtp`](https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp),
Apache-2.0) copies the 15 `blk.64.*` tensors of the Qwen3.8-27B MTP block — a full attention block
plus the `nextn` projections — from
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q2_K_XL`
(they live in the last ~0.35 GB of that file, fetched with HTTP range requests) and appends them to
your Bonsai GGUF, setting:

```
qwen35.block_count         64 -> 65
qwen35.nextn_predict_layers  (new) = 1
```

```bash
python graft_mtp.py Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf
# [6/6] OK   tensors=866 (851 + 15)  block_count=65  nextn_predict_layers=1
```

The tensors are **not** rotated and are deliberately not added to `prism.hadamard.weight_names`:
Bonsai's RMSNorm weights match stock Qwen3.8 elementwise, so the residual stream is in the original
basis (only `token_embd` is latent, which is exactly what PR #205 fixes).

Alternative, if you prefer a ready-made 7.6 GB file: `decent-jawfish/bonsai-2-27b-mtp`
(`Bonsai-2-27B-PQ2_0-MTP.gguf`) or `ProCreations/Ternary-Bonsai-2-27B-MTP`
(`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`, the file used in PR #205's repro).

## Validation (both gates must pass)

1. **The trunk must not change.** With the grafted file, greedy logits dumps (32 tokens) are
   bit-identical to the original file: `BIT-IDENTICAL (6 208 000 logits)`.
2. **Greedy output must not change with MTP on.** Every draft is verified against the target, so
   `--temp 0` output is byte-identical with and without `--spec-type draft-mtp` — verified at d=0
   (64 and 256 tokens, n-max 1/2/3/4), 32K and 96K.

`../tools/ab_mtp.ps1` runs both gates plus a 256-token throughput comparison.

## Results on a V100 (see `../docs/results.md` for the full tables)

| depth | no MTP | MTP (best n-max) |
|---|---:|---:|
| d=0 | 60.9 t/s | **86.7 t/s (+41%)**, n-max 4 |
| 32K | 52 t/s | **68.1 t/s (+33%)**, n-max 3 |
| 96K | 37.6 t/s | 44.5 t/s (+18%), n-max 1–3 |
| 100K chat + summarize (server) | 35.3 t/s | 31.6 t/s (**−10%**) |

Rules of thumb: n-max 4 for short prompts, 3 around 32K, 1–3 at 96K+ (there is a cliff at n-max 4
for long context), and turn MTP off entirely for long-document work — at ~100K the draft head has
to attend over the whole KV by itself and acceptance drops to ~36%.
