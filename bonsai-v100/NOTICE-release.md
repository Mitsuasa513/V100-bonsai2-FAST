This repository contains patches, documentation and tooling. It does not contain model weights.

Third-party components and their licenses:

* llama.cpp — MIT License, Copyright (c) 2023-2026 The ggml authors.
  https://github.com/ggml-org/llama.cpp

* PrismML fork of llama.cpp — MIT License. `patch/state-path-fusions.patch` is a diff against a
  snapshot of that fork (2026-09-26, mainline build b11004) and is meant to be applied on top of it.
  https://github.com/PrismML-Eng/llama.cpp

* MTP support (upstream side) — PrismML PR #205 "qwen35: apply the Hadamard inverse to the MTP
  token-embedding lookup", merged 2026-09-21. Required for `--spec-type draft-mtp` on
  Hadamard-folded ternary models.
  https://github.com/PrismML-Eng/llama.cpp/pull/205

* mtp/graft_mtp.py — from decent-jawfish/bonsai-2-27b-mtp (Apache-2.0), which grafts the Qwen3.8-27B
  MTP block into the Bonsai GGUF.
  https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp

* The MTP tensors themselves come from unsloth/Qwen3.8-27B-GGUF (UD-Q2_K_XL); they are downloaded at
  graft time and are not redistributed here.
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF

* Bonsai 2 27B model — prism-ml/Ternary-Bonsai-2-27B-gguf (Apache-2.0).
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf

* ProCreations/Ternary-Bonsai-2-27B-MTP — the pre-built bundled target+MTP file referenced by PR #205.
  https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP
