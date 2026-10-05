> [!IMPORTANT]
> **Unofficial V100-tuned build — not affiliated with, and not endorsed by, ggml-org or PrismML.**
>
> This tree is a modified copy of the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
> (base revision `prism-b10743-adfffbe`, 2026-09-25) plus unpublished single-GPU (Tesla V100 / sm_70)
> decode tuning, and a few files ported from the [cyyself fork](https://github.com/cyyself/llama.cpp).
> Every upstream and third-party license and copyright notice is retained unchanged:
> llama.cpp and the PrismML fork are MIT (© 2023-2026 The ggml authors). See [NOTICE](NOTICE) for the
> full attribution list and [bonsai-v100/README.md](bonsai-v100/README.md) for what was changed and
> how it was measured (Bonsai 2 27B ternary: +5.7% decode from the CUDA work, +41% more with MTP).
>
> If you want the unmodified base, use the upstream repositories above.
>
> Unofficial Tesla V100 (sm_70) tuning of the PrismML llama.cpp fork for ternary Bonsai 2 27B: bit-exact CUDA fusions + MTP speculative decoding, with the full measurement log.  
>【重要提示！】
非官方 V100 调优版本 —— 与 ggml-org 及 PrismML 无关联，亦未获其认可。
本代码树是 llama.cpp 的 PrismML 分支（基础版本 prism-b10743-adfffbe，2026-09-25）的修改副本，加入了未公开的单 GPU（Tesla V100 / sm_70）解码调优，以及若干从 cyyself 分支 移植的文件。所有上游及第三方许可证与版权声明均原样保留：llama.cpp 与 PrismML 分支均为 MIT 许可（© 2023-2026 The ggml authors）。完整的署名列表见 NOTICE，具体改动了什么、如何测量，见 [README.md](bonsai-v100/README.md)（Bonsai 2 27B 三值模型：CUDA 优化使解码速度提升 5.7%，配合 MTP 再提升 41%）。
若需要未修改的基础版本，请使用上述上游仓库。
面向三值 Bonsai 2 27B 的 PrismML llama.cpp 分支非官方 Tesla V100（sm_70）调优：位精确（bit-exact）CUDA 融合 + MTP 推测解码，并附完整测量日志。
> ## V100 Bonsai2 Fast —— MAXPOWER UP TO 85 tokens on single v100 card.
> ## V100 Bonsai2 Fast —— 单卡火力全开最高可达每秒钟85词元.
> ## Performance / 性能实测

**Test bed / 测试环境**: Tesla V100-SXM2 32 GB (sm_70) · CUDA 12.6 · MSVC 14.44 · Windows ·
PrismML llama.cpp fork `prism-b10743-adfffbe` · model **Ternary-Bonsai-2-27B-PQ2_0** (2.13 bpw
ternary, hybrid attention).
**Flags / 参数**: `-ngl 99 -fa 1`, greedy (`--temp 0`), single stream / 单流.
**Method / 方法**: every A/B is interleaved over multiple rounds and reported as best-of + median —
the machine is shared, single runs drift by ~1.4%. Reproduce with `bonsai-v100/tools/`.

### 1. Decode & prefill / 解码与预填 (`llama-bench` tg128, graphs off)

| Configuration / 配置 | d=0 | 90K | d=0 pp512 | 90K pp512 |
|---|---:|---:|---:|---:|
| Handover baseline / 交接基线 | 58.89 | 38.31 | 787.65 | 488.92 |
| Reference arm of this build / 本构建参考臂 (`GGML_CUDA_GDN_ROWS_READ=0`) | 63.03 ± 0.19 | 39.99 ± 0.08 | 814.75 | 504.47 |
| **This build, all fusions on / 全开** | **66.47 ± 0.10** | **41.21 ± 0.07** | **818.47** | **505.90** |
| Δ vs reference arm / 相对参考臂 | **+5.46%** | **+3.05%** | +0.46% | +0.28% |
| Δ vs handover baseline / 相对交接基线 | **+12.9%** | **+7.6%** | +3.9% | +3.5% |

At 32K the handover baseline was 49.27 t/s / 647.30 t/s prefill; the MTP table below was measured
with a different harness (`llama-cli`, 128 generated tokens) and is not additive with this one.
32K 的交接基线是 49.27 / 647.30；下面 MTP 那张表用的是 `llama-cli`（不同口径），两者不可相加。

### 2. Where the decode time goes / 单个 token 的时间去向 (ncu, 260 matvec launches ≈ 1 token)

| Metric / 指标 | Value / 数值 |
|---|---:|
| Weights read per token / 每 token 读取权重 | 6.80 GB |
| Matvec kernel time / 权重 matvec 内核时间 | 10.57 ms (68% of the token) |
| Effective bandwidth / 等效带宽 | 607 GB/s |
| DRAM utilisation / 显存带宽占用 | 61–75% |
| SM / issue utilisation / 发射槽占用 | 42–64% |

### 3. Individual fusions / 各项融合 (A/B on the same build)

| Fusion / 融合项 | Switch / 关闭开关 | Gain / 收益 |
|---|---|---:|
| GDN state read by row index / GDN 行索引读状态 | `GGML_CUDA_GDN_ROWS_READ=0` | +2.9% |
| conv-state gather + concat, write-back folded / conv 状态合并与写回折叠 | `GGML_CUDA_CONV_STATE_FUSION=0` | +1.9% |
| `[ADD, RMS_NORM, MUL]` on Volta | `GGML_CUDA_ADD_RMS_NORM=0` | +0.4…0.8% |
| `ssm_alpha` / `ssm_beta` matvec pair / 配对 launch | `GGML_CUDA_MMVF_PAIR=0` | +0.6…1.8% (acceptance run +3.00%) |
| de-interleaving GLU store / GLU 去交织写 | `GGML_CUDA_GLU_PERMUTE=0` | +0.5…0.9% |
| L2-norm folded into the pair launch | `GGML_CUDA_CONV_L2=0` | ≈ +0.5% |
| FWHT → q8_1 pre-quantisation / 量化折进 FWHT | `GGML_CUDA_FWHT_Q8_1=0` | 0.0% (d=0) · +0.71% (90K) |

### 4. MTP speculative decoding / MTP 投机解码

Grafted Qwen3.8-27B MTP head (`bonsai-v100/mtp/`), `llama-cli` greedy, graphs off; numbers are t/s
with the gain over the same harness without MTP. / 嫁接 MTP 头，`llama-cli` 贪心，括号内为同口径收益。

| Depth / 深度 | no MTP | n-max 1 | n-max 2 | n-max 3 | n-max 4 | n-max 6 |
|---|---:|---:|---:|---:|---:|---:|
| d=0 (512 tok, 4 rounds) | 60.90 | 74.60 (+22.5%) | 72.30 (+18.7%) | 83.90 (+37.1%) | **86.70 (+40.7%)** | 82.40 (+35.0%) |
| 32K (128 tok, 3 rounds) | 51.2 | 65.6 (+28.1%) | 67.5 (+31.8%) | **68.1 (+33.0%)** | 65.0 (+25.7%) | — |
| 96K (128 tok, 2 rounds) | 37.6 | **44.5 (+18.4%)** | 44.4 (+18.1%) | 44.2 (+17.6%) | 37.0 (−2.9%) | 34.8 (−7.4%) |

Acceptance rate / 验收率: **75%** at d=0 (n-max 4) · 64% short chat · 36% at ~100K.
Correctness / 正确性: greedy output is **byte-identical** with and without MTP at every depth, and
the grafted trunk is bit-identical to the stock file (6 208 000 logits).
贪心输出与不开 MTP **逐字节相同**；嫁接后的主干与原文件 logits **逐位一致**。

### 5. Full-window server / 满窗服务 (262144 ctx, q8_0 KV, `-np 1`)

| Metric / 指标 | Value / 数值 |
|---|---:|
| VRAM | 19.3 GB |
| Prefill / 预填 | 577 t/s (101 003 tok in 175 s) ⇒ ~7.5 min to fill 262 K |
| 100K chat + summarize, no MTP | 35.3 t/s |
| 100K chat + summarize, MTP n-max 3 / 1 | 31.6 / 31.6 t/s (acceptance 36% / 59%) |

⇒ keep MTP on for chat and short/medium prompts, turn it off for long-document work.
聊天与短中上下文开 MTP；超长文档改用 `start-server-nospec.cmd`。


# llama.cpp

> [!IMPORTANT]
> **This is the PrismML fork of llama.cpp**, the main line behind the [Bonsai](https://huggingface.co/collections/prism-ml/bonsai) models (branch `prism`, developed as `prism-v7`). It tracks current mainline llama.cpp and adds the fork's low-bit formats and runtime features on top.
>
> **New here? Start with the [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) repo.** It downloads the right models and the correct prebuilt binaries for your hardware/backend automatically.
>
> **Which ternary model file to use:**
>
> - `*-PQ2_0.gguf` (fork group-128, ggml id 142): preferred on Metal, CUDA, HIP and CPU. About 6% smaller than group-64.
> - `*-Q2_0_g64.gguf` / 27B `*-Q2_g64.gguf` (official group-64, ggml id 42): runs on every backend here AND on mainline llama.cpp. If unsure, use this. Newer model releases name this file plain `*-Q2_0.gguf`.
> - `*-Q2_0.gguf` on OLDER model repos is the **deprecated legacy format** (group 128 stored as id 42). It does not load on these builds; the error tells you which file to get instead. If you must run it, use the frozen [`prism-v5`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-v5) line and its final release [`prism-b9601`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9601-68faa14).
>
> **Speculative decoding (dspark)** is supported via mainline's draft-dspark plus fork patches. Drafters published for older model releases need a one-time conversion with `gguf-dspark-to-dflash` (see [SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) in Bonsai-demo); newer releases ship ready-to-use drafters.
>
> Do NOT build from `prism-v6` (stale mid-migration snapshot) and do NOT mix this fork's `ggml-*` libraries with a stock llama.cpp build.

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
