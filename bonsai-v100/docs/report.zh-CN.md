# 步骤①：GDN 行索引状态读取（`src[6]`）CUDA 实现方案

2026-09-27 · 树 `<llama.cpp-bonsai>` · Tesla V100-SXM2-32GB(sm_70) · MSVC 14.44 + CUDA 12.6

本文只做三件事：**(a)** 用本机实测数据核对交接文档的前提；**(b)** 给出内核层与放行层可落地的改动；
**(c)** 给出验收/基准脚本与风险清单。没有改任何源码。

---

## 0. TL;DR（先看这四条，其中两条推翻了交接文档的前提）

1. **`src[6]` 这条路在你的基准配置下今天走不到。**
   `qwen35.cpp:478` 的条件是 `gdn_state_rows_env && gdn_state_rows_dev_ok && cparams.n_rs_seq > 0`；
   而 `cparams.n_rs_seq` 只由 `common.cpp:1722` 从 `params.speculative.need_n_rs_seq()` 得到，
   即**只有开了 draft 类投机解码（draft-simple / draft-mtp / eagle3 / dflash / dspark）才 > 0**。
   `-ngl 99 -fa 1` 跑 `llama-bench`（无投机）→ `n_rs_seq = 0`（我抓的日志：`llama_context: n_rs_seq = 0`，
   `llama_memory_recurrent: ... (1 cells, 64 layers, 1 seqs  0 rs_seq)`），rows 分支恒为 false。
   另外 **Bonsai 2 PQ2_0 这个 GGUF 没有 NextN/MTP 张量**（851 个张量名里 `nextn` 出现 0 次、没有 `blk.64`），
   所以 `--spec-type draft-mtp` 直接失败（我实测：`failed to create MTP context`）。要跑 `n_rs_seq>0` 只能配一个 draft 模型走 `draft-simple`。
2. **写回侧 CUDA 已经融合了**，rows 只能省"读"。
   fork 自带 `ggml_cuda_try_gdn_cache_fusion()` + `ggml_cuda_op_gated_delta_net_fused_cache()`
   （`gated_delta_net.cu` 与 prism 原树**逐字节相同**，也就是说这是 fork 原有能力，不是我们加的）。
   实测证据：解码图里 `CPY cache_s_l*(copy of new_state)` 节点存在，但运行时**没有** 12288-block 的 `cpy_scalar`
   （3 MB 拷贝应为 `ceil(786432/64)=12288` 个 block），也没有 memcpy 行 → 它要么被 epilogue 融合、要么走 `cudaMemcpyAsync`（`cpy.cu:466-475` 对连续同类型拷贝有 memcpy 快路径，ncu 的 per-kernel 摘要里看不到）。
   两种情况下都**没有**"3 MB 拷贝内核"可以省。
3. **所以 `src[6]` 本身的收益是 `GET_ROWS` 那一次 3 MB gather**：实测 8.45 µs/层（`k_get_rows_float_vec`，grid `(1,768,1)`）
   → **0.41 ms/token ≈ +2.4%（d=0）/ +1.6%（90K）**，不是文档估的 5–9%。
4. **要拿到 5–9%，需要打"整条 state 路径"**（都是 3~6 µs 的发起开销型小内核，删掉即净赚）：

| 项 | 每层 | 每 token | 占 d=0 解码 |
|---|---:|---:|---:|
| s 状态 **读**（GET_ROWS 3 MB） | 8.45 µs | 0.41 ms | **2.4%** |
| conv 状态 **读**（GET_ROWS 123 KB） | 4.16 µs | 0.20 ms | **1.2%** |
| conv 状态 **写回**（CPY 123 KB） | 6.09 µs | 0.29 ms | **1.7%** |
| s 状态写回 | 已融合 | ~0 | 0% |
| s/conv 清零（`SCALE`） | 稳态为空 | 0 | 0% |
| `CONT`（final_output/gate_reshaped，64/步） | ~1.3 次 | 0.34 ms | **2.0%** |
| 合计可动 | | **1.24 ms** | **≈7.3%** |

→ 建议顺序：**先做"后端内 GET_ROWS→GDN 融合"（方案 B，拿 2.4%，风险最小、今天就能生效；方案 A/B 的定义见 §4）**，
再做 conv 那条（+2.9%），最后才考虑图级 `src[6]`（方案 A，用于上游对齐 / 将来跑 draft 的 K>1 路径）。

---

## 1. 证据（本机实测，可复现）

### 1.1 当前解码图真实构成（4637 节点，48 个 GDN 层，`-p 0 -n 8`，`GGML_V100_DUMP_NODES=1`）

```
RESHAPE 1187 / MUL_MAT 755 / VIEW 738 / MUL 483 / RMS_NORM 209 / GET_ROWS 193 / CPY 192
ADD 128 / GLU 112 / SCALE 96 / PERMUTE 96 / UNARY 64 / CONT 64 / TRANSPOSE 48 / CONCAT 48
SSM_CONV 48 / L2_NORM 48 / GATED_DELTA_NET 48 / ROPE 32 / SET_ROWS 32 / FLASH_ATTN_EXT 16
```

每个循环层（共 48）相关的节点（`ne` 是实测值）：

| 节点 | ne | 说明 |
|---|---|---|
| `GET_ROWS conv_states-l` | (30720,1,1,1) | conv 状态 gather（123 KB） |
| `GET_ROWS node_*` (extra) | (30720,**0**,1,1) | 空（`n_rs - n_seqs = 0`），运行时被 `ggml_is_empty` 跳过 |
| `CPY cache_r_l*(view)(copy of )` | (30720,0,1,1) | 空 |
| `CPY conv_state_update-l` | (30720,1,1,1) | conv 写回（真跑，见 1.2） |
| `GET_ROWS node_*` | (786432,1,1,1) | **s 状态 gather（3 MB）** ← `src[6]` 要干掉的就是它 |
| `GET_ROWS node_*` (extra) | (786432,0,1,1) | 空 |
| `CPY cache_s_l*(view)(copy of )` | (786432,0,1,1) | 空 |
| `CPY cache_s_l*(view)(copy of new_state)` | (786432,1,1,1) | s 写回（3 MB）——运行时无对应内核，已融合/memcpy |
| `SCALE cache_r_l0/1/…` | (30720,1,1,1) 或 **(0,1,1,1)** | **首个 pass 3 MB 清零，稳态为空** |
| `SCALE cache_s_l0/1/…` | (786432,1,1,1) 或 **(0,1,1,1)** | 同上 |
| `GATED_DELTA_NET` | (6144,129,1,1) | 129 = 1 个 attn 行 + 128 个 state 行 ⇒ **K=1** |
| `SET_ROWS cache_k_l*/cache_v_l*` | (1024,256,1,1) | FA 的 KV 写入（16 层 ×2） |

"首个 pass 清零、稳态为空"这一条很关键（我用同一个 dump 抓了 4 个 pass 对比）：

```
[graph 0] SCALE ne=(30720,1,1,1) x48  SCALE ne=(786432,1,1,1) x48    ← 序列起点，真清零
[graph 1] 同上
[graph 2] SCALE ne=(0,1,1,1) x96                                     ← 稳态：空节点，零开销
[graph 3] 同上
```

结论：**不要在成本模型里算"每 token 清零 3 MB×48"**，它只在序列开头发生。

### 1.2 ncu 内核耗时（你的 `bonsai_A.csv`：400 个 launch，其中 10 个 GDN ⇒ 约 0.21 token）

（我按 CSV 的 **Average** 列重算；你原来的 `analyze_ncu.py` 取的是 `Minimum` 列（`parts[12]` 应为 `parts[14]`），
所以它报的 3.58 ms 偏低，真实是 **3.93 ms**。下面"占比"用采样窗口内的份额，"每 token"按 48 层折算。）

| 内核 | grid / block | 采样窗口内 | µs/次 | 份额 | 每 token（折算） | 备注 |
|---|---:|---:|---:|---:|---:|---|
| `mul_mat_vec_q<142,…>`（3 个变体） | — | 74 次 | 19–68 | **57.8%** | 10.9 ms | 权重侧，已接近带宽极限 |
| `fwht_cuda_block<1024,…>` | — | 54 | 4.1 | 5.5% | 1.05 ms | Hadamard |
| `rms_norm_f32`（2 个变体） | — | 43 | 3.8–7.5 | 6.7% | 1.27 ms | |
| `quantize_q8_1` | — | 53 | 3.0 | 4.2% | 0.79 ms | B1 已做过一轮 |
| `scale_f32` | (3072,1,1)/(120,1,1) | 22 | 9.55/3.27 | 3.6% | 0.68 ms | **只在首 pass，稳态为 0** |
| `cpy_scalar`（30720, grid 480） | (480,1,1) | 11 | 6.15 | 1.7% | 0.33 ms | conv 写回 |
| `cpy_scalar`（6144, grid 96） | (96,1,1) | 13 | 5.43 | 1.8% | 0.34 ms | 实为 `CONT` 节点（final_output/gate_reshaped；图里没有 6144 元素的 CPY 节点） |
| `gated_delta_net_cuda<128,0,0,1,0>` | (48,1,32)/(32,4,1) | 10 | 10.1 | 2.6% | 0.49 ms | 已含融合写回 |
| `k_get_rows_float_vec` | (1,768,1) | 11 | **8.45** | **2.4%** | **0.41 ms** | **s 状态 gather ← `src[6]` 目标** |
| `k_get_rows_float` | (1,120,1) | 11 | 4.16 | 1.2% | 0.20 ms | conv gather |
| `flash_attn_tile` + `combine` | — | 3+3 | 27.9/4.0 | 2.4% | 0.46 ms | |
| `concat_cont` / `ssm_conv_f32` / `l2_norm_f32` / `unary_silu` | — | 11/10/10/15 | 4.3/3.7/4.4/3.1 | 3.4% | 0.64 ms | |
| `k_set_rows<float,longlong,__half>` | (4,1,1) | 6 | 2.92 | 0.4% | 0.08 ms | FA KV 写入 |
| 其余（`mul_mat_vec_f`、`rope`、`bin_bcast`…） | — | — | — | 5% | |

窗口总耗时 3.93 ms；按 48/10 折算 ≈ 18.9 ms/token，比实测的 17.0 ms 高约 11%——
差异来自窗口里含**首 pass 的 0.14 ms 清零**和窗口边界效应，所以**看份额、看每层的 µs，不要直接用折算值**。
各条目的"每层 µs"是直接测量值：`8.45 µs × 48 = 0.41 ms`（占 17.0 ms 的 **2.4%**）这类算法是可用的。

> 旁证（写回是否真的融合）：`gated_delta_net_cuda` 每层 10.13 µs。融合形态只需搬
> 读 cache 行 3 MB + 写回 cache 行 3 MB（+ q/k/v/g 各 48×128×4 B ≈ 50 KB 与 attn 24 KB）≈ 6.1 MB
> → 600 GB/s，V100 上合理；若**未**融合，内核还要多写 3 MB 的 dst 快照尾部（≈9 MB）→ 应约 15 µs。
> 因此可以较有把握地认为写回已经融合；最终确认还是靠 `GGML_CUDA_DEBUG=1` 那一行日志。

### 1.3 我这次新做的 A/B：`GGML_CUDA_DISABLE_FUSION`

`llama-bench -m … -ngl 99 -fa 1 -p 512 -n 128 -d 0 -r 3`：

| 配置 | pp512 | tg128 |
|---|---:|---:|
| 融合开（默认） | 782.72 ± 23.4 | **58.13 ± 0.39** |
| `GGML_CUDA_DISABLE_FUSION=1` | 755.96 ± 23.9 | **52.16 ± 0.25** |
| 差 | −3.4% | **−10.3%** |

这个开关关掉的是 `ggml_cuda_try_fuse()` 里**全部**模式（含 GDN→cache、RMS_NORM+SCALE、RMS_NORM+MUL+ROPE+SET_ROWS、
FWHT 符号融合、multi add/mul、topk-moe…），是一个 bundle，**不能**把 10.3% 都算到状态路径上。
但它说明一件事：这套融合是承重的，**新写融合时不要破坏既有匹配**（尤其不要为实现 rows 把 GDN 后面那个 CPY 变成别的形状）。

### 1.4 与交接文档的三处差异（供后续文档更正）

| 交接文档的说法 | 实测 |
|---|---|
| "CUDA 只能走老路径，产出每步 192 个 `cache_s` 拷贝（≈1 ms/步，占 6%）" | 192 个 CPY 节点 = 48 真（`conv_state_update`）+ 96 空（`ne[1]=0`）+ 48（s 写回，运行时已融合/走 memcpy）。**没有** ~1 ms 的 3 MB 拷贝可省 |
| "todo#1 预计 5–9%" | `src[6]` 只覆盖 s 状态**读**（0.41 ms = 2.4%）；5–9% 需要连 conv 读/写回（2.9%）一起做 |
| "开关 `GGML_GDN_STATE_GATHER=1` 做 A/B" | 该开关只在 `n_rs_seq>0` 时才有语义；你现在的基准 `n_rs_seq=0`，关/开它**没有任何区别** |

---

## 2. 参考实现导读：CPU 与 Metal 的 rows 模式（本方案的语义来源）

### 2.1 CPU：`ggml/src/ggml-cpu/ops.cpp:11084-11315`（`ggml_compute_forward_gated_delta_net_one_chunk`）

```cpp
const ggml_tensor * src_rows = dst->src[6];                       // I32 [n_seqs] 行索引
const int32_t * state_rows_idx = src_rows ? (const int32_t *) src_rows->data : nullptr;
const int64_t state_seq_stride = src_rows ? 0 : (int64_t)(src_state->nb[3] / sizeof(float));
const int64_t state_row_size   = src_rows ? (int64_t)(src_state->nb[1] / sizeof(float)) : 0;
...
const float * s_in = state_rows_idx
    ? state_in_base + (int64_t) state_rows_idx[iv3] * state_row_size + iv1 * S_v * S_v
    : state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
memcpy(s_out, s_in, S_v * S_v * sizeof(float));                   // 之后完全在 scratch 上原地算
```

要点：

* `src[6]` 只是**行索引**（每 sequence 一个），rows 模式下行距 = `src_state->nb[1]/sizeof(float)` = `n_embd_s`；
  非 rows 模式下每 sequence 的行距是 `nb[3]`（gather 缓冲里 `[S_v,S_v,H,n_seqs]`）。
* 状态在**缓存行内是转置存储**：`s_out[j*S_v + i] = S[i][j]`，所以"列 j"是连续的（Metal/CUDA 都按这个顺序读）。
* 输出布局不变：`[attn_scores S_v*H*T*B] ++ [new_states S_v*S_v*H*B*K]`，slot 0 = 最新状态，`target_slot = n_tokens-1-t`。
* **CPU 图里没有 SET_ROWS**：反向写回是 `delta-net-base.cpp` 用 `ggml_set_rows(...)`（rows 模式）或 strided `ggml_cpy`（gather 模式）发出的，算子本身只管"读哪一行"。

### 2.2 Metal：`ggml-metal/kernels/gated_delta_net.metal:12-201` + `ggml-metal-ops.cpp:1969-2180`

```metal
// metal:52-55 —— rows 模式读
const uint state_seq_base = HAS_ROWS
    ? ((uint)((device const int *) rows)[i23]) * (uint)(args.ne21*S_v*S_v)   // 行距 = H*S_v*S_v = n_embd_s
    : (i23*args.ne21)*S_v*S_v;
const uint state_in_base = state_seq_base + i21*S_v*S_v + i20*S_v;

// metal:147-171 —— 写回（WRITE_ROWS，编译期常量）
device float * dst_state = dst + attn_size + target_slot*state_size_per_snap + state_out_base;  // 契约：op 自己的快照尾部照写
... dst_state[is] = ls[j];
if (WRITE_ROWS) {
    const uint64_t row = ((device const int64_t *) write_rows)[target_slot*args.ne23 + i23];
    device float * dst_rows = state_dst + row*(S_v*S_v*args.ne21) + i21*S_v*S_v + i20*S_v;
    dst_rows[is] = ls[j];                                             // 直接散进状态缓存
}
```

要点（这些直接决定 CUDA 版怎么写）：

* rows 读用 `int32` 索引、**行距 = `n_embd_s`**；`i21` 是头号、`i20` 是行号，行内布局同上。
* 写回用 **`int64` 行号** `write_rows[slot*n_seqs + seq]`；**即使融合了也仍然写 op 自己的快照尾部**（注释明说是为了其它消费者/回调不拿到未初始化区域）。
* K==1 分支同样支持 `WRITE_ROWS`（metal:183-193）。
* Metal host 侧 `ggml_metal_gdn_write_rows()`（ops.cpp:1969-2100）把 `GDN → SET_ROWS` 折进 epilogue：它反查 SET_ROWS 的 dst 视图、`write_rows`、`state_dst`，并保证"行不重叠"才融合（重叠时留给真正的 SET_ROWS 跑）。
* rows/write_rows 在 Metal 是**函数常量**（`FC_gated_delta_net_rows` / `..._write_rows`），所以两条分支各自编出干净的内核。

### 2.3 与 CUDA 现状的差异（本任务要补的就是这个缺口）

| 能力 | CPU | Metal | CUDA（今天） |
|---|---|---|---|
| rows 读（`src[6]`） | ✅ | ✅ | ❌（`supports_op` 显式拒绝；内核只认 gather 缓冲） |
| 写回融合 | —（用图节点） | ✅ `WRITE_ROWS`（写 2 份） | ✅ 形态不同：`ggml_cuda_try_gdn_cache_fusion()` 匹配 `GDN→CPY`，**只写 1 份**（直接写 cache，跳过 dst 尾部）——"运行时确实融合了"是从内核清单反推的，需 `GGML_CUDA_DEBUG=1` 构建确认 |
| `SET_ROWS` 写回 | ✅（图节点） | ✅ 折进 epilogue | ✅ 图节点会跑，但**没有**融合进 GDN |

---

## 3. 内核层改动（方案 A / B 共用，这是本任务的核心）

### 3.1 `ggml/src/ggml-cuda/gated_delta_net.cu`

现在（`gated_delta_net.cu:69-74`）：

```cpp
    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
```

改成（新增两个尾部参数，语义与 CPU `ops.cpp:11133-11139`、Metal `gated_delta_net.metal:52-55` 完全对齐）：

```cpp
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
__global__ void ... gated_delta_net_cuda(
        ...,
        int64_t state_slot_stride, int K,
        const int32_t * state_rows,   // NEW: 2D cache 行索引 [n_seqs]；nullptr = 老的 gather 缓冲
        int64_t         state_row_stride)  // NEW: 缓存每行浮点数 = n_embd_s (= S_v*S_v*H)
{
    ...
    // rows 模式：状态存在循环缓存里（2D，每行 D 个 float 连续），
    // 第 sequence 条序列的活状态在第 state_rows[sequence] 行。
    // 布局与 CPU/Metal 一致：行内 h 号头占 [h*S_v*S_v, (h+1)*S_v*S_v)，行内转置存储 M[col][i] = S[i][col]。
    const int64_t state_seq_off = state_rows
        ? (int64_t) state_rows[sequence] * state_row_stride
        : (int64_t) sequence * H * S_v * S_v;
    const int64_t state_in_offset = state_seq_off + h_idx * S_v * S_v;
    curr_state += state_in_offset;   // 之后 84-91 行的加载循环、写回逻辑都不用改
```

要点：

* `sequence = blockIdx.y`、`h_idx = blockIdx.x`（`gated_delta_net.cu:53-54`），所以 `state_rows[sequence]` 是**块内 uniform** 的一次标量读，没有 warp 分歧、没有额外寄存器压力（模板参数不用动，5 个变体的 cubin 数量不变）。
* `state_rows` 必须是 `const int32_t *`（`s_copy` 是 I32；Metal 也是 `(device const int *)`）。
* 写回侧**完全不动**：`keep_rs_t`（K>1）仍写 `state + target_slot*state_slot_stride`，K==1 仍写 `state`（`state` 此时要么指向 dst 尾部，要么指向 cache，取决于是否融合）。
* KDA 分支（`else` 那段）与 scalar-gate 分支共用同一个 `curr_state`，所以一处改动同时覆盖两种 gate。

### 3.2 host 侧：`ggml_cuda_op_gated_delta_net_impl()`

```cpp
    ggml_tensor * src_state = dst->src[5];   // rows 模式：2D cache view [D, n_rows]；否则：4D [S_v,S_v,H,n_seqs]
    ggml_tensor * src_rows  = dst->src[6];   // I32 [n_seqs] 或 nullptr
    ...
    const int32_t * rows_d            = nullptr;
    int64_t         state_row_stride  = 0;
    if (src_rows != nullptr) {
        GGML_ASSERT(src_rows->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src_rows));
        GGML_ASSERT(src_rows->ne[0] >= n_seqs && src_rows->data != nullptr);
        GGML_ASSERT(ggml_is_contiguous(src_state));
        GGML_ASSERT(src_state->nb[0] == sizeof(float));
        state_row_stride = (int64_t) (src_state->nb[1] / sizeof(float));   // 与 CPU 的 state_row_size 一致
        GGML_ASSERT(state_row_stride == S_v * S_v * H);                    // 布局假设，和 CPU/Metal 相同
        rows_d = (const int32_t *) src_rows->data;
    }
```

`launch_gated_delta_net(...)` 与 `GDN_LAUNCH` 宏各加两个参数直通即可（`gated_delta_net.cu:215-270`、`366-382`）。
**建议**：同时把 `GGML_USE_MUSA` 的情形挡在外面（MUSA 现在整个 op 都返回 false）。

### 3.3 性能注意

* rows 模式下内核少读一个中间缓冲（`state_predelta`），但**总 DRAM 字节数几乎不变**（3 MB 从 cache 行读 vs 从 gather 缓冲读）。收益来自"少一个内核"，不要指望内核本身变快。
* `cols_per_warp` / `warp_size` / `__launch_bounds__` 都不用动；`state_rows` 为空时编译器保留原分支，老的 gather 路径零风险。

### 3.4 为什么这条路天然逐字节一致

gather 路径是把 cache 行的 N 个 bit 用 `memcpy` 语义搬进 `state_predelta`，GDN 再按同样顺序读。
rows 路径让 GDN 直接读同一段内存的同一顺序 → **同一组 float 值、同一运算顺序**，因此输出应与老路径逐字节相同
（前提：索引张量取的就是 `s_copy_main`，行距 = `n_embd_s`）。这也是本任务验收条件可成立的原因。

---

## 4. 放行路径（两种方案，可叠加）

### 4.1 方案 A：图级 rows（交接文档的路线）

文档列的三处之外，**还必须补两处**，否则要么走不到、要么净亏：

| # | 位置 | 改动 | 为什么必须 |
|---|---|---|---|
| A1 | `ggml/src/ggml-cuda/ggml-cuda.cu:5962` | `src[6] != NULL` 时改为调用新的 `ggml_cuda_gated_delta_net_rows_supported(dev, op)`（校验 I32/连续/`nb[1]` 行距/`S_v∈{16,32,64,128}`），而不是 `return false` | 放行给 CUDA |
| A2 | `src/models/qwen35.cpp:141-157` | 现在是 `is_gpu && reg_name != "MTL" → gdn_state_rows_dev_ok = false`；改成允许 `"CUDA"`（或做一次 `ggml_backend_dev_supports_op` 探针） | 否则 `gdn_state_rows` 恒 false |
| A3 | `src/models/delta-net-base.cpp:556` | `GGML_ASSERT(state_rows == nullptr || keep)` — `keep = cparams.n_rs_seq > 0`。**K=1（无 rollback）时现在是断言死路**，要放开才能覆盖你的基准配置 | 不然 `n_rs_seq=0` 时仍然只能 gather |
| A4 | `delta-net-base.cpp:558-570`（`!keep` 分支） | 走 rows 时不要再用 `ggml_cpy(new_state → cache)`，改成 `ggml_set_rows(ssm_states_all, snaps_view, write_rows)`（`build_rs_write_rows(inp, K=1, n_tokens, n_seqs)`） | 否则既读 cache 又写 dst 再拷贝，多一次 3 MB |
| A5 | `ggml-cuda.cu` + `gated_delta_net.cu` | 给 `ggml_cuda_gated_delta_net_fused_cache` 加 `int64_t * write_rows`，把 rows 模式的 `SET_ROWS` 也融进 GDN epilogue（Metal `WRITE_ROWS` 的等价物：`row = write_rows[target_slot*n_seqs + seq]`，写 `cache + row*(S_v*S_v*H) + h*S_v*S_v + i`） | K>1（draft 场景）下若不融合，rows 会 **多** 48 次 SET_ROWS + 双写 12 MB/层，净亏 |

代价与收益（K=1，d=0）：省 `GET_ROWS` 8.45 µs/层 = **+2.4%**；A4 不做则 −6 µs/层左右（SET_ROWS + 多半次 3 MB），A5 不做在 K>1 时净亏。
改动面：跨 `src/models/`（`qwen35.cpp`、`delta-net-base.cpp`）与 `ggml/src/ggml-cuda/`；而 `build_recurrent_attn`
被 6 个架构共用（qwen35 / qwen35moe / qwen3next / qwen4exp / kimi-k3 / bailingmoe3），回归面比方案 B 大得多。

### 4.2 方案 B（推荐先做）：后端内 `GET_ROWS → GATED_DELTA_NET` 融合

思路：**图不变**，在 CUDA 后端把"每层的状态 gather + GDN"识别成一个融合模式，跳过 gather 内核，把行索引直接喂给 GDN 内核。
索引张量完全相同（`build_rs` 传给 `get_state_rows` 的就是 `inp->s_copy_main`，与 `src[6]` 是同一个张量），所以**语义等价、逐字节一致**，
但**不需要**动 `qwen35.cpp` / 后端能力查询 / `supports_op` / `delta-net-base.cpp`，并且 `n_rs_seq=0` 和 `>0` 都生效。

落地要点：

* **per-pass 容器**（照 Metal `fused_set_rows` 的做法，`ggml-metal-ops.cpp:78-83`）：
   `std::unordered_set<const ggml_tensor *> gdn_state_read_elided;`，在 `ggml_cuda_q8_1_cache_begin()` 同一个位置 `clear()`（`ggml-cuda.cu:4629` 附近）。
* **在计算主循环里加一个锚点**（`ggml-cuda.cu:4739` 之前，那里已经有 GB10 的 `i += 2; continue;` 先例）：

```cpp
// GDN rows-read: 每层一次 3 MB 的 get_rows 只是把 cache 行搬到 scratch，
// GDN 自己按行读更省一次内核；把索引和行距记下来，等 GDN 节点消费。
if (node->op == GGML_OP_GET_ROWS && !is_concurrent_event_active) {
    ggml_cuda_gdn_rows_read_match m;
    if (ggml_cuda_match_gdn_rows_read(cgraph, i, m)) {     // 见下
        cuda_ctx->gdn_state_read_elided.insert(node);      // 标记：本 pass 不执行这个节点
        i += 0;                                            // 只跳过自己，后续节点照常
        continue;
    }
}
```

  匹配条件（`ggml_cuda_match_gdn_rows_read`）：
  * `node->op == GGML_OP_GET_ROWS`、`node->type == GGML_TYPE_F32`、`src[1]->type == GGML_TYPE_I32`、`src[1]->ne[0] == n_seqs`、`src[1]` 连续且 `data != nullptr`；
  * `src[0]` 是 `cache_s_l*` 的 2D 视图/reshape：`ne[0] == n_embd_s`、`nb[1] == n_embd_s*sizeof(float)`；
  * **唯一的消费者**是一路 view/reshape 到某个 `GATED_DELTA_NET` 的 `src[5]`（扫 `i+1..n_nodes`，确认没有别的节点引用 `node`；中间只允许 VIEW/RESHAPE/PERMUTE）；
  * `ggml_cuda_check_fusion_memory_ranges(...)`（或至少确认 GDN 的 `dst` 与 `src[0]` 不重叠）。
* **GDN 执行时消费标记**：`ggml_cuda_op_gated_delta_net_impl()` 里沿 `src[5]` 的 view 链往上找到那个 `GET_ROWS`，若它在 `gdn_state_read_elided` 里 → `rows_d = (int32_t*)getrows->src[1]->data; state_row_stride = getrows->src[0]->nb[1]/sizeof(float);` 并走第 3 节的内核路径。找不到就照旧。
* **A/B 开关**：`GGML_CUDA_GDN_ROWS_READ=0` 关闭该模式（不要复用 `GGML_CUDA_DISABLE_FUSION`，会连带关掉别的融合，污染对比）。
* 与既有 `ggml_cuda_try_gdn_cache_fusion()`（GDN→CPY 写回融合）**天然共存**：一个省读、一个省写，互不干扰。

方案 B 的风险面：只在 `ggml-cuda/` 内部（约 80–120 行），不碰模型代码、不碰 CPU/Metal；唯一"非典型"的地方是它是个**向后**融合（取数节点在前、消费节点在后），需要那个 elided 集合 + 主循环的一个分支。

### 4.3 两条路的对比

| | 方案 A（图级 `src[6]`） | 方案 B（后端融合） |
|---|---|---|
| 改动文件 | `ggml-cuda.cu`, `gated_delta_net.cu(h)`, `qwen35.cpp`, `delta-net-base.cpp`(+`llama-graph.h`) | `ggml-cuda.cu`, `gated_delta_net.cu(h)` |
| 生效配置 | 需 `n_rs_seq>0`（且要额外放开 A3 才能覆盖 `n_rs_seq=0`） | `n_rs_seq = 0 / >0` 都生效 |
| d=0 收益 | +2.4%（K=1）/ K>1 需 A5 才不亏 | **+2.4%** |
| 其它后端回归风险 | 有（共享 `build_recurrent_attn`，6 个模型架构） | 无 |
| 上游对齐 / 可维护性 | 好（CPU/Metal 同构） | 弱（本地实验性融合） |
| 建议 | 第二优先（或只做 A1+A2 备用） | **第一优先** |

---

## 5. 正确性验收

1. **逐字节 A/B**（你的既定标准）：
   ```
   # 新路径（默认）
   llama-cli -m <gguf> -ngl 99 -fa 1 -c 4096 -p "<固定 prompt>" -n 256 --temp 0 --seed 7 -st > new.txt
   # 老路径
   $env:GGML_CUDA_GDN_ROWS_READ='0'   # 方案 A 则用 GGML_GDN_STATE_GATHER=1
   llama-cli ... > old.txt
   fc.exe /b old.txt new.txt
   ```
   两条路径读的是同一段内存、同一顺序 → 期望**完全一致**；若不一致，先查"索引张量是否就是 `s_copy_main`"和"行距是否 `n_embd_s`"。
2. **数值面**：再跑一次 `llama-perplexity`（同一小文本）对比 PPL 小数位，作为第二道闸。
3. **内核数验证**（不需要 admin）：用 `GGML_V100_DUMP_NODES=1` 对比前后节点表（应当**完全不变**，方案 B 不改图）；再在 `GGML_CUDA_DEBUG=1` 的调试构建里确认打印/计数里 `k_get_rows_float_vec` 每层少了 1 次、`gated_delta_net_cuda` 不变。
4. **边界用例**（这次要一并测）：
   * 多序列（`-np 2` 或 `llama-batched-bench`）：`rows[seq]` 逐序列索引，`n_seqs>1`；
   * 首 token / 序列切换（`rs_z` 保护那次清零还在，注意 A4 改动后不要把它删掉）；
   * `n_tokens>1`（预填 / draft 校验批）：K=1 与 K>1 两条写回；
   * 缓存回绕：`-c` 小到触发 cell 复用后仍与老路径一致；
   * （如果有 draft 模型）`n_rs_seq=3, K=4` 的 rollback 路径。

---

## 6. 基准命令与预期

```powershell
# 基线复现（我实测融合开 58.13 / 关 52.16，和你文档的 58.89 同量级）
llama-bench -m <gguf> -ngl 99 -fa 1 -p 512 -n 128 -d 0     -r 3
llama-bench -m <gguf> -ngl 99 -fa 1 -p 512 -n 128 -d 90112 -r 3

# 方案 A 的 A/B（注意：必须让 n_rs_seq>0 才有意义）
$env:GGML_GDN_STATE_GATHER='1'   # 回老路
```

**怎么得到 `n_rs_seq > 0`**：`n_rs_seq` 只由投机解码类型驱动，Bonsai 2 又没有 MTP 头。三条路：
1. `--spec-type draft-simple --spec-draft-model <draft.gguf> --spec-draft-n-max 3`（真实验收路线，需要一个 draft 模型）；
2. 构建一个**临时调试开关**（例如在 `common.cpp:1722` 后加 `if (getenv("GGML_FORCE_RS_SEQ")) cparams.n_rs_seq = atoi(...)`），用来做纯 `llama-bench` 的 A/B——只用于实验，不进最终提交；
3. 干脆接受"方案 B 覆盖 `n_rs_seq=0`"，不做 A 的 K>1 验证。

预期（d=0；以文档基线 58.89 t/s = 16.98 ms/token 起算，省下的 ms 直接相加）：

| 做完 | 每 token 省 | 累计 ms/token | t/s | 相对 |
|---|---:|---:|---:|---:|
| 方案 B（s 读） | 0.41 ms | 16.57 | 60.3 | **+2.4%** |
| + conv 读/写回 | 0.49 ms | 16.08 | 62.2 | +5.3% |
| + CONT/PERMUTE（todo#5） | 0.34 ms | 15.74 | 63.5 | +7.3% |
| （K>1 场景另有 SET_ROWS 融合的收益，待 draft 环境实测） | | | | |

---

## 7. 风险清单

| 风险 | 说明 | 处置 |
|---|---|---|
| CUDA Graph 捕获 | `graph reused = 31` 说明解码走图重放；elided 集合必须 per-pass 清理，且融合决策在捕获时一次性确定（指针稳定） | 在 `q8_1_cache_begin` 同点 `clear()`；先在 `GGML_CUDA_DISABLE_GRAPHS=1` 下验证 |
| 索引张量落在 host/pinned 内存 | `llama-graph.cpp:379` 断言 `s_copy` 在 host buffer；但 GET_ROWS 今天已经把它当 device 指针读，说明 UVA 下可读 | 与今日 GET_ROWS 同一约束，不新增风险；仍保留 `data != nullptr` 断言 |
| 多消费者 | 万一 `state_predelta` 还被别的节点读，跳过 gather 就会读到垃圾 | 匹配时强制"唯一消费者 + 中间只允许 view/reshape/permuate" |
| 破坏既有写回融合 | A4 把 `CPY` 换成 `SET_ROWS` 会让 `ggml_cuda_try_gdn_cache_fusion` 失配（1.3 节显示融合 bundle 值 10.3%） | 方案 A 必须先落 A5；或干脆选方案 B |
| `n_embd_s == S_v*S_v*H` 假设 | 其它架构（kimi-k3 / qwen4exp / bailingmoe3）共用 `build_recurrent_attn` | 内核/匹配里加断言，不成立就走老路 |
| MUSA/ROCm | 后端差异 | `src[6]` 放行时保持 MUSA false；ROCm 走原有 `GGML_CUDA_CC_IS_NVIDIA` 判断 |
| 其它后端一致性 | 方案 A 改了共享图代码，CPU/Metal 也在跑 | 每次改完用 CPU（`-ngl 0`）跑同一 prompt 对比输出 |

---

## 8. 需要你定的三件事

1. **目标配置**：只优化 `n_rs_seq=0` 的纯解码（方案 B，+2.4% 起步），还是要同时覆盖 `n_rs_seq>0` 的 draft/MTP 场景（那就要 A5）？
   —— 目前 Bonsai 2 PQ2_0 没有 MTP 头，`draft-mtp` 无法启动；要走投机得准备一个 `draft-simple` 的草稿模型。
2. **是否接受"方案 B（本地后端融合）"作为第一步**？它今天就能在 `-ngl 99 -fa 1` 基准上生效，且不改模型/共享代码。
3. **5–9% 的口径**：是否接受把它拆成 s 读（2.4%）+ conv 读/写回（2.9%）+ CONT/PERMUTE（2.0%）三步交付（每步都可独立 A/B，逐字节验收）？

定了我就按顺序动手（先内核层 + 方案 B，然后把 conv 那条也做掉）。

---

### 附：本次用到的可复现脚本（都在 `work/`）

| 文件 | 用途 |
|---|---|
| `work/dump_graph.py` | 解析 `GGML_V100_DUMP_NODES` 输出，按 op/ne 统计节点（本报告 1.1 的表） |
| `work/scan_graphs.py` | 逐 pass 对比节点 `ne`（发现"清零只在首 pass"） |
| `work/ncu_detail.py` | 解析 `ncu --print-summary per-kernel --csv`，按 kernel/block/grid 汇总（本报告 1.2 的表） |
| `work/ncu_totals.py` | 同上的按内核汇总 + 总耗时（用 Average 列） |
| `work/tg_dump.log` / `work/mtp_dump.log` | 本次抓的节点 dump（decode / MTP 失败现场） |

> 顺手修一个旧脚本的坑：`<work-old>/analyze_ncu.py` 用 `parts[12]` 当平均耗时，
> 但 CSV 表头里 12 是 `Minimum`、14 才是 `Average` → 它报的总量（3.58 ms）偏低，真实是 3.93 ms。

---

## 9. 实施结果（2026-09-27，已编译 + 已验收）

按 §4.2 **方案 B** 落地（内核层 rows 读 + 后端内 `GET_ROWS → GDN` 融合），未改 `qwen35.cpp` / `supports_op` /
`delta-net-base.cpp`（那些属于方案 A，仍未动）。改动 4 个文件、全部在 `ggml-cuda/` 内：

| 文件 | 改动 |
|---|---|
| `ggml-cuda/gated_delta_net.cu` | 内核新增 `state_rows` / `state_row_stride`：`state_seq_off = rows[sequence]*stride`（否则仍 `sequence*H*S_v*S_v`）；launcher 直通；`impl` 里支持 src[6]（方案 A 用，暂未启用）与"融合 rows"两条来源，并把读指针从被跳过的 scratch 改到 cache 基址；尾部有 env 门控的调试探针 |
| `ggml-cuda/gated_delta_net.cuh` | `ggml_cuda_gated_delta_net_fused_cache` 增加 `state_base` / `state_rows` / `state_row_stride`，并补默认初始化（原来 `data`/`slot_stride` 是未初始化的） |
| `ggml-cuda/ggml-cuda.cu` | 新增 `ggml_cuda_match_gdn_state_read()`（匹配条件见 §4.2）+ `ggml_cuda_concurrent_region_overlaps()`；主循环加两个锚点（取数节点跳过、GDN 消费并折叠写回）；每 pass 清空容器；并发区域位图；pass 结束校验"每个被跳过的取数都被消费" |
| `ggml-cuda/common.cuh` | 新增 `ggml_cuda_gdn_state_read` 结构体与 `ggml_backend_cuda_context::gdn_state_read` per-pass 容器 |

开关：`GGML_CUDA_GDN_ROWS_READ=0` 关闭（A/B）；`GGML_CUDA_GDN_ROWS_DEBUG=1` 打印每次融合与探针；
`GGML_CUDA_GDN_ROWS_NO_ELIDE=1` 只调试用（保留取数、仍走 rows 读，便于比对两个缓冲）。

### 9.1 验收（按你加的约束：两种图模式 + 逐字节）

| 项 | graphs ON (`GGML_CUDA_DISABLE_GRAPHS=0`) | graphs OFF (`=1`) |
|---|---|---|
| d=0、贪心 192 token，生成文本 sha256（去掉时延行） | `2bb3265c747dcd0b…` 两侧一致 ✓ | `2bb3265c747dcd0b…` 一致 ✓ |
| 32K prompt、贪心 32 token，文本 sha256 | `169bab47ab96fd9c…` 一致 ✓ | `169bab47ab96fd9c…` 一致 ✓ |
| `llama-perplexity`（1536 token，3 chunk） | `[1]1.8639,[2]1.8376,[3]1.7776` 两侧一致；剔除时间戳日志后输出 diff = **0 行** ✓ | 同上 ✓ |
| 融合覆盖率（`GGML_CUDA_GDN_ROWS_DEBUG`） | 每 pass **48/48** 层融合（48 个 GDN 全部命中） ✓ | 48/48 ✓ |
| 多流安全 | 实测该 decode 图上并发区域数 = 0（`concurrent region(s)` 未出现）；守卫按 [取数, GDN] 区间判定，一旦跨区域就放弃融合（`kept because they cross a concurrent region`），本次未触发 | 同 |

### 9.2 性能（同机、同模型、`-r 3`，与 gather 路径对比）

| 深度 / 图模式 | rows（新路径） | gather（`GGML_CUDA_GDN_ROWS_READ=0`） | Δ |
|---|---:|---:|---:|
| d=0 解码，graphs ON | **62.00** | 59.80 | **+3.7%** |
| d=0 解码，graphs OFF | **61.81** | 59.74 | **+3.5%** |
| 90K 解码，graphs OFF | **38.67** | 38.09 | **+1.5%** |
| d=0 预填，graphs ON/OFF | 807.4 / 807.5 | 807.7 / 808.6 | 噪声（±0.15%） |
| 90K 预填，graphs OFF | 496.08 | 494.83 | 噪声 |

比 §6 预估的 +2.4% 略好（实测 +3.5%），与"省掉每层 8.45 µs 的 3 MB gather"的量级一致。

### 9.3 落地过程中踩到并修掉的一个真实坑（供你 review 时留意）

第一版实现里内核读的基址仍取自 `dst->src[5]->data` —— 在**图级 rows 模式**下 src[5] 就是那张 cache 视图
（所以 CPU/Metal 的写法没问题），但在**方案 B** 下 src[5] 是那个**已被跳过**的 gather scratch，于是内核读到的是
未初始化显存：输出不崩但语义错（我第一轮 A/B 就抓到了：文本逐字节比对失败，且 llama-cli 报了格式错）。
修法是让匹配器把**取数源**的基址（`gr->src[0]->data`）一并交给 GDN（`state_base`），并把 `s_d` 指过去；
修完后 `GGML_CUDA_GDN_ROWS_NO_ELIDE=1` 的探针显示两个缓冲的探针和逐位吻合（`scratch_sum == cache_sum`）。

教训（也是你加的那条验收要求的价值）：**只有逐字节比对才能抓住这类"能跑但读错内存"的问题**，
"文本通顺"完全抓不到 —— 第一版跑 `llama-bench` 时性能还"变快"了 3%，指标一切正常。

### 9.4 仍未做 / 建议下一步

1. **多序列（`n_seqs>1`）的逐字节验收**：内核按 `state_rows[sequence]` 逐序列取行，逻辑上通用，但本轮只测了单序列；
   建议用 `llama-batched-bench -np 2,4`（或 server 并发两路）跑一遍，仍与 `GGML_CUDA_GDN_ROWS_READ=0` 对比。
2. **方案 A（图级 `src[6]`）**：内核与 `impl` 已经支持，但还没有放行 `supports_op`/`qwen35` 门控，
   也还没有 §4.1 的 A3/A5（K=1 断言、`SET_ROWS` 写回融合）—— 在拿到 draft 模型跑 `n_rs_seq>0` 之前不建议启用。
3. **conv 状态那条（todo#2，约 +2.9%）**：`GET_ROWS conv_states`(4.16 µs) + `CPY conv_state_update`(6.09 µs) 每层，
   同一套"按行读 + 融合写回"的思路可以照搬（consumer 是 `SSM_CONV` 而不是 `GDN`，匹配器要新写一个）。
4. **`CONT`（final_output/gate_reshaped，约 +2.0%）**：每步 62 次 6 KB 的 cpy，属于 todo#5。

复现脚本（都在 `work/`）：`build_incremental.ps1`（vcvars+ninja，日志写回 `build-v100\build-log.txt`）、
`cc_check.ps1`（用 compile_commands.json 单文件语法检查）、`ab_matrix2.ps1`（两种图模式 × 正确性 + 性能）、
`verify_final.ps1`（文本哈希 + PPL 哈希复核）。

---

## 10. 第二步：delta-net conv 状态融合（2026-09-28，已编译 + 已验收）

todo#2 落地：把每层 conv 状态那三个小内核（`GET_ROWS conv_states` 4.16 µs + `CONCAT conv_input` 4.29 µs +
`CPY conv_state_update` 6.09 µs ≈ 14.5 µs/层）合成**一个 launch**。

| 文件 | 改动 |
|---|---|
| `ggml-cuda/concat.cu` | 新增 `conv_state_update` 内核 + `ggml_cuda_op_conv_state_update()`：读 cache 行活状态 → 与 qkv^T 拼成 `conv_input`（SSM_CONV 的输入）→ 把新状态（`conv_input` 的最后 `conv_kernel-1` 行，即行区间 `[T, T+KS)`）散回同一 cache 行。两个输出都由输入直接算出，无内核内依赖 |
| `ggml-cuda/concat.cuh` | 声明该 op |
| `ggml-cuda/ggml-cuda.cu` | 新增 `ggml_cuda_match_conv_state_update()`：在 `GET_ROWS`（conv 状态取数）处标记并在 `CONCAT` 处消费、跳过中间的 2 个 view + 写回 `CPY`；`CONCAT` 锚点执行融合 |
| `ggml-cuda/common.cuh` | per-pass 容器改成通用的 `ggml_cuda_state_rows_read` / `state_rows_read`（同时服务 GDN 与 conv 两条），加 `dst_view` / `skip` 字段 |

开关：`GGML_CUDA_CONV_STATE_FUSION=0` 单独关掉这一步；`GGML_CUDA_GDN_ROWS_READ=0` 关掉第一步。
于是有三种可对比配置：`all`（两个融合都开）/ `gdn`（只开第一步）/ `none`（全部回退到 gather 老路）。

匹配条件（刻意保守）：唯一消费者链 → `CONCAT(dim 0)`；`CONCAT` 后第一个真节点必须是那个 strided `CPY`；
`CPY` 的源必须是 `conv_input` 的"最后 KS 行"视图、目的必须**是同一个 cache 张量**（行距 = `n_embd_r`）；
**K>1（rollback 环）会出现多个这样的 CPY，直接放弃**；取数与 `CONCAT` 之间任何非空节点碰 cache 也放弃。

### 10.1 验收（与 §9 同一套口径）

| 项 | 结果 |
|---|---|
| d=0 贪心 192 token，`all` / `gdn` / `none` 三种配置 × graphs ON/OFF | 六个 sha256 **全部相同**（`0d1e23e50f920cd1…`） ✓ |
| 融合覆盖（`GGML_CUDA_GDN_ROWS_DEBUG=1`，4 个 pass） | gdn 192 次 + conv 192 次 = **48+48 层/ pass**，`unconsumed=0` ✓ |
| 多序列（llama-server `-np 2`，两路并发贪心） | 2-seq 批次 1392–1488 个；两路文本与 gather 路径**逐字节一致**（graphs ON/OFF 各一次） ✓ |

### 10.2 性能（graphs OFF，`-r 3`）

| 配置 | d=0 tg128 | 90K tg128 | pp512 |
|---|---:|---:|---:|
| `none`（全 gather） | 59.95 | 38.11 | 806.64 |
| `gdn`（只做第一步） | 61.72（**+3.0%**） | — | — |
| `all`（两步都做） | **62.36（相对 gather +4.0%）** | **39.26（+3.0%）** | 805.64（±0%） |

（同一轮的另一次直接对比里 `all` = 63.10 vs `none` = 59.72 → **+5.7%**；两次的差值主要来自机器抖动，
报告里用哪一次都行，但**两处都是正的**。预填另测 pp8192：`all` 804.40 vs `none` 797.82 = **+0.8%**。）

### 10.4 收尾时改掉的两处（都在同一轮 A/B 里被抓到）

1. **状态行布局对齐规范**：第一版融合内核用自己一套（[r][c]，tap 在外）的读写约定 —— 模型输出**逐字节一致**
   （因为读和写自洽），但 cache 行的物理布局与图（strided CPY + reshape 得来，`cache[c*KS + r]`，tap 在内）
   不一致，会在 **state 保存/加载、跨配置（融合开/关）互操作**时被误读。已改成规范布局并重新验收：文本仍逐字节一致，
   且与非融合路径写出同一份 cache 内容。
2. **预填回退 → 已彻底修好**：第一版融合内核按 (row, c, b) 顺序铺线程，与张量"dim0 在最内"的显存顺序相反 →
   写和读全部非合并，T=512/8192 时 pp 掉 0.4–1.2%。改成**按各张量自身的内存顺序铺线程**（dim0 最内）后：
   拼接写、状态散回写、活状态读全部合并访存 → 门控 `T <= 8` 已删除，融合对长预填同样生效，
   pp8192 = **+0.8%**（原来 −0.65%）、pp512 打平。这个修复也说明：一开始那 −0.8% 不是"三内核更优"，
   而是我自己把访存搞歪了。

### 10.3 这一步踩的坑（都记下来给以后）

1. `conv_state_last` 那个视图**不连续**（它的行跨在父张量的平面上），所以不能按"连续切片"校验 —— 第一版就是死在这个判断上；
2. 该视图的**偏移是元素级** `T * sizeof(float)`，不是行级 `T*C*B*sizeof(float)`（`ggml_view_3d` 的 offset 只沿 dim0 走 s_idx 个元素）；
3. 匹配器"为什么没匹配"必须能自证：给 matcher 加了 `const char ** why` 诊断参数（env 门控打印），
   否则只能靠"融合没生效"这种间接现象猜 —— 这次正是靠它一句话定位到上面两条。

---

## 11. 第三步：todo#5（`CONT`）—— 查清了，也**证伪了**预期收益（2026-09-28 晚）

### 11.1 先把 `CONT` 的来源查清（工具：dump 增强 + `work/analyze_cont.py`）

给 `GGML_V100_DUMP_NODES` 的 dump 加了 `[cuda-src]` 行（每个节点列出 src 的 op/名字/形状），
于是不用猜接线。解码图里每 pass **64 个 `CONT`**，两类：

| 类别 | 数量 | 链路 | 出处 |
|---|---:|---|---|
| `final_output-*` | 48 | `PERMUTE(128,3,16) → CONT → RESHAPE(6144) → MUL(signs) → MUL_MAT`（FWHT/Prism Hadamard 旋转，MUL 已被既有 FWHT-sign 融合吃掉） | `llama-graph.cpp:1566-1571`（`build_lora_mm` 的 Hadamard 分支） |
| `gate_reshaped-*` | 16 | `VIEW(Qcur_full) → CONT(6144) → UNARY(sigmoid) → MUL(attn_out)` | `qwen35.cpp:353-358`（FA 层 QG 联合投影的 gate 半边） |

### 11.2 实现的融合与**负结果**

给第二类做了 `CONT → UNARY(SIGMOID)` 融合（`unary.cu` 的 `unary_strided_kernel` + 匹配器/锚点，
开关 `GGML_CUDA_CONT_SIGMOID`，**默认关**）：

* 命中正常：16 层/pass 全部融合、逐字节一致（两种图模式、192 token）；
* 但**解码掉 0.6–0.9%**（62.58 vs 62.96；另一轮 62.68 vs 63.23，两次可复现），
  而且第一次实现里每元素两次 int64 除法，改成"按行分块免除法"后**仍然是负的**。

### 11.3 为什么是负的（这次学到的关键一条）

1. **它会抢占既有的 `UNARY+MUL` 融合**（`ggml-cuda.cu:4836` `ggml_cuda_can_fuse(cgraph, i, {UNARY, MUL})`）：
   我在 `CONT` 节点先把 sigmoid 执行掉，后面的 `sigmoid×attn_out` 就再也合并不了，净亏。
2. 成本模型本身也要修正：`CONT` 每个只搬 24 KB，5.4 µs 里几乎全是 launch/teardown；
   而**这一版先把 sigmoid 从既有 `UNARY+MUL` 融合里抢走**（第 1 条），两项相加就是负的。
   （更正：当天早些时候我把这归因于 PDL 重叠 —— 那是错的：**PDL 在本树要求 `__CUDA_ARCH__ >= 90`，
   在 sm_70/V100 上是空操作**，见 `common.cuh:134`。)

### 11.4 对 todo#5 的结论与下一步方向

* 第一类（`final_output-*`，48 个，**仍是最值得做的下一步**）：它和 sigmoid 那条不同 ——
  没有"抢占既有的 UNARY+MUL 融合"问题（它的下游是既有 FWHT-sign 融合），而 PDL 在 sm_70 是空操作，
  所以这 48 个 5 µs 级 launch 是**真的可加**的 → 预期 **+1~1.5%**。
  做法：让 `fwht_cuda_block` 支持"带步长的源"（按 dim0 最内的映射读 `PERMUTE` 后的张量），
  匹配 `[PERMUTE → CONT → RESHAPE → MUL(signs) → RESHAPE → MUL_MAT(HADAMARD)]` 时跳过 CONT，
  把步长交给既有 `ggml_cuda_op_fwht_signed`。改动集中在 `fwht.cu` + 一条匹配臂，可逐字节验收。
* 真要继续提速，应该盯**有实际数据量**的地方（与本次两类成功案例一致：s 状态读 3 MB/层、
  conv 三合一 287 KB/层）。剩下的候选：
  1. `qkv_mixed` 的 `TRANSPOSE`（48/步，`delta-net-base.cpp:475`）——它本身是 view（零开销），
     但它逼着 `CONCAT` 走 strided 读；可考虑把转置并进 conv 融合内核（共享内存 tiling），顺带把
     `conv_input` 的拼接也省一层；
  2. `PERMUTE`（96/步：KV cache 布局 + `final_output`）与 `CONT final_output` 一起处理，收益需先测；
  3. 更值得做的是**权重侧**：ncu 里 `mul_mat_vec_q<142,…>` 仍占 ~58%（10.9 ms/token 折算），
     而 RS/状态路径那 4.7% 已经吃完了。

### 11.5 本轮结束时的最终数字（graphs OFF，`-r 3`）

| 配置 | d=0 tg128 | pp512 |
|---|---:|---:|
| 默认（s 状态行读 + conv 三合一） | **62.96** | 809.70 |
| 全关（`GGML_CUDA_GDN_ROWS_READ=0`） | 60.13 | 810.34 |
| 再加 `GGML_CUDA_CONT_SIGMOID=1` | 62.58（−0.6%） | 810.58 |

→ 相对全程 gather：**解码 d=0 +4.7%、90K +3.0~3.4%、预填 ±0%（pp8192 曾测到 +0.8%）**；
逐字节一致（两种图模式 × 三种配置 × 192 token，以及多序列 server 2-seq）。

---

## 12. 第四步：把 fork 的 `[ADD, RMS_NORM, MUL]` 融合在 Volta 上打开（2026-09-28 晚，已验收）

### 12.1 怎么发现的

用增强版 dump + `work/count_add_norm.py` 数解码图：每 pass **128 个 `ADD`，全部是"残差相加 → 归一化"的形状**
（`attn_residual-*` / `l_out-*`），其中 **96 个的后继正好是 `[ADD → RMS_NORM → MUL(weight)]`**，
而且权重形状满足融合条件（`ne[0] == 行宽`、单行）。

树里**早就实现了**这个融合（`ggml_cuda_op_add_rms_norm_fused`，注释写着"保留残差和给后续消费者、同一次
launch 里完成归一化"），但匹配臂被 `cc == GGML_CUDA_CC_DGX_SPARK` 门住，Volta 用不到。

### 12.2 改动与逐位一致性的论证

* 把该门放宽为 `cc == DGX_SPARK || cc == VOLTA`（开关 `GGML_CUDA_ADD_RMS_NORM=0` 可关）；
* 但**原臂的 `ggml_cuda_check_fusion_memory_ranges()` 在残差复合形状上会拒绝**（它要求融合窗口的输出
  不被窗口外的节点读取，而残差 ADD 的输出本来就要留给后面用）—— 所以另加了一条条件等价、但不做
  内存范围检查的匹配臂：融合内核**写出的正是原两个算子写的那两块内存**（残差和 + 归一化乘权重），
  读的也只有原算子的输入，`ggml_can_fuse_subgraph()` 仍然保证中间结果（RMS_NORM 输出）在窗口外没有消费者。
* 逐位一致的依据（都核对过源码）：`add_rms_norm_f32` 与 `rms_norm_f32` 的**逐线程累加顺序和
  `block_reduce<SUM>` 归约顺序完全相同**；残差用 `__fadd_rn` 落盘，与独立 ADD 算子一致；
  乘法则满足 `(scale*sum)*weight == (sum*scale)*weight`（IEEE 乘法可交换）。

### 12.3 验收与收益

* 命中：**96 层/pass**（`GGML_CUDA_GDN_ROWS_DEBUG=1` 计数），逐字节比对 on/off 在两种图模式下**全同**；
* 性能（graphs OFF）：d=0 `63.34 vs 62.85`（**+0.78%**）、pp512 `813.5 vs 807.2`（**+0.78%**）、
  90K `40.02 vs 39.97`（+0.13%，另测一轮 +0.4%）。
  最终整体数字（见 §12.4）：d=0 **63.56 vs 60.29 = +5.4%**、90K **39.86 vs 37.95 = +5.0%**、
  pp512 中性。

### 12.4 本轮收尾时的整体账（graphs OFF，`-r 5`/`-r 3`）

| | 默认（四个融合全开） | 全关（`GGML_CUDA_GDN_ROWS_READ=0`） | Δ |
|---|---:|---:|---:|
| d=0 解码 tg128 | **63.56** | 60.29 | **+5.4%** |
| 90K 解码 tg128 | **39.86** | 37.95 | **+5.0%** |
| d=0 预填 pp512 | 811.45 | 811.66 | ±0% |
| 90K 预填 pp512 | 497.76 | 496.33 | +0.3% |

四个融合的贡献阶梯（d=0）：GDN 行读 ≈ +3.0~3.3%、conv 三合一 ≈ +2.0%、`ADD+RMS+MUL` ≈ +0.4~0.8%、
`CONT→sigmoid`（默认关）=−0.6%。**这已经落在最初 5–9% 目标区间的下半段**，而且这四条全部通过
逐字节验收（两种图模式 + 多序列 + PPL 交叉验证）。

### 12.5 一条流程教训（值得写下来）

排查 `[ADD,RMS_NORM,MUL]` 时我有两次"命中 0"，最后发现是**我的构建脚本输出被截断、构建其实失败了，
而我拿旧二进制在测**。之后固定成：每次构建后先 grep `build-log.txt` 的 `: error|FAILED` 再测。

---

## 13. 第五步：Hadamard `CONT` 省略（2026-09-28 深夜）——实现了、逐位一致、但**不划算**

### 13.1 目标和做法

`build_lora_mm()`（`llama-graph.cpp:1566-1571`）为 Hadamard 折叠权重构造
`RESHAPE → PERMUTE → CONT → RESHAPE → MUL(signs) → RESHAPE → MUL_MAT(hint)`。
后端已经把"符号翻转 + 旋转矩阵乘法"整段替换成**一次 FWHT**（`ggml_cuda_mul_mat` 的 hint 分支 +
`ggml_cuda_op_fwht_signed` 融合臂），所以那个 `CONT` 的唯一作用就是**把 permute 后的布局物化成连续内存**
给 FWHT 读。48 个/ pass。

实现：新增 `ggml_cuda_op_fwht_signed_view()`（`fwht.cu`）——按张量自身的内存顺序、用**步长**
直读 permute 视图（蝶形变换的代码与 `fwht_cuda_block` 逐行相同，保证逐位一致）；匹配器在 `CONT` 处登记
（键 = 后面的 `MUL(signs)` 节点）并跳过该节点，FWHT 臂在命中登记时改调 strided 版本。
开关 `GGML_CUDA_CONT_FWHT=1`（**默认关**）。

### 13.2 结果：命中 48/pass、逐字节一致，但解码略亏

* 命中：`GGML_CUDA_GDN_ROWS_DEBUG=1` 计到 **144 = 48×3 pass** ✓；
* 逐字节：on/off 在两种图模式下生成文本 sha 全同 ✓；PPL（含全关对照）也全同 ✓；
* 性能：d=0 `62.52 / 63.14`（on）vs `63.32 / 63.35`（off）→ **−0.3% ~ −1.3%**；
  pp512 `817.1 / 816.4` vs `813.0 / 813.1` → **+0.5%**。**净为负/打平 → 默认关。**
* 第一版还用 int64 做索引（每元素 3 次除法，V100 上 64 位除法很贵，且该 kernel 只有 6 个 block，
  时长≈单线程延迟）→ **−0.8%**；改成 32 位索引后回到打平。这条也说明：**当 kernel 的 block 数远小于 SM 数时，
  它测的是延迟，不是吞吐。**

### 13.3 顺带量到的两条硬数据（对以后有用）

1. **FWHT 当前实现值 +5.5%**：`GGML_CUDA_FWHT_LEGACY=1`（老 warp 版）d=0 **60.12** vs 现有 block 版 **63.45**
   （预填 795.4 vs 813.9）。也就是说 Hadamard 变换在这个模型上是**真实成本（≈1 ms/token，6%）**，
   而现有实现已经是最快的一版；254 次/ token 的变换属于模型量化方案的内生开销，不是"可省的 launch"。
2. **我对 ncu 小内核计时的修正**：24 KB 的拷贝在 ncu 里是 5.4 µs，但**剥掉它对 wall-clock 几乎没有影响**。
   综合本轮三次"合并小内核"实验（CONT→sigmoid、CONT→FWHT、conv 三合一）：
   **只有真正减少数据搬运/冗余计算才涨价**（GDN gather 3 MB/层、conv 三合一省掉两次 123 KB 往返、
   ADD+RMS+MUL 省掉一次残差重读），单纯合并 launch 基本是噪声。

### 13.4 全链路数值一致性（最终复核）

在最终二进制上把所有开关都做了一遍 PPL（1536 token，3 chunk）：`默认 / ADD 融合关 / conv 融合关 /
状态+conv 全关（回 gather 老路）` 四个配置给出**完全相同**的 `[1]1.8642,[2]1.8377,[3]1.7776`；
生成文本在两种图模式下也逐字节一致。⇒ **整条融合后的管线与原路径逐位一致**，不是"差不多"。

---

## 14. 当前状态、账本与下一步（2026-09-28 收尾）

### 14.1 最终数字（graphs OFF，`-r 5`，最终二进制）

> ⚠ **2026-09-28 晚更正**：本节数字来自**带并发缺陷**的 conv 三合一版本（见 §16）。缺陷只影响数值正确性、
> 不影响速度：修好之后同一台机器复查 d=0 `63.55`（本节 63.59）、90K `39.69`（本节 39.86），
> 而参考路径 `60.61` / `38.13`。也就是说这一节的速度结论仍然成立，但**当时"逐字节一致"的验收结论是错的**
> —— 文本与 PPL 都过了，只有逐位 logits 比对才暴露出来。请以 §16 为准。

| | 默认（四条融合） | 全关 `GGML_CUDA_GDN_ROWS_READ=0` | Δ |
|---|---:|---:|---:|
| d=0 解码 tg128 | **63.59** | 60.50 | **+5.1%** |
| d=0 预填 pp512 | 814.55 | 812.16 | +0.3% |
| 预填 pp8192 | 807.42 | 806.90 | +0.1% |
| （另一轮）90K 解码 tg128 | **39.86** | 37.95 | **+5.0%** |

图开关现在几乎无差别（graphs OFF 63.49 vs ON 63.40）：融合掉一批小内核后，图的发射开销本来就不再是瓶颈。

### 14.2 时间去哪了（修正后的模型）

* 权重侧 MMVQ 仍是绝对大头（约 68%），已接近该卡的有效带宽；
* Hadamard FWHT ≈ 6%（**内生**，且已是快版本，见 §13.3）；
* 归一化 / q8_1 量化 / GDN 状态 / FA 合计约 20%，其中 GDN 状态读写在 3 MB/层 的量级、**也接近带宽**；
* 小内核发射开销**不是**杠杆（三条融合实验证明）。

### 14.3 下一步候选（按性价比）

1. **要再压解码，只剩权重侧的 MMVQ**：需要一次 admin 权限的 `ncu` 采集（当前构建、PQ2_0 的
   `mul_mat_vec_q<142,…>`：dram__bytes + sm__throughput + stall 原因），先判断它到底是**带宽顶**还是
   **ALU/取指受限**。若是后者，Volta 上还能试 dp4a/更少指令的 dequant 序列；若是前者，就该收手。
2. **预填**：本轮三个融合对预填基本中性（±0.3%），历史收益都在 FA/MMA 路径（P1-a/P1-b），可以回去继续挖。
3. **FWHT+quantize 融合**（把蝶形折进 Q8_1 量化器）：理论省掉 254 次/ token 的"一次内核延迟"≈1%，
   但 §13.3 的教训说这类"合并 launch"大概率是噪声 —— 建议先拿 ncu 确认 FWHT 的真实 wall-clock 占比再决定。

### 14.4 交接开关速查

| 开关 | 作用 |
|---|---|
| `GGML_CUDA_GDN_ROWS_READ=0` | 状态路径全部回退（GN D行读 + conv 融合一起关），做 A/B 基线 |
| `GGML_CUDA_CONV_STATE_FUSION=0` | 只关 conv 三合一 |
| `GGML_CUDA_ADD_RMS_NORM=0` | 只关 `[ADD,RMS_NORM,MUL]` 融合 |
| `GGML_CUDA_CONT_SIGMOID=1` | 开 CONT→sigmoid（默认关，实测 −0.6%） |
| `GGML_CUDA_CONT_FWHT=1` | 开 Hadamard CONT 省略（默认关，实测 −0.3~−1.3%） |
| `GGML_CUDA_GDN_ROWS_DEBUG=1` | 打印每次融合/被跳过原因 |
| `GGML_CUDA_DISABLE_GRAPHS=0/1` | 图开/关（现在两者几乎等价） |

`GGML_V100_DUMP_NODES=1` 的 dump 现在带 **类型 / flags / 每个 src 的 op+类型+名字+形状**
（`[cuda-node]` + `[cuda-src ]` 两行一节点），配合 `work/analyze_*.py` 用来做图分析：
`analyze_cont.py`（CONT 归类）、`count_add_norm.py`（ADD+RMS 三元组计数）、`analyze_matmul.py`
（按权重类型统计 MUL_MAT）。今晚正是靠它们把"96 个 bf16 matvec = 每层 ssm_alpha/beta"这类问题在几分钟内定死。

---

## 15. 给下一次的"证据请求"：在管理员终端跑一次 ncu（这是当前最大的瓶颈）

今晚三次"合并小内核"实验都打平/为负，说明**剩下的空间不在 launch 数**；而我在这个沙箱里
拿不到 GPU 性能计数器（`ERR_NVGPUCTRPERM`），所以下一步应该先取一份**当前构建**的实测画像，
再决定要不要动 MMVQ。建议的命令（沿用你原来的 `profile_bonsai_decode.bat` 风格，只改两处：
**跳过首 token** 拿稳态，以及同时收 DRAM 字节数）：

```bat
set BIN=...\llama.cpp-b11004-bonsai\build-v100\bin
set MODEL=<models>\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PQ2_0.gguf
set CUDA_VISIBLE_DEVICES=0

REM 稳态解码：跳过前 ~8 个 token（每个 token ~1900 次启动），再抓 1 个完整 token
ncu --target-processes all -o %TEMP%\bonsai_steady --force-overwrite ^
    --launch-skip 15000 --launch-count 1900 ^
    --metrics gpu__time_duration.sum,dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ^
    llama-bench.exe -m %MODEL% -ngl 99 -p 0 -n 40 -d 0 -r 1 -fa on

ncu --import %TEMP%\bonsai_steady.ncu-rep --print-summary per-kernel --csv > steady.csv
```

（如果 15000 跳过头了，日志里会显示没有内核；把 `--launch-skip` 减半再试。）

我要用它回答**一个问题**：当前 `mul_mat_vec_q<142,…>`（PQ2_0 权重）到底是
**(a) DRAM 带宽顶住了**（`dram__bytes.sum / gpu__time_duration` ≈ 700–900 GB/s，`sm__throughput` 低），
还是 **(b) ALU/取指受限**（`sm__throughput` 高而 DRAM 不满）。前者说明该卡上解码已到极限、可以收手；
后者才值得去改 Volta 的 dequant/点积序列（dp4a、更少指令、更优的 `MMVQ_PARAMETERS_VOLTA`）。

顺带还能一次性确认：`fwht_cuda_block` 的真实 wall-clock 份额、`add_rms_norm_f32` 合并后的占用、
以及 `gated_delta_net_cuda` 的状态读写在 DRAM 上到底占多少 —— 有了这三个数，下一轮的取舍就不用再猜了。

---

## 16. 步骤六（2026-09-28 晚）：抓到一个真实的并发缺陷、修掉它，并把验收升级成"逐位 logits 比对"

这一节推翻了 §10/§14.1 的一条结论：**conv 三合一在内核里既有竞争、又是不可复现的**，而当时那套
"文本逐字节 + PPL" 的验收**全部放行**了它。下面按"怎么发现的 → 根因 → 怎么修 → 修完的逐位证据 →
修完的性能 → 方法论结论"写全，方便复核。

### 16.1 怎么发现的：跑交接过来的那套验收脚本，第一次就红了一项

接手后第一次跑（19:48，二进制 = 交接时那版 19:45:45 的 DLL）给出：

```
[FAIL] text.gON   sha=117e6e6d76413b69
[PASS] text.gOFF  sha=b3b903713dc0bc46
[PASS] ppl        [1]1.8642,[2]1.8377,[3]1.7776
```

同一个 prompt、同一个 seed、贪心解码，**只有 graphs ON 的这一格**与 gather 路径不同；差异只有一行
thinking 文本（`User asks: … Final: Paris.` vs `We need answer simple question. Need final concise.`），
argmax 很快又收敛回同一条轨迹。也就是说：**差的是"恰好落在近似并列附近的那一次 argmax"**，
而"文本逐字节"这个判据本身就有噪声（§10.1 里六个 sha 全同，是当时运气好）。

### 16.2 把判据升级：给后端加 logits dump 钩子（`GGML_V100_DUMP_LAST`）

文本只看到 argmax，看不见 logits 里的小漂移，所以先造一个能看见全量的尺子：

* 位置：`ggml_cuda_graph_evaluate_and_capture()`（`ggml-cuda.cu`）执行完节点/图之后，
  把**本 pass 最后一个节点**（就是 `result_output`，`MUL_MAT`，`ne=(248320,1,1,1)`，本模型
  `n_vocab=248320`）的 f32 内容 `cudaMemcpyAsync` 到 host 并按 append 写进 `GGML_V100_DUMP_LAST` 指定的文件；
* graph capture 那一次跳过（那时内核只是被"记录"、并未执行，读出来是脏的）；`cudaMemcpyAsync` + 同流
  `cudaStreamSynchronize`，与图启动天然有序；
* 于是"两个配置逐位相同" ⇔ 每个 token 的 248320 个 logits 全同。上面的用例是 25 个 pass
  （21 个预填位置 + 4 个生成 token）⇒ 文件 25 × 248320 × 4 B = **24 832 000 B = 6 208 000 个 logits**。

第一次测量（19:56 的二进制 = 交接版内核 + 只加了 dump 钩子，graphs OFF，每个配置跑两遍，
与"两个状态融合全关"的参考对比）：

| 配置 | 两次运行之间 | 与参考对比 |
|---|---|---|
| `ref`（都关） | bitDiff **0** / 6 208 000 | — |
| `gdn`（只开第一步行读） | bitDiff **0** | bitDiff **0**（逐位相同） |
| `conv`（含三合一） | bitDiff **6 207 963**，maxAbsDiff **1.844E-001** | bitDiff 6 207 967，maxAbsDiff 1.887E-001 |
| `default`（两个都开） | bitDiff **6 207 962**，maxAbsDiff **2.251E-001** | 第 0 行相同、之后每一行都不同 |

两条结论当场定死：

1. **第一步（GDN 行索引读）是干净的**：可复现，而且与非融合路径逐位相同；
2. **conv 三合一有问题**：*同一个配置跑两次结果都不一样* —— 这不是舍入顺序，这是**竞争**。

### 16.3 根因：三合一内核里，拼接部分与写回部分在同一个 cache 上争同一批行

把 `concat.cu` 里那个内核的索引展开（修复前的版本）：

* 拼接部分（`i < n_cat`）：`dst_cat[row,c,b]`，其中 `row < KS` 时**读** `cache[c*KS + row]`（旧状态），
  `row >= KS` 时读 qkv；写 `dst_cat`；
* 写回部分（`i >= n_cat`）：新状态第 `drow` 行 = `dst_cat` 的第 `T+drow` 行，**写** `cache[c*KS + drow]`
  （`drow ∈ [0,KS)`）；当 `T+drow < KS`（解码时 `T=1, KS=3` 必然如此）时还要**读** `cache[c*KS + T+drow]`。

于是：

* 拼接**读**的行集合是 `[0, KS)`，写回**写**的行集合也是 `[0, KS)` → **无条件重叠**；
* 当 `T < KS` 时，写回自身的读集合 `[T, KS)` 与写集合 `[0, KS)` 再重叠一次；
* 同一个 launch 内不同 block 之间没有任何定序，谁先谁后由调度决定 ⇒ 拼接部分可能读到"已经被写回覆盖"
  的旧状态，解码就漂了。**每次运行漂的幅度不同**，这正是"同一配置两次不一致"的来源。

为什么之前没被抓到：错的是 conv 输入里最老的 1–2 个 tap，量级 1e-3~2e-1，
贪心解码的 argmax 大多数时候不变（所以文本 sha 全同、PPL 四位小数全同），
而 `gdn` 那条路径本来就没有第二块"重叠写"，所以它逐位干净。

### 16.4 修复：写回不跟拼接同 launch；最终折进**消费端**（净收益相同、且无竞争）

**第一步（3→2）**：`concat.cu` 只做"读活状态 + 拼成 `conv_input`"（`ggml_cuda_op_conv_state_concat()`），
把写回交还给图里本来就有的那个 strided `CPY`。逐位立刻干净（见 16.5），但 d=0 只有 **62.72**（相对
`all-off` 60.28 = +4.05%）：48 层 × 每 token 多一个 launch ≈ 1.3%，正好是 §10 里那 6.09 µs 的 `CPY`。

**第二步（3→1，最终版）**：把写回折进**消费这块 `conv_input` 的 `SSM_CONV` 内核**，而不是折进拼接内核。
安全性论证很直接：

* `ssm_conv_f32` 只**读** `conv_input`（`dst_cat`）、只**写**状态 cache —— 两块内存不相交，**不存在竞争**；
* 该内核每个线程本来就按自己的通道把 `conv_input` 整行读过（`x_block[tid*stride_x + n_t + j]`），
  新状态第 `j` 个 tap 恰好就是第 `n_t + j` 个位置，**同一批字节**，不引入任何算术；
* 写回目的地址用 `CPY` 目的视图自己的 `data` / `nb[1]`（`sv->data + bidx*sv->nb[1] + c*KS + j`），
  与图上那个 CPY 写的位置**完全一致**；
* 只在短内核路径（`n_t <= 32`，即解码）折；长预填（`n_t > 32`）继续走图里的 `CPY`，长内核里显式
  `GGML_ASSERT(state_dst == nullptr)`。

匹配器相应加了 `ssm_conv_idx` 与前向扫描：要求在"写回 CPY"到"conv 消费节点"之间**没有任何节点读或写这块
cache**（这一层里 GDN 状态的 7 个节点确实夹在中间，所以"必须紧跟"的写法一开始没命中 —— 调试输出
`[gdn-rows] conv state write-back folded into conv_output_raw-N` 命中数为 0 才发现的）。
另外加了一条保守门（对着你说的多流风险）：**如果被省掉的那个写回 CPY 落在某个多流并发区
（concurrent region）里，就放弃折叠、保留图里的 CPY** —— 那些区域按自己的映射重排并派发区间内的每个节点，
不能让其中一个位置被跳过（判据 `ggml_cuda_concurrent_region_overlaps`，与第一步同款）。

交付物：

| 文件 | 改动 |
|---|---|
| `ggml-cuda/concat.cu` / `.cuh` | 内核改名 `conv_state_concat`（只做拼接），删除写回分支；头文件里写明"为什么写回不能留在这个 launch" |
| `ggml-cuda/ssm-conv.cu` / `.cuh` | `ssm_conv_f32` 增加可选写回（`state_dst`/`state_nb1`）；`ggml_cuda_op_ssm_conv()` 增加 `state_wb` 形参并校验目的视图几何 |
| `ggml-cuda/ggml-cuda.cu` | matcher 增加 `ssm_conv_idx` + 前向"cache 未被碰"扫描；`ggml_cuda_take_conv_state_wb()`（3 个 SSM_CONV 派发点都接上，并置 `consumed`）；`GGML_V100_DUMP_LAST` 钩子 |
| `ggml-cuda/mmvf.cu` / `.cuh` | §17：`mul_mat_vec_f` 加 PAIR 模式（`x2`/`dst2`）+ 导出 `ggml_cuda_mul_mat_vec_f_pair()` |
| `ggml-cuda/common.cuh` | §17：`ggml_cuda_mm_fusion_args_device` 加 `x2` / `dst2` 两个字段 |

命中与自检（`GGML_CUDA_GDN_ROWS_DEBUG=1`）：`conv state write-back folded into conv_output_raw-N`
**48 层 × 11 pass = 528 次**，`elide conv state gather` 576 次（48×12，第 1 个 pass 形状不同未折），
`unconsumed=0`，无 error。

### 16.5 修完之后的逐位验收（这才是这次真正的证据）

同一套 25 pass / 6 208 000 logits，四配置 × 各两遍、两种图模式，全部互相比：

```
work/dump_matrix.ps1（graphs OFF，浮点级比对，每个配置跑两遍）
  ref_a     vs ref_b     : floats=6208000 bitDiff=0 maxAbsDiff=0.000E+000
  gdn_a     vs gdn_b     : floats=6208000 bitDiff=0 maxAbsDiff=0.000E+000
  conv_a    vs conv_b    : floats=6208000 bitDiff=0 maxAbsDiff=0.000E+000   ← 修复前：6207963 / 1.844E-001
  default_a vs default_b : floats=6208000 bitDiff=0 maxAbsDiff=0.000E+000   ← 修复前：6207962 / 2.251E-001
  gdn_a     vs ref_a     : bitDiff=0      ← 第一步（GDN 行读）仍然逐位相同
  conv_a    vs ref_a     : bitDiff=0
  default_a vs ref_a     : bitDiff=0      ← 修复前：第 0 行相同、之后每行都不同

work/dump_gon.ps1（graphs ON，字节级比对）
  gon default_a vs default_b : BYTE-IDENTICAL (24832000 bytes)
  gon ref_a     vs ref_b     : BYTE-IDENTICAL (24832000 bytes)
  gon default_a vs ref_a     : BYTE-IDENTICAL (24832000 bytes)
  gon default_a vs 上述 graphs-OFF 的 gdn_a / default_a / ref_a : 全部 BYTE-IDENTICAL
```

graphs ON 侧的 4 份 dump 与 graphs OFF 的 4 份**也逐字节相同**（`work/dump_gon.txt`），
即 **"图开关 × 融合开关 × 重复运行" 8 份 dump 全同**。另用 32K prompt（走长预填内核路径、
`n_t > 32` 不折写回）复核：`gOFF/default vs gOFF/gather`、`gON/default vs gOFF/gather` 同样逐位相同。

**多序列（graphs ON/Off × rows/gather，共 5 次 `llama-server -np 2`）**也补了逐位版（`work/ms_dump.ps1`）：
每次 dump = 50 行（两路各 25 个 pass）× 248 320 logits = **49 664 000 B**，
* 同一配置跑两次：一致；
* graphs ON 的 rows vs gather、graphs OFF 的 rows vs gather、graphs ON vs graphs OFF：**值全部逐位一致**；
* 5 次里有 1 次（graphs OFF/gather）**行序不同**（服务器把两路请求交错进图的顺序变了）—— 所以脚本同时给
  "整文件字节比较"和"每行哈希排序后的多重集合比较"两种口径，按后者比较 5 次完全一致。
  （顺带说明：正因为服务器调度会影响行序，多序列这一格的判据不能只写"文件字节相同"。）

这些都已固化进 `work/run_acceptance.ps1`（判据 = logits 逐位，文本/PPL/多序列只作为附加项），
运行记录写在 `work/acceptance.txt`。

### 16.6 修完之后的性能（graphs OFF，`-r 3`，d=0 用 `-r 3`、90K 用 `-r 2`）

| 配置 | d=0 tg128 | 90K tg128 | d=0 pp512 | pp8192 |
|---|---:|---:|---:|---:|
| `all-off`（`GGML_CUDA_GDN_ROWS_READ=0`） | 60.61 | 38.13 | 813.72 | 807.01 |
| `conv-off`（只开第一步） | 62.39（**+2.94%**） | — | 815.04 | — |
| `default`（两步都开） | **63.55（+4.85%）** | **39.69（+4.09%）** | 815.06（+0.16%） | 807.56（+0.07%） |

（中间那版 3→2 是 `62.72 / 61.87 / 60.28`：写回折进 conv 消费端把 conv 那一步从 +1.37% 拉回 **+1.86%**。
90K 预填 `pp512@d90112` = 495.26 vs 494.87，+0.08%。）

和交接文档里的原始基线（`-ngl 99 -fa 1`：d=0 **58.89**、pp512 787.65、90K 38.31 / 488.92）比：
**d=0 +7.9%**、pp512 +3.5%、90K tg128 +3.6%、90K pp512 +1.3% —— 解码落在 5–9% 目标区间内；
而"本轮这条状态路径自身"的贡献（同机 A/B）是 **+4.85%**。

### 16.7 三条方法论结论（这次最值钱的部分）

1. **"文本逐字节 + PPL"不是并发缺陷的可靠判据**。这次两者全过，缺陷却是真的（1e-3~2e-1 的
   logits 漂移）。要抓这类问题必须看**逐位（或至少张量级）输出**：`GGML_V100_DUMP_LAST` 一挂，
   两分钟就定位。以后任何"融合/跳过/改派发"的改动，验收都该走这条路。
2. **"同一配置跑两次结果不同"是最短的根因判据**。一旦出现，先怀疑 launch 内的跨 block 依赖、
   未初始化读、被覆盖的读，而不是舍入顺序 —— 舍入顺序是确定性的，不会自己变。
3. **只有真正减少数据搬运、或砍掉依赖链上的 launch 才涨钱**：本例 48 个/ token 的小 `CPY`
   值 1.3~1.9%；但"折进已有内核"没有额外代价 —— 关键是把写回放在**只读那份数据的内核**里，
   而不是放在**既读又写同一块内存的内核**里。

### 16.8 交接（增量部分）

| 开关 | 作用 |
|---|---|
| `GGML_CUDA_GDN_ROWS_READ=0` | 状态路径全部回退（行读 + conv 两步一起关），A/B 参考 |
| `GGML_CUDA_CONV_STATE_FUSION=0` | 只关 conv 那一步（含写回折叠），保留 GDN 行读 |
| `GGML_CUDA_MMVF_PAIR=0` | **新**：关掉 §17 的 `ssm_alpha/ssm_beta` 配对 launch（左右两侧都保留时不影响 A/B 结论） |
| `GGML_CUDA_GLU_PERMUTE=0` | **新**（§20）：关掉"去交织 GLU 写"（**默认开**；修好"写太早"之后 32K 也逐位通过） |
| `GGML_CUDA_CONV_L2=0` | **新**（§21）：关掉"L2 折叠进 pair launch"（**默认开**） |
| `GGML_V100_DUMP_LAST=<file>` | **新**：每次图 pass 追加最后一个节点（logits）的 f32 原始内容；逐位比对用 |
| `GGML_V100_OP_HIST=1` / `GGML_V100_OP_TIME=1` | **新**（§19）：派发计数 / 逐算子 GPU 时间（graphs OFF 下用） |
| `GGML_CUDA_GDN_ROWS_DEBUG=1` | 打印每次匹配/跳过原因、写回折叠命中 |

新增/更新的脚本：`work/logits_dump_probe.ps1`（最初的单文件探针）、`work/flip_probe.ps1`（文本复现性探针）、
`work/dump_matrix.ps1`（四配置 × 两遍，graphs OFF）、`work/dump_gon.ps1`（graphs ON + 跨图模式比对）、
`work/perf_after_fix.ps1`（三配置性能矩阵）、`work/run_acceptance.ps1`（**一键验收**，含逐位 logits、
32K prompt、文本、PPL、多序列、性能）。

### 16.9 补丁文件与"怎么打"（这次顺手把补丁做成了可验证的）

`outputs/state-path-fusions.patch`（100 KB，14 个文件：`common.cuh`、`ggml-cuda.cu`、
`gated_delta_net.cu/.cuh`、`concat.cu/.cuh`、`ssm-conv.cu/.cuh`、`mmvf.cu/.cuh`、`unary.cu/.cuh`、`fwht.cu/.cuh`）
由 `work/regen_patch.ps1` 生成：把 `<work-old>\merged-tree`（基线）与当前树的文件按
**同一相对路径**摆到临时目录的 `a/` `b/` 两侧再 `git diff --no-index`，因此可以直接在树根用

```bat
git -c core.autocrlf=false apply -p1 ..\outputs\state-path-fusions.patch
```

打上；本次实测：在基线树的副本上打完，**14 个文件与当前树哈希全部一致**。

两个坑记下来：

1. **必须加 `-c core.autocrlf=false`** —— 这台机器的**系统级** gitconfig 里 `core.autocrlf=true`，
   否则 `git apply` 会把每个 LF 换成 CRLF，打完后文件"看着对"但字节全变（本次核对时 68675 B → 70460 B）；
2. **生成补丁时不能用 PowerShell 的 `Out-String` / `Set-Content`** —— 它们会把行尾改写成 CRLF 并加 BOM，
   结果就是 `patch does not apply`；现在走 `cmd /c "git diff … > file"` 原样落盘（脚本里有注释）。

（与那棵基线树相比，`ggml-cuda/` 下还有 `mmq.cuh`、`mmvq.cu`、`norm.cu` 三个文件不同 —— 那是你更早的
P1/V100 调优，不是本轮的状态路径改动，**故意没有打进这个 patch**，免得把两批改动混在一起。）

---

## 17. 步骤七（2026-09-28 深夜）：把每层的 `ssm_alpha`/`ssm_beta` 两个 matvec 合成一次 launch（**+0.6~1.3%**）

### 17.1 先量化上限，再决定要不要动手（"探针"）

图里每层有两个 **(d_inner=5120 → 48) 的 bf16 `MUL_MAT`**（`blk.N.ssm_alpha.weight` /
`blk.N.ssm_beta.weight`，节点 40/43），96 个 / token，而且**读同一个激活**（`attn_norm-N`）。
按 §16.7 第 3 条的经验（"依赖链上砍掉一个 launch ≈ 1%~2%/48 层"），这里可能值钱。

动手前先用一个**故意算错**的探针把上限量出来：`GGML_CUDA_SKIP_BETA_PROBE=1` 直接跳过每层
beta 的 dispatch（beta 输出保留上一 token 的旧值，结果当然是错的，只测 wall clock）：

| d=0 解码（graphs OFF，`-r 2`，交替两轮） | tg128 |
|---|---:|
| 探针关 | 62.26 / 63.00 |
| 探针开（少一个 launch） | **64.22 / 64.10** |

⇒ **+1.5%~3.1%（均值 +2.4%）**；同一轮 pp512 = 813.8 → 861.9（+5.9%，预填那侧是 GEMM 路径，见 17.4）。
探针用完立刻从源码删除 —— 交付的树里没有它（这条也写进 §17.5 的教训）。

### 17.2 实现：**靠构造**就逐位一致（不是"应该一致"）

关键观察：`mul_mat_vec_f<...>` 内核里 `x = src0`（权重矩阵）、`y = src1`（激活）、`dst` 是输出；
而 `block_size_best` **只由 `ncols` 决定**，`channel_ratio/sample_ratio` 与 `channel_dst` 的偏移在
`channel_dst_eff = 0` 时全部退化。所以就有一条"顺手"的路：**把 `grid.y` 从 1 改成 2，让
`blockIdx.y == 1` 复用完全相同的代码路径，只把 `x`、`dst` 指针换成第二个 matvec 的**。

* `common.cuh`：`ggml_cuda_mm_fusion_args_device` 加两个字段 `x2` / `dst2`（PAIR 模式；注意
  `has_fusion` 的判定只看 `gate/x_bias/gate_bias`，所以模板实例化与非 pair 路径完全相同）；
* `mmvf.cu`：内核里加
  `pair_second = fusion.x2 != nullptr && channel_dst == 1`、`channel_dst_eff = pair_second ? 0 : channel_dst`，
  取 `x`/`dst` 时按 `pair_second` 选择；新增导出 `ggml_cuda_mul_mat_vec_f_pair()`，用
  `nchannels_dst = 2` 调同一条 `mul_mat_vec_f_cuda()` 链路（其余参数与单节点调用逐字相同）；
* `ggml-cuda.cu`：新增 matcher `ggml_cuda_match_mul_mat_vec_f_pair()` —— 第一个节点必须是
  `MUL_MAT`、dst 单列、`src0` 走 mmvf 路径（用与派发完全相同的 `ggml_cuda_should_use_mmvf()` 判定）、
  `src0` 不是 view；第二个节点必须**紧跟在 view/noop 之后**、`src1` **指针相同**、`src0` 形状/类型相同、
  dst 形状相同且同样走 mmvf；命中后一次 launch 把两个节点都算完（`i = second_idx; continue`）。
  开关 `GGML_CUDA_MMVF_PAIR=0`（默认开）。

**踩到的坑（值得记）**：第一版把第二个权重塞进了内核的 `y`（那是**激活**，不是权重），
logits 直接差到 `maxAbsDiff = 2.0E+1`；逐位对比当场暴露，改名 `x2` 后全同。
*教训：字段名要跟内核里那个变量的语义走（`x = 权重`），别跟我脑子里的名字走。*

### 17.3 逐位验收（`work/dump_pair.ps1`，6 份 dump 互比）

```
pair ON  vs pair OFF（状态融合开）   : BIT-IDENTICAL (6208000 logits)
pair ON 重复运行                    : BIT-IDENTICAL
pair ON（状态融合关）vs 纯基线       : BIT-IDENTICAL
全部融合 vs 纯基线                   : BIT-IDENTICAL
graphs ON vs graphs OFF（全开）      : BIT-IDENTICAL
```

其中"纯基线" = `GGML_CUDA_GDN_ROWS_READ=0` + `GGML_CUDA_MMVF_PAIR=0`（§16 里那条老路）。

### 17.4 性能（graphs OFF，`-r 3`，d=0）

| 配置 | tg128 | pp512 |
|---|---:|---:|
| 纯基线（状态融合关 + pair 关） | 60.49 | 812.62 |
| 状态融合（pair 关） | 63.32（+4.68%） | 813.43 |
| 全部融合（pair 开） | **64.13（相对纯基线 +6.02%）** | 813.51 |

⇒ pair 自身 = 63.32 → 64.13 = **+1.28%**；预填中性（预填时 `ssm_alpha/beta` 是 ne11=512 的 GEMM，
不走 mmvf 这条路，matcher 的 `ne[1] == 1` 条件自然把它排除）。
（交付版最后一次验收里同一对测到 `64.09 vs 63.69 = +0.63%` —— 这个量级本来就在噪声里，取区间
**+0.6%~+1.3%**，两次都是正的，见 §18。）

### 17.5 与交接文档基线的关系

* d=0 解码：**64.13 vs 58.89 = +8.9%**（落在你定的 5–9% 目标区间的上沿）；
* 同机 A/B（相对"状态路径全关 + pair 关"这条最保守的基线）：**+6.02%**；
* 90K 解码、预填数字见 §16.6 与 `work/acceptance.txt`（本轮验收脚本现已在 2b/7b 两处加入 pair 的
  逐位与性能检查）。

### 17.6 下一步候选（都带量化依据，留给下一轮）

1. **同一个 pair 机制还能再吃两处**（都用 `ggml_cuda_should_use_mmvf` 判定，改的是 matcher 不是内核）：
   注意力里那些 `ne[1] == 1` 的成对投影（若它们共享同一激活）——先用 `analyze_matmul.py` 数一遍再动；
2. **`z` 投影（`blk.N.attn_gate.weight`，5120→6144，pq2_0）** 与 alpha/beta 无关，但它和 GDN 输出
   之间还有一次 `RESHAPE`+`MUL`，属于 PQ2_0 的 MMVQ 路径，收益只能靠 ncu 判断（§15）；
3. **预填侧**：本轮三个融合 + pair 对预填基本中性（±0.1%），历史收益都在 FA/MMA（P1-a/P1-b），
   可以回去继续挖 —— 但需要 §15 那份 ncu 画像。

---

## 18. 最终验收记录（2026-09-28 21:51–22:13，交付版二进制）

二进制：`build-v100\bin\ggml-cuda.dll` @ **2026-09-28 21:49:33**（源码见 §16.9 的 patch，
14 个文件；`build-log.txt` 无 error）。一键脚本：`work\run_acceptance.ps1`，
原始输出：`work\acceptance.txt`。

```
SUMMARY: 18/18 checks passed
```

| # | 检查 | 结果 |
|---|---|---|
| 1 | dll 比源码新 / 日志无 error | PASS |
| 2 | 逐位 logits：`gOFF/gON` × `default/gather/conv-off` + 重复运行（6 格） | 全部 BIT-IDENTICAL（6 208 000 logits） |
| 2b | 逐位 logits：matvec pair（on/off × 状态 on/off × 图 on/off，5 格） | 全部 BIT-IDENTICAL |
| 3 | 逐位 logits：32K prompt（长预填路径，两种图模式） | 全部 BIT-IDENTICAL（1 489 920 logits） |
| 4 | 文本逐字节（两种图模式） | PASS（sha `b3b903713dc0bc46…` 相同） |
| 5 | PPL（1536 token ×3 chunk） | PASS（`[1]1.8642,[2]1.8377,[3]1.7776` 相同） |
| 6 | 多序列 `llama-server -np 2`（两种图模式，两路文本） | PASS（IDENTICAL） |
| 7 | 性能 d=0 / d=90112（`-r 3`，graphs OFF） | PASS（见下表） |
| 7b | matvec pair on/off d=0 | PASS（+0.63%） |

最终性能（同一台机器、`-ngl 99 -fa 1`、graphs OFF）：

| 配置 | d=0 tg128 | 90K tg128 | d=0 pp512 | 90K pp512 |
|---|---:|---:|---:|---:|
| **交付版（全开）** | **63.92** | **40.04** | 814.68 | 498.24 |
| 参考（`GGML_CUDA_GDN_ROWS_READ=0` + `GGML_CUDA_MMVF_PAIR=0`） | 60.65 | 38.44 | 813.44 | 497.42 |
| Δ | **+5.39%** | **+4.16%** | +0.15% | +0.16% |

分项（同一轮成对测量）：matvec pair **+0.63%**（另一次独立测量 +1.28%，取区间 **+0.6~1.3%**），
其余分项见 §16.6（GDN 行读 ≈ +2.9%、conv ≈ +1.9%、`ADD+RMS+MUL` ≈ +0.4~0.8%）。

对照交接文档里的原始基线（`-ngl 99 -fa 1`：d=0 58.89、90K 38.31、pp512 787.65、90K pp512 488.92）：
**d=0 +8.5%**、90K 解码 +4.5%、pp512 +3.4%、90K pp512 +1.9%。

> 备注：多序列那一格（第 6 项）走的是文本判据；它的逐位版本在 `work/ms_dump.ps1` 里单独做过一次
> （5 次 `llama-server -np 2`、每次 49 664 000 B 的 logits dump，按"每行哈希排序后的多重集合"全部一致，
> 见 §16.5）。之所以不放进一键脚本：服务器把两路请求交错进图的**行序**偶尔会变，整文件字节比较会假红。

> **2026-09-30 增量**：交付二进制现在是 **2026-09-30 12:24** 那版（在 §16~§18 的基础上加了 §19 的
> 度量工具和 §20 的 `GGML_CUDA_GLU_PERMUTE`）。新改动的逐位验收已单独做完（`work/dump_glu.txt`
> 6 份 dump 全同 + §16 的 8 份 dump 复跑全同 + 文本/PPL 一致）；**18 项一键验收需要在机器安静时
> 重跑一次**（今天进程间抖动 ±3~5%，性能那两格的数字不可信，见 §20.4）。

---

## 19. 步骤八（2026-09-30 中午）：把"时间到底去哪了"量出来 —— 结论是**一半时间不在已派发的 kernel 里**

这一轮没有改算法，先补上缺了两次的那把尺子：既然 `ncu` 拿不到（§15），就用 CUDA event 在后端内部
自己量。新增两个 env 门控（默认关，零开销）：

* `GGML_V100_OP_HIST=1`：统计**真正到达 `ggml_cuda_compute_forward()` 的算子**（= 一次 kernel 派发；
  被融合臂/被状态路径省掉的节点不会到那里），每个图 pass 打印一次；
* `GGML_V100_OP_TIME=1`：给每个派发的算子前后各记一个 event，pass 结束时按算子汇总 GPU 时间，
  并额外报出**整个 node loop 的 GPU 时间**（两者之差 = 融合臂自己的 launch + kernel 之间的空隙）。

（都只在 `GGML_CUDA_DISABLE_GRAPHS=1` 下有效；graphs ON 时 capture 阶段不能插 event。测量本身会加
约 1.3 ms/token 的 event 开销，所以下面用"占比"看，别把绝对值当最终性能。）

### 19.1 稳态解码一个 token 的实测（`-p 0 -n 6`，graphs OFF，d=0）

```
[op-time] pass 7: 508 ops, 7.896 ms timed GPU time
[op-time]   node-loop GPU time: 16.989 ms (so 9.094 ms is fused-op launches + inter-kernel gaps)
[op-time]   MUL_MAT                 6.487 ms  x242  (0.027 ms each, 82.2%)
[op-time]   GLU                     0.322 ms  x72   (0.004 ms each, 4.1%)
[op-time]   CONT                    0.320 ms  x64   (0.005 ms each, 4.0%)
[op-time]   FLASH_ATTN_EXT          0.240 ms  x16   (0.015 ms each, 3.0%)
[op-time]   L2_NORM                 0.223 ms  x48   (0.005 ms each, 2.8%)
[op-time]   ROPE                    0.145 ms  x32   (0.005 ms each, 1.8%)
[op-time]   SET_ROWS                0.144 ms  x32   (0.005 ms each, 1.8%)
[op-time]   MUL                     0.009 ms  x1
[op-time]   GET_ROWS                0.005 ms  x1
```

三条结论：

1. **派发出去只有 508 个 kernel/token**（图里 3437 个节点：1237 个 view/noop/空节点被跳过，
   其余约 1700 个被融合臂吃掉 —— 融合网已经织得很密，`MUL/ADD/RMS_NORM` 几乎一个都不剩：
   历史 `MUL 483 / ADD 128 / RMS_NORM 209` 在派发统计里只剩 `MUL 1`、`ADD 0`、`RMS_NORM 0`）。
2. **MUL_MAT 占 82%（6.49 ms）**，其中绝大部分是 PQ2_0 权重（每 token 要把 6.7 GB 权重读一遍，
   900 GB/s 的理论下限就是 7.4 ms）→ 权重侧确实已经在墙上了，符合 §14.2/§15 的判断。
3. **剩下 ~9.1 ms 是"融合臂自己的 launch + kernel 之间的空隙"**，而单个小 kernel 的实测是
   **4~5 µs**（`L2_NORM`/`ROPE`/`SET_ROWS`/`CONT` 都是这个量级，其中绝大部分是 launch 延迟，
   因为它们的数据量只有 1.5 KB~16 KB）。这正是这一轮之前每一步融合都能赚 0.5~1.5% 的原因：
   **每砍掉 48 个/token 的 launch ≈ 0.25~0.5 ms ≈ 1.5~3%**。

补一条实测：`CONT (128,3) × 48` 这种"384 个元素也要发一个 kernel"的小算子还有 48 个，
`L2_NORM (128,32) × 48`、`SET_ROWS/ROPE × 32/32` 同理 —— **小 launch 才是剩下的主要杠杆**。

### 19.2 顺手排除掉的两个"看起来有肉"的方向（都白跑一趟，写下来免得下次再踩）

1. **每层的两个 `SCALE`（状态清零）不是问题**。图里每层有 2 个
   `SCALE cache_{r,s}_lN ... = 0`（`llama-graph.cpp:3597` 的"清一个 slot 再拷给其它 slot"），
   而 CUDA 的 `ggml_cuda_op_scale` 没有 identity 快路。我一度以为它们每 token 要读+写 ~6.5 MB/层。
   实测（`GGML_V100_DUMP_NODES=1` 逐 pass 统计）：**只有会话最开始 2 个 pass 是"活的"
   （ne=(786432,)），之后的 pass 全变成 ne=(0,) 空视图被跳过** → 折算到 tg128 上约 0.03%，不构成优化点。
   （如果哪天要在 llama-server 里频繁建/退序列，这一格才值得再看。）
2. **graphs ON/OFF 已经完全等价**：同一轮 `-r 3` 实测 `graphs OFF 63.55 ± 0.07` vs
   `graphs ON 63.59 ± 0.11`。即 CPU 派发已经不是瓶颈（GPU 侧每 kernel 的 launch/依赖延迟才是），
   所以别指望 CUDA graph 再挤出东西；反过来也说明 §16 的"融合减少 launch"才是正路。

### 19.3 下一步的候选清单（按"每砍 48 个 launch ≈ 1.5~3%"换算，已标注实测代价）

| 候选 | 每次/token 的 launch | 实测代价 | 备注 |
|---|---:|---:|---|
| ~~`CONT (128,3) ×48` 那批（GDN 小视图物化）~~ | 48 | ≈0.19 ms ≈1.2% | ✅ **已完成**（§20，默认开）：折进 GLU 的写地址，且写入挪到 CONT 节点执行 |
| `GLU (128,48) ×64`（`z * silu(gate)`） | 64 | ≈0.29 ms ≈1.8% | 若能把 silu+乘折进上面的 `z`/gate 投影 MMVQ（fork 已有 `epilogue` 机制，§4 的 `mul_mat_glu_ops`） |
| ~~`L2_NORM (128,32) ×48`~~ | 48 | ≈0.22 ms ≈1.4% | ✅ **已完成**（§21，默认开）：搭在 `ssm_alpha/ssm_beta` pair launch 上，归约顺序与 `l2_norm_f32<32>` 逐位相同 |
| `SET_ROWS ×32` / `ROPE ×32` | 64 | ≈0.29 ms ≈1.8% | ROPE 的 Q/K 可以对儿发（同 §17 的 pair 手法）；SET_ROWS 是 KV 写回，K/V 能否合一次要看 cache 布局 |
| 16×`FLASH_ATTN_EXT` | 16 | ≈0.24 ms | d=0 时单发 15 µs；90K 时会变成大头，另有优化空间（P1 已做过一部分） |

> **两条通用经验（这一轮最值钱的部分，写进下一轮的方法论）**：
> 1. **"跳过节点 + 提前写它的输出"必须检查生命期**：ggml 分配器按节点位置算生命周期，
>    在目的张量的生产者节点之前写它，随时可能被借出去的显存覆盖（§20.7）。正确做法是
>    **把融合算子挪到那个目的节点执行**，并用 `add_alloc_dep()` 把它读的操作数保活到那里；
> 2. **收益评估要看 launch 账 + 安静环境的墙钟**：这台机器上有远程桌面 agent时，
>    `llama-bench` 的抖动可达 ±3%（甚至 43 t/s 的离群点），任何 <2% 的结论都必须多轮交错 + best-of/中位数。

顺序建议：`CONT(128,3)` → `GLU(128,48)` → `L2_NORM`。三个都改完之后，再回头看预填侧
（预填对这三类都不敏感，收益仍在 FA/MMA）。

跑法（本轮的尺子已经留在树里）：

```bat
set GGML_CUDA_DISABLE_GRAPHS=1
set GGML_V100_OP_TIME=1
llama-bench.exe -m %MODEL% -ngl 99 -fa 1 -p 0 -n 6 -r 1
```

---

## 20. 步骤九（2026-09-30）：把 GDN 输出的"去交织拷贝"折进 GLU 的写地址（`GGML_CUDA_GLU_PERMUTE`，**最终默认关**）

> ✅ **本节在当天晚些时候修好了并改为默认开启**：短 prompt 全过，但 **32K prompt 那一格先失败了**
> （开着它 `default vs gather` 差 744960/1489920）。根因是**写入时机**：第一版在 GLU 节点就写
> `CONT` 的输出，而那块 buffer 要到 CONT 节点才算"活的"，ggml 的分配器可以把它在中间借给别人 ⇒
> 被覆盖。修法：**把融合算子挪到 CONT 节点执行**（那时目的地已活），并用
> `ggml_backend_optimize_params::add_alloc_dep()` 把 GLU 的两个操作数保活到 CONT 节点。
> 修完后：短 prompt、32K prompt、两种图模式全部逐字节相同。详见 §20.6/§20.7、§22。

### 20.1 目标：那 48 个 `CONT (128,3,16)`

§19 的直方图里 `CONT ×64`、实测 0.32 ms（≈4%），其中 **48 个是每层一个的"去交织"**：

```
55 GLU  node_57 (128,48)            z ⊙ silu(rms(gdn_out))
56 RESHAPE final_output-0 (6144,1)
57 RESHAPE final_output-0 (128,16,3)
58 PERMUTE (0,2,1)  -> (128,3,16)
59 CONT                               ← 纯拷贝，24 KB/层
60..63 RESHAPE -> MUL(signs) -> RESHAPE -> MUL_MAT(hadamard hint)
```

这个 `CONT` 存在的唯一理由就是**把 GDN 输出摆成 Hadamard 路径要的布局**。
既然算子在它前面（GLU 是逐元素的），那就**让 GLU 直接按置换后的地址写**：
数学一个字都没变（同样的两个操作数、同样的 `silu(x)*g`、同样的顺序），只是**写地址**不同，
于是这个 `CONT` 连它的 launch 一起消失，下游 FWHT 读到的仍是**同一块连续 buffer 的同一批字节**。

### 20.2 实现

* `unary.cu`：新增 `unary_gated_permuted_op_kernel` + `ggml_cuda_op_swiglu_permuted()`
  —— 对每个**写出位置** `q` 反解出来源下标 `i` 再算同一个 GLU（避免"写侧散列"）：
  ```
  i0 = q % rows;  j1 = (q/rows) % no;  j2 = q/(rows*no);
  i  = i0 + j2*rows + j1*rows*ni;        // ggml 是 dim0 最内层
  dst[q] = silu(x[i]) * g[i];
  ```
* `ggml-cuda.cu`：新增匹配器 `ggml_cuda_match_glu_permuted_store()`（GLU 必须是**双操作数、连续、
  单 token 的 F32 SWIGLU**；GLU → 只含 view/noop → `CONT`，且中间的 `RESHAPE`/`PERMUTE(0,2,1)`
  形状必须严格对得上；GLU 输出与中间视图**不能被任何别的节点引用**），命中后
  `ggml_cuda_op_swiglu_permuted()` + 把 `CONT` 的位置记进 `permuted_store_skip`（该位置直接跳过）。
* 开关 `GGML_CUDA_GLU_PERMUTE=0`（默认开）；调试打印 `[gdn-rows] GLU ... stores deinterleaved into ...`。

**踩到的两个坑**（都靠逐位 dump 当场抓到，值得写下来）：

1. 第一版把 `swapped`（`op_params[1]`）也用上了 —— 但**双操作数**时普通路径根本不看这个标志，
   于是操作数被对调，模型直接提前 EOS（dump 只有 4 行）；照抄普通路径后一致；
2. 第一版按"dim1 最内层"推下标 —— **ggml 是 dim0 最内层**（`nb[0]=4`），整个置换写错了；
   改成 `i0 = q % rows` 那一套之后才逐位相同。

### 20.2b 命中与 launch 账（`GGML_V100_OP_HIST=1`，稳态解码一个 pass）

```
glu-permute OFF : 508 次派发 | CONT 64 | GLU 72
glu-permute ON  : 412 次派发 | CONT 16 | GLU 24      ← 48 个去交织 CONT + 48 个 GDN GLU 被折掉
```

派发数少了 96，但融合后的 GLU 本身也是一次 launch（不进 `ggml_cuda_compute_forward`，所以不计入上表）
⇒ **每 token 净少 48 个 launch** + 48 × 24 KB 的拷贝，与 §19 的"4~5 µs/小 launch"折算到 ≈ +1.2% 一致。

### 20.3 验收（`work/dump_glu.ps1`，6 份 dump）

```
glu OFF 重复运行            : BIT-IDENTICAL (6208000 logits)
glu ON  vs glu OFF          : BIT-IDENTICAL
glu ON  重复运行            : BIT-IDENTICAL
graphs ON vs graphs OFF     : BIT-IDENTICAL
```

另外在最终二进制上复核：文本 sha `0d1e23e50f920cd1…`（on/off 相同）、PPL
`[1]1.8639,[2]1.8376,[3]1.7776`（on/off 相同；注意 PPL 第 4 位小数本身有 ±3e-4 的抖动，
因为 `llama-perplexity` 的多线程累加顺序不定 —— 判据要看 logits dump，不是 PPL）、
§16 的状态路径 8 份 dump 仍全同。

### 20.4 性能：**结构性只减不增，但今天这台机器的抖动盖住了它**

砍掉的是 48 个 launch + 48 × 24 KB 的拷贝，**没有增加任何计算**，所以方向上是确定的收益
（按 §19 的 4~5 µs/小 launch 折算 ≈ +1.2%）。实测：

* 干净的两对（`-p 0 -n 128 -r 2`，交替）：**+1.19% / +1.12%**；
* 第一对（`-p 512 -n 128 -r 3`）：+4.16%；
* 但今天**进程间抖动高达 ±3~5%**（同一配置 5 次跑出 57.8~64.4 t/s，`nvidia-smi` 采样确认
  V100 上没有别的进程、无降频、温度 37~46 ℃，所以是**分配/放置层面**的抖动，不是外部干扰），
  后续几轮 A/B 因此互相矛盾。

**结论（据实写）**：这是一个"只减少不增加"的改动，逐位一致已验证；收益量级 **≈ +1.2%
（干净环境下）**，需要在机器安静时用 `work/run_acceptance.ps1` 的第 7 节重测确认。
本轮不再基于抖动数据下"涨了几个点"的结论。

### 20.5 下一步（§19.3 的清单更新）

`CONT(128,3,16)` 这一项已经做完。**（当天晚些时候更新：`L2_NORM` 也做完了，见 §21.4——
但实现方式不是"折进 `SSM_CONV`"，而是搭在 pair launch 上，原因见 §20.7。）**
清单剩下：`GLU (128,48) ×64` 里还没折的部分、`ROPE/SET_ROWS ×32+32`（Q/K 对儿发）、
以及预填侧的 FA/MMA。**（最新清单位于 §19.3 的两条通用经验之后。）**

### 20.6 第一次更正（2026-09-30 下午）：32K prompt 那一格没通过 ⇒ 当时先改成默认关闭

§20.3 的"逐位一致"只覆盖了**短 prompt（25 pass）**。同一天把 `run_acceptance.ps1` 的
完整套件跑起来后，**第 3 格（32K prompt）报警**：

```
[FAIL] logits32k.gOFF  bitDiff=744960/1489920  maxAbsDiff=5.747E-001  firstAt=744960
```

定向复测（同一份 prompt、只切 `GGML_CUDA_GLU_PERMUTE`）：

| 配置 | 32K `default vs gather` |
|---|---|
| `GLU_PERMUTE=0` | **BYTE-IDENTICAL** |
| `GLU_PERMUTE=1` | **bitDiff=744960**（6 个 pass 里的第 3/4/5 个全行不同） |

⇒ 这个融合在 32K 场景下确实会改变数值（大概率是与状态路径的"跳过节点"机制在长 prompt 图上的
交互，短 prompt 图上看不出来）。**已把默认改成关**（`GGML_CUDA_GLU_PERMUTE=1` 才启用），
重新构建后复核：`dump_matrix`（状态路径 8 份 dump）逐位相同 ✓、
32K 的 `default vs gather` 在 graphs ON/OFF **两种模式下都逐字节相同** ✓。

两条教训（写给下一次）：

1. **短 prompt 的逐位验收不足以证明一个"跳过节点"类融合是安全的** —— 这次和 9/28 的 conv 竞争
   是同一类陷阱：**只有长 prompt（32K）那一格能看出来**。以后这类改动必须先跑 32K 再谈收益；
2. 收益再好，只要有一格逐位不过就**不能默认开**——这轮的 launch 账（508→412）本身是真的，
   但正确性优先。

### 20.7 最终修复（2026-09-30 晚）：**根因是"写得太早"，改成在 CONT 节点写 + `add_alloc_dep` ⇒ 默认开启**

晚上顺着"两处失败（本节的 GLU、§21 的 L2）都是'把某个节点的输出提前到别的节点去写'"这条线索，
定位到共同根因：

> **ggml 的图分配器按"节点位置"算张量生命周期**——一个张量从它的**生产者节点**开始才算活着，
> 在此之前它占的显存可以借给别的张量。所以**提前写**一个还没到生命周期的张量，随时可能被覆盖。
> 这也解释了为什么"短 prompt 过、32K 不过"：不同图的分配布局不同，32K 那张图刚好把这块借出去了。

修法（`ggml_backend_cuda_graph_optimize` + 节点循环）：

1. **把融合算子挪到目的张量的"生产者节点"执行**：GLU 折叠不再在 GLU 节点写 CONT 的输出，而是
   在**CONT 节点**执行（`pending_glu_idx/pending_cont_idx` 记录，走到 CONT 时才发那一次融合 launch）；
2. **用 `params->add_alloc_dep(user_data, tensor, until)` 把 GLU 的两个操作数保活到 CONT 节点**
   （这个 API 正好是"把 `tensor` 至少保活到 `until` 节点"）；
3. 另外加一条保守门：若 `[GLU, CONT]` 跨到多流并发区就放弃（与 §16 的状态路径同款）。

验收（全部在最终二进制上）：

```
work/dump_glu.ps1 : glu OFF 重复 ✓ / glu ON vs OFF ✓ / glu ON 重复 ✓ / graphs ON vs OFF ✓   (6 208 000 logits 全同)
work/dump_matrix.ps1 : 状态路径 8 份 dump 全同 ✓
32K prompt（graphs OFF 与 ON）：fused vs gather 逐字节相同 ✓
完整 18 项验收：正确性 17 项全 PASS（唯一 FAIL 是当晚受干扰的性能格，见 §22.2）
```

launch 账（`GGML_V100_OP_HIST=1`，稳态解码一个 pass）：**508 → 412（只开 GLU 折叠）→ 364（GLU + L2，§21）**
—— 每 token 净少 96 个 launch（GLU 折叠：`GLU+CONT` 两次 launch 合成一次；L2 折叠：干脆搭在已有的
pair launch 上，0 新增）。

---

## 21. 步骤十（2026-09-30）：`L2_NORM` 折叠 —— 第一次失败、定位根因后**改成搭在 pair launch 上，已通过并默认开启**

> ✅ **最终状态（当天晚些）**：不是"折进 SSM_CONV"，而是**搭在后面的 `ssm_alpha/ssm_beta` pair launch 上**
> （§17 的融合），因为**那个节点才是 L2 输出 buffer 已经活着的位置**（根因见 §20.7）。
> 现在默认开启（`GGML_CUDA_CONV_L2=0` 可关），短 prompt/32K/两种图模式全部逐位相同。
> 下面 21.1–21.3 保留第一次失败的过程记录（它正是定位根因的线索）。

### 21.1 为什么看起来是最值钱的一项

`qwen35.cpp:530-535` 把 conv 输出的前 32 个 head 组（q/k 拼一起）做了一次 L2 归一化：

```cpp
ggml_tensor * qk_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, 2*num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_qkv_mix->type, head_k_dim),   // nb1 = 128*4
        nb1_qkv, nb1_qkv * n_seq_tokens, 0);
qk_conv = ggml_l2_norm(ctx0, qk_conv, eps_norm);
```

独立内核 `l2_norm_f32<32>`（`norm.cu`，ncols=128 < 1024 → **单 warp、每 lane 4 个跨步元素**）：
`tmp = 0; tmp += x[tid]²; tmp += x[tid+32]²; tmp += x[tid+64]²; tmp += x[tid+96]²;`
再 `block_reduce<SUM,32>`（= `warp_reduce_sum<32>`）、`scale = rsqrtf(fmaxf(tmp, eps²))`、
`dst[col] = scale * x[col]`。

而 conv 内核的一个 block 正好持有"一个 head 的 128 个通道"（threads=128 = 头内通道数，
`bidy` = 通道组 = head）。所以**把 128 个值塞进 shared、让 warp 0 用同样的顺序做 4 次加法 +
同样的 `warp_reduce_sum<32>`**，理论上逐位一致，代价是 48 个 launch/token 消失（≈1.4%）。

### 21.2 实现与实测：**不逐位一致，而且不可复现** → 判定为存在竞争/未定义写，立即回退

实现（内核里按 token 循环内加 shared + 一次 `__syncthreads()`，匹配器要求
`ne[0]==128`、`warp_size==32`、`n_t<=32`、无 bias 融合等）编好后，`work/dump_l2.ps1` 的结果：

```
l2 OFF 重复运行      : BIT-IDENTICAL (6208000 logits)
l2 ON  vs l2 OFF     : bitDiff=5462994/6208000 maxAbsDiff=2.542E-001 firstAt=744960
l2 ON  重复运行      : bitDiff=5463005/6208000 maxAbsDiff=3.456E-001   ← 同一配置两次都不同！
graphs ON vs OFF     : bitDiff=5463007
关闭 PDL 后（GGML_CUDA_PDL=0）: bitDiff=5214691  ← PDL 不是原因
```

"同一配置两次都不一样" ⇒ 一定有竞争或未定义数据，哪怕把 PDL 关掉也一样。已排除的假设：
PDL 提前发射（否）、视图偏移（模型代码里 offset = 0，我核对了）、地址推导
（`(i*nheads + h)*128 + tid + 32k` 与 `qk_conv` 的 4D 视图/连续输出都对得上）。
没时间继续定位，**按纪律回退**：`ssm-conv.cu/cuh`、`ggml-cuda.cu` 已复原，
重新构建后 `dump_glu`（§20）与 `dump_matrix`（§16）的逐位验收**全部恢复为 BIT-IDENTICAL**。

### 21.3 留给下一次的三条线索

> （本节的三条线索已在当天晚上用掉：线索 1/2 的结论就是 §20.7 的"写太早"根因，最终实现见 §21.4。）

1. 复现方式：`work/dump_l2.ps1`（含 PDL 开关的对照）—— 现在的补丁里**没有**这块代码，
   需要重新实现；建议先把"只写 L2 输出、不与 silu 写在同一循环里"的版本做出来
   （例如把 L2 放到 `ssm_conv_long_token` 那种按 token 分块的布局里，减少 shared 复用窗口）；
2. 已知唯一"可疑但未证实"的点：**L2 输出 buffer 与 silu 输出的生命周期在图中相邻**
   （L2 早写 10 个节点），可以先用一个 `GGML_V100_DUMP_LAST` 风格的按名字 dump 把
   `qk_conv_l2` 的内容抓出来，与 OFF 路径逐位比，直接看是"写错位置"还是"被覆盖"；
3. 若最终无法做到逐位，也可以退一步：**不折 L2，只把它的 launch 与 silu 一起发**
   （fork 已有 `[SSM_CONV, UNARY]` 融合臂，L2 仍单独发）—— 没有收益，所以此路不值得走。

### 21.4 最终实现（默认开启）

* **matcher** `ggml_cuda_match_l2_norm_fold()`（在 L2 节点上判定）：`L2_NORM`、F32、连续、
  `ne[0] == 128`、`ne[1] >= 1`、`ne[2] == ne[3] == 1`（只做单 token 解码图）；它的输入必须是
  一个 view（`view_src != nullptr`，`nb[0]=4`、`nb[1]=512`、形状 (128, heads)）；
  然后**向后找第一个 `MUL_MAT`**，要求它正好是 §17 的 alpha/beta pair（否则放弃，不动图）；
* **执行**：在 L2 节点处**跳过**它（记 `pending_l2_idx`），走到那个 pair 节点时把 L2 节点作为
  参数传给 `ggml_cuda_mul_mat_vec_f_pair()`（不增加 launch！）；万一 pair 没匹配上，就地调用
  原来的 `ggml_cuda_op_l2_norm()` 兜底（那时目的地也已活）；
* **内核**（`mmvf.cu`）：pair 内核的前 `l2_nheads` 个 block（`blockIdx.y == 0`）各用一个 warp
  做**与 `l2_norm_f32<32>` 完全相同**的归约：每 lane 依次加 4 个跨步平方 → `warp_reduce_sum<32>`
  → `rsqrtf(fmaxf(sum, eps²))` → 写回 4 个元素；
* **逐位验收**（`work/dump_l2.ps1`，6 份 dump）：l2 OFF 重复 ✓ / l2 ON vs OFF ✓ / l2 ON 重复 ✓ /
  graphs ON vs OFF ✓，全部 6 208 000 logits 相同；32K prompt 在 graphs ON/OFF 下 `fused vs gather`
  也逐字节相同 ✓。

### 21.5 两件事的收益（诚实版）

launch 数是确定性证据：**508 → 364 次派发/token**（净 −96 个 launch + 少 48×24 KB 拷贝）。
但当晚的**墙钟测量不可信**：这台机器上跑着远程桌面 agent（累计约 4300 s CPU），
把 `llama-bench` 打出 ±3% 的抖动（甚至出现 43.44 t/s 的离群点）。当晚 8 对 `-p 512` 交错测量的
结果是 best-of **+0.83%**、中位数 **+0.52%**、均值 −0.21%（被离群点带偏）；三配置（off/glu/both）
那轮里 `both` 最好、`glu` 反而更低，明显自相矛盾 ⇒ **两个融合合计的墙钟收益只能给"≈ +0.5~1%"，
需要在没有远程桌面/安静环境下用 `work/run_acceptance.ps1` 复测**。

---

## 22. 本轮收尾记录（2026-09-30 晚，交付版）

### 22.1 二进制与验收

* 二进制：`build-v100\bin\ggml-cuda.dll` @ **2026-09-30 21:10:24**（GLU 去交织写 + L2 折叠**默认开启**；
  日志 0 error）；
* 一键验收 `work/run_acceptance.ps1` → `work/acceptance.txt`：**17/18 PASS**，唯一 FAIL 是 `perf.d90112`，
  而那一条是**测量事故**（同一格报告 `30.60 ± 8.71`，± 值本身就说明那轮被干扰；干净复测见 22.2）；
* 正确性 16 项全 PASS：短 prompt 逐位（4 配置 × 两图模式 + 重复）、pair 逐位（5 格）、
  **32K prompt 逐位（两图模式）**、文本逐字节、PPL、多序列（两图模式）。

### 22.2 性能（含"这台机器今晚不可信"的说明）

| 项目 | 数字 | 备注 |
|---|---|---|
| 派发 launch 数/token | **508 → 412（+GLU 折叠）→ 364（+L2 折叠）** | 确定性证据；每 token 净 −96 launch、−48×24 KB 拷贝 |
| d=0 tg128（本轮验收那一对） | 63.75 vs 61.44（+3.76%） | 那一轮的 gather 参考偏低（61.44，平时 60.5~60.7），故偏乐观 |
| d=0 tg128（pair on/off 同轮） | 63.93 vs 63.55（+0.60%） | 同轮成对，可信度高于跨轮 |
| **90K tg128（干净复测，-r 2）** | **40.26 ± 0.10 vs 38.44 ± 0.04 = +4.7%** | 全栈 A/B（`GGML_CUDA_GDN_ROWS_READ=0` + 两个新开关全关） |
| 两个新融合单独的墙钟 | best-of **+0.83%** / 中位数 **+0.52%**（8 对交错，`-p 512`） | 机器上有远程桌面 agent（累计约 4300 s CPU），抖动 ±3% |

> 今晚认真踩过的两个"假信号"，记下来省下一次：
> 1. `perf.d90112` 报 −19.98%（`30.60 ± 8.71`）——**± 值过大时应直接判定该测量作废**，
>    干净复测是 **+4.7%**，说明这条路径没有回归；
> 2. 中途一次"每个融合单独都快（39.6/40.3）、两个一起就崩（28.1）"的诡异结论，同样是干扰；
>    追这条假信号花掉了约 15 分钟，后来在同一配置上复测得到 40.21 才排除。
>
> 干扰源也查清楚了：**同机上有另一个任务在用 V100**——`nvidia-smi` 显示
> `<work>\<other-job>\...\<other-cuda-job>.exe` 常驻占用 **5128 MiB**（截取时为 0% util，
> 会间歇性活跃），另外还有远程桌面 agent `<remote-desktop-agent>`（累计 4271 s CPU）。
> **下次复测性能前先确认这张卡是空的**（`nvidia-smi --query-compute-apps=pid,process_name,used_memory`）。

### 22.3 交付物

* 补丁：[`outputs/state-path-fusions.patch`](state-path-fusions.patch)（134.6 KB / **14 个文件**；
  `git -c core.autocrlf=false apply -p1` 实测打完 14/14 文件哈希与当前树全同）；
* 报告：本文件（§19 度量工具与画像、§20 GLU 去交织写 + 根因修复、§21 L2 折叠 + 根因修复、§22 收尾记录）；
* 脚本（都在 `work/`）：`run_acceptance.ps1`（一键 18 项）、`dump_matrix.ps1`（状态路径 8 份 dump）、
  `dump_glu.ps1`、`dump_l2.ps1`、`dump_pair.ps1`、`dump_gon.ps1`、`ms_dump.ps1`（多序列逐位）、
  `perf_after_fix.ps1`、`regen_patch.ps1`、`cc_check.ps1`（单 TU 语法检查）、`build_incremental.ps1`；
* 新增开关：`GGML_CUDA_GLU_PERMUTE=0`、`GGML_CUDA_CONV_L2=0`（都默认开，可关做 A/B）、
  `GGML_V100_OP_HIST=1` / `GGML_V100_OP_TIME=1`（画像）、`GGML_V100_DUMP_LAST` / `GGML_V100_DUMP_NODES`
  （逐位验收与图分析）。

### 22.4 下一步（按价值/风险排序）

1. **在安静环境下复测**（关掉远程桌面 agent）：一轮 `run_acceptance.ps1` 就能把 d=0/90K 的性能格钉死；
2. `GLU (128,48) ×64` 里剩下的 24 个（16 个注意力层的 silu-gate + 8 个其他）与
   `z`/gate 投影的 MMVQ epilogue 合并（fork 已有 `mul_mat_glu_ops` 机制）；
3. `ROPE ×32` 对儿发（同 §17 的 pair 手法，预期 ≈0.2~0.4%）；
4. 预填侧 FA/MMA（§19 的画像显示 d=0 时 16 次 FA 只占 1.3%，但 90K 时会成大块）；
5. 仍然待办的那份 **admin `ncu`**（§15）：判断 PQ2_0 的 MMVQ 是带宽顶还是取指受限 ——
   这是"能不能再压解码"的唯一决定性证据。

---

## 23. 2026-10-01：干净机器上的最终验收 + "l aunch 到底有多少个"（真相比 dispatch 直方图大 4 倍）

### 23.1 干净环境下的 18/18（推翻昨晚的 17/18）

昨晚那条 `perf.d90112` FAIL 确认是干扰。今天 V100 空出来（`nvidia-smi --query-compute-apps` 无外部进程）
重跑一键验收：

```
SUMMARY: 18/18 checks passed
```

| 项目 | 交付版（全开） | 参考（`GGML_CUDA_GDN_ROWS_READ=0` + 两个新开关关） | Δ |
|---|---:|---:|---:|
| d=0 tg128 | **64.95 ± 0.26** | 61.44 ± 0.14 | **+5.71%** |
| 90K tg128 | **40.45 ± 0.08** | 38.91 ± 0.08 | **+3.96%** |
| d=0 pp512 | 816.66 | 814.36 | +0.28% |
| 90K pp512 | 505.16 | 501.09 | +0.81% |
| matvec pair on/off | 64.34 | 63.21 | +1.79% |

（正确性 16 项同样全 PASS：短 prompt/32K 逐位、pair 逐位、文本、PPL、多序列。）

### 23.2 新增工具：真正的 kernel launch 计数（`GGML_V100_LAUNCH_COUNT=1`）

§19 的 `OP_HIST` 只统计**到达 `ggml_cuda_compute_forward()` 的算子**，看不到"算子内部自己发的 kernel"
（q8_1 量化器、融合臂里的辅助 kernel 等）。这次在 `ggml_cuda_kernel_launch()` 这个总入口上加了一个
计数器（C++17 inline 变量，跨 TU 唯一），并按算子统计"该算子一共发了几个 launch"：

```
[op-hist] pass 7: 364 dispatched launches
[op-hist]   (kernel launches issued from inside those ops: 517)
[op-hist]   -- launches per op (incl. internal) --
[op-hist]   MUL_MAT           379  (1.57 per op)     ← 137 个是 MMVQ 内部的 q8_1 量化
[op-hist]   FLASH_ATTN_EXT     32  (2.00 per op)     ← 每个 FA 还带一个内部成对 kernel
[op-hist]   其余（CONT/ROPE/SET_ROWS/GLU/MUL/GET_ROWS）都是 1.00 per op
```

而**整个 pass 的真实 launch 总数是 1431**（`[launch-count]`，稳态解码每 token 稳定 1431）。
也就是说：

> **364 个节点被派发、其中 517 个 launch 是这些算子自己发的；剩下 1431 − 517 = 914 个 launch
> 来自"融合臂"（fork 自带的 `[RMS_NORM,MUL]`、`[ADD,RMS_NORM,MUL]`、`[UNARY,MUL]`、
> `[SSM_CONV,UNARY]`… 以及本轮的两处折叠）。**

把两个数字和 §19 的时间画像放在一起：

* 已派发算子的 kernel 时间 ≈ 7.9 ms（其中 MUL_MAT 占 82%）；node loop ≈ 15.7 ms（真实值）；
* 差的 ≈ 7.8 ms = 914 个融合/内部 launch 的 kernel 时间 **+ kernel 之间的空隙**；
* 1431 × ~5 µs（V100 上同流依赖 kernel 的启动延迟）≈ **7 ms** —— 与上面那个差值吻合。
  ⇒ **解码大约一半时间花在"kernel 之间的启动/依赖延迟"上**，而 `graphs ON/OFF` 实测等价
  （63.55 vs 63.59，§19.2）说明瓶颈在 **GPU 侧**、不是 CPU 派发。

### 23.3 这条数据怎么用（结论：剩下的路要么"大规模融合"，要么动 MMVQ 本身）

1. **权重带宽已经接近实际可用上限**：d=0 = 15.4 ms/token，权重 7.2 GB/token ⇒ 若把它全算成权重读，
   等效 ~468 GB/s；扣掉非权重部分（~3-5 ms）后 ≈ 650 GB/s，即 V100 上**实际可达带宽（~700-720 GB/s）
   的 ~90%** ⇒ 权重这条路基本到顶，**除非** ncu 证明 `mul_mat_vec_q<142,…>` 是 ALU/取指受限（§15 的问题）；
2. **launch 数量是剩下唯一的大头**：1431 个/token，其中 **914 个来自融合臂**。但实测"每砍一个 launch"
   只值 ~1 µs（§21.5：−96 launch ≈ +0.5~0.85%），所以想再拿 5% 得砍掉 ~700 个 launch ——
   那需要把**多个融合臂再合并**（例如 `[RMS_NORM,MUL]`+`[MUL_MAT,GLU]` 之类的跨臂融合），属于
   "大工程 + 需要 ncu 先证明这些 launch 真的在关键路径上"；
3. **MTP / 投机解码 —— 不是"不支持"，但当前跑不起来，值得单独查一次**：
   * 这棵树**支持**投机解码（`README.md`："Speculative decoding (dspark) is supported via mainline's
     draft-dspark plus fork patches"，drafter 按模型版本发布、老版本需 `gguf-dspark-to-dflash` 转换）；
     `common/speculative.cpp` 里有四种 draft 类型：`draft` / `dflash` / `eagle3` / **`mtp`**，
     其中 `mtp` 是**针对主模型再建一个 `LLAMA_CONTEXT_TYPE_MTP` 轻量上下文**（不需要额外权重文件）；
   * 但 9/27 的探针（`work/mtp_dump.log`）在 `llama_init_from_model(model_tgt, cparams)` 这一步就失败：
     `failed to create MTP context` —— 最可能是**显存**（第二个上下文 + KV/计算 buffer 在 32 GB 上放不下，
     探针当时用的是默认 `-c`/`-ngl 99`）或模型元数据缺 MTP 标记，**不是"实现不支持"**；
   * 另外本地 `<models>\Ternary-Bonsai-2-27B-gguf\` 里**没有 drafter 文件**（只有 PQ2_0/PTQ1_0/mmproj），
     所以"外部 drafter"这条路需要先拿到 drafter（或转换），而 `mtp` 那条不需要。
   ⇒ 值得花 30~60 分钟单独排查（把 draft 上下文的 `-c` 调小、看是否只是 OOM）；若跑通，
     贪心采样下**逐 token 等价**，且权重每 token 只读一遍 ⇒ 2~3× 的潜力，是剩下最大的一块，
     但属于另一个范畴（不属于 CUDA 状态路径）。

### 23.3b 两个候选的收益预估（据本轮实测的边际值）

边际值来自本轮两次落地：**−96 launch ⇒ +0.5~0.85%**（即 **~1~1.4 µs / launch**，远小于"
5 µs 的裸延迟"，因为 GPU 已经在重叠一部分）。

| 方向 | 可行范围 | 预估收益 | 依据 |
|---|---|---|---|
| (a) 跨融合臂合并（把 914 个 arm launch 再并） | 一次会话 ~100~200 个 launch | **+0.3~1%** | 按实测 1~1.4 µs/launch |
| (a) 天花板（理论上全并掉） | 全 914 个 | ≈ +6% | 0.9 ms / 15.4 ms；但多数 arm 已是一组一个 kernel，需要"跨组"融合（量化折进生产者、norm 折进 matmul 读等），每个都有逐位风险 |
| (b) MMVQ 内核（ncu 指引） | 取决于 ncu 结论：**带宽顶 ⇒ ≈0%**；**ALU/取指受限 ⇒ 有空间** | 现实 **0~+3%**，理论上限 ≈ +13% | 权重 7.2 GB/token、等效 580~690 GB/s ≈ 实际可达带宽 700~720 GB/s 的 80~95% |
| (c) MTP / 投机解码 | 需要先排查 30~60 min | **0（现在）→ 2~3×（若跑通）** | 树支持，但 MTP 上下文创建失败（大概率 OOM）；本地无 drafter 文件 |

⇒ **CUDA 这条路上现实剩下的空间约 +1~3%**；想要更大的收益，方向是**预填**：
pp512 = 816 t/s 对应约 44 TFLOPS 等效吞吐（V100 FP16 tensor core 峰值 125 TFLOPS 的 ~35%），
而 PQ2_0 这种 2.13 bpw 格式本来就吃不到 tensor core，P1-a/P1-b 已经在 FA/MMA 上做过一轮 —— 
那里是"还能大幅挖"的地方，但同样需要 ncu 先把预填的瓶颈画像拿出来。

### 23.4 本轮交付物状态

* 二进制：`build-v100\bin\ggml-cuda.dll` @ 2026-10-01 14:54（含 §22/§23 的仪表；**默认行为与
  18/18 验收的那一版完全一致**——新加的 `GGML_V100_LAUNCH_COUNT` 默认关、零开销）；
* 补丁：[`outputs/state-path-fusions.patch`](state-path-fusions.patch)（14 文件；`git -c core.autocrlf=false
  apply -p1` 实测 14/14 哈希全同）；
* 报告：本文件（§19→§23）；
* 新仪表开关：`GGML_V100_LAUNCH_COUNT=1`（真实 launch 计数 + 每算子 launch 归属）、
  `GGML_V100_OP_HIST=1`、`GGML_V100_OP_TIME=1`、`GGML_V100_DUMP_LAST` / `GGML_V100_DUMP_NODES`。


---

## 24. 步骤十一（2026-10-01 下午）：FWHT -> q8_1 预量化（默认开、逐位通过）+ **"launch 数不是杠杆"的实测结论** + 第一次真正的单 token 成本分解

这一轮的出发点是交接文档里"剩下的只有 launch 数"的判断（§23.3）：稳态解码 1431 个 launch 里 914 个
来自融合臂，按 1~1.4 µs/launch 折算，砍掉 257 个量化 launch 应该值 +1.7~2.3%。做完之后结论**反转**了：
launch 数确实砍掉了 18%，墙钟**一点没动**；而新的计时器第一次把"时间到底花在哪个臂上"量清楚了。

### 24.1 目标：每个 Hadamard 旋转后面那 257 次 q8_1 量化

从节点图看得很清楚（`GGML_V100_DUMP_NODES=1`，`work/parse_nodes.py` 可复现）：

```
 70 MUL       node_72   = attn_post_norm-0 * prism.hadamard.signs.5120      (5120,1)
 71 RESHAPE   (reshaped)= (1024,5)                       <- 视图
 72 MUL_MAT   node_74   = FWHT(node 71)                 <- hint=SRC0_IS_HADAMARD，被融合臂吃掉
 73 RESHAPE   (reshaped)= (5120,1)                       <- 视图
 74 MUL_MAT   ffn_gate-0: src1 = node 73                <- MMVQ：src1 量化成 q8_1（一次 launch）
 75 MUL_MAT   ffn_up-0  : src1 = node 73                <- 同一个张量，缓存命中，不再量化
```

`ggml_cuda_q8_1_cache_get()` 的 key 就是**节点 73 这个视图张量**，而它的数据就是 FWHT 的输出。
所以把量化折进 FWHT 内核，就能让 74/75 两个 matvec 一个量化 launch 都不用发。

### 24.2 实现（`ggml_cuda_fwht_q8`）

* **数学上逐位可行**：q8_1 的块是 32 个连续元素；块内核 `fwht_cuda_block` 用 NT=256 时，
  元素 `(i*NT + tid)` 落在第 `(i*NT + tid)/32` 个块里，也就是 **第 `tid/32` 个 warp 在寄存器 `i` 上的
  那 32 个元素**。蝶形算完之后，每条 warp 直接对自己那 32 个值跑
  `warp_reduce_max<32>` / `warp_reduce_sum<32>`，再按 `d = amax/127`、`q = roundf(xi/d)`、
  `ds = make_half2(d, sum)` 写块 —— 与 `quantize.cu` 的 `quantize_q8_1` **同归约、同顺序、同舍入**，
  而且写进 dst 的和参与归约的是**同一个寄存器值**，所以是构造性的一致，不是"应该一致"。
* 新增 `ggml_cuda_fwht_q8{buf, blocks_per_row, chunks_per_row}`，
  `ggml_cuda_op_fwht_signed{,_view}()` 多一个可选实参；只有块内核（N ≥ 512、无 `GGML_CUDA_FWHT_LEGACY`）
  支持，其它情况**直接返回 false**，避免"预留了槽位却没写"。
* 图侧新增匹配器 `ggml_cuda_match_fwht_q8()`：只接受 FWHT 输出的**连续视图**（`view_src == mm` 或
  `data` 相同）、`ggml_nelements` 相同、`ne10 % N == 0`、**无行填充**（`GGML_PAD(ne10,512) == ne10`）、
  `ne[1] <= 8`，并且至少有一个消费者会走 MMVQ（复制 `ggml_cuda_mul_mat()` 的判据：
  `!should_use_mmvf && should_use_mmvq && quantized src0`）。行号映射是
  `row = r / chunks_per_row`、`chunk = r % chunks_per_row`，对 `(1024,5)->(5120,1)` 和直接
  `(1024,5)` 两种形态都对。
* 槽位用 `ggml_cuda_q8_1_cache_get()` 以**消费者张量**为 key 预定，把裸指针交给内核；
  MMVQ 随后 `needs_quantize == false` 直接复用 ⇒ 量化 launch 真正消失。
* 兜底：`ggml_cuda_q8_1_cache_drop()`（预留但 launch 落空时把条目删掉，否则后续消费者会拿到脏数据）。
* 开关 `GGML_CUDA_FWHT_Q8_1=0`（默认开）。

### 24.3 逐位验收（`work/dump_fwhtq8.ps1`）

| 对比 | 结果 |
|---|---|
| 短 prompt：OFF 两遍 / ON 两遍 / graphs ON | 5 份 dump，各 6 208 000 logits，**全部逐位相同** |
| 32K prompt：graphs OFF/ON × ON/OFF | 4 份 dump，各 1 489 920 logits，**全部逐位相同** |

（多序列 2 路、文本、PPL 由 `work/run_acceptance.ps1` 覆盖，同样全过。）

### 24.4 launch 数掉 18%，墙钟不动

`GGML_V100_LAUNCH_COUNT=1`（稳态解码一个 pass）：

```
ON : 1414 1414 915 819 1174 1174 1174 ...   ← 稳态 1174 / token
OFF: 1671 1671 915 819 1431 1431 1431 ...   ← 稳态 1431 / token
```

正好 **−257**（= 每 token 的 q8_1 量化次数；预填 pass 同样 −257）。

交错 A/B（`work/ab_fwhtq8.ps1`，8/4/3 轮，graphs OFF）：

| d | ON best / median | OFF best / median | Δbest | Δmedian |
|---|---:|---:|---:|---:|
| 0 | 65.85 / 65.27 | 65.87 / 65.34 | -0.03% | -0.11% |
| 32K | 54.68 / 53.51 | 53.84 / 53.54 | +1.56%(离群) | -0.05% |
| 90K | 41.33 / 41.04 | 40.81 / 40.75 | +1.27% | +0.71% |

pp512（d=0 / 90K）：+0.10% / +0.40%（都在噪声里）。
⇒ **d=0、32K 是平局；90K 有一个 ~+0.7% 的小信号**（三份 ON 全大于三份 OFF，可以留着，但不该当卖点）。
拐点现象本身比数字重要：**"砍 launch"这条路在这台机器上不涨钱。**

### 24.5 根因：新的 per-arm 计时器（`GGML_V100_OP_TIME` 现在也计时融合臂）

之前 `GGML_V100_OP_TIME` 只把 `ggml_cuda_compute_forward()` 包在 event 对里；融合臂是在
`ggml_cuda_try_fuse()` 里发的 kernel，**完全落在"未解释的 9 ms"里**。这轮把 event 对也包住
`ggml_cuda_try_fuse()`，并按"臂的第一个节点的 op"归档（新增 `GGML_V100_ARM_DUMP=1` 打印每次融合的
op 序列，`work/armdump.err` 是这次的记录）。稳态解码一个 token（ON）：

| 项 | ms / pass | 次数 | 每次 | 说明 |
|---|---:|---:|---:|---|
| `MUL_MAT`（已派发的 MMVQ 权重 matvec） | 6.40 | 242 | 26.5 µs | 6.80 GB/token 权重的主要部分 |
| `arm:MUL_MAT` | 5.36 | 120 | 44.7 µs | 80×`[MUL_MAT,ADD]` + 40×`[MUL_MAT,MUL_MAT,GLU]`，**里面是真正的权重 matvec**（fork 的 epilogue 融合） |
| `arm:MUL` `[MUL,RESHAPE,MUL_MAT]` | 1.83 | 257 | 7.1 µs | Hadamard（FWHT）臂 |
| `arm:RMS_NORM` `[RMS_NORM,MUL]` | 1.12 | 161 | 6.9 µs | |
| `arm:ADD` `[ADD,RMS_NORM,MUL]` | 0.44 | 48 | 9.2 µs | |
| `arm:SSM_CONV` `[SSM_CONV,UNARY]` | 0.25 | 48 | 5.2 µs | |
| `FLASH_ATTN_EXT` / `ROPE` / `SET_ROWS` / `GLU` / `CONT` | 0.25 / 0.15 / 0.14 / 0.13 / 0.09 | 16/32/32/24/16 | | |

合计 16.25 ms，与**不开仪表**的 node-loop（16.27 ms）一致 ⇒ 这套账是可信的。

ON 与 OFF 的差：

| 桶 | ON | OFF | Δ |
|---|---:|---:|---:|
| `MUL_MAT`（派发） | 6.220 | 6.677 | **-0.457** |
| `arm:MUL`（FWHT） | 1.784 | 1.401 | **+0.383** |
| 其它 | | | ≈0 |

⇒ **量化的工作是"原地搬家"**：删掉 257 个量化内核（每个 1.17 µs），FWHT 每个涨 1.09 µs。
量化内核不是"纯启动开销"：它要读 20 KB 激活、写 10 KB、跑两个 32 宽归约，这些活换个地方干还是要干。
**§23 里"914 个融合臂 launch ≈ 7 ms 启动延迟"的解释不成立** —— 融合臂的时间是真计算
（每个 5~9 µs，5~17 个 block 的小内核是延迟受限，不是带宽、也不是 launch 间隙）。

### 24.6 顺带做的 FWHT 线程数实验（负结果，写下来免得再试）

块内核一个 block 一行，这个模型每次只有 5~17 行 ⇒ 只用得上 5~17 个 SM。按"少一层 shared 阶段就少
两个 `__syncthreads`"的想法试了 NT=64 / 128 / 256 / 512：

| NT | `arm:MUL` / pass | 每次 |
|---|---:|---:|
| 64 | 2.32 ms | 9.0 µs（**+27%**） |
| 256（原值） | 1.83 ms | 7.1 µs |
| 512 | 1.79~1.86 ms | 7.0~7.2 µs |

⇒ 这个内核是**延迟受限**（内存往返 + 蝶形链），阶段数不是瓶颈、warp 数量才是；NT=256 已经到地板，
**改线程数没有肉**。要动它只能"少发几次"（例如把它折进消费端 matvec，见 §24.8）。

### 24.7 权重带宽账（给 (b) 用）

* 从节点图按张量算：每 token 读的量化权重 = **6.800 GB**（401 个量化 `MUL_MAT`，6 张/层 × 64 层 + 输出投影；
  `work/parse_nodes.py nodedump.err 5 ops 5120` 之类可复算）。
* 权重 matvec 的总时间 = `MUL_MAT` 6.40 + `arm:MUL_MAT` 5.36 = **11.76 ms** ⇒ **578 GB/s**。
* 剩下的 ~4.4 ms 才是那些延迟受限的小臂 + FA/ROPE/SET_ROWS/GLU。
* V100 SXM2 理论 900 GB/s、实测 stream 型 roof 一般 780~850 GB/s ⇒ 现在等效 **68~74%**。
  但这是"内核串行链里的平均带宽"；两个 matvec 桶各自的真实占用只有 ncu 能说清（§24.9）。

### 24.8 (a) 还剩什么（按"能不能真的少干活"排序）

1. **把 FWHT 折进消费端 matvec**（唯一能真正删掉这 1.83 ms 的办法）：matvec 的每个 block 需要的是
   整条激活（5120 个 float = 20 KB，L2 完全放得下），在 block 里"读原激活 -> 做 5 个 1024 点 Hadamard ->
   量化 -> 点积"就能把 FWHT 臂和量化**一起**删掉。代价是每个 block 重复做变换/量化（激活读会放大 ~4 倍，
   但都在 L2 里），收益上限约 1.8~2.5 ms ≈ **+12~16%**。风险：块内归约顺序必须与 `fwht_cuda_block`、
   `quantize_q8_1` 逐位相同（可做，但要小心），属于"下一个大件"。
2. `arm:RMS_NORM` 1.12 ms（161 次）：把 norm 的输出**直接写进消费者要的布局**（类似 §20 的 GLU 手法），
   省掉一次读+写；上限 +1~2%，实现难度中等。
3. `arm:MUL_MAT` 里的 `[MUL_MAT,MUL_MAT,GLU]`（40 次，含真权重读）已经是最优形态；`[MUL_MAT,ADD]`（80 次）
   可以再想：ADD 的加数能否用 `epilogue` 折进 matvec（若还没折）。
4. **别再按 launch 数做规划**：这轮的 257 个 launch = 0%，而 §20/§21 的 -96 个 launch = +0.5~0.85%，
   差别在于**后者同时删掉了一次内存搬运**。判据换成"这一刀删掉了多少字节"。

### 24.9 (b) MMVQ 的现状与"证据请求"（需要管理员跑一次 ncu）

* 我在沙箱里试了一次 ncu（`work/ncu_probe.bat`）：**`ERR_NVGPUCTRPERM`** —— 与 §15 一致，
  这台机器上非管理员拿不到性能计数器。
* 已经把命令写好：**`work/ncu_mmvq_roof.bat`（右键 -> 以管理员身份运行）**。
  它只 profile `mul_mat_vec_q`（`--launch-skip 200 --launch-count 260`，约一个稳态 token 的量），
  抓 `gpu__time_duration.sum` / `dram__bytes.sum` / `sm__throughput` / `smsp__issue_active`，
  并导出 `桌面\bonsai_mmvq_roof.csv`。
* 判读规则：
  * `dram__bytes.sum / gpu__time_duration.sum` ≈ 700~850 GB/s，且 `sm__throughput` 低
    ⇒ **带宽顶到了**，(b) 收手（现实收益 0%）；
  * `sm__throughput` / `smsp__issue_active` 高而 DRAM 不满 ⇒ 值得改 vec_dot
    （PQ2_0 每个 32 元素块是 8×`dp4a` + 16×`__byte_perm`，符号展开占了 2/3 的指令）。
* 现在的经验值：整条解码的等效带宽 **578 GB/s**；每 32 元素 8 条 dp4a（=32 MAC）对应
  **1.75 TMAC/s ≈ V100 dp4a 峰值的 5.6%**，指令发射占比也只有几个百分点
  ⇒ 如果 ncu 显示"发射/ALU 都很空、带宽只有 ~60%"，那瓶颈就是**访存并行度（MLP）/占用**，
  对应手段是"加每线程独立加载数 / 调 rows_per_block / 提高 block 并发"，**而不是减指令**。

### 24.10 本轮交付物

* 代码：`fwht.cu/.cuh`（q8_1 输出 + 支持性判据）、`mmvq.cu/.cuh`（导出 q8_1 槽位 get/drop）、
  `ggml-cuda.cu`（`ggml_cuda_match_fwht_q8()` + 融合臂改动 + per-arm 计时 + `GGML_V100_ARM_DUMP`）；
* 补丁：[`outputs/state-path-fusions.patch`](state-path-fusions.patch)（**16 文件**，168 KB；
  `git -c core.autocrlf=false apply -p1` 到 pristine 副本后 **16/16 SHA256 与工作树全同**）；
* 脚本：`work/dump_fwhtq8.ps1`（逐位验收）、`work/ab_fwhtq8.ps1`（交错 A/B）、
  `work/parse_nodes.py`（节点图查询）、`work/ncu_mmvq_roof.bat`（(b) 的证据请求）、
  `work/armdump.err`（本轮的融合清单）；`run_acceptance.ps1` 的 `TgOf` 解析改成按列取数（± 的编码会变）；
* 新开关：`GGML_CUDA_FWHT_Q8_1=0`、`GGML_V100_ARM_DUMP=1`。

### 24.11 一键验收（`work/run_acceptance.ps1`，交付版二进制，graphs OFF）

```
SUMMARY: 18/18 checks passed
```

| 项目 | 交付版（全开） | 参考（`GGML_CUDA_GDN_ROWS_READ=0`） | Δ |
|---|---:|---:|---:|
| d=0 tg128 | **66.47 ± 0.10** | 63.03 ± 0.19 | **+5.46%** |
| 90K tg128 | **41.21 ± 0.07** | 39.99 ± 0.08 | **+3.05%** |
| d=0 pp512 | 818.47 | 814.75 | +0.46% |
| 90K pp512 | 505.90 | 504.47 | +0.28% |
| matvec pair on/off（d=0） | 66.27 | 64.34 | +3.00% |

正确性 15 项全 PASS（短 prompt 5 配置逐位、32K 逐位、文本、PPL、多序列）。
对比 §23.1 的 64.95 / 40.45：**今天下午这台机器更干净**（63.03 vs 61.44 的参考臂同样上移），
所以跨天的绝对值不能直接比，A/B 的百分比才是可比的（+5.46% vs 上午的 +5.71%，一致）。

（顺手修掉了 `run_acceptance.ps1` 里 `TgOf()` 的一个解析 bug：llama-bench 的 `±` 是 UTF-8，
不同控制台代码页下可能到达为其它字节，早先的正则匹配会静默取到 `6.70 GiB` 那一列，
把性能三项误判为 0.00% 而报 FAIL。现在改成"取 `tg128` 后面那一列"。）

---

## 25. 步骤十二（2026-10-05）：PQ2_0 权重对齐重排（b1）——逐位通过、**但收益为负**；以及 (b) 的最终判定

### 25.1 动机：ncu 给出的第一手画像 + SASS

你按 §24.9 跑了 `work/ncu_mmvq_roof.bat`（报告在桌面，CSV 导出那步没成功，不影响：`ncu --import`
直接可读，已存 `work/ncu_mmvq_roof.csv`）。稳态解码里 260 次 `mul_mat_vec_q<142,…>` 的账：

| grid | 次数 | 读入 | 耗时 | 等效 | DRAM% | SM/发射% |
|---|---:|---:|---:|---:|---:|---:|
| 124160（输出投影） | 1 | 365 MB | 567 µs | 644 GB/s | 71.8 | 63.7 |
| 8704（ffn, GLU 融合臂） | 29 | 1584 MB | 2369 µs | 669 GB/s | 74.5 | 50.7 |
| 8704 | 35 | 1039 MB | 1738 µs | 598 GB/s | 67.7 | 55.5 |
| 5120 | 35 | 674 MB | 1169 µs | 577 GB/s | 64.5 | 50.5 |
| 3072 | 35 | 460 MB | 839 µs | 549 GB/s | 60.9 | 44.0 |
| 2560 | 35+57 | 459+1511 MB | 820+2444 µs | 560/618 GB/s | 62/68 | 42/45 |
| 6144 | 11 | 245 MB | 416 µs | 590 GB/s | 65.5 | 52.7 |
| **512（attn_k/attn_v）** | 22 | 72 MB | 206 µs | **350 GB/s** | **38.4** | **17.7** |
| 合计 | 260 | 6.41 GB | 10.57 ms | **607 GB/s** | ~67 | ~50 |

⇒ 权重 matvec 占一个 token 的 **68%**；DRAM 只到 61~75%，而发射槽也占到 42~64%：**两头都半满**。

`cuobjdump` 出的 SASS（`mul_mat_vec_q<142,1,0,0,0,0>` 主循环，134 条指令处理 64 个权重）：

```
 16 IDP.4A      ← 真正的 MAC（每 4 个权重一条）
 40 PRMT + 16 LOP3 + 8 SHF   ← 2-bit → int8 的符号展开（48%！）
 11 LDG.E.U16   ← 权重：**16 位**加载
  8 LDG.E       ← q8_1 激活
 17 IMAD …      ← 地址/循环
```

根因看起来很清楚：`block_pq2_0 = {ggml_half d; uint8_t qs[32]}` 是 **34 字节**，所以 lane 需要的 8 字节
（`34*kbx + 2 + 8*iqs`）永远只能 2 字节对齐，编译器只能发 16 位加载；一个 warp 的 32 条 lane 每条指令
只用到扇区的 1/4（8 个扇区/指令，256 字节里用 64 字节）。**假设**：把 qs 对齐到 8 字节 ⇒ 每 lane 一条
`LDG.64`，L1 波前降 4×、发射槽降 ~5% ⇒ 期望 +5~10%。

### 25.2 实现（已落地，默认关）

* 重排布局：**qs 连续 32 字节/块（8 字节对齐）+ scale 单独 2 字节/块**，总字节数与原来完全一样
  （34 B / 128 权重，0 额外带宽）；
* 懒加载的 per-tensor 重排缓存（`cudaMalloc` 一次、进程内不释放，因为捕获的 CUDA graph 会引用它）；
  重排内核把 `17×uint16`（[0]=scale，[1..16]=qs）拆成两个数组，16 位访问（源本身只有 2 字节对齐）；
* 新增 `vec_dot_pq2_0_q8_1_repacked()`：`uint2` 一次取 4 个 int16，把 `q` 按原样重建
  （`(int)(int16_t)(uint16_t)…` 精确复刻原来的符号扩展），后面的展开/归约/舍入**一字不改** ⇒ 构造性逐位；
* `ggml_cuda_mm_fusion_args_device` 增加 4 个指针（x_qs/x_ds/gate_qs/gate_ds）；
  MMVQ 派发链加 `pq2_repacked` 模板参数（只有 PQ2_0 多一份实例化），入口处 `if fusion.x_qs != nullptr` 转一次；
* 主机侧在 `ggml_cuda_mul_mat_vec_q()` 里懒创建（GB10 的 AoS 路径不参与）；开关 `GGML_CUDA_PQ2_0_REPACK`。

### 25.3 踩到的坑（写下来防复发）

第一次跑出来**同一配置两次 dump 都不一样**（`bitDiff=248319`，maxAbs 0.088）——不是舍入顺序，是竞争：

> 重排内核被我发在 **null stream**（`stream = 0`）上，而 matvec 发在后端的 **non-blocking stream** 上，
> 两者之间没有隐式同步 ⇒ matvec 可能读到还没重排完的数据。

改成在 `ctx.stream()` 上发（并把捕获状态检查也移到那条流上）之后，**6 份 dump 全部逐位一致**。
教训：任何旁路 buffer 的"生产者 kernel"必须和消费者同流，null stream 的隐式同步不能依赖。

### 25.4 逐位验收（`work/dump_pq2repack.ps1`）

| 对比 | 结果 |
|---|---|
| 短 prompt：repack ON vs packed / ON 跑两遍 / graphs ON / graphs ON(packed) | 4 份 6 208 000 logits **全同** |
| 32K prompt：repack ON vs packed / graphs ON vs OFF | 2 份 1 489 920 logits **全同** |

### 25.5 性能：**负结果**

交错 A/B（`work/ab_pq2repack.ps1`，graphs OFF）：

| d | repack ON（best / median） | packed OFF（best / median） | Δbest | Δmedian |
|---|---:|---:|---:|---:|
| 0 | 63.67 / 54.34 | 64.73 / 54.40 | **-1.64%** | -0.11% |
| 32K | 53.72 / 48.09 | 53.53 / 51.00 | +0.35% | -5.7%(噪声) |

（这次机器抖得厉害：同一个 arm 的样本在 50~64 t/s 之间跳，但两个 arm 里"快样本"成对出现且
packed 始终高 ~1.5% ⇒ 结论可信：**重排没有收益，反而略慢**。）

SASS 侧确认改动**确实生效**：主循环 134 → **127 条**指令、权重加载 **11×`LDG.E.U16` → 2×`LDG.E.64`**、
寄存器 47 → 48（占用率不变）。

⇒ **假设被证伪**：这个内核不是 L1/发射受限，2 字节权重加载不是瓶颈。它是**访存延迟 / DRAM 受限**
（DRAM 61~75%、占用率 ~62%、每个线程只做 1~2 次 K 迭代）。
加上代价是 **+5.7 GB 显存**（实测：packed 18.3 GB vs repack 24.0 GB），所以**默认关**，
只留 `GGML_CUDA_PQ2_0_REPACK=1` 给以后的 A/B。

### 25.6 (b) 的最终判定，以及 b2 的实证

* 已证伪的假设：**指令数**（§24 的 q8_1 实验）、**L1 波前 / 16 位加载**（本轮 b1）。两者都没有带来墙钟收益。
* 现状：matvec 10.57 ms/token（68%）、6.41 GB、607 GB/s（DRAM 61~75%）、发射 42~64%、占用 ~62%。
* **b2 的实证（ncu + 节点图）**：`grid=512` 那批是 **attn_k / attn_v**（每层 2 个、共 32 个/token，
  5120×1024 权重 = 1.36 MB），它们只有 **38.4% DRAM / 17.7% 发射**，每次 launch 却有 **3.28 MB** DRAM
  流量 —— 其中只有 1.36 MB 是权重，**其余 ~1.9 MB 是激活重读**（每个 block 都要重读整条 5120 宽的
  q8_1 激活；512 个 block × 5.8 KB ≈ 3 MB）。修法有二：
  1. 对"输出行数少"的 matvec 提高 `rows_per_block`（block 数减半 ⇒ 激活重读减半）；
  2. 把相邻的 K/V 两个 matvec 做成 **pair launch**（§17 的 `x2`/`dst2` 机制现成，ssm_alpha/beta 已经这么做），
     一次 launch 算两条，顺带省掉 16 次 launch。
  预估合计 **+0.5~1%**（32 个 launch × ~3 µs 的固定开销）。
* **但是**：这台机器现在的测量噪声达到 **±25~30%**（同一 arm 的样本 50 → 64 t/s），任何 <1% 的改动
  都无法在这里测出来。b2 的具体实现（以及任何 <1% 的收尾）应该等机器安静时再做，否则只是烧电。

### 25.7 本轮交付物状态

* 代码：`mmvq.cu`（重排缓存 + 重排内核 + 第二套 vec_dot 调用路径 + 派发链的 `pq2_repacked`）、
  `vecdotq.cuh`（`vec_dot_pq2_0_q8_1_repacked`）、`common.cuh`（fusion 结构体 +4 指针）；
* 开关：`GGML_CUDA_PQ2_0_REPACK=1`（**默认关**，实验用）；
* 脚本：`work/dump_pq2repack.ps1`（逐位）、`work/ab_pq2repack.ps1`（交错 A/B）、`work/ncu_mmvq_roof.csv`
  （ncu 导入结果）、`work/sass_pq2_0.txt` / `work/sass_pq2_repacked.txt`（两版 SASS）；
* 结论一句话：**PQ2_0 的 matvec 不是"加载太窄"，是"延迟没盖住"**；下一步（b2 或预取）都应当以
  "提高访存并行度"为目标，而不是再减指令。

### 25.8 收尾：交付版仍然 18/18

重排改回**默认关**之后重跑一键验收（`work/acceptance.txt`，备份 `work/acceptance_1005_pq2repack_off.txt`）：

```
SUMMARY: 18/18 checks passed
```

* 正确性 15 项全 PASS（短 prompt 5 配置逐位、32K 逐位、文本、PPL、多序列）；
* perf：d=0 **61.61 vs 60.92 = +1.13%**、90K **40.98 vs 39.48 = +3.80%**、pair +4.46%。
  ⚠️ 注意这一轮 d=0 的绝对值和增益都被噪声压扁了：**同一个配置（default 与 pair-on 都是全开）
  在相隔 2 分钟的两次测量里分别是 61.61 和 65.15（差 5.7%）**，所以这一格的 +1.13% 不能当结论；
  今天上午同一版二进制的干净值是 **+5.46%**（§24.11）。跨天/跨时段只能比 A/B 的相对值，且要交错多轮。

补丁已重新生成并验证：`outputs/state-path-fusions.patch`（**17 文件、194 KB**），
`git -c core.autocrlf=false apply -p1` 到 pristine 副本后 **17/17 SHA256 与工作树全同**。

---

## 26. 步骤十三（2026-10-05）：MTP 投机解码 —— **d=0 +41%、32K +33%、96K +18%**，且贪心输出逐字节不变

### 26.1 先侦察"作者有没有更新"（这一步做对了，省了大量无用功）

| 事实 | 证据 |
|---|---|
| **官方模型文件里没有 MTP 头** | `prism-ml/Ternary-Bonsai-2-27B-gguf`（HF 最后更新 2026-09-25，就是我们本地这份）：851 张量，无 `blk.*.nextn.*`，无 `nextn_predict_layers` 元数据。F16 文件同样 851 张量（用 HTTP Range 拉了它的头部核对）。 |
| 9/27 那次探针的失败原因**不是 OOM** | `src/llama-context.cpp:4093`：`ctx_type == MTP && hparams.n_layer_nextn == 0` → 直接返回 nullptr，日志是 "model doesn't contain MTP layers"。 |
| fork 的 MTP 代码**我们这棵树已经有** | `src/models/qwen35.cpp` 的 `graph_mtp` + Hadamard 逆变换（PR #205，2026-09-21 合并；我们的快照是 9/26）。模型卡 `KNOWN_ISSUES.md` 里"MTP is refused for these files — fixed in source (#205)"说的就是这条。 |
| MTP 头由社区打包提供 | `ProCreations/Ternary-Bonsai-2-27B-MTP`（作者 PR #205 的复现就用它）、`decent-jawfish/bonsai-2-27b-mtp`（含嫁接脚本）、`killy369/...-MTP-drafter-GGUF`（sidecar drafter，但按 PR #205 的分析会重复词表、rho≈0.43 反而亏）。 |
| **零字节开销的嫁接** | 15 张 `blk.64.*` 张量（一个完整 transformer 块 + nextn 投影）来自 `unsloth/Qwen3.8-27B-GGUF` 的 `UD-Q2_K_XL`，只占源文件**最后 0.35 GB**，可以 Range 下载。 |

作者侧的更新（9/29 提交 `5244ceade`）确实提到"Bonsai 2 27B … with the MTP head"，用的就是这种打包文件；10/02 的 `2459f68b5`"cuda: fused FWHT quantizer for 64-wide warps" 与我 §24 做的 FWHT→q8_1 融合是同一件事（他们是给 64 宽 warp 的 AMD 路径做的）。

### 26.2 嫁接（而不是下载 7.6 GB 整包）

用 `decent-jawfish` 的 `graft_mtp.py`（已存 `work/mtp/graft_mtp.py`）：
* 读本地 PQ2_0 头部 → 扫 `unsloth/Qwen3.8-27B-UD-Q2_K_XL` 头部（Range 48 MB）→ 定位 15 张 `blk.64.*`；
* 元数据只改两处：`qwen35.block_count` 64→65、新增 `qwen35.nextn_predict_layers = 1`；
* 张量原样复制（**不做 Hadamard 旋转**，也不加入 `prism.hadamard.weight_names`——Bonsai 的残差流在原基下，只有 `token_embd` 是 latent 的，这正是 PR #205 修的那条）；
* 输出 `work/mtp/Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf`（7.56 GB，866 张量）。

**结果自检**：`block_count=65 / nextn_predict_layers=1 / blk.64 ×15` 全部通过。
**主干不受影响（逐位）**：同一 prompt、greedy、32 token，用**原文件**和**嫁接文件**各 dump 一次 logits — `BIT-IDENTICAL (6 208 000 logits)`。
⇒ 我们前面所有 CUDA 优化、验收基线与这份 MTP 文件完全兼容。

### 26.3 正确性：贪心输出逐字节不变

`--spec-type draft-mtp` 的每一步都由目标模型验证，所以贪心（`--temp 0`）下输出必须与不开 MTP 完全相同：

| 对比 | 结果 |
|---|---|
| d=0，64 token，n-max 1 / 2 / 3 | 文本 SHA256 **全部相同**（`f693fd44…`） |
| d=0，256 token（n-max 1/2/3/4） | 文本 SHA256 **全部相同** |
| 32K prompt，128 token，n-max 4 | **IDENTICAL** |
| 96K prompt，128 token，n-max 4 | **IDENTICAL** |

（MTP 的 acceptance 用 `--verbose` 可看：d=0、n-max 4 时 **24 drafted / 18 accepted = 75%**，比社区在 MIG 切片上的 60% 还高。）

### 26.4 性能（V100，`-ngl 99 -fa 1`，greedy，单流，llama-cli 内建 server 路径）

| 深度 | 不开 MTP | 最佳 n-max | 开 MTP | 增益 |
|---|---:|---:|---:|---:|
| **d=0**（短 prompt，512 token，4 轮交错） | 60.90 | **4** | **86.70** | **+40.7%** |
| **32K**（128 token，3 轮交错） | 51.2–51.7 | **3** | **68.10** | **+33.0%** |
| **96K**（128 token，2 轮交错） | 37.6 | **1–3** | **44.50** | **+18.4%** |

n-max 曲线（同一台机器，单次）：

| 深度 \ n-max | 0 | 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| d=0 | 61.8 | 74.6 (+22.5%) | 72.3 (+18.7%) | 83.9 (+37.1%) | **86.7 (+40.7%)** | 82.4 (+35.0%) |
| 32K | 51.2 | 65.6 (+28.1%) | 67.5 (+31.8%) | **68.1 (+33.0%)** | 65.0 (+25.7%) | — |
| 96K | 37.6 | **44.5 (+18.4%)** | 44.4 (+18.1%) | 44.2 (+17.6%) | 37.0 (**−2.9%**) | 34.8 (−7.4%) |

**一个值得记的现象**：96K 下 n-max 3 → 4 有一个**断层**（+17.6% 掉到 −2.9%）。3 的 verify 批次是 4 个 token、4 是 5 个 token —— 很可能是小批次（5 列）踩到了某条慢路径（作者在 9/29 的 `5242ceade`/`5244ceade` 里专门加了 "batch-invariant small-batch kernels" 和 "hybrid PTQ1_0 dispatch"，我们这棵树还没有）。**调参结论：短上下文用 4，32K 用 3，长上下文用 1–2；不要盲目加大。**

### 26.5 怎么用

```bat
set GGML_CUDA_DISABLE_GRAPHS=1
llama-cli  -m <...>-PQ2_0-MTP.gguf -ngl 99 -fa 1 -c 32768 ^
           --spec-type draft-mtp --spec-draft-n-max 3 -p "..." -n 256 --temp 0
```
`llama-server` 同样支持（加 `--spec-type draft-mtp --spec-draft-n-max N`）。
注意：**`llama-bench` 不支持投机解码**，所以 MTP 的吞吐只能用 llama-cli / llama-server 的 timings 量。

显存代价：MTP 会多建一个 draft context。实测峰值 32K ≈ 11.8 GB、96K ≈ 16.9 GB（32 GB 卡都放得下；但 90K 下 `-np` 的槽位数会比原来少）。

### 26.6 复查完，这一步的收益是**数量级提升**

| 阶段 | d=0 | 90K | 说明 |
|---|---:|---:|---|
| 原始基线（交接文档） | 58.89 | 38.31 | |
| §14–§25 的 CUDA 状态路径 + 融合 | 64.95–66.47 (**+5.5~5.7%**) | 41.2–41.3 (**+3~5%**) | 逐位验收 18/18 |
| **+ MTP（本轮）** | **86.7 (+31%)** | **44.5 (+8%)** | 相对 *已优化* 的那一版再涨 |

### 26.7 交付物与下一步

* 模型：`work/mtp/Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf`（7.56 GB；主干逐位等于官方 PQ2_0）
  与 `work/mtp/graft_mtp.py`（231 行，可复现，只需 0.35 GB 下载）；
* 脚本：`work/mtp/ab_mtp.ps1`（三道门验收）、`work/mtp/bench_mtp.ps1`（n-max 扫描）、
  `work/mtp/bench_mtp_depth.ps1`（32K/96K）；
* 下一步可选：① 用 `llama-server -np N` 复测社区提到的"恰好 2 并发会掉"的现象；
  ② 查 96K 下 n-max≥4 那个断层（很可能是小批次 kernel 路径，作者新版已改）；
  ③ 把 MTP 纳入一键验收（把 `--spec-type draft-mtp` 当作第四个配置，判据仍是贪心文本逐字节相同）。
