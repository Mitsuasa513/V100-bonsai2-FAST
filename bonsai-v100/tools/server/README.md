# Bonsai 2 27B（PQ2_0 + MTP）V100 局域网服务

两个启动脚本，按用途挑：

| 脚本 | MTP | 适合 |
|---|---|---|
| **start-server.cmd** | 开（n-max 3） | 聊天、短/中上下文 —— 实测快 20~40% |
| **start-server-nospec.cmd** | 关 | 超长上下文 / 长文档总结 —— 100K 深度下 MTP 反而慢约 10% |

停止用 **stop-server.cmd**（或直接关窗口）。

## 接到哪里

启动后窗口会打印本机 IPv4，用那个地址：

| 用途 | 地址 |
|---|---|
| OpenAI 兼容 API（LM Studio / Cherry Studio / Chatbox / Open WebUI 都能接） | http://IP:8080/v1 |
| 健康检查 | http://IP:8080/health |
| 模型信息 | http://IP:8080/props |

注意：这个 build 没有编进内置 WebUI（构建时没有网络，UI 资源没下到），所以浏览器直接打开
:8080 看不到聊天页面；请用上面那些客户端，或者 curl。

## 这套参数怎么定的（都在本机实测过）

| 参数 | 值 | 原因 |
|---|---|---|
| 模型 | Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf | 官方 PQ2_0 主干（逐位相同）+ 15 张 blk.64.* MTP 层（取自 Qwen3.8-27B UD-Q2_K_XL）；block_count=65、nextn_predict_layers=1 |
| -c | 262144 | 模型满窗 |
| -ctk/-ctv | q8_0 | f16 KV 在 262K 下要 ~17 GB 放不下；q8_0 约 9 GB，画质几乎无损。显存紧张就改 q4_0（约 5 GB，作者自己的 262K 复现用的就是 q4_0） |
| -fa on | | KV 量化必须配 flash attention |
| --spec-type draft-mtp --spec-draft-n-max 3 | | 投机解码；实测 d=0 +37%、32K +33%、96K +18%（n-max 4 在 >=96K 会掉成负数） |
| GGML_CUDA_DISABLE_GRAPHS=1 | | MTP 下 graphs=off 比 on 快 9%（每步 batch 形状不同，图缓存反而拖累） |
| -np 1 | | 单槽最快；要并发就加大，-c 会按槽均分（例如 -np 4 则每槽 64K） |

## 实测（这台 V100，262144 ctx + q8_0 KV，单流）

| 场景 | 不开 MTP | 开 MTP | 验收率 |
|---|---:|---:|---:|
| 短 prompt（512 token，4 轮交错） | 60.9 t/s | **86.7 t/s（+41%，n-max 4）** | 75% |
| 32K（128 token） | 52 t/s | **68 t/s（+33%，n-max 3）** | — |
| 96K 纯续写（128 token） | 37.6 t/s | 44.5 t/s（+18%，n-max 4；n-max ≥4 会掉成负的） | — |
| **100K chat + 总结（服务端实测）** | **35.3 t/s** | 31.6 t/s（−10%） | 36% |

预填（服务端实测）：101003 token / 175 s = **577 t/s**；按此外推灌满 262144 约 **7.5 分钟**（之后同一会话走 KV 复用，很快）。
对照：官方文件纯解码基线 58.89 t/s；我们这一路的 CUDA 状态路径优化加到 ~66 t/s，MTP 在短中上下文再叠加 20~40%。

**为什么长上下文 MTP 会亏**：draft 头是"主模型里的一个块"，它自己也要对整条 KV 做注意力，上下文越长这一步越贵；同时验收率从短上下文的 75% 掉到 100K 的 36%。所以长文档场景用 `start-server-nospec.cmd`。

## 注意事项

* 首字延迟：吃满 262K 上下文时预填要几分钟（V100 上约 300-400 t/s）；同一会话的后续消息复用 KV，很快。
* 贪心输出不变：MTP 每一步都由主模型验证，temperature=0 下输出与不开 MTP 逐字节相同（已验收 d=0 / 32K / 96K）。
* 采样建议（模型卡）：thinking 模式 temperature=1.0、top_p=0.95、top_k=20、min_p=0.0；
  想少思考用 reasoning_effort: "medium"；输出上限给大（max_tokens >= 16384），否则会在思考中途被截断。
* n-max 调参：短上下文用 4 最快；96K 以上用 1-3（n-max 4 会踩到 5-token verify 批次的慢路径）。
* 显存实测：32K 约 11.8 GB、96K 约 16.9 GB、262K（q8_0 KV）约 19 GB；本卡 32 GB（另有一个任务占 ~5 GB）。
* 局域网连不上：Windows 防火墙放行端口（管理员执行一次）：
  `netsh advfirewall firewall add rule name="llama-server 8080" dir=in action=allow protocol=TCP localport=8080`

## 文件

```
<service-dir>\
  start-server.cmd                     启动（双击）
  start-server-nospec.cmd              启动（不带 MTP，长上下文用）
  stop-server.cmd                      停止
  Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf  7.56 GB 模型
  server.log                           运行日志（启动时生成）
  README.md                            本文件
```
