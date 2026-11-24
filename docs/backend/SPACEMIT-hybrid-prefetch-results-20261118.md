# DeepSeek V4.1 Flash 单层：混合加载与异步 mmap 页预热实测

日期：2026-11-18。测试设备：k3-0，SpacemiT K3，15 GiB RAM，无 swap。源码：hz-snode3 的 feat/prefetch 工作树，基线 HEAD 02e0783ef，包含当前未提交修改。本轮没有 commit。

模型：`/data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf`，真实单层，文件 8,815,791,776 字节；普通权重 1,590,565,336 字节，三个专家张量合计 7,219,445,760 字节（6.724 GiB）。MXFP4 专家通过 CPU fallback 执行，本次结果不代表已实现 MXFP4 专家加速内核。

## 测试方法

通过 k3-submit 排队独占执行，正式性能矩阵任务 1388 成功结束。五种方式各三个独立进程，三轮轮换顺序。每次运行前仅对该模型文件执行 POSIX_FADV_DONTNEED，并通过 mincore 验证模型文件缓存为零。没有全局 drop_caches，没有更改 CPU 频率配置。

A100 计算线程为 8，预热使用一个可用 X100 搬运线程，外加 dispatcher。context 256，batch/ubatch 128，n_gpu_layers=99，路由专家通过 CPU override 执行。禁用预先模型推理热身；首次 PP 输入固定 token IDs 100..227，再执行 16 次单 token decode，随后清空 KV，重复 PP128 三次。使用 steady_clock 和 llama_synchronize 测量，热 PP 为每进程三次平均，表中再取三个进程平均。decode 计时包含贪心选词和 logits 有限值检查。

## 性能结果（三轮均值）

| 加载方式 | 加载 s | 首次 PP128 s | 加载＋context＋首次 PP s | 热 PP128 token/s | decode16 token/s | 峰值 RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| 全部权重复制到内存 | 9.745 | 3.868 | 13.619 | 33.59 | 15.06 | 8.34 |
| 原有懒 mmap | 1.821 | 5.691 | 7.517 | 33.62 | 12.84 | 1.79 |
| 加载时预填充 mmap | 4.995 | 3.847 | 8.847 | 33.57 | 15.02 | 9.09 |
| 普通权重常驻＋专家 mmap，关闭异步预热 | 3.125 | 4.094 | 7.224 | 33.57 | 14.31 | 2.46 |
| 普通权重常驻＋专家 mmap，开启异步预热 | 3.104 | 4.018 | 7.229 | 33.60 | 15.17 | 8.34 |

热 PP128 各方式均约 3.81 秒。首次 PP128 范围：全部内存 3.840–3.918 秒；原有懒 mmap 5.635–5.744 秒；预填充 mmap 3.837–3.855 秒；混合关闭预热 4.084–4.106 秒；混合开启预热 4.014–4.025 秒。

- 混合加载＋异步预热相对原有懒 mmap：首次 PP 耗时减少 29.4%，加载＋context＋首次 PP 耗时减少 3.8%。这包含加载策略、mmap advice 和异步预热共同影响。
- 在相同混合加载策略下，异步预热开启相对关闭：首次 PP 耗时减少 1.84%（75 ms）；context 创建从 5.24 ms 增至 106.70 ms，加载＋context＋首次 PP 为 7.224 与 7.229 秒，本轮看不到启动到首次 PP 完成的收益。
- decode16 从 1.118 秒降到 1.055 秒，耗时减少 5.67%；这是接在首次 PP 之后的一段 decode，专家页驻留程度影响结果，不可外推为任意上下文下的固定提升。
- 异步扫描所有专家后，峰值 RSS 从 2.46 GiB 增至 8.34 GiB。64 MiB 仅是在途任务上限，不是页缓存或权重驻留上限。与加载时预填充 mmap 相比，总启动到首次 PP 减少约 18.3%，首次 PP 本身仍较慢。

## 正确性与实际触发证据

15 次正式运行全部退出 0，所有 logits 有限；每次的首个 PP、decode 末尾及各次重复 PP 的完整 129,280 维 float32 logits 跨方式、跨轮次逐字节一致；生成 token IDs 一致。

开启模式的生产运行时均报告 `source warmup enabled for 3 routed weight ranges`。context 创建后未预热，QKV BEGIN 回调激活后台预热；首次 PP 后 RSS 已达 8.345 GiB，而关闭模式约 2.404 GiB。首次 PP 的 major faults，关闭为 2930/3009/2936，开启为 188/192/159。这里是复用原有专家 mmap 的后台触页，不是复制一份专家权重，也不替换 tensor->data。

测试期间修复了生产运行时的资格检查：核心 CPU/CPU_Mapped buffer 可以没有关联 device；原逻辑将其误判为不可预热。现在保留 loader allowlist、host buffer 和地址范围检查，只拒绝明确关联到非 CPU device 的 buffer。正式矩阵 1388 使用修复后的库；早期 1387 检测到未激活并失败，其结果不用于本报告。

## 复现与原始数据

交叉构建目录：`hz-snode3:/home/yuxuewen/0_workspace/builds/prefetch-transfer-20261118`。
源码目录：`hz-snode3:/home/yuxuewen/0_workspace/worktrees/feat-prefetch`。
板端运行目录：`k3-0:/root/hybrid-prefetch-validation-20261118`。
正式原始数据及完整 logits：板端 `matrix-r2/`；本地 `summary.json`、`results.json` 和 `matrix-r2-text-results.tar.gz` 保留汇总、逐次结果、日志及运行脚本。

新基准 `test-spacemit-hybrid-prefetch` 直接使用生产 model/context 参数和生产预热管理器，不安装替代回调。运行方式：

```sh
export LD_LIBRARY_PATH="$PWD"
export GGML_SPACEMIT_BACKEND=1 GGML_SPACEMIT_WORKERS=8
./test-spacemit-hybrid-prefetch \
  /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  hybrid-on output-prefix 128
```

在 k3-0 上必须通过 k3-submit 运行以上推理命令。模式包括 resident、mmap、mmap-eager、hybrid-off、hybrid-on；完整三轮驱动为归档中的 run-matrix.py。
## 编译与回归测试

交叉构建成功：llama-bench、llama-cli、llama-completion、test-prefetch-config、test-spacemit-prefetch-plan、test-spacemit-prefetch-runtime、test-spacemit-transfer、test-spacemit-transfer-x100、test-spacemit-expert-prefetch、test-spacemit-hybrid-prefetch。

队列任务 1384 的五个回归程序全部通过：

- test-prefetch-config：CLI/环境变量/默认值及覆盖规则。
- test-spacemit-prefetch-plan：语义匹配、层号及边界条件。
- test-spacemit-transfer：异步搬运、取消、并发与 mmap 页预热。
- test-spacemit-transfer-x100：真实 X100 provider，可用 CPU 0–7。
- test-spacemit-prefetch-runtime：实际执行图事件顺序、缓存图重放、暂停恢复及输出逐位一致。

最终源码 `git diff --check` 通过；保留所有修改供 review，没有 commit。二进制 SHA256 见 validation-binaries.sha256。
## 命令行推理补充

队列任务 1395 使用修复后的同一份动态库重新运行五个回归程序，全部通过。`llama-completion` 在真实单层权重上分别带 `--no-prefetch` / `--prefetch` 生成 32 token，两个进程退出码均为 0，生成文本逐字节一致。

该次短提示词为 7 token，并非 PP128 性能测试。关闭预取时 prompt eval 673.39 ms、decode 31 次 2167.55 ms（14.30 token/s）；开启参数时 prompt eval 327.97 ms、decode 31 次 2344.49 ms（13.22 token/s）。此处的开关同时改变加载策略，不能单独归因于异步预热；它也说明短提示词下不能保证 decode 加速。

命令行默认日志阈值为 3，库 INFO 被映射到 trace（4），因此默认日志不会显示 source warmup enabled。正式 PP128 矩阵独立检查运行时 manager enabled 和启用日志，三轮 hybrid-on 均通过；队列 1398 的详细日志复查已成功完成，明确出现 source warmup enabled for 3 routed weight ranges，两个开关运行均退出 0，32 token 输出仍逐字节一致。该复查仅提高日志等级，不参与上表统计，日志已补入归档。

回归和命令行日志归档：`validation-final-logs.tar.gz`。