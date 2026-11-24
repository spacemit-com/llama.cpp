# K3 单层全部专家异步预取实验（2026-11-18）

已完成实现及实测，代码未提交。异步请求开销很小，但在本层 QKV 前才开始搬运全部专家，无法隐藏全部搬运时间。本次首次 PP128 比按需 mmap 慢；权重驻留后的 PP128 各模式接近。

## 测试条件

- 设备 k3-0，通过 k3-submit 串行执行。任务 1370（matcher 回归）、1371（原始四模式）、1372（顺序预取）均成功，退出码 0。
- 模型 /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf，真实单层权重，文件 8,815,791,776 字节。
- 384 个专家，gate/up/down 三个 MXFP4 张量，共 7,219,445,760 字节（6.724 GiB）。
- PP128，8 个计算线程，ctx256，16 步 greedy decode。每模式三个独立进程，每进程额外测三次清空 KV 后的热 PP128；表格均为三轮均值。
- 每进程启动前仅驱逐目标模型的文件页缓存，mincore 确认每次驻留为 0 字节。这不代表 SSD 控制器断电冷启动。前四模式轮换顺序，最终顺序预取在后续独立任务测量。
- 当前 MXFP4 专家 GEMM 由 CPU 执行，其余支持的节点使用 SpacemiT。各模式强制相同的专家放置，并安装相同的观测回调。

## 结果

单位为秒；RSS 为完整字节校验前的进程驻留值。加载＋首次包含 context 创建等首次前开销。完整字节校验及释放资源不计入这些推理时间。

| 模式 | 加载 | 首次 PP128 | 加载＋首次 | 热 PP128 | decode 16 步 | 测量末 RSS / GiB |
|---|---:|---:|---:|---:|---:|---:|
| 全量加载到内存 | 9.748 | 3.912 | 13.668 | 3.827 | 1.069 | 8.34 |
| mmap 提前触页（lazy off） | 5.066 | 3.836 | 8.908 | 3.805 | 1.053 | 8.34 |
| mmap 按需读（lazy on） | 1.809 | 5.692 | 7.507 | 3.815 | 1.228 | 1.79 |
| QKV 前异步预取全部专家（顺序读） | 1.840 | 9.321 | 11.167 | 3.799 | 1.093 | 13.82 |

首次 PP128 三轮范围：全量内存 3.837–4.014 秒，提前触页 mmap 3.833–3.840 秒，按需 mmap 5.672–5.708 秒，顺序预取 9.278–9.363 秒。

相对按需 mmap，预取的首次 PP128 慢 63.7%，加载到首次完成多 3.660 秒。相对全量内存，预取减少了加载等待，加载到首次完成缩短 2.502 秒，但首次推理本身更慢。

预取同时保留 mmap 源页面及匿名 DDR 目标，匿名部分约 7.68 GiB、文件映射约 6.13 GiB。文件页可以回收，RSS 不等于系统独占物理内存。

## 预取时序

1. 控制线程创建一个可用 X100 搬运 worker，为三个专家张量预留匿名 DDR，在 context 建立前固定目标地址。
2. 首次层 0 的 QKV BEGIN 回调在 A100（日志 CPU8）提交三个描述符，均值 27.36 微秒。A100 不复制权重内容。
3. X100 顺序读取 mmap 源并复制全部专家到 DDR。后续 QKV 不再提交，权重持续驻留。
4. 首个 CPU 专家 MUL_MAT_ID 所在 split 执行前等待全部事件完成，防止读取未初始化权重。没有调用取消 API，清理时等待已接受的请求完成。

从提交到等待点间隔平均 2.637 秒；到达等待点后额外阻塞 5.401 秒；提交到全部权重可用 8.038 秒。最后一项包括排队、缺页、读取、目标页分配和复制，不是纯 memcpy 带宽测试。按专家总字节/该时长约 0.84 GiB/s。

本层计算只提供约 2.64 秒的重叠窗口，远短于全部搬运所需的约 8.04 秒。按需 mmap 也无需将全部专家复制到新缓冲，本实验中耗时和 RSS 均更少。更早触发、分批搬运或只搬预测会使用的专家，需要另行设计与测量。

## 原始 RANDOM 映射问题

lazy loader 默认将专家映射设为 MADV_RANDOM。直接全量复制会失去顺序预读，原始方案首次 PP128 平均 52.475 秒，约 176 万次主缺页。最终方案在控制线程仅改为 MADV_SEQUENTIAL 提示，不触页、不发 WILLNEED/POPULATE；实际读取仍由 QKV 后启动的 X100 完成。首次 PP128 降到 9.321 秒，原始结果以 prefetch-random 保留。

## 实现位置与范围

仓库：hz-snode3:/home/yuxuewen/0_workspace/worktrees/feat-prefetch。

- ggml/src/ggml-spacemit/spacemit-graph.cpp：按精确投影权重名补充 QKV 起点识别，覆盖命名输出落到其他 backend 的分图；仍检查同层 attn_norm 依赖。
- src/models/deepseek41.cpp：三个专家张量标记为可 lazy 加载，是否启用由现有 loader 模式控制。
- tests/test-spacemit-expert-prefetch.cpp：专用实验策略、搬运及消费前等待、计时和全量字节校验。
- tests/test-spacemit-prefetch-plan.cpp：增加投影分图、层号错误和近似权重名不得匹配的回归。
- scripts/spacemit-expert-prefetch-bench.py：文件缓存检查、轮换模式、三轮测量、输出一致性验证及汇总。

本次全部专家搬运策略挂在专用实验程序中。普通 llama-cli 的 --prefetch 仍是模块事件开关，单独打开它不会自动启用本策略。实验限定一个模型层，尚无多层容量和淘汰策略。不能把单层测试直接外推到完整模型。

## 校验和数据

- matcher 回归通过，交叉编译成功，git diff --check 通过，代码未提交。
- 15 个进程的 first/decode/warm 共 45 份最后 token 的全量 logits 文件逐字节一致。进程内三次热 PP128 均验证与首次 logits hash 一致。
- 每次预取额外比较全部 7,219,445,760 个专家字节，校验通过；该校验在计时区间外。
- 内核没有 /proc/self/io，读取字节计数不可用。results.json 已标为 null，不能将原始日志中的 0 理解成没有存储读取。
- 原始二进制曾将对照模式的 all_expert_bytes_verified 标为 true；对照模式实际验证 logits，未复制及比较全部专家字节。合并结果已纠正该标记，原始日志保留供追溯。
- results.json：15 次运行归一化数据及跨模式校验结果。summary.json：三轮均值与范围。measurement-artifacts.tgz：原始日志及 JSON。baseline-results.raw.json：未经修正的原始对照数据。

## 复现

交叉编译 test-spacemit-expert-prefetch 与 test-spacemit-prefetch-plan，连同依赖库及仓库测量脚本部署到板上，然后排队运行：

```sh
k3-submit --name whole-expert-compare --owner codex-expert-prefetch \
  --cwd /root/prefetch-experiment --timeout 1800 -- \
  python3 spacemit-expert-prefetch-bench.py \
  --bin-dir /root/prefetch-experiment \
  --model /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  --output-dir /root/prefetch-experiment/results
```
