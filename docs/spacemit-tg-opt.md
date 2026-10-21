# Historical SpacemiT M1 TG optimizations

Current PR branch status (2026-10-21): the TG paths in
`kernel-agent/pr-base-v1.1` have been aligned to
`kernel-agent/v1.1@d7ae8aa548dcca41cd79b6cf6210d3557c34b80d`.
Q4_0 uses N128 cyclic ownership and the original N32 HP M1 kernel;
Q4_K and Q6_K use their original M1 instruction order. The original dynamic
N scheduler counter layout and initialization barrier are restored, and the
shape-specific Q4_K staging and Q4_1/Q8_0 static scheduling are removed.
The N64 Q4_0 kernel is removed. PR dispatch adaptations and PP implementation
remain; v1.1's Q4 PP streaming and expanded weights are not imported.

The pre-alignment tree is preserved in Git stash commit
`4a58c1665a0db656d58b5c9c887403a26173b164`.
The aligned backend builds successfully with GCC15.2/toolchain v1.2.4;
its SHA256 is `6d0947f3271784e5b23ad108a814371c789802cc7d297de48d05c9a94328e487`.
K3-0 job 86 passes all 19,447,808 FP32 logits bitwise for each of Qwen3-0.6B
Q4_0 and Q4_K_S over 128 TG steps. In the ABBA TG128 comparison (10 samples
per variant/model), v1.1 -> aligned PR medians are 60.97975 -> 59.22285 tok/s
for Q4_0 and 62.3254 -> 60.7705 tok/s for Q4_K_S. This restores the TG kernel
and scheduling code, not equal end-to-end throughput; the upper llama/ggml
versions still differ. Full commands, library hashes and results are under
`/root/yuxuewen/v11-tg-exact-20261021/results/20261021-120144/` on k3-0.
The rest of this document records historical experiments and measurements;
it does not describe the active PR implementation or validate the new build.

Review branch: `kernel-agent/v1.1-tg-opt-closeout`, based directly on
`kernel-agent/v1.1@d7ae8aa548dcca41cd79b6cf6210d3557c34b80d`.
This single code submission contains the four implementation files from
`ff7dc96c11128975615f69018f9baa4582fa2c2c`, byte for byte. That implementation
is also identical to the previously validated `dcfda0a4b075ec9af4bd973de9646e797e2dd1fc`.
The original integration and diagnostic branches remain available.

The independent 2026-10-14 closeout adds no kernel change, build, deployment,
or native measurement. The performance Goal remains paused and unmet.
Full provenance, numerical evidence, failure records and reproduction details
are in the separate task repository, `kernel-opt-agents/workspace/tg-opt`,
starting at `docs/closeout.md` and `logs/CLOSEOUT-20261014/`.

## Code and scope

| Format | Retained change and reason | Dispatch scope | Standalone evidence |
| --- | --- | --- | --- |
| Q4_K | C003 places independent zero-point demand before conversion; C009 stages unchanged compressed N32 weights using a measured three-load copy | C003 existing M1 ZPDS path; staging only M1/N4096/K8192 with TCM/VLEN/alignment/capacity guards and direct fallback | Final B0 pair 843.333 -> 716.917 us, 1.1763x; three predecessor repeats 1.0794-1.0921x |
| Q6_K | C008 moves independent activation demand before weight decode in a TG-only copy | M1/VLENB128; original shared PP function remains byte-identical | Final B0 pair 1032.625 -> 1026.459 us, 1.0060x; predecessor repeats 0.38-0.58% |
| Q4_0 | Q40-C01 reduces cyclic ownership granularity from N128 to N32 | Existing M1 K256 HP direct path | Final B0 pair 670.208 -> 658.208 us, 1.0182x; three repeats 1.0126-1.0200x |
| Q4_1 | FMT-C01 removes dynamic claims with cyclic N128 ownership | Legacy K32 M1/N4096/K8192 only | Three pairs 1.0646-1.0673x |
| Q8_0 | FMT-C01 removes dynamic claims with cyclic N32 ownership | K32 M1/N4096/K8192 only | Three pairs 1.0950-1.0973x |

Implementation files: `ggml/src/ggml-spacemit/ime.cpp`, `ime2_kernels.cpp`,
`memory-copy.h` and `memory-copy-u3.h`. The separate Q6 TG function preserves
PP instruction order; factoring this duplicate would be a new unvalidated
code change. Q4_0/Q4_1/Q8_0 arithmetic kernels and all weight layouts,
quantizers, FP16 intermediate rounding and ordered FP32 arithmetic are unchanged.
TCM staging costs remain inside the recurring timer.

Q4_K staging and Q4_1/Q8_0 static scheduling previously regressed actual
models when applied broadly. The retained shape guards follow selective
replay; other shapes use original service. Later rejected or incomplete
experiments, cache changes, instrumentation and expanded layouts are excluded.
Q8_1/Q8_KS remain deferred pending separately specified source.

## Retained validation and measurements

Historical `INT-C02-validation` passes 110/110 complete-backend cases bitwise,
including all five formats, seeds42/123, required-shape patterns, edge shapes,
K65536 fallback and M4/M16/M128 PP. `INT-C02-model-check` passes exact finite
PP128/TG8 logits for all five models. Standalone checks include independent
FP64 inherited-error references; candidate-added error is zero.

Complete graph mean process medians, B0 -> candidate in us:
Q4_K 892.013 -> 794.658; Q6_K 1174.718 -> 1183.884;
Q4_0 791.575 -> 746.259; Q4_1 865.835 -> 832.129;
Q8_0 1487.213 -> 1378.886. Q6_K has no stable graph gain.
These older measurements use one reused weight buffer and include recurring
activation conversion/copy and graph service; standalone uses four rotating
weight matrices and excludes initial activation conversion.

Latest existing five-model measurement: `logs/v11-d27-affinity-bench-v2/`,
three alternating pairs, five repetitions, PP128/TG128, batch2048/ubatch128,
one host thread, eight workers, SPACEMIT0/ngl99, flash attention on, mmap off.
Values are model-specific mean process throughputs in tokens/s.

| Model | TG B0 -> candidate | TG change | PP B0 -> candidate | PP change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q4_K_S | 62.905 -> 63.282 | +0.600% | 666.773 -> 674.263 | +1.123% |
| HY-MT1.5-1.8B-XH Q4_K_M | 23.130 -> 23.197 | +0.290% | 117.057 -> 117.082 | +0.021% |
| Qwen3-0.6B Q4_0 | 61.187 -> 61.856 | +1.094% | 716.180 -> 708.030 | -1.138% |
| Qwen3-0.6B Q4_1 | 61.344 -> 61.388 | +0.072% | 710.779 -> 704.836 | -0.836% |
| Qwen3-0.6B Q8_0 | 40.835 -> 40.887 | +0.128% | 683.852 -> 683.662 | -0.028% |

Q4_0 TG paired95%CI is [+0.346%,+1.843%]; PP CI is [-2.179%,-0.094%].
The PP regression remains a review risk; retention is not a production rollout
recommendation. Other latest TG intervals include zero. Pure Q6_K model
coverage is absent; Q4_K_S/Q4_K_M provide mixed Q6_K tensor coverage.
The latest window also has an exact before-inventory copy receipt gap; its
preflight-before-deploy chain is retained separately. Earlier measurement
windows and their limitations remain in the task report and are not pooled.
The saved d27 commands use cpu_mask=0x0 and cpu_strict=false; they do not
establish enforced GP CPU0 placement. Historical observations are not the
later strict caller-affinity receipt contract.

During local closeout replay, the current baseline backend-graph artifact
differs from its saved d27 identity receipt. Seven other benchmark, backend
and runtime artifact hashes match. The historical remote receipts agree with
the saved analysis; no binary was rebuilt or overwritten to repair the current
local gap. Current deployment is not qualified by this closeout.

Measured sustained compatible full-unit VMADOT references are 14.718/29.440
TOPS depending on arithmetic mode. Older full graph useful-work ratios are
0.274-0.385%; theoretical MFU remains unverified. Neither >=70% MFU nor
>=1.5x actual-model TG is established.

## Build and reproduction

The task's `docs/closeout-reproduction.md` separates local source/evidence
checks from future native commands. Native recipes are documentation only;
no new run is authorized by the paused closeout. Cross-build uses GCC15.2,
toolchain v1.2.4 and SPERT7895f075. Historical artifact hashes and source
commits remain recorded; a rebuilt commit identifier does not reuse an old
binary identity automatically.

K3 binding follows llama.cpp: each executing AI worker opens
`/proc/set_ai_thread`, writes one ASCII `0`, closes it, then self-binds with
`pthread_setaffinity_np`. GP caller self-binds CPU0. Require singleton masks,
actual CPU8--15, VLENB128 and FRM0. Never use `taskset` for K3 launch/binding.
Production SPERT owns worker registration; historical harness return-value
receipt limits are stated in the task report.
