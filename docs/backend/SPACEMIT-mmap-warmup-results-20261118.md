# llama.cpp mmap + QKV-triggered expert warmup on K3

Date: 2026-11-18. Worktree: `/home/yuxuewen/0_workspace/worktrees/feat-prefetch`, branch `feat/prefetch`, base HEAD `e4455005e`. Changes remain uncommitted.

## Result

The existing model-loader mapping can support asynchronous expert page warmup. Model loading establishes the mappings; QKV BEGIN submits three source-only requests; an X100 worker touches the pages. Tensor addresses stay unchanged. The expert computation proceeds while one request is still pending, and ordinary demand faults supply pages that are not ready. There is no expert-wide wait and no second anonymous expert buffer.

The attribution control matters: most cold PP128 improvement in this experiment came from changing the expert VMA advice from RANDOM to NORMAL. Adding the asynchronous whole-layer warmup reduced PP128 by only about 73 ms (1.8%) relative to NORMAL alone, while increasing end-of-inference process RSS from about 1.80 GiB to 7.69 GiB. This supports the mechanism, but does not justify enabling whole-layer warmup universally.

## Setup and measurement definitions

- Board: k3-0. All inference and real-provider tests ran through `k3-submit`.
- Real single-layer model: `/data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf`, 8,815,791,776 bytes.
- Three expert tensors (gate/up/down), 384 experts, total 7,219,445,760 bytes (6.724 GiB).
- Eight SpacemiT compute workers and one X100 transfer worker. Submission was observed on CPU 8; the provider discovered available X100 CPUs 0–7.
- Expert tensors use the CPU buffer type in every mode. These measurements include CPU expert fallback, not a newly implemented MXFP4 accelerator kernel.
- Each mode runs in three independent processes, with rotated order. Each process executes first PP128, 16 greedy decode steps, and three warm PP128 repetitions with llama memory cleared.
- Before each process, evict only the model file's cache with POSIX_FADV_DONTNEED and check with mincore. Every measured run started with 0 cached bytes. This is OS file-cache cold, not power-on SSD/controller cold.
- `load + context + first PP` is the sum of those three measured timers. It excludes backend/worker initialization, setup/advice, subsequent decode/warm runs, output verification, and teardown; it is not full process startup latency or user-visible TTFT.
- Warm PP is the mean of three repetitions per process. Decode is the total for 16 tokens, measured after the first PP.
- RSS is a process snapshot after the inference measurements and before full-copy verification. It is not total system RAM usage or page-cache residency. Mapped and anonymous copies can both contribute to the copy-mode RSS.
- `/proc/self/io` is unavailable on this board. JSON I/O byte counts are null; no storage bandwidth is inferred from them.

## Five-mode comparison: queue job 1376

All times below are seconds; entries are means across three processes. RSS is GiB.

| Mode | Load | First PP128 | Load + context + first PP | Warm PP128 | Decode 16 | End RSS |
|---|---:|---:|---:|---:|---:|---:|
| resident | 9.759 | 3.840 | 13.608 | 3.799 | 1.079 | 8.344 |
| mmap | 1.807 | 5.690 | 7.502 | 3.818 | 1.238 | 1.791 |
| mmap-eager | 5.013 | 3.840 | 8.859 | 3.801 | 1.067 | 8.345 |
| prefetch | 1.827 | 9.219 | 11.051 | 3.806 | 1.102 | 13.799 |
| mmap-prefetch | 1.800 | 4.002 | 5.808 | 3.813 | 1.104 | 7.692 |

Modes:

- `resident`: mmap disabled; all weights loaded into anonymous memory before inference.
- `mmap`: existing lazy mmap with RANDOM expert advice; no worker.
- `mmap-eager`: mmap enabled and lazy mode disabled; pages populated during loading.
- `prefetch`: existing QKV-triggered copy into a separate anonymous DDR buffer, SEQUENTIAL source advice, and wait at expert consumption.
- `mmap-prefetch`: lazy loader mapping, NORMAL expert advice, QKV-triggered X100 page touching, original tensor pointers, and no whole-request wait at expert consumption.

Compared with default lazy mmap, mmap-prefetch reduced first PP128 by 29.7% and the measured load/context/first-PP sum by 22.6%. These differences include the advice change; they must not be attributed solely to the asynchronous worker.

## Advice-only attribution: queue job 1378

This second matrix uses the same final binary for all three modes. `mmap-normal` uses lazy mmap and NORMAL expert advice but creates no transfer worker and submits no warmup.

| Mode | First PP128 mean, s | First PP128 range, s | Load + context + first PP, s | End RSS, GiB |
|---|---:|---:|---:|---:|
| mmap | 5.643 | 5.610–5.689 | 7.469 | 1.791 |
| mmap-normal | 4.072 | 4.048–4.088 | 5.896 | 1.804 |
| mmap-prefetch | 3.999 | 3.971–4.022 | 5.815 | 7.692 |

NORMAL alone accounts for approximately 1.571 s of the cold PP improvement. The additional whole-layer asynchronous warmup accounts for approximately 0.073 s (1.8% of the NORMAL-only latency) in this three-run control. This is a small-sample, single-layer result; a full-model benefit needs a layer-window and capacity experiment.

## Correctness and overlap

- Jobs 1376 and 1378 succeeded. All 24 processes produced byte-identical first, decode, and warm logits, including a cross-matrix byte comparison (517,120 bytes per saved logits vector).
- First logits hash: `16204105775265430327`; decode logits hash: `11882100468404711487`.
- In all mmap-prefetch runs, expert computation was reached with one request still pending; after first PP all requests were complete. Original tensor addresses were unchanged.
- `wait_ms=0` means no explicit whole-request wait at expert consumption. Page faults can still block individual reads. `ready_ms=0` is unused in this mode and does not measure full warmup completion.
- Final-control submission of all three requests averaged about 29.7 microseconds. The interval from submission to the expert observer averaged about 2.655 s; this is an overlap window, not a worker-completion timer.
- Requests drain after all measured inference and before model unmapping. Cancellation is supported but deliberately unused by these inference experiments.
- COMPLETE means every intersecting page was touched. Pages are not pinned and can subsequently be evicted.
- Mock transfer tests and host ASan/UBSan/leak validation passed. Real X100 provider validation passed in job 1379, including guarded read-only cold pages, unaligned boundaries, event reuse and concurrent producers. Job 1377 initially failed to launch because SCP removed the executable bit; chmod and resubmission fixed it.
- Cross-build, Python syntax validation and `git diff --check` passed.

## Code for review

- `ggml/include/ggml-spacemit-transfer.h`: new source-only `ggml_spacemit_transfer_prefetch` API and lifetime/completion semantics.
- `ggml/src/ggml-spacemit/spacemit-transfer.cpp`: reuse bounded queue/events and 64 KiB chunk cancellation protocol for copy and page warmup jobs.
- `ggml/src/ggml-spacemit/spacemit-transfer.h` and `spacemit-transfer-x100.cpp`: private provider API and page touching on X100 workers. No payload reads occur on the submitting A100 thread.
- `tests/test-spacemit-transfer.cpp`: worker dispatch, failure, zero-length, cancellation, reuse, guarded/read-only pages and residency checks.
- `tests/test-spacemit-expert-prefetch.cpp`: mmap-prefetch and NORMAL-only controls, pointer/completion checks, pending observations and nonblocking expert consumption.
- `scripts/spacemit-expert-prefetch-bench.py`: mode selection, cold rotated matrices, byte-for-byte logits validation and JSON summaries.
- `docs/backend/SPACEMIT-prefetch.md`: API and experiment documentation.

The warmup policy is wired into the single-layer experimental harness. Production CLI `--prefetch` does not yet automatically warm every layer. Full-model deployment still needs a layer window, byte budget, repeat-request policy and selection/eviction strategy.

## Reproduction and artifacts

Main matrix (job 1376):

```sh
k3-submit --name mmap-warmup-cold5modes --owner codex-mmap-warmup \
  --cwd /root/mmap-warmup-20261118 --timeout 1800 -- \
  python3 spacemit-expert-prefetch-bench.py \
  --bin-dir /root/mmap-warmup-20261118 \
  --model /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  --output-dir /root/mmap-warmup-20261118/cold5modes \
  --modes resident mmap mmap-eager prefetch mmap-prefetch
```

Attribution matrix (job 1378), using the final harness with NORMAL-only support:

```sh
k3-submit --name mmap-warmup-normal-control --owner codex-mmap-warmup \
  --cwd /root/mmap-warmup-control-20261118 --timeout 1200 -- \
  python3 spacemit-expert-prefetch-bench.py \
  --bin-dir /root/mmap-warmup-control-20261118 \
  --model /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  --output-dir /root/mmap-warmup-control-20261118/cold3modes \
  --modes mmap mmap-normal mmap-prefetch
```

Raw summaries, per-run measurements, stdout/stderr and logits remain in the two board output directories. Local copies of the four aggregate JSON files accompany this report; K3 queue logs identify jobs 1376, 1378 and 1379.
