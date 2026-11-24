# SpacemiT graph prefetch events

The graph planner can annotate existing execution boundaries with module events.
Register a callback on a SpacemiT backend to enable recognition:

```cpp
static void on_prefetch(const ggml_spacemit_prefetch_event * event, void * user_data) {
    auto & prefetcher = *static_cast<MyPrefetcher *>(user_data);
    if (event->module == GGML_SPACEMIT_MODULE_ATTENTION &&
        event->phase == GGML_SPACEMIT_PREFETCH_BEGIN) {
        prefetcher.enqueue_candidates(event->layer); // return promptly
    }
    if (event->module == GGML_SPACEMIT_MODULE_ROUTER &&
        event->phase == GGML_SPACEMIT_PREFETCH_READY) {
        // Copy the I32 indices now, respecting ne[]/nb[] and backend access
        // rules. The background task must own its copy, not borrow tensor data.
        prefetcher.enqueue_selected(copy_expert_ids(event->tensor), event->layer);
    }
    if (event->module == GGML_SPACEMIT_MODULE_EXPERTS &&
        event->phase == GGML_SPACEMIT_PREFETCH_BEGIN) {
        prefetcher.wait_for_weights(event->layer);
    }
}

ggml_backend_spacemit_set_prefetch_callback(backend, on_prefetch, &prefetcher);
// Run inference. Before destroying prefetcher, synchronize inference and detach:
ggml_backend_spacemit_set_prefetch_callback(backend, nullptr, nullptr);
```

`MyPrefetcher` and `copy_expert_ids` above are application-supplied. The backend
supplies event timing, not a weight cache, transfer queue, or prediction policy.
The setter is also available through the backend registry's proc-address lookup.
Backend handles for the same device share the callback setting.

## Command line and environment

`llama-cli`, `llama-completion`, `llama-server` and the common argument parser
accept `--prefetch` / `--no-prefetch`. `llama-bench` accepts the same switches.
The environment equivalent is `LLAMA_ARG_PREFETCH=1` / `0`; command-line
switches override the environment, and the default is disabled.

```sh
llama-cli -m model.gguf --prefetch -p "Hello"
LLAMA_ARG_PREFETCH=1 llama-bench -m model.gguf -p 128 -n 0
LLAMA_ARG_PREFETCH=1 llama-bench -m model.gguf -p 128 -n 0 --no-prefetch
```

The setting is passed through `common_params.prefetch` to both
`llama_model_params.prefetch` and `llama_context_params.prefetch`. Model loading
uses allocated buffers for ordinary tensors and the existing lazy mmap path for
routed expert gate/up/down tensors. QKV, Attention output, Norm, Router and shared
expert tensors follow the ordinary allocated-buffer path; other tensor classes
with their own lazy-read policy keep that policy. Routed experts bypass the
usual AUTO size threshold, so small individual expert tensors are also lazy.
The file mapping may cover the GGUF file, but ordinary weights are read into
their own buffers rather than used through that mapping.

With a supported SpacemiT backend, the context installs one expert-prefetch
manager. Both prefill and decode policies live in
`src/llama-spacemit-prefetch.cpp` and share the same model-owned slot buffer,
transfer service and dispatcher. QKV BEGIN opens the current layer's window.
Callbacks snapshot IDs and update request metadata under a short mutex; they do
not copy weights, wait for transfer completion, create mappings or drain work.
An unpinned dispatcher owns job descriptors and submits payload work to an
available X100 worker. Missing backend functions or unavailable transfer CPUs
leave inference on the existing mmap fallback.

Four reusable events carry chunks of at most 16 MiB, with at most 64 MiB in
flight. BUSY retries keep the cursor unchanged. The dispatcher polls completion
with a 1 ms condition-variable timeout. Cancelled destinations remain reserved
until every accepted write reaches a terminal state; late completions cannot
publish an aborted expert or overwrite a reused slot. Shutdown detaches the
callback, joins dispatch and drains accepted requests before releasing buffers.

DeepSeek41 graphs with one token per sequence set add a speculative predictor
before QKV, including batched decode. It applies the current layer's FFN norm
and router to the pre-attention stream, then uses the normal group/top-k policy.
This is an approximation using existing weights, not a trained predictor or a
promise that predictions match the actual router. The predictor branch is
explicitly expanded before QKV and does not feed model outputs. Its READY event
snapshots the union of predicted IDs across the microbatch; QKV BEGIN starts
those copies. At actual router READY, decode closes its window and requests
cancellation immediately. Only fully copied actual experts are exposed from
slots; misses and unfinished experts use their original mmap addresses. No
router-time demand copy or transfer wait is added. The first local expert node
also snapshots actual IDs when the router was in another backend split.

Without a ratio, prefill warms all current-layer expert pages through the same
transfer service; decode warms predicted IDs and cancels at the router. Page
warmup does not pin OS file-cache pages or allocate an expert slot buffer. Both
modes retain mappings for normal demand faults. Only banks allowlisted by
`model.can_prefetch` and backed by contiguous CPU mappings are eligible.

## Recognized windows

| Module | Recognition | Boundary |
| --- | --- | --- |
| QKV | Named Qcur/Kcur/Vcur/qkv or DeepSeek41 q/kv, or local GEMMs using exact Q/K/V projection weight roles; dependencies must reach attn_norm of the same layer | Available local projection/preparation branches, including normalization and RoPE |
| RMSNorm | Named RMS_NORM, optionally followed by its scaling MUL | Normalization start through scaling completion |
| Attention | kqv_out/attn_out dependencies reaching attn_norm, containing matmul and softmax/flash attention | After input normalization through the named output; overlaps QKV |
| Router | I32 ffn_moe_topk dependencies reaching ffn_norm, containing matmul | Router start through selected-index production |
| Experts | MUL_MAT_ID with an index dependency named ffn_moe_topk | First through last routed expert GEMM in this backend split |

Names must have an exact nonnegative layer suffix. Scheduler copy wrappers
`backend#name-layer#copy` are unwrapped. Dependencies and operation types provide
additional checks; matching names alone do not establish a module window.

The planner maps original tensors, including fused members, to execution nodes.
BEGIN runs before the first mapped node; END runs after the last mapped node.
A top-k view gets READY after its actual local compute producer, not at the view.
If a boundary lies inside a fused node, timing is rounded to that node's boundary.

Recognition is conservative and specific to these llama graph naming patterns.
A backend split missing the named output or required local work is not reported
as a complete module. QKV reports the available named branches, not a guarantee
that all three projections are present. A split containing only expert work can
still get EXPERTS events through its copied index dependency, without a local
ROUTER READY event. Names and operators for other model builders need additional
rules. This API does not coordinate events across different backends.

## Execution and ownership

Only worker 0 calls the callback. Workers synchronize before executing nodes with
BEGIN events. END/READY callbacks run after the existing all-worker completion
barrier, followed by an event barrier before any worker advances. A READY
callback can therefore copy transient indices before later nodes reuse storage.
BEGIN's boundary tensor is not yet computed; it must not be read as an output.

Submit asynchronous transfer work and return promptly to overlap transfer with
subsequent computation. The only intended blocking point is EXPERTS BEGIN when
weights must be ready. The caller must arrange actual weight storage, cache
installation, transfer completion and background task lifetimes. Do not write
active compute buffers, throw exceptions, or reenter the backend from callbacks.

Accurate current-token expert IDs become available at ROUTER READY. Prefetching
them during earlier attention work requires a candidate prediction policy.

Recognition runs on plan construction when prefetch is enabled. Events are
cached with the graph UID; replays invoke them on every execution. Changing the
callback invalidates the event plan. Removing the callback disables recognition
and all added event barriers, unless planning is explicitly enabled again.
Event windows do not consume or reorder compute nodes.

## Validation

`test-spacemit-io-uring` checks buffered reads, QD slot reuse, a short final
read, payload integrity and page-cache residency observed through the same mmap.
It is Linux-specific and skips on other platforms.

`test-spacemit-prefetch-plan` checks data dependencies, fused-member mapping,
view top-k producer timing, scheduler copy names, split graphs and layer mismatch.
It is registered when tests and the SpacemiT backend are enabled. Trait queries
are stubbed in this planning test so matching does not depend on IME eligibility.

`test-spacemit-prefetch-runtime` is a hardware test for worker-0 callbacks,
READY-before-expert ordering, cached replay, disabling, and bitwise output
equivalence. On K3 it must be run through k3-submit.


## Asynchronous transfer service (X100 provider)

`ggml-spacemit-transfer.h` supplies a core-independent byte-copy interface.
`spacemit-transfer.cpp` owns a bounded queue and completion events;
`spacemit-transfer-x100.cpp` is the replaceable provider. There are no core IDs,
AI-thread operations or model policies in the public interface. All functions
are also exposed by backend registry proc-address lookup using their full names.

Create the service and events on a normal **non-AI control thread before
inference**, not inside a graph callback or `compute_forward`. Creation discovers
X100 model entries in `/proc/cpuinfo`, intersects them with online CPUs and the
creator's allowed CPU set, then samples per-CPU `/proc/stat` twice 100ms apart.
Only CPUs with at most 10% non-idle time are returned in `cpus`; busy CPUs,
missing samples and backwards counters are excluded. Iowait is conservatively
counted as non-idle, and guest time is not double counted. Discovery therefore
may sleep for 100ms, only on the control-thread initialization path, never in
`submit`. It starts persistent workers and verifies affinity before returning.
No CPU indices are hardcoded, no caller/cpuset restrictions are widened, and no
fallback onto A100 is permitted. No sampled available X100 CPUs or a binding failure returns
UNAVAILABLE. Current discovery deliberately accepts only the verified K3 model
name `Spacemit(R) X100`; other firmware names need an explicit provider update.
Each worker may use any of the sampled available X100 CPUs so the OS can
load-balance it. Availability is a **load snapshot**, not an exclusive reservation:
other work can arrive after discovery. The mask is selected at creation, not
refreshed on each request; long-running services must be recreated on a control
thread if a new load snapshot is needed. One worker is the conservative default.

```cpp
// Control thread, before inference:
ggml_spacemit_transfer_status status;
ggml_spacemit_transfer_config config{1, 64};
auto mover = ggml_spacemit_transfer_create(&config, &status);
auto done = ggml_spacemit_transfer_event_create();
// Check mover/status/done. Keep buffers and event alive for accepted requests.

// Worker-0 BEGIN callback, when the relevant M threshold/policy matches:
ggml_spacemit_transfer_request request{cache_slot, weight_src, weight_bytes};
status = ggml_spacemit_transfer_submit(mover, &request, done);
// OK: return promptly and compute current nodes concurrently.
// BUSY: nothing accepted; retry at a later boundary or skip speculative work.
// Record whether this invocation accepted the request. Do not retry-spin here.

// Before consuming THIS copy, and only if submission was accepted:
if (ggml_spacemit_transfer_wait(done) != GGML_SPACEMIT_TRANSFER_OK) {
    // Propagate failure; never use an incomplete cache slot.
}

// Control thread, after inference/callbacks are quiescent:
ggml_spacemit_transfer_free(mover); // drains accepted work and joins workers
ggml_spacemit_transfer_event_free(done);
```

Submission copies only a descriptor to a preallocated ring using try-locks;
it performs no payload copy, event allocation, queue-space wait or completion
wait. BUSY also covers short mutex contention; the service is not lock-free and
does not promise real-time wakeup latency. Allocate events outside the hot path.
Only the provider workers execute `memmove`. Acquire query of COMPLETE or a
successful wait publishes the destination bytes. Each event has at most one
pending request. Reuse it only after all previous observers have finished.
Multiple workers can complete requests out of order; arrange dependencies at the
consumer, not by assuming queue order. The owner must keep src/dst alive and
avoid conflicting accesses until completion. Event destruction waits for a
pending copy; service destruction drains rather than cancels accepted jobs.

The first provider supports only CPU-accessible ordinary coherent memory. It
is **not** a page-cache prefetch, DMA engine, NUMA cache-warmup operation or a
TCM/MMIO transfer. Cache installation, prediction, buffer-slot ownership,
router-index snapshots, thresholds and graph-event wiring remain policy work.
No automatic transfer or new graph synchronization is enabled by adding this
service. Do not send transient graph outputs to the background worker unless
you can pin their lifetime and prevent allocator reuse.

`test-spacemit-transfer` tests the portable queue/events with a controlled stub
provider (full queue, pending-event protection, failures, reuse, overlap,
concurrent producers and draining shutdown). `test-spacemit-transfer-x100` uses
the real provider and verifies startup binding plus byte-copy correctness. Run
that hardware executable through `k3-submit`, not directly on the board.

### Optional cooperative cancellation

`ggml_spacemit_transfer_cancel(event)` requests cancellation of that event's
pending request. This interface is available for future policies; graph callbacks,
prefetch disabling and the CLI do **not** call it automatically.

The call uses an event try-lock and never waits for the provider. `BUSY` means
lock contention; retry later. `OK` means the request was accepted, or that this
event already acknowledges cancellation. NULL/IDLE/COMPLETE/FAILED returns
`INVALID`. An accepted cancellation wins over completion/failure if the worker
has not published a terminal state yet, even if all bytes have already been
copied. There is no rollback and no promise that a cancelled destination is empty.

A queued request is skipped when a worker dequeues it; cancellation does not
immediately reclaim its queue slot. Active transfers use chunks of at most
64 KiB, checking cancellation before each chunk. The current provider call must
finish before cancellation can be acknowledged; this is not a wall-clock latency
guarantee. Chunk order preserves overlapping-buffer `memmove` semantics.

```cpp
// Optional future policy; not connected to graph events today:
auto status = ggml_spacemit_transfer_cancel(done);
if (status == GGML_SPACEMIT_TRANSFER_OK) {
    // Acknowledgment is asynchronous: keep buffers and event alive here.
    auto result = ggml_spacemit_transfer_wait(done); // only at a safe boundary
    if (result == GGML_SPACEMIT_TRANSFER_CANCELLED) {
        // Worker stopped. Discard the destination rather than installing it.
    }
}
```

Until acknowledgment, `query` still returns `PENDING`. The terminal state is
`GGML_SPACEMIT_TRANSFER_CANCELLED_STATE`, and `wait` returns status
`GGML_SPACEMIT_TRANSFER_CANCELLED`. Acquire observation of any terminal state
means the worker no longer accesses the request's buffers. Event destruction
waits for this acknowledgment; it never initiates cancellation. Service drain
and destruction finish all accepted requests, including cancelled requests,
and continue copying requests that were not cancelled. Event reuse clears the
cancellation flag; all old observers and cancellers must leave before reuse.

Tests cover queued cancellation with no provider copy, cancellation during a
blocked chunk, safe event destruction, partial destinations, unaffected requests,
reuse, completion races and large overlapping moves in both directions.


## Whole-expert QKV experiment

`test-spacemit-expert-prefetch` connects the graph events to the X100 transfer
service for a **single-layer DeepSeek41 model**. This is an explicit experimental
policy in the test executable. Its whole-expert copy destinations and waits are
separate from the CLI hybrid mmap/page-warmup policy described above.

The control thread reserves three anonymous DDR destinations for all gate/up/down
experts and sets stable tensor addresses before creating the llama context. At
the first layer-0 QKV BEGIN, an A100 worker submits three descriptors and returns.
One available X100 worker copies all expert bytes. Later QKV events do not submit
again: the weights stay available throughout this experiment. No cancellation API
is called, including on shutdown; accepted copies drain before buffers are freed.

MXFP4 expert GEMMs currently execute on CPU. Consequently, waiting only in a
SpacemiT EXPERTS event would be incorrect. The scheduler evaluation observer waits
before the CPU split containing the first MUL_MAT_ID, then returns false so the
scheduler retains the original split. The experiment requires QKV to have
submitted first; otherwise it fails instead of consuming uninitialized memory.
All comparison modes install the same event/evaluation instrumentation and use
identical CPU expert placement and SpacemiT placement for supported operations.

The source is still a file mapping. Copying it faults storage pages and writes
a second, anonymous copy into DDR; this is not DMA. Lazy expert mappings normally
have RANDOM advice. For the `prefetch` mode the control thread changes the source
VMA advice to SEQUENTIAL without touching bytes or issuing WILLNEED/POPULATE;
the later X100 copy performs the actual reads. `prefetch-random` retains RANDOM
advice to measure that overhead explicitly.

| Mode | Loading and expert consumption |
| --- | --- |
| `resident` | Explicit full weight loading, lazy off; experts read anonymous host memory |
| `mmap-eager` | mmap with lazy off; loader's normal population/early-read behavior |
| `mmap` | mmap with lazy on; experts fault on demand, no extra DDR copy |
| `prefetch` | lazy mmap plus QKV-triggered whole-expert DDR copy, sequential source advice |
| `prefetch-random` | Same copy with the loader's original random source advice |

DeepSeek41 expert tensors are now marked eligible for lazy reads. The loader's
existing lazy mode controls whether this is used; this marker is not an automatic
transfer policy. The graph matcher can also recognize an initial projection GEMM
from an exact `blk.N.attn_q_a.weight`/Q/K/V weight role when the named q/kv output
is on a different backend. It still checks the same-layer attn_norm dependency,
including scheduler-copy names. A regression test covers split graphs, wrong
layers and weight-role suffixes that must not match.

Build `test-spacemit-expert-prefetch` and `test-spacemit-prefetch-plan`, and package
them with their libraries on K3. Run through the board queue, for example:

```sh
k3-submit --name whole-expert-compare --owner codex-expert-prefetch \
  --cwd /root/prefetch-experiment --timeout 1800 -- \
  python3 spacemit-expert-prefetch-bench.py \
  --bin-dir /root/prefetch-experiment \
  --model /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  --output-dir /root/prefetch-experiment/results
```

The runner is `scripts/spacemit-expert-prefetch-bench.py`. It executes three
independent processes per mode in rotated orders. Before each process it evicts
only the specified model's file cache with POSIX_FADV_DONTNEED, verifies residency
with mincore, and rejects residual residency above 1%. This does not clear the
SSD/controller cache or simulate power-on storage. It does not globally drop
caches. Use `--modes prefetch` for a focused three-process run.

Each process executes PP128, 16 greedy decode steps, then three PP128 repetitions
with cleared llama memory. JSON records loading, context creation, first/warm
PP128, decode, submit/wait timing, page faults and process memory. First/warm
last-token logits must match; first/decode/warm logits must match byte-for-byte
across modes. Prefetch additionally verifies every expert byte against its mapped
source, outside the reported inference timing. The process wall time includes
this expensive validation and teardown; do not treat it as inference latency.
`maxrss_kib` is sampled before full-byte validation. File-mapped RSS includes
mapped source pages and can double count weights also present in anonymous DDR.
Unavailable `/proc/self/io` counters are flagged and the runner emits nulls;
zero counters on a kernel without this interface do not mean zero storage I/O.

Measured K3 single-layer results and experiment limits: [2026-11-18 results](SPACEMIT-prefetch-results-20261118.md).

### Concurrent page warmup with expert slots

With `--prefetch --prefetch-expert-ratio 0.65` and at least two prefetch workers,
prefill now runs page warmup and mmap-to-slot copies within the same QKV-to-router
window. One request lane touches the current layer's mapped expert pages; the
remaining lanes copy expert payloads into the existing CPU slot buffer. Both use
the same X100 worker service, so the worker count does not increase. Router READY
cancels both kinds of work without waiting; only fully copied experts are
published to slots. Incomplete copies continue to use mmap. Pagewarm jobs never
reserve, publish or discard slots. A single worker retains copy-only behavior;
decode retains the predictor-driven copy policy. Omitting the ratio retains
pagewarm-only behavior, and ratio zero retains demand mmap.

### Reusing the loader mmap: asynchronous page warmup experiment

The single-layer experiment accepts `mmap-prefetch`. It reuses the loader's
mappings with lazy experts enabled. Before context creation it changes expert
VMA advice from RANDOM to NORMAL, without reading pages or allocating another
expert buffer. At layer 0 QKV BEGIN it submits three
`ggml_spacemit_transfer_prefetch(engine, source, bytes, event)` requests.
Only an X100 worker touches the pages. Expert tensor addresses remain unchanged;
the CPU expert observer records pending requests and proceeds without waiting.
Expert kernels fault missing pages on demand.

The API shares the nonblocking bounded queue, events and cooperative cancellation
protocol. This experiment deliberately does not cancel requests. COMPLETE means
all pages were touched; it does not pin pages or guarantee future residency.
There is no second anonymous expert buffer. The harness drains requests after
all measured inference and before model unmapping. `pending_at_experts`,
`pending_after_first` and `final_drain_ms` expose overlap and cleanup. Here
`wait_ms=0` means no explicit whole-request wait at consumption; `ready_ms=0`
is unused and does not measure warmup completion. Copy verification applies
only to copy modes. Warmup checks completion, unchanged tensor addresses and
byte-identical output logits against the baseline.

This section describes the single-layer experimental harness. The CLI now has
a separate context-owned page-warmup manager, installed by `--prefetch`; its
ordinary-weight allocation and request lifecycle are described above. The
historical experiment measurements do not validate the new production path.
The current implementation has not been compiled or run after these edits, as
requested. Full expert selection, a resident-memory budget and eviction remain
outside this change.

```sh
k3-submit --name mmap-warmup-cold5modes --owner codex-mmap-warmup \
  --cwd /root/mmap-warmup-20261118 --timeout 1800 -- \
  python3 spacemit-expert-prefetch-bench.py \
  --bin-dir /root/mmap-warmup-20261118 \
  --model /data/models/deepseek-v4.1-flash/deepseek-v4.1-flash-1layer-mxfp4-real.gguf \
  --output-dir /root/mmap-warmup-20261118/cold5modes \
  --modes resident mmap mmap-eager prefetch mmap-prefetch
```
For attribution, `mmap-normal` is an additional control: the same lazy mapping
and MADV_NORMAL expert advice as `mmap-prefetch`, with no transfer worker or page
warmup. Comparing it with `mmap-prefetch` isolates the asynchronous page touching
from the VMA readahead-advice change. The control experiment runs three cold
rotated rounds of `mmap mmap-normal mmap-prefetch` with the same final binary.
Measured mmap warmup and NORMAL-only attribution: [2026-11-18 mmap results](SPACEMIT-mmap-warmup-results-20261118.md).


## Bounded expert slots

Pass `--prefetch --prefetch-expert-ratio 0.65` to allocate a separate CPU expert buffer.
The ratio must be finite and in [0, 1]. The equivalent environment variable is
LLAMA_ARG_PREFETCH_EXPERT_RATIO. --prefetch is required; use --cpu-moe to keep all
routed expert banks on the CPU lazy mappings.

One slot contains all routed projection matrices (gate, up, down, or fused
gate/up) for one expert in one layer. Shared experts and routers do not count.
The slot count is floor(ratio * sum(expert_count_per_layer)). Layers may have
different projection sizes: slot stride is the largest complete expert payload
across layers, with each matrix aligned to 64 bytes. The single CPU buffer
is slot_count * slot_stride bytes (backend allocation alignment may add padding).
Zero reserves no expert buffer; one reserves all experts. Omitting the ratio
keeps the existing mmap page-warming mode and allocates no separate slot buffer.

A positive ratio must reserve at least one complete layer's experts; otherwise
loading reports an error. Zero remains mmap-only. For example, two 256-expert
layers with ratio 0.65 reserve floor(512 * 0.65) = 332 slots, and the fixed
prefill region occupies slots [0, 256). No separate prefill buffer is allocated.

Prefill writes expert `e` only to slot `e` in that fixed prefix, replacing the
previous layer's unpinned cached experts. Slots outside the prefix are never
prefill destinations. Prefill attempts all current-layer experts, with mmap
fallback if compute reaches an expert before its slot copy completes. On transition
to decode, the prefix and any retained complete experts become ordinary parts
of the same full 332-slot LRU cache; no buffer migration is needed.

Decode first protects predicted hits, then admits misses into empty slots or
the least recently used eligible slots across the entire buffer. Actual router
hits update LRU and are pinned through their local EXPERTS execution window.
For CPU-offloaded experts without a SpacemiT EXPERTS END event, the next layer
or token's QKV BEGIN releases the prior pins; its input depends on completion
of the prior FFN.
Writing slots are never evicted, including while cancellation awaits worker
acknowledgement. One expert is published only after all gate/up/down matrices
and chunks complete. CPU MUL_MAT_ID uses the resolver for complete pinned hits;
all other reads retain the original mmap slice. Completed contents belong to
the model and survive context recreation.

Positive-ratio slots are filled by X100 background workers copying the exact
expert slices from the model's existing mmap into the slots buffer. The slot
path does not reopen the model, use `pread` or `O_DIRECT`, or read sector
padding. Workers can still block on cold source page faults; unlike page
warming alone, this mode also copies and stores a second copy of the payload.

During prefill with a positive slot ratio and at least two transfer workers,
one event lane independently warms the current layer's mmap pages while the
remaining lanes copy experts into slots. With the default four workers this is
one page-warmup lane and three copy lanes, sharing the same worker service.
The warmup lane can copy when it has no eligible warmup job. Warmup progress is
tracked separately and owns no slot reservation; only complete copies publish
slots. Both job kinds are cancelled at router READY. Decode and single-worker
prefill keep the copy-only policy.

`--prefetch-threads N` (environment: `LLAMA_ARG_PREFETCH_THREADS`) selects the
number of background X100 transfer workers, default 4, minimum 1. Each worker
binds to a distinct online, affinity-eligible CPU sampled as available by the
provider. Requesting more workers than available CPUs disables warmup with a
warning; busy CPUs and A100 cores are not used as substitutes. Each copy job
has at most one in-flight chunk, and each worker handles at most one 16 MiB
chunk. The independent warmup lane may touch the same expert as a copy lane.
This option applies to both slot copies and mmap page warmup.

Both prefill and decode close the prefetch window and request cancellation as
soon as the current layer's router is ready. No more reads are submitted for
that window. Complete experts remain usable; unfinished experts use mmap.
Workers check cancellation between 64 KiB copy chunks. A copy already in
progress, including any page fault it triggers, must return before the worker
acknowledges cancellation and the slot can be reused. Failed or cancelled
transfers do not publish an expert. Omitting the ratio still uses mmap page
warming, unchanged.

The ratio bounds only the separate expert buffer, not RSS or OS file page cache.
Native accelerator expert buffers are outside this CPU slot mode. Only one
manager can own the shared SpacemiT callback; use one active inference context
per model with this mutable cache. A shared model executing concurrent CPU
expert graphs in other contexts is not supported by the current reader pins.


## Ideal-compute prefetch debugging

```bash
GGML_PREFETCH_DEBUG_IDEAL=1 GGML_PREFETCH_DEBUG_IDEAL_US=0 <PP command>
# Substitute a fixed 1000 us sleep for each handled heavy operator.
GGML_PREFETCH_DEBUG_IDEAL=1 GGML_PREFETCH_DEBUG_IDEAL_US=1000 <PP command>
```

This CPU / Spacemit debug mode produces synthetic outputs and routing. Its
outputs must not be used to evaluate inference correctness or model quality.
`GGML_PREFETCH_DEBUG_IDEAL` accepts 0 / 1; unset means disabled. The optional
`GGML_PREFETCH_DEBUG_IDEAL_US` is an integer in [0, 60000000], default 0.
Configuration is read once per process; restart after changing it.

- `MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT` and `LIGHTNING_INDEXER` skip their
  arithmetic and fill F32 outputs with 1. CPU / Spacemit fusion is disabled in
  this mode so that fused dispatch cannot bypass the hook. Other operations
  execute normally, and prefetch callbacks and synchronization remain active.
- Router ARGSORT nodes named `ffn_moe_topk*` / `ffn_moe_argsort*` receive a
  deterministic round-robin permutation. This prevents constant outputs from
  routing every token to the same experts. Full expert coverage still requires
  sufficient token rows and must be measured in each experiment.
- Expert reads retain the normal resolver boundary: published slots are used
  when available, otherwise original mmap slices. One volatile byte per 4096
  bytes and the final byte are read for each selected expert. Dense weights and
  attention K/V inputs are also touched pagewise. This preserves page-fault and
  residency costs, but does not reproduce full GEMM DDR traffic or cache reuse.
- All workers sleep concurrently for the requested duration per handled heavy
  operator, followed by the executor's normal barrier. Sleep scheduling may
  overshoot, particularly for very short waits. Zero removes intentional wait;
  page access, output writes, remaining operators and synchronization still
  contribute to PP time. Measured times are simulated, not real-model speedups.
