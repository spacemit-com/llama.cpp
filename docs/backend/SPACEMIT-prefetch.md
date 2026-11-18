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

The setting is passed through `common_params.prefetch` and
`llama_context_params.prefetch` to the optional backend registry function
`ggml_backend_set_prefetch(backend, enabled)`. Backends without this capability
ignore it. SpacemiT caches the recognized module windows when enabled, including
when no callback is installed. Without a callback there are no new event barriers
or weight copies: this switch enables the planning entry point, not an automatic
expert-cache policy. The X100 transfer service still needs explicit requests.

`ggml_backend_spacemit_set_prefetch` can pause/resume an installed callback while
idle and invalidates the cached event plan. Installing a callback also enables
prefetch for compatibility with existing callers; removing it disables prefetch.
As with the callback API, handles for the same device share this configuration.
The added context-parameter field requires rebuilding callers against this header.

## Recognized windows

| Module | Recognition | Boundary |
| --- | --- | --- |
| QKV | Named Qcur/Kcur/Vcur/qkv or DeepSeek41 q/kv, with matmul dependencies reaching attn_norm of the same layer | Available local projection/preparation branches, including normalization and RoPE |
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
