# Acceptance evidence

Use this contract for changes to `ggml/src/ggml-spacemit/**` or the maintained multimodal integration under `tools/**`.
Select checks from the affected behavior and failure risk; a larger test count is not stronger evidence by itself.
Record the baseline, intended behavior, affected paths, selected cases and acceptance criteria before evaluating the candidate.
Use [the validation record](templates/validation.md) to make the result reproducible and reviewable in a PR.
Existing build and test facilities outside the primary source paths may be used without changing them.
If required coverage is unavailable within the agreed scope, record the gap rather than silently expanding the change.

## Select the necessary evidence

Apply every relevant row and explain exclusions briefly.

| Change | Build and correctness evidence | Additional acceptance evidence |
| --- | --- | --- |
| Documentation or build configuration | Documentation-only changes need link/command consistency checks; build changes need affected RISC-V configuration, linking and backend loading checks. | Confirm intended IME features and runtime dependency resolution; no model benchmark for prose alone. |
| Operator bug fix | Build the backend and selected test targets; reproduce the defect on the baseline and pass the same input and criterion on the candidate against an independent reference. | Exercise the failing shape/layout/dtype plus the nearest relevant boundary and an unaffected supported case; run a representative model when graph integration can change. |
| Numerical or repack change | Independent arithmetic/dequantization reference, exact input identity, declared error criteria and relevant format/layout boundaries. | Separate packed-representation correctness from error against original weights; validate affected quantization/model behavior and any changed memory footprint. |
| Dispatch or kernel optimization | Correctness on the selected path, eligibility boundaries and fallback; prove that the intended implementation executed. | Paired operator and representative production-model measurements; include relevant prefill/decode and worker configurations, retaining regressions and inconclusive results. |
| Memory, workspace or concurrency change | Allocation/size/alignment/lifetime checks and affected numerical cases; exercise relevant tails, allocation failure/fallback, reuse and overlapping requests. | Demonstrate synchronization and ownership under the claimed concurrency; check leaks, corruption, hangs and resource release across repeated lifecycle operations, plus relevant latency/throughput. |

## Scope-to-test mapping

Select the rows for every changed area; do not use a broad build as a substitute for the affected behavior.

| Changed area | Required targeted evidence | Add when applicable |
| --- | --- | --- |
| `ggml/src/ggml-spacemit/**` | Backend build, selected operator or independent numerical cases, evidence that `SPACEMIT0` and the intended dispatch path executed, and end-to-end correctness on the relevant model or models that exercise the changed path. | Run comparable baseline/candidate performance measurements on the relevant model or models, covering the affected prefill/decode or request path and retaining regressions and inconclusive results. |
| `tools/mtmd/**` | Build the affected mtmd targets and run `test-mtmd-c-api` / `test-mtmd-impl` or the closest selected cases. | Run the affected image, audio, video, or model smoke with fixed media and expected semantic behavior. |
| `tools/smt-mtmd/**` | Build with `LLAMA_SERVER_SMT_MTMD=ON`, verify matching `SPACEMIT_ORT_DIR` and runtime artifacts, and run the affected SMT media pipeline. | Cover vision, audio, multi-ASR, or TTS separately when only one pipeline changes; include K1/K3 where the path differs. |
| Media-related `tools/server/**` | Build `llama-server` and the server component target; run the relevant `tools/server/tests` pytest selection. | Run at least one real media request for vision/audio/SMT changes and exercise concurrency when request lifetime or scheduling changes. |
| Cross-layer backend plus tools change | Satisfy every affected lower-layer row and run one end-to-end model/media request through the actual server or CLI path. | Add a production-like model and target-board run when the PR claims user-visible behavior or performance. |

Every source change under `ggml/src/ggml-spacemit/**` must include both end-to-end correctness and model-level performance validation for the relevant model or models that exercise the changed path. Operator or kernel checks are supporting evidence and do not replace the model run. If no relevant model, input, runtime, or hardware is available, record the model correctness or performance check as `NOT RUN` with the reason and its effect on the conclusion; do not treat the missing model evidence as a pass.

The `tools/**` umbrella does not require all tools to be tested for every change; narrow the selection to the changed target and its callers.
Changes to other repository paths, such as `tests/**`, public headers, top-level build files, or CI configuration, are possible but strongly discouraged.
For each such path, record evidence that the task cannot be completed without changing it, the alternatives considered and why they fail, its exact necessity, and the corresponding validation.
Do not develop or optimize the retired `ggml/src/ggml-cpu/spacemit/**` backend; backend optimization work belongs in `ggml/src/ggml-spacemit/**`.

Keep regressions focused on the modified contract and its callers.
Use broader model or operation coverage when shared dispatch, representation or memory ownership makes the impact broad, and state that reason.
For bug fixes without a reproducible baseline failure, describe the missing reproduction and supporting evidence instead of claiming a demonstrated fix.

## Build and platform identity

For executable or build changes, follow [the SpacemiT build guide](../docs/build-riscv64-spacemit.md) with the agreed toolchain and runtime versions.
Record source revision plus local diff identity, compiler, CMake configuration, detected IME capabilities, runtime version and build commands with exit status.
Build with `GGML_SPACEMIT=ON` and confirm that the standalone backend is produced and selected; the legacy `GGML_CPU_RISCV64_SPACEMIT` path is outside this source scope.
Rebuild every changed component and deploy a coherent package with executable, backend and dependency hashes; record the libraries actually loaded at runtime.
Avoid mixing a current executable with stale libraries found through the working directory, RPATH or `LD_LIBRARY_PATH`.
Host-only builds or QEMU runs can provide specific correctness evidence but do not establish K1/K3 hardware or performance acceptance.
Identify whether K1/IME1, K3/IME2, shared RVV/scalar code or shared dispatch is affected.
Test each affected platform where practical and report unavailable coverage separately; K3 success does not certify K1 behavior.
For platform-specific changes, explain why the other platform's path is unaffected rather than requiring every model on both boards.

## Correctness and path evidence

Preserve exact operands or reproducible input generation, including seeds, shapes, dtypes, strides, quantization metadata and relevant attributes.
Choose error criteria from the operation's semantics and precision before inspecting results; report non-finite values and worst errors as applicable.
Never regenerate expected values from the candidate to erase a failure or relax tolerances merely to obtain a pass.
For lossy repacking or requantization, measure both the represented operation and the deviation from the original representation; one passing comparison does not prove the other.
For a changed fast path, retain evidence of `SPACEMIT0` execution and the relevant dispatch branch, not just device discovery or a successful support query.
Record CPU fallback or unsupported cases explicitly and distinguish expected fallback from failure to exercise the candidate.
Use targeted logs, tracing, profiling or a verified dispatch predicate to establish execution; keep intrusive instrumentation outside performance timing.

[test-backend-ops](../tests/test-backend-ops.cpp) is an existing integration test tool with operation and parameter filters.
It requests the CPU reference implementation through `ggml_backend_cpu_set_use_ref`, which alone is not proof that the comparison is independent.
Verify that the reference does not call the implementation being tested, including architecture-specific CPU dispatch or shared helpers changed by the patch.
Use an independently justified scalar/arithmetic reference or isolated reference configuration when that independence cannot be established.
The separate standalone SpacemiT numerical suite under development elsewhere is not an available dependency of this branch.
If no independent reference is available, report numerical correctness as unverified and describe exactly what the integration comparison establishes.

Reconcile the selected case list with actual executed, passed, failed, skipped, unsupported and missing results.
An exit code of zero, empty selection, all-skipped run or absent output is not acceptance evidence.
Preserve crashes, timeouts and partial results; do not count them as skipped successes.
Text readability is a smoke check; use task-specific expected behavior for model correctness and distinguish it from numerical comparison.

## Performance evidence

State the workload, timing boundary, expected benefit and acceptable regression range before comparing versions.
Identify baseline and candidate artifacts separately and keep all other inputs and dependencies matched, including model/input hashes, batch and microbatch sizes, context depth, KV types and attention settings.
Record CPU threads and SpacemiT workers separately, together with board identity, affinity, frequency and relevant environment overrides.
Compare on the same idle board under matching conditions; monitor for unrelated work and preserve evidence for excluded or contaminated runs.
Use the same warmup policy and repeat measurements, alternating baseline/candidate order across rounds to expose drift.
Retain raw samples, commands, exit codes and exclusions, and report sample count plus a suitable central value and spread.
When a difference is comparable to run-to-run variation, repeat the affected comparison or report it as inconclusive; do not choose only favorable retries.
Measure affected representative production models in addition to isolated kernels or operators when claiming a model-level benefit.
Report local kernel gains, end-to-end results and regressions separately; a faster kernel does not demonstrate a faster model.
Do not call a numerically failing or differently loaded workload an accepted performance improvement.

Read [llama-bench usage](../tools/llama-bench/README.md) and verify the deployed binary's options before running commands.
In [the current implementation](../tools/llama-bench/llama-bench.cpp), `-p P -n N` creates separate prompt and generation tests; with the default `-d 0`, the generation test starts with an empty KV cache.
The `-p` value does not establish context for that separate generation test.
For generation after a synthetic context, `-p 0 -n N -d C` prepares or restores depth `C` before timing; record this depth and input protocol.
The combined `-p 0 -n 0 -pg C,N` measures prompt plus generation together, so it is not a direct decode-only rate.
For natural-input or routing-sensitive claims, use fixed real inputs and report the actual context and generated work; synthetic benchmark tokens are not a semantic model test.

## Command patterns

Set each variable to a reviewed binary, artifact or case selection and retain the resolved command with its output.
These are invocation patterns, not a complete acceptance suite or evidence that any test has run.
The backend test's current argument parser prints usage and exits nonzero for `--help`; that invocation is not a failed correctness case.

```bash
"$OPS_BIN" --list-ops
"$OPS_BIN" -b SPACEMIT0 -o "$OP_FILTER" -p "$CASE_FILTER" --output csv
"$BENCH_BIN" --help
"$BENCH_BIN" --list-devices
"$BENCH_BIN" -m "$MODEL" -dev SPACEMIT0 -ngl 99 -t "$CPU_THREADS" -b "$BATCH" -ub "$UBATCH" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPETITIONS" -o json
```

Confirm nonempty executed coverage and actual backend/dispatch behavior after applying filters.
Capture benchmark warmup, worker settings and remaining defaults alongside the command rather than assuming that `-t` controls every backend worker.

## Report the outcome

Separate build, execution, numerical correctness, performance and platform coverage in the validation record.
Attach enough commands, identities, counts and evidence links for a reviewer to reproduce and assess the claim without access to a private session.
State unavailable boards, non-independent references, missing cases, known baseline failures and inconclusive performance explicitly.
Only label the checks actually satisfied as passed; describe unresolved acceptance requirements as unverified.
