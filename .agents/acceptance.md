# Acceptance evidence

Use this contract for changes to `ggml/src/ggml-spacemit/**` or the maintained multimodal integration under `tools/**`.
Select checks from the affected behavior and failure risk; a larger test count is not stronger evidence by itself.
Record the baseline, intended behavior, affected paths, selected cases and acceptance criteria before evaluating the candidate.
Use [the validation record](templates/validation.md) to make the result reproducible and reviewable in a PR.
Existing build and test facilities outside the primary source paths may be reused. Temporary local cases or minimal reproducers are allowed under the [test-case policy](../CONTRIBUTING.md#test-case-policy); do not submit their test-code changes in the PR.
If required coverage is unavailable within the agreed scope, record the gap rather than silently expanding the change.

## Select the necessary evidence

Apply every relevant row and explain exclusions briefly. Classify the change by its effect, not its filename. A mechanical edit must preserve computation, dispatch, layout, ownership, and execution cost; justify that classification with the diff and affected contracts. Build flags or dependencies that alter those properties also require execution-path checks. If the impact is uncertain, keep the affected checks required until the uncertainty is resolved.

| Change | Build and correctness evidence | Additional acceptance evidence |
| --- | --- | --- |
| Documentation or mechanical change | Check links/commands for prose; compile affected targets and check changed references or interfaces for mechanical source edits. Explain why runtime behavior is unchanged. | Model correctness and performance can be `N/A` with this impact-based justification; a source-file edit alone does not require a benchmark. |
| Build configuration | Configure and build affected targets; check linking, backend loading, intended IME features, and runtime dependency resolution. | Add execution-path and model checks if compiler flags, feature selection, or dependencies can change execution or numerics; performance checks follow the risk or claim. |
| Operator bug fix | Build the backend and selected existing test targets or local reproducer; reproduce the defect on the baseline and pass the same input and criterion on the candidate against an independent reference. | Exercise the failing shape/layout/dtype, nearest relevant boundary, an unaffected supported case, and affected model behavior. Measure performance when execution cost may change. |
| Numerical or repack change | Independent arithmetic/dequantization reference, exact input identity, declared error criteria and relevant format/layout boundaries. | Separate packed-representation correctness from error against original weights; validate affected quantization/model behavior and any changed memory footprint. |
| Dispatch or kernel optimization | Correctness on the selected path, eligibility boundaries and fallback; prove that the intended implementation executed. | Paired operator and representative production-model measurements; include relevant prefill/decode and worker configurations, retaining regressions and inconclusive results. |
| Memory, workspace or concurrency change | Allocation/size/alignment/lifetime checks and affected numerical cases; exercise relevant tails, allocation failure/fallback, reuse and overlapping requests. | Demonstrate synchronization and ownership under the claimed concurrency; check leaks, corruption, hangs and resource release across repeated lifecycle operations, plus relevant latency/throughput. |

## Scope-to-test mapping

Select the rows for every changed area; do not use a broad build as a substitute for the affected behavior. The runtime rows below apply when execution behavior changes; mechanical and build-only edits follow the corresponding impact rows above.

| Changed area | Required targeted evidence | Add when applicable |
| --- | --- | --- |
| `ggml/src/ggml-spacemit/**` execution-path changes | Backend build, selected operator or independent numerical cases, evidence that `SPACEMIT0` and the intended dispatch path executed, and end-to-end correctness on the relevant model or models. | Require comparable baseline/candidate model-level performance when execution cost may change or a benefit is claimed. Cover affected prefill/decode, workers, and platforms; retain regressions and inconclusive results. |
| `tools/mtmd/**` | Build the affected mtmd targets and run `test-mtmd-c-api` / `test-mtmd-impl` or the closest selected cases. | Run the affected image, audio, video, or model smoke with fixed media and expected semantic behavior. |
| `tools/smt-mtmd/**` | Build with `LLAMA_SERVER_SMT_MTMD=ON`, verify matching `SPACEMIT_ORT_DIR` and runtime artifacts, and run the affected SMT media pipeline. | Cover vision, audio, multi-ASR, or TTS separately when only one pipeline changes; include K1/K3 where the path differs. |
| Media-related `tools/server/**` | Build `llama-server` and the server component target; run the relevant `tools/server/tests` pytest selection. | Run at least one real media request for vision/audio/SMT changes and exercise concurrency when request lifetime or scheduling changes. |
| Cross-layer backend plus tools change | Satisfy every affected lower-layer row and run one end-to-end model/media request through the actual server or CLI path. | Add a production-like model and target-board run when the PR claims user-visible behavior or performance. |

For execution-path changes, operator or kernel checks supplement the affected model or media validation; they do not replace it. Performance measurements are required for changes that may affect execution cost, such as kernel, dispatch, repack, allocation, or worker changes, and for performance claims. Explain why a comparison is `N/A` when no performance impact or claim applies. Missing hardware, models, or inputs is `NOT RUN`, not `N/A`, and follows the merge-decision rules below.

The `tools/**` umbrella does not require all tools to be tested for every change; narrow the selection to the changed target and its callers.
Changes to other repository paths, such as public headers, top-level build files, or CI configuration, are possible but strongly discouraged.
For each such path, record evidence that the task cannot be completed without changing it, the alternatives considered and why they fail, its exact necessity, and the corresponding validation.
This exception does not permit submitting test-code changes. Describe the cases, local adaptations, commands, and results in the PR instead.
Do not develop or optimize the retired `ggml/src/ggml-cpu/spacemit/**` backend; backend optimization work belongs in `ggml/src/ggml-spacemit/**`.

Keep regressions focused on the modified contract and its callers.
Use broader model or operation coverage when shared dispatch, representation or memory ownership makes the impact broad, and state that reason.
For bug fixes without a reproducible baseline failure, describe the missing reproduction and supporting evidence instead of claiming a demonstrated fix.

## Required evidence and merge decisions

Required checks come from the impact analysis and tables above. Record which checks are required, which do not apply and why, and which provide additional coverage before interpreting results. Changing a check to optional or `N/A` requires an impact-based justification, not an unavailable board or an inconvenient result.

- A demonstrated new correctness or safety defect blocks merge until fixed. Preserve the failed result and the successful revalidation.
- A required check without valid passing evidence leaves merge blocked until it is satisfied or a maintainer explicitly accepts the validation gap. This includes `NOT RUN`, inconclusive `FAIL`, unavailable board/model checks, and missing proof that the intended path ran. Disclose known baseline failures separately; they do not silently waive a required check. Confirmed new correctness or safety defects still require a fix.
- Report the missing evidence, reason, affected paths, available substitute evidence and its limits, and residual risk in the PR. The contributor or agent must not grant its own exception or claim merge readiness merely because the gap is documented.
- Record any maintainer exception with a link to the decision, the checks and candidate revision it covers, accepted risk, and follow-up owner/action. Without an explicit decision, the gap remains unresolved. Reassess the exception after changes to the affected scope or candidate.
- An accepted exception does not turn `NOT RUN` or inconclusive evidence into `PASS`. Additional checks outside the justified required scope do not become blockers merely by being listed; explain their applicability and retain observed failures.
- A performance improvement claim always needs a comparable baseline/candidate run. If that evidence is absent, withdraw the claim; a maintainer exception cannot establish an unmeasured improvement.

These are review requirements, not a new automated gate. Maintainers decide whether to accept a validation exception and whether the PR can merge.

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
The separate standalone SpacemiT numerical suite under development elsewhere is not an available dependency of this branch. Permanent backend-specific cases can be contributed there once it is available. For now, explain the case construction and local reproduction in the PR and keep test-code changes out of the submitted diff.
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

## Preserve execution records

Use the existing validation template and test tools; these requirements apply whether records are collected by hand, by an agent, or by a script. Record actual execution, not a proposed command or an expectation copied from the plan.

- Give each attempt a distinct local run label and record start/end times with timezone, working directory, resolved command, relevant environment, and exit status or termination reason. Link it to the affected behavior in the development-map note and the selected check. A directory name or record heading is sufficient; no case registry is required.
- Bind each attempt to its base/candidate source revision and local patch identity, tested binaries and loaded libraries, model/input identity, runtime, and hardware. Shared metadata may be referenced when it is identical across runs; record any differences explicitly.
- Retain raw stdout/stderr, actual execution counts, dispatch evidence, reference and error criteria, observed outputs, and raw performance samples as applicable. If a tool cannot expose an item, record the gap and its effect on the conclusion; do not fabricate counts or path evidence.
- Keep failed, interrupted, excluded, contaminated, and retried attempts with their reasons and links to any replacement run. A later success does not erase an earlier failure; explain whether it was resolved, environmental, or still unclassified.
- After source, build, dependency, input, or configuration changes, check whether earlier evidence still applies. Rerun affected checks or mark their coverage unverified. Evidence carried forward needs a concrete reason that the tested artifact and behavior remain applicable to the final candidate; never relabel an old run as a new one.

Use `PASS` only when the check's declared criterion is established by valid evidence. Use `FAIL` when an executed check does not establish that criterion, with a reason distinguishing an observed defect from interrupted, contaminated, or inconclusive evidence. Use `NOT RUN` when the required check did not execute, including empty selections or all-skipped runs, and retain any attempted command and its output. Use `N/A` only for a requirement that does not apply, with a scope-based reason; missing hardware or inputs are not `N/A`.
A check's status describes its acceptance evidence, not a diagnosis of the code. If execution occurred but the conclusion is unverified, say so explicitly; do not disguise it as `NOT RUN` or `PASS`.
Status and merge disposition are separate: apply [required evidence and merge decisions](#required-evidence-and-merge-decisions) to every unresolved required check.

## Report the outcome

Separate build, execution, numerical correctness, performance and platform coverage in the validation record.
Attach enough commands, identities, counts and evidence links for a reviewer to reproduce and assess the claim without access to a private session.
State unavailable boards, non-independent references, missing cases, known baseline failures and inconclusive performance explicitly.
Only label the checks actually satisfied as passed; describe unresolved acceptance requirements as unverified.
