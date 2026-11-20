# SpacemiT backend development

This guide covers implementation changes in `ggml/src/ggml-spacemit/**` and the maintained multimodal integration under `tools/**` for the SpacemiT fork.
Use [acceptance.md](acceptance.md) to assess results and [delivery.md](delivery.md) to prepare a reviewable change.

## Establish the task

State the affected behavior, the reason for changing it, and the expected result before editing.
Record the base revision, existing local changes, target hardware, and intended source scope.
Prefer implementation edits in `ggml/src/ggml-spacemit/**` or the relevant `tools/**` subdirectory.
Reading dependencies and running existing build or test facilities elsewhere does not expand the modification scope.
Changes elsewhere in the repository are possible but strongly discouraged.
Before making one, establish with code-path evidence why the task cannot be completed without it, identify alternatives considered and why they fail, and name the exact necessary paths.
Include the effect of each such path in the planned diff, validation, and PR explanation; do not silently broaden the change.
This general exception excludes `ggml/src/ggml-cpu/spacemit/**`, which is a retired backend and must not receive new backend development or optimization.

## Before implementation: choose the approach

Apply these principles before editing, using the development map and execution-path analysis below. Keep relevant decisions in the existing impact note at a level proportional to the task.

- **Address the cause.** Identify the failing contract or the behavior the feature must establish. Choose the smallest coherent solution that addresses it; line count alone is not the goal. Do not plan arbitrary delays, retries, hardcoded exceptions, or swallowed errors merely to hide a failure. A necessary compatibility measure needs a specific technical reason, bounded applicability, and validation in the development record or PR.
- **Choose reliable reference code.** Prefer maintained implementations with matching responsibilities and preconditions. Check whether a copied pattern is a supported design, a historical workaround, or a special case. Existing code is evidence to examine, not automatic permission to repeat it. Do not propagate retired backend patterns or expand a special-case solution beyond its valid conditions.
- **Identify hot-path costs.** Determine whether the change affects initialization, graph preparation, repacking, per-operator execution, or per-token execution. Plan to avoid repeated allocation, conversion, configuration lookup, logging I/O, and unnecessary synchronization in frequently executed paths. Move work to a suitable preparation stage only when input validity and resource lifetimes allow it; justify and measure necessary runtime costs. This is not a blanket ban on allocation throughout the backend.
- **Bound each rule.** State the hardware, dtype, shape, layout, resource, or lifecycle conditions under which a new restriction applies. A local correction must not become a global ban without evidence. Preserve valid neighboring paths and select checks for the boundary of the restriction.

## Source map

Paths in the backend table below are relative to `ggml/src/ggml-spacemit/`.

| Files | Responsibility |
| --- | --- |
| `ggml-spacemit.cpp` | Backend registration, buffers, supported operations, graph preparation, runtime launch, and synchronization. |
| `ime.cpp`, `ime.h` | Tensor traits, repack selection, workspace requirements, operator routing, and matrix execution. |
| `ime1_kernels.cpp`, `ime2_kernels.cpp`, `rvv_kernels.cpp`, `scalar_kernels.cpp` | Architecture-specific and scalar compute implementations. |
| `repack.cpp`, `repack.h` | Packed weight representations, conversion, and readback. |
| `spe_kernel_dispatch.cpp`, `spe_kernels*.h`, `ime_kernels.h`, `rvv_kernels.h`, `scalar_kernels.h` | Kernel interfaces and supporting dispatch. |
| `spacemit-opnode.h`, `spacemit-fused.cpp`, `spacemit-fused.h` | Execution nodes and fused operator behavior. |
| `spacemit-session.h`, `spacemit-context.h`, `spacemit-env.h`, `spine_barrier.h` | Session state, worker context, configuration, and synchronization support. |
| `spine_mem_pool.cpp`, `spine_mem_pool.h`, `CMakeLists.txt` | Backend allocation and build integration with the prebuilt runtime. |

## Multimodal source map

| Files | Responsibility |
| --- | --- |
| `tools/mtmd/**` | Generic multimodal model loading, preprocessing, tokenization, graph construction, and encode/decode APIs. |
| `tools/smt-mtmd/media-worker.*`, `tools/smt-mtmd/smt/smt-media-common.*` | Shared media worker and SpacemiT provider/configuration helpers. |
| `tools/smt-mtmd/smt/smt-vision-*`, `smt-audio-*`, `multi-asr/**`, `qwen3-tts/**` | SpacemiT vision, audio, multi-ASR, and TTS pipelines. |
| `tools/server/server-media.*`, `server-context.*`, `server-common.*`, and SMT-gated server files | Media request handling, multimodal context lifetime, and server integration. |
| `tools/smt-mtmd/CMakeLists.txt`, `tools/server/CMakeLists.txt`, top-level CMake and SpacemiT workflow | Build switches, ORT/SPERT dependencies, target wiring, and deployment configuration. |

Read the applicable `tools/mtmd/README-dev.md`, `tools/server/README-dev.md`, and build workflow before changing these paths.

The similarly named `ggml/src/ggml-cpu/spacemit/` directory is a retired legacy backend; do not use it for new backend development or optimization.
Route SpacemiT backend optimization through `ggml/src/ggml-spacemit/`.
Some local code retains historical namespaces; determine ownership from the source path and build target rather than the namespace alone.

## Keep the change located on the development map

Before editing, the agent must understand and be able to explain what will change and where its effects sit on the source maps above. A list of edited files alone does not establish the impact. Record a concise impact note in the existing [validation record](templates/validation.md):

- Identify the changed responsibility and concrete entry points or symbols, with links to the relevant source-map rows and code.
- Trace callers, dispatch, data representation, resource ownership, and downstream consumers as relevant. Include affected behavior in files that do not need editing.
- Name the model or feature paths, inputs, and hardware conditions that can exercise the change. Explain which nearby paths should remain unaffected and why; treat unresolved reachability as a coverage gap.
- Connect each affected behavior or contract to the selected checks in [acceptance.md](acceptance.md), including required model correctness and performance checks. Explain exclusions and unavailable coverage.

For example, a repack change can affect tensor upload, traits, packed kernels, readback, and model output even when only `repack.cpp` is edited. Follow those dependencies before deciding the verification scope.
Keep this understanding current while developing. If the call path or impact expands, update the note and verification selection before relying on earlier results. At handoff, reconcile the final diff with the map and evidence; explain every changed responsibility and every remaining coverage gap.
Update the source-map descriptions when responsibilities or entry points change. Use the existing maps and record; no separate case-definition files or `dry-run` step are required.

## Trace the affected execution path

Read the caller, support predicate, tensor traits, selected kernel, and relevant fallback before changing an implementation.
Start with `ggml_backend_spacemit_device_supports_op` and `ggml_spacemit_get_tensor_traits` when changing which operations or tensor layouts the backend accepts.
Follow `ggml_backend_spacemit_graph_compute` through workspace sizing, runtime launch, and `ggml_spacemit_compute_opnode` when changing execution.
Follow buffer initialization, tensor upload, repack selection, and tensor readback when changing weight storage.
For fusion, inspect graph optimization, opnode construction, allocation dependencies, and the unfused behavior together.
Write down the assumptions the change relies on, including tensor dimensions, byte strides, types, alignment, vector length, worker count, and available scratch space.
Check that support predicates and dispatch agree with the implementation's actual preconditions.

## During implementation: enforce the contracts

Apply the following principles to the code being changed. Implement concrete safeguards where the task exposes a relevant problem; these principles do not require a new framework, a new lint tool, or unrelated repository-wide cleanup.

- **Make interfaces hard to misuse.** Express preconditions, resource ownership, and lifecycle through existing interfaces, types, support predicates, and boundary checks where practical. Keep shared rules such as workspace sizing and buffer slicing consistent rather than duplicating assumptions at each caller. Introduce a helper or interface change only when it is needed for the task.
- **Respect module boundaries.** Place behavior in the layer responsible for it and connect layers through their established interfaces. Keep model or media business decisions out of low-level kernels, and do not bypass dispatch, allocation, or synchronization contracts. If a boundary must change, explain its necessity and update the development-map impact and affected validation.
- **Turn recurring mistakes into checks.** When fixing a reproducible defect or repeated misuse, use a focused regression test, support-condition check, type/interface constraint, or existing static check that can detect it. Prefer existing test and build facilities. A comment or a new prose rule does not replace a required regression test or validation; if a constraint cannot be checked mechanically, explain its reasoning and review evidence in the development record.
- **Make failures actionable.** At the appropriate validation or error boundary, identify the violated condition and enough actual/expected information to locate the cause, such as dtype, shape, or required/available workspace. Preserve a truthful error or an explicitly supported fallback. Do not suppress a failing check, relax a criterion merely to pass, or fabricate success. Keep diagnostics out of the normal compute hot path and performance timing.

## Keep the implementation focused

Make the smallest coherent change that fixes the problem or implements the requested behavior.
Preserve unaffected interfaces and behavior, and avoid unrelated formatting, renaming, or framework work.
Use existing dispatch, allocation, and synchronization mechanisms when they express the required behavior.
Hardware specialization is appropriate when its capability, vector length, type, layout, shape, alignment, or resource conditions are explicit and justified.
Keep unsupported inputs on a valid existing fallback or reject them through the appropriate support predicate.
Do not use model names, file paths, or benchmark identities as substitutes for technical dispatch conditions.
Separate compiler instruction support from runtime hardware support when selecting IME1, IME2, or RVV code.
Default to no new code comments. Add only brief data/memory-layout explanations or essential usage notes for genuinely new interfaces or features, following the [comment standards and examples](../CONTRIBUTING.md#coding-guidelines). Keep other design rationale and validation details in the development record or PR.

## Preserve numerical and layout contracts

Track the logical tensor representation separately from the bytes stored in a packed buffer.
Check block sizes, padding, row strides, alignment, scales, zero points, signedness, accumulator width, and tail handling where they affect the change.
Distinguish lossless rearrangement from conversion or requantization that changes represented values.
For a packed kernel, compare its arithmetic against an independent reference evaluated on the values actually represented by that packed format.
For conversion or repack changes, also compare against the original tensor values to measure any representation error.
Passing the packed-format comparison alone does not establish equivalence to the original model weights.
Ensure the reference does not dispatch to the same implementation under test, and document how its independence was established.
Choose numerical tolerances from the operation, accumulation, and representation contract before inspecting candidate results.
Cover the changed dimensions and relevant boundaries, including partial tiles, non-contiguous views, broadcast or batch dimensions, and multi-chunk accumulation when supported.
Check upload and readback behavior when storage or traits change, including partial access where the interface permits it.

## Preserve memory and execution ordering

Keep workspace sizing consistent with every buffer region used by the compute path.
Account for alignment, padding, worker slices, TCM capacity, and simultaneous buffer lifetimes.
Check allocation failure and insufficient-resource behavior for affected paths.
Preserve synchronization across workers and the completion contract of graph execution.
For fusion or execution reordering, verify aliasing, views, producer-consumer dependencies, and allocator reuse before assuming an input remains live.
Treat cached graph state and shared session resources as part of the change when their ownership or lifetime is affected.

## Plan verification before implementation

Use [templates/validation.md](templates/validation.md) to record the baseline, intended cases, references, metrics, and known limitations at a level proportional to the change.
Select checks that can expose the failure or validate the intended behavior rather than merely repeat the implementation.
Follow the [SpacemiT build guide](../docs/build-riscv64-spacemit.md) and record the actual compiler, CMake options, and prebuilt spine-runtime package used.
The standalone backend uses `GGML_SPACEMIT=ON` and `SPERT_DIR`; the legacy CPU SpacemiT path is not an interchangeable build target.
For every source change under `ggml/src/ggml-spacemit/**`, select the relevant model or models before editing and plan both end-to-end correctness and model-level performance checks. Operator and kernel tests supplement these model runs; they do not replace them.
Use the scope-matched facilities in [acceptance.md](acceptance.md): backend operators and numerical cases for `ggml-spacemit`, mtmd tests and media smoke for `tools/mtmd`, SMT build and pipeline checks for `tools/smt-mtmd`, and server component/API tests for media-related `tools/server` changes.
Existing `test-backend-ops`, `llama-cli`, `llama-bench`, `llama-server`, mtmd tests, server pytest cases, and SpacemiT build workflows may be used as evidence.
Verify the intended backend, dispatch path, binaries, and loaded libraries before interpreting results.
Choose architecture coverage from the changed paths, and state explicitly which affected hardware or fallback paths remain untested.
For performance work, define the workload and comparison conditions in advance and validate correctness before using timing results as acceptance evidence.
A successful build, an emulator run, and a run on target hardware provide different evidence and must be reported separately.

## Prepare the result for review

Review the final diff against the task scope and the development-map impact note, and explain the concrete behavior change. Revisit the approach decisions and implementation contracts above against the final code and evidence. Reassess earlier evidence after further edits; rerun affected checks or mark their coverage unverified as required by [acceptance.md](acceptance.md#preserve-execution-records).
Apply the relevant checks in [acceptance.md](acceptance.md) and report their actual outcomes, including failures and unavailable coverage.
Keep the rationale, reproducible evidence, and remaining limitations consistent with [delivery.md](delivery.md).
