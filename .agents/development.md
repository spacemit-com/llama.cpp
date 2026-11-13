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

## Trace the affected execution path

Read the caller, support predicate, tensor traits, selected kernel, and relevant fallback before changing an implementation.
Start with `ggml_backend_spacemit_device_supports_op` and `ggml_spacemit_get_tensor_traits` when changing which operations or tensor layouts the backend accepts.
Follow `ggml_backend_spacemit_graph_compute` through workspace sizing, runtime launch, and `ggml_spacemit_compute_opnode` when changing execution.
Follow buffer initialization, tensor upload, repack selection, and tensor readback when changing weight storage.
For fusion, inspect graph optimization, opnode construction, allocation dependencies, and the unfused behavior together.
Write down the assumptions the change relies on, including tensor dimensions, byte strides, types, alignment, vector length, worker count, and available scratch space.
Check that support predicates and dispatch agree with the implementation's actual preconditions.

## Keep the implementation focused

Make the smallest coherent change that fixes the problem or implements the requested behavior.
Preserve unaffected interfaces and behavior, and avoid unrelated formatting, renaming, or framework work.
Use existing dispatch, allocation, and synchronization mechanisms when they express the required behavior.
Hardware specialization is appropriate when its capability, vector length, type, layout, shape, alignment, or resource conditions are explicit and justified.
Keep unsupported inputs on a valid existing fallback or reject them through the appropriate support predicate.
Do not use model names, file paths, or benchmark identities as substitutes for technical dispatch conditions.
Separate compiler instruction support from runtime hardware support when selecting IME1, IME2, or RVV code.
Explain non-obvious layout, arithmetic, or ordering requirements with concise comments near the relevant code.

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

Review the final diff against the task scope and explain the concrete behavior change.
Apply the relevant checks in [acceptance.md](acceptance.md) and report their actual outcomes, including failures and unavailable coverage.
Keep the rationale, reproducible evidence, and remaining limitations consistent with [delivery.md](delivery.md).
