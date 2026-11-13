# SpacemiT validation record

Complete this record for an executable or backend change. Keep the detailed record in `.agents/workspace/` while working, then link a reviewer-accessible copy or summarize the relevant evidence in the PR.

## Change identity

- Change / issue:
- Source scope: `ggml/src/ggml-spacemit/**` / `tools/**` subpaths:
- Paths outside the primary scope, if any:
- Evidence the task cannot be completed without those paths:
- Alternatives considered and why they do not work:
- Why each additional path is the minimum necessary change:
- Confirmation that no retired `ggml/src/ggml-cpu/spacemit/**` backend development was added:
- Scope-to-test rows selected: backend / mtmd / smt-mtmd / server media / cross-layer:
- Base revision:
- Candidate revision / patch identity:
- Affected behavior and expected result:
- Hardware paths in scope: K1 / K3 / RVV / scalar / other:

## Artifact identity

- Compiler and toolchain:
- CMake configuration and relevant environment:
- `GGML_SPACEMIT` / `SPERT_DIR` configuration:
- Runtime package and version:
- Executable, backend library, and dependency hashes:
- Model and input identifiers or hashes:
- Loaded-library / dispatch evidence:

## Plan

- Baseline case and expected failure or reference:
- Candidate cases and acceptance criteria:
- Independent reference and why it is independent:
- End-to-end model correctness cases and expected behavior:
- Model-level performance cases and baseline/candidate comparison:
- Selected build, correctness, performance, and platform checks:
- Excluded checks and reason:

## Results

Use exactly one status for each selected check: `PASS`, `FAIL`, `NOT RUN`, or `N/A`.

| Check or case | Status | Command / input identity | Observed result | Evidence link |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Record counts separately for passed, failed, skipped, unsupported, missing, timed out, and contaminated cases.
An empty selection or all-skipped result is not a pass.

## Performance

For source changes under `ggml/src/ggml-spacemit/**`, complete the model-level
performance comparison even when no improvement is claimed. Use `NOT RUN` with a
reason when the relevant model or environment is unavailable. For documentation,
build-only, or tools-only changes, use `N/A` with a reason when no model
comparison is applicable.

- Case identity (hardware, model/input, shape/context, backend, threads, workers, flags):
- Baseline and candidate artifact identities:
- Timed region and unit:
- Warmup, repetitions, process count, run order, affinity, frequency, and load checks:
- Statistic and spread definition:
- Delta formula and favorable direction:
- Raw samples, exclusions, and model-level result:

| Case | Timed region / unit | Baseline | Candidate | Sample count | Delta / conclusion |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Do not claim a model improvement from an isolated kernel result. Do not report a percentage when the measurements are not comparable.

## Limits

- Untested hardware or dispatch paths:
- Non-independent reference or incomplete output coverage:
- Known baseline failures:
- New failures, regressions, or inconclusive results:
- Evidence that reviewers cannot access:
- Follow-up required:
