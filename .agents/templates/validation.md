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

## Development-map impact

Keep this note concise and update it as the change develops. Use the [development map](../development.md#keep-the-change-located-on-the-development-map); no separate case-definition file or `dry-run` is required.

- Changed responsibilities, entry points/symbols, and source-map/code links:
- Affected callers, downstream consumers, and contracts, including unedited paths:
- Models/features, inputs, and hardware conditions that exercise the change:
- Nearby paths expected to remain unaffected, with code-path reasons:
- Impact-to-check mapping, exclusions, and unresolved coverage:
- Final diff reconciliation and earlier evidence to rerun or justify carrying forward:

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

## Execution records

Repeat this block or link an equivalent existing record for each attempt, including retries. Reference shared metadata above only when it matches the actual run. Follow [execution-record requirements](../acceptance.md#preserve-execution-records).

- Run label, selected check, and affected behavior:
- Start/end times with timezone, working directory, resolved command, and relevant environment:
- Source revision / local patch, artifact, runtime, model/input, and hardware identities:
- Exit status or termination reason:
- Raw stdout/stderr, actual coverage counts, and dispatch evidence:
- Independent reference, criterion/tolerance, and observed result, where applicable:
- Raw performance samples, where applicable:
- Failure, interruption, exclusion, contamination, or retry reason; related run labels:
- Applicability to the final candidate; carried-forward evidence justification or checks to rerun:

## Results

Use exactly one status for each selected check: `PASS`, `FAIL`, `NOT RUN`, or `N/A`, as defined in [acceptance.md](../acceptance.md#preserve-execution-records). An executed but inconclusive check is `FAIL` with an explicit unverified-evidence reason; this does not by itself diagnose a code defect. Keep required checks without results visible and state why they remain unverified.

| Check or case / run label | Status | Command / input identity | Observed result / reason / limits | Evidence link |
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

## Delivery traceability

- Summary claims and corresponding check/run labels and raw evidence locations:
- Final candidate identity and record used to prepare the summary:
- Reviewer-accessible evidence copies or summaries; any verification limits:
- Contributor checks of reported statuses, counts, calculations, and remaining gaps:

## Limits

- Untested hardware or dispatch paths:
- Non-independent reference or incomplete output coverage:
- Known baseline failures:
- New failures, regressions, or inconclusive results:
- Evidence that reviewers cannot access:
- Follow-up required:
