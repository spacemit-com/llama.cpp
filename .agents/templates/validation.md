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
- Impact classification and justification: execution-path / mechanical / documentation / build-only; runtime effects of build changes:
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
- Case construction: operation/feature, inputs, shape/dtype/layout boundaries, seed and expected behavior:
- Existing test command or minimal local reproducer/adaptation; local test-patch identity separate from production changes:
- Confirmation that added/modified test code is excluded from the commit and PR diff:
- Independent reference and why it is independent:
- Intended backend/dispatch path and how actual execution will be demonstrated:
- End-to-end model correctness cases and expected behavior:
- Model-level performance cases and baseline/candidate comparison:
- Selected build, correctness, performance, and platform checks; required versus additional coverage:
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

| Check or case / run label | Applicability and reason | Status | Command / input identity | Observed result / reason / limits | Evidence link |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Record counts separately for passed, failed, skipped, unsupported, missing, timed out, and contaminated cases.
An empty selection or all-skipped result is not a pass.

## Performance

Complete a comparable model-level performance check when execution cost may change or a performance benefit is claimed. Use `NOT RUN` with a reason when a required comparison cannot run, and `N/A` with an impact-based reason when none applies. Missing hardware or models does not make a required check inapplicable.

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

## Merge disposition

Apply [required evidence and merge decisions](../acceptance.md#required-evidence-and-merge-decisions). An unresolved required check blocks merge pending evidence or an explicit maintainer exception; its result status does not grant an exception.

- Required checks not satisfied, reason, and affected behavior:
- Available substitute evidence, its limits, and residual risk:
- Disposition: satisfied / blocked pending evidence / maintainer exception accepted:
- Maintainer decision link, checks and candidate revision covered, or pending if absent:
- Follow-up owner/action and whether a later source change invalidates the exception:

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
