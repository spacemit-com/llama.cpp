## Overview

<!-- For ggml/src/ggml-spacemit or maintained tools/** contributions, use .agents/templates/pull-request.md as the detailed checklist. -->

- Problem and trigger:
- Resulting behavior:
- Related issue or discussion:

## Scope and design

- Changed paths and why each is necessary:
- Development-map responsibilities, affected callers/models/features, and corresponding checks:
- Paths outside ggml/src/ggml-spacemit and relevant tools paths, if any:
- Code-path evidence they are necessary and alternatives considered:
- No new development or optimization in the retired ggml/src/ggml-cpu/spacemit backend:
- Minimum necessary change and key tradeoffs:
- Impact classification: execution-path / mechanical / documentation / build-only, with justification:
- No added/modified test code in this implementation PR; temporary local cases excluded from the submitted diff:

## Validation

- Base / candidate revisions and artifact identity:
- Build and configuration:
- Case construction and local reproduction: inputs/boundaries, expected behavior or reference, command or local test adaptation, and baseline/candidate results:
- Required checks and any impact-based exclusions or additional coverage:
- Correctness evidence and independent reference:
- Actual backend/path execution and affected model/media correctness for execution-path changes (NOT RUN if missing, or justified N/A):
- Comparable baseline/candidate performance when execution cost may change or a benefit is claimed (NOT RUN if required but unavailable, otherwise justified N/A):
- Detailed validation record and reviewer-accessible logs, with check/run labels for each claim:
- Evidence applicability to the final candidate, including reruns, carried-forward evidence, and gaps:

<!-- Apply .agents/acceptance.md status and merge-decision rules. Keep failures, retries, and inconclusive results visible; NOT RUN is not permission to merge. Follow CONTRIBUTING.md#test-case-policy: document cases and local reproduction rather than submitting test sources. -->

| Check | Applicability and reason | Status (PASS / FAIL / NOT RUN / N/A) | Result and evidence |
| --- | --- | --- | --- |
| Build |  |  |  |
| Correctness |  |  |  |
| Performance |  |  |  |
| Platform coverage |  |  |  |

## Risks and limits

- Untested paths, known failures, and inconclusive results:
- Evidence unavailable to reviewers:
- Follow-up work:
- Unresolved required checks and merge disposition (blocked pending evidence or explicit maintainer exception):
- Maintainer exception, if any: decision link, covered checks/revision, accepted risk, and follow-up owner/action:

## AI assistance and responsibility

- AI assistance and scope (code, tests, analysis, report, PR text):
- Contributor review of code and evidence:
- I understand and take responsibility for the submitted code and claims.
