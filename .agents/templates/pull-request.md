<!-- For ggml/src/ggml-spacemit or maintained tools/** contributions to spacemit-com/llama.cpp.
Follow ../delivery.md and ../acceptance.md; use validation.md for detailed evidence.
Replace placeholders; use NOT RUN or N/A with a reason instead of empty claims.
Derive results from execution records; retain missing, inconclusive, failed, and superseded evidence with explanations.
Use reviewer-accessible links, not private workspace paths. -->

## Problem and behavior

- Issue / reproduction: <link and exact trigger>
- Before -> after: <observed behavior and why it matters>

## Change and design

- Scope: <backend/tools paths, entry points, and why each change is needed>
- Development-map impact: <changed responsibilities, affected callers/models/features including unedited paths, and corresponding checks; exclusions with reasons>
- Additional paths outside ggml/src/ggml-spacemit and relevant tools paths: <none, or exact paths and code-path evidence that the task cannot be completed without them>
- Alternatives to those additional changes and evidence they are insufficient: <details or N/A>
- Approach and key tradeoff: <numerics/layout/lifetime/dispatch/fallback as applicable>
- Retired ggml/src/ggml-cpu/spacemit backend development/optimization: none
- Supporting harness changes or external dependencies: <details or none>
- Impact classification: <execution-path / mechanical / documentation / build-only; justification and any runtime effects of build changes>
- Submitted test-code changes: none; temporary local cases are excluded from the commit and PR diff

## Validation

- Base / candidate: <commit IDs; uncommitted patch identity if applicable>
- Environment / artifacts: <hardware, compiler, build flags, runtime, binary/library identities>
- Case construction and reproduction: <operation/feature, inputs, shape/dtype/layout boundaries, reference or expected behavior, command and any minimal local test adaptation; no committed test sources>
- Regression check: <observed baseline failure and candidate result; distinguish proposed from executed cases and identify any local test patch separately>
- Required checks and exclusions: <map to actual impact; justify additional/optional checks and N/A>
- Path execution and end-to-end model correctness: <required for execution-path changes; dispatch proof, relevant model/input, expected behavior and result; NOT RUN with reason if missing, or justified N/A>
- Model-level performance: <required when execution cost may change or a benefit is claimed; comparable baseline/candidate result, NOT RUN with reason, or justified N/A>
- Detailed validation record and logs: <reviewer-accessible links with check/run labels for each claim>
- Evidence applicability to the final candidate: <rerun checks, justified carried-forward evidence, and remaining gaps>

| Acceptance check / selected cases | Applicability | Status | Observed result and evidence |
| --- | --- | --- | --- |
| <build / execution / correctness; separate rows> | <required / additional / not applicable; reason> | <PASS / FAIL / NOT RUN / N/A> | <run labels, counts, errors/tolerances/reference, raw evidence link; failure/unverified reason, or reason if NOT RUN or N/A> |

## Performance

Complete a comparable model-level performance check when execution cost may change or a performance benefit is claimed. Use `NOT RUN` when a required comparison cannot run, and `N/A` with an impact-based reason when none applies. An unmeasured improvement must not be claimed.

- Case identity: <hardware; input/model hash + quantization; shape/tokens/context; backend; threads/workers/flags>
- Protocol: <artifact IDs; run order; warmups; repeats; independent processes; frequency/affinity/load observations>
- Timing and statistics: <timed region; unit; statistic; population/count; exact meaning of +/- or interval>
- Delta definition: <formula and favorable direction; omit delta if measurements are not comparable>
- Raw samples and analysis: <links; interruptions/exclusions and reasons>

| Case identity | Timed region / unit | Baseline statistic (+/- defined spread) | Candidate statistic (+/- defined spread) | Sample count (base / candidate) | Delta, if comparable |
| --- | --- | --- | --- | --- | --- |
| <case ID> | <boundary; unit> | <value> | <value> | <n / n> | <value or inconclusive> |

## Risks and limits

- Untested paths / remaining risks: <details and impact>
- Known failures: <baseline / new / unclassified; evidence links>
- Evidence unavailable to reviewers: <what and how it limits the conclusion, or none>
- Unresolved required checks and merge disposition: <none, or blocked pending evidence / explicit maintainer exception; NOT RUN alone does not allow merge>
- Maintainer exception, if any: <decision link, covered checks/revision, accepted risk, follow-up owner/action; otherwise pending, not self-approved>

## AI assistance and contributor responsibility

- AI assistance: <none, or tools and scope: code/tests/analysis/report/PR drafting>
- Contributor verification: <what was personally reviewed and how evidence was checked>
- I take responsibility for the submitted code and claims and can explain their design, validation, and limitations.
