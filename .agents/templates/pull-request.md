<!-- For ggml/src/ggml-spacemit or maintained tools/** contributions to spacemit-com/llama.cpp.
Follow ../delivery.md and ../acceptance.md; use validation.md for detailed evidence.
Replace placeholders; use NOT RUN or N/A with a reason instead of empty claims.
Use reviewer-accessible links, not private workspace paths. -->

## Problem and behavior

- Issue / reproduction: <link and exact trigger>
- Before -> after: <observed behavior and why it matters>

## Change and design

- Scope: <backend/tools paths, entry points, and why each change is needed>
- Additional paths outside ggml/src/ggml-spacemit and relevant tools paths: <none, or exact paths and code-path evidence that the task cannot be completed without them>
- Alternatives to those additional changes and evidence they are insufficient: <details or N/A>
- Approach and key tradeoff: <numerics/layout/lifetime/dispatch/fallback as applicable>
- Retired ggml/src/ggml-cpu/spacemit backend development/optimization: none
- Supporting harness changes or external dependencies: <details or none>

## Validation

- Base / candidate: <commit IDs; uncommitted patch identity if applicable>
- Environment / artifacts: <hardware, compiler, build flags, runtime, binary/library identities>
- Regression check: <case/command link; observed baseline failure and candidate result>
- End-to-end model correctness: <relevant model/input, expected behavior, candidate result; or NOT RUN with reason>
- Model-level performance: <baseline/candidate comparison for the relevant model path; or NOT RUN with reason>
- Detailed validation record and logs: <reviewer-accessible links>

| Acceptance check / selected cases | Status | Observed result and evidence |
| --- | --- | --- |
| <build / execution / correctness; separate rows> | <PASS / FAIL / NOT RUN / N/A> | <counts, errors/tolerances/reference, log link; reason if not run or N/A> |

## Performance

For source changes under `ggml/src/ggml-spacemit/**`, complete a model-level
comparison even when no improvement is claimed. Use NOT RUN with a reason when
the relevant model or environment is unavailable. For documentation, build-only,
or tools-only changes, use N/A with a reason when no model comparison applies.

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

## AI assistance and contributor responsibility

- AI assistance: <none, or tools and scope: code/tests/analysis/report/PR drafting>
- Contributor verification: <what was personally reviewed and how evidence was checked>
- I take responsibility for the submitted code and claims and can explain their design, validation, and limitations.
