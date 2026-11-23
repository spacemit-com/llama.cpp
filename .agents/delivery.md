# Delivering SpacemiT changes

This guide covers contributions to `ggml/src/ggml-spacemit/**` and the maintained
multimodal integration under `tools/**` in the `spacemit-com/llama.cpp` fork.
It explains how to make a change reviewable;
[acceptance.md](acceptance.md) defines the required verification.
Use the [PR template](templates/pull-request.md) and attach a completed
[validation record](templates/validation.md) where the evidence needs more space.

## Explain the change

Lead with the concrete problem and resulting behavior. Name the trigger: an
operator, dtype, shape, layout, hardware path, or sequence of calls. Give a short
before/after example when it makes the failure or benefit easier to understand.
Explain why the behavior matters and link the issue or reproducible report.

Keep the diff to the minimum necessary change. Identify the affected backend or
multimodal entry points and explain why each changed component is needed. Describe
the final implementation, not the conversation that produced it. Explain any
supporting harness changes separately. If a solution needs source changes outside
`ggml/src/ggml-spacemit/**` and `tools/**`, identify that dependency instead of
presenting it as routine scope. Such changes are strongly discouraged; give the
code-path evidence that the task cannot be completed without each changed path,
the alternatives considered and why they fail, and the minimal change needed.
The retired `ggml/src/ggml-cpu/spacemit/**` backend is not a target for new
backend development or optimization.

Explain the key design choice and its tradeoff: dispatch coverage, numerical
semantics, layout, memory lifetime, synchronization, fallback, or maintenance cost.
Mention an alternative only when rejecting it explains the chosen design.
Avoid unrelated cleanup and speculative extensibility.

## Supply reproducible evidence

Link the bug reproduction and the regression check. State the observed baseline
failure and candidate result, with commands, selected cases, expected behavior,
and logs. If the baseline was not executed, say so; do not infer a before/after
result from a plausible explanation. Reuse an applicable existing check when it
can expose the regression. Identify any missing regression coverage.

Follow the [test-case policy](../CONTRIBUTING.md#test-case-policy): put case construction, inputs, boundaries, reference or expected behavior, commands, and any local test adaptation in the PR description. Keep added or modified test code out of the submitted diff, including temporary changes to existing tests. Identify local reproducer/test patches separately from production code, and distinguish proposed cases from executed results. Permanent backend-specific cases belong in the separate SpacemiT suite once it is available.

For each applicable acceptance requirement, report PASS, FAIL, or NOT RUN;
use N/A only with a reason. Separate build success, execution success, numerical
correctness, and performance. Record actual executed, failed, and skipped cases.
An empty selection, skipped case, or successful process exit alone is not a
numerical pass. Identify the reference, tolerance, and observed error where used.

State the impact classification and required checks. Execution-path changes need proof that the intended path ran and correctness results for affected models or media. Include comparable baseline/candidate model performance when execution cost may change or a benefit is claimed. Mechanical and build-only changes need checks matched to their impact, with a reason for any `N/A` model check. Missing required model or board evidence is `NOT RUN` and remains a merge blocker pending evidence or an explicit maintainer exception.

Record the base and candidate commits, including any uncommitted patch identity,
compiler/build configuration, runtime dependencies, and tested binary/library
identities. Name the actual hardware and backend path exercised. Distinguish
source inspection and host checks from hardware execution. Follow the validation
record for commands, inputs, raw outputs, and the limits of each result.

Separate evidence status from merge disposition using [the acceptance decision rules](acceptance.md#required-evidence-and-merge-decisions). For each unresolved required check, report the reason, affected behavior, substitute evidence and limits, residual risk, and follow-up. Link any explicit maintainer exception and its covered revision/checks; otherwise state that the gap remains unresolved. Neither documenting a gap nor drafting a report grants an exception, and an exception cannot justify an unmeasured performance claim.

## Derive delivery summaries from evidence

Prepare validation summaries and PR text from the recorded runs, using the existing templates. This applies equally to manual summaries, agent drafting, and any report-generation tool; no new runner or report command is required by this guide.

- Map each correctness, coverage, or performance claim to a recorded check, run label, and raw evidence location. Include the source revision and artifact identity to which the claim applies, and reconcile the final changed responsibilities with the development-map impact note.
- Preserve result status, execution counts, failures, exclusions, retries, regressions, and limitations. Required checks that lack evidence remain visible. Do not turn a successful report-generation command, a blank field, or an expected result into a validation pass.
- Keep required/optional applicability and merge disposition separate from result status; carry forward only maintainer exceptions that still cover the candidate and affected checks.
- Compute performance summaries only from identified raw samples under the comparison rules below. Preserve units, sample counts, aggregation and delta definitions, and exclusion reasons. If measurements cannot support a comparison, state that it is inconclusive.
- Separate observed facts from interpretation. Design rationale, necessity of scope, reference independence, and the meaning of results require contributor review; template completion or an agent assertion does not prove them.
- Before delivery, check that the evidence still applies to the final candidate. Update summaries after changed results, replace stale claims, and identify carried-forward evidence and its justification. Keep local raw records in `.agents/workspace/`; provide reviewer-accessible copies or summaries with explicit verification limits, as described below.

Generating a report prepares review material. The contributor must check its claims and own the result under the existing AI-assistance and publication rules.

## Make performance claims comparable

For each comparison, assign a case identity that fixes the hardware, model/input
hash and quantization, operator dimensions or token/context lengths, backend
selection, thread and worker settings, and relevant flags. Reference the exact
baseline/candidate artifacts. Report execution order, warmups, repetitions and
independent processes, plus relevant frequency, affinity and competing-load
observations. Disclose differences or interruptions that affect comparability.

Define the timed region and unit: kernel time, operator time, prefill throughput,
decode at a stated context, or request latency are different measurements.
Provide raw samples and identify the statistic actually computed. If using
`mean +/- spread`, define the spread (for example sample standard deviation),
the sample population and count; a confidence interval needs its level and method.
Do not label a tool's spread without checking its definition. State whether lower
or higher is better and show how any reported delta was calculated.

Report regressions alongside gains. Do not turn an operator result into a model
claim. Without comparable measurements, report performance as unmeasured or
inconclusive; do not invent a percentage, a zero delta, or a no-regression claim.

## Leave a usable review record

List remaining risks, untested paths, and known failures. Distinguish a confirmed
baseline failure from a new failure or an unclassified result, and link supporting
evidence. Explain the practical limit of the conclusion, not just the next task.
For any change outside `ggml/src/ggml-spacemit/**` and the relevant `tools/**`
paths, put the necessity argument, rejected alternatives, exact affected paths,
and path-specific tests in the PR so reviewers can assess why the broader diff is
required.

Put essential conclusions in the PR itself. Attach logs or artifact links that
reviewers can access without chat history, a private checkout, or local credentials.
Use stable commit links for code and durable links for evidence. Redact credentials
and sensitive data before sharing; preserve commands, identities and measurements
needed to assess the result. Summarize any evidence that cannot be shared and state
what reviewers cannot independently verify.

Disclose AI assistance and its scope, including code, tests, analysis or PR drafting.
For this fork's backend contributions, contributors may use AI to prepare reports
and PR text, but must check the resulting claims and evidence themselves. The
contributor remains responsible for every submitted line and must be able to
explain the code, design choices, validation results, and limitations.
