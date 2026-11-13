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

For each applicable acceptance requirement, report PASS, FAIL, or NOT RUN;
use N/A only with a reason. Separate build success, execution success, numerical
correctness, and performance. Record actual executed, failed, and skipped cases.
An empty selection, skipped case, or successful process exit alone is not a
numerical pass. Identify the reference, tolerance, and observed error where used.

For source changes under `ggml/src/ggml-spacemit/**`, include end-to-end
correctness results for the relevant model or models and model-level performance
results against the baseline. Operator or kernel results are supporting evidence
and do not replace these model checks. If a model check cannot run, record `NOT
RUN`, the reason, and the limit on the conclusion.

Record the base and candidate commits, including any uncommitted patch identity,
compiler/build configuration, runtime dependencies, and tested binary/library
identities. Name the actual hardware and backend path exercised. Distinguish
source inspection and host checks from hardware execution. Follow the validation
record for commands, inputs, raw outputs, and the limits of each result.

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
