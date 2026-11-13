# SpacemiT contribution harness

This harness is a review aid for the `spacemit-com/llama.cpp` fork.
It is intentionally organized around three questions: how to make a scoped change, how to prove the result, and how to present it so a maintainer can review it without private chat history.

## Scope

The primary source scope for this first version is `ggml/src/ggml-spacemit/**` and the maintained multimodal integration under `tools/**`.
The common paths are `tools/mtmd/**` for the multimodal core, `tools/smt-mtmd/**` for SpacemiT media wrappers, and media-related `tools/server/**` code.
Other `tools/**` paths are allowed when the change is part of the same maintained feature, but the PR must narrow the changed paths and explain their connection.
The similarly named `ggml/src/ggml-cpu/spacemit/**` path is a retired legacy backend, not an alternative implementation target.
Do not add features or optimizations there; all SpacemiT backend optimization work belongs in `ggml/src/ggml-spacemit/**`.
Existing build systems, test tools, model runners, and board procedures outside the primary source paths may be read or run as evidence.
Changes outside `ggml/src/ggml-spacemit/**` and the relevant `tools/**` paths are not categorically forbidden, but are strongly discouraged.
Before making such a change, show why the task cannot be completed without it, which alternatives were considered, and the specific evidence that rules those alternatives out.
Name the exact out-of-scope paths, explain why each one is necessary, and include their effects and tests in the delivery record and PR.
This general exception does not apply to the retired CPU backend; changing its development status requires a separate explicit policy decision.

## Use the documents

1. Read [development.md](development.md) before editing.
2. Fill [templates/validation.md](templates/validation.md) before running the selected checks.
3. Use [acceptance.md](acceptance.md) to select evidence for the affected behavior.
4. Use [delivery.md](delivery.md) and [templates/pull-request.md](templates/pull-request.md) to prepare the review.

The documents describe the evidence expected from a change; they do not register agent roles, prescribe a delegation API, or replace the project's build and CI systems.
Use the planning, delegation, and review capabilities of the agent tool available in the current environment.

## Evidence principles

- Separate build, execution, numerical correctness, performance, and platform coverage.
- Identify the exact source, binary, library, model, input, and runtime used for each claim.
- Prefer an independent reference and preserve the baseline failure for a bug fix.
- Report skipped, unavailable, contaminated, inconclusive, and known failing cases explicitly.
- Keep raw commands and results in `.agents/workspace/`; put conclusions and reviewer-accessible links in the PR.

The harness is guidance, not an automated acceptance gate. A green command or an agent report does not replace a human review of the code and evidence.
