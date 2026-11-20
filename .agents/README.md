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

## Rule ownership

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution policy and coding/comment standards, [AGENTS.md](../AGENTS.md) for agent behavior and authorization, [development.md](development.md) for approach and implementation principles, [acceptance.md](acceptance.md) for validation requirements, and [delivery.md](delivery.md) for presenting evidence. Templates and review skills apply these rules; they must not add conflicting requirements from upstream workflows. Update the owning document and its affected references together when a policy changes.

## Use the documents

1. Read [development.md](development.md) before editing and map the change to the source responsibilities, affected execution paths, and model or feature behavior.
2. Use [acceptance.md](acceptance.md) to select evidence for that impact, then record the plan in [templates/validation.md](templates/validation.md).
3. Run the selected checks with existing tools and preserve each attempt's identity, raw output, and actual outcome in the validation record.
4. Use [delivery.md](delivery.md) and [templates/pull-request.md](templates/pull-request.md) to derive review summaries from those records, checking every claim against its evidence.

The source maps in `development.md` are the development map for this workflow. Keep a concise impact note in the existing validation record; a separate case-definition system or `dry-run` stage is not required.

The documents describe the evidence expected from a change; they do not register agent roles, prescribe a delegation API, or replace the project's build and CI systems.
Use the planning, delegation, and review capabilities of the agent tool available in the current environment.

## Evidence principles

- Separate build, execution, numerical correctness, performance, and platform coverage.
- Identify the exact source, binary, library, model, input, and runtime used for each claim.
- Prefer an independent reference and preserve the baseline failure for a bug fix.
- Report skipped, unavailable, contaminated, inconclusive, and known failing cases explicitly.
- Keep raw commands and results in `.agents/workspace/`; put conclusions and reviewer-accessible links in the PR.

The harness is guidance, not an automated acceptance gate. A green command or an agent report does not replace a human review of the code and evidence.
