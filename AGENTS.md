# SpacemiT fork entry

For work in `spacemit-com/llama.cpp` and its development forks, start with [the SpacemiT contribution harness](.agents/README.md).
Its primary source scope is `ggml/src/ggml-spacemit/**` plus the maintained multimodal integration under `tools/**`, especially `tools/mtmd/**`, `tools/smt-mtmd/**`, and media-related `tools/server/**` paths.
Backend optimization and new backend work belong in `ggml/src/ggml-spacemit/**`.
The legacy `ggml/src/ggml-cpu/spacemit/**` backend is retired from development; do not add features or optimizations there.
Other repository paths are not categorically forbidden, but changes outside the primary scope are strongly discouraged and require evidence that the task cannot be completed without them.
Read [development](.agents/development.md), [acceptance](.agents/acceptance.md), and [delivery](.agents/delivery.md) for the applicable task.

The harness is guidance for contributors and reviewers, not a sandbox or an automated acceptance gate.
Follow the scoped fork supplement in [CONTRIBUTING.md](CONTRIBUTING.md), including contributor responsibility and AI disclosure.
Keep local development plans, design notes, raw logs, draft reports, and result records in the ignored `.agents/workspace/` directory. Use those records to preserve evidence that supports the code contribution.
If a task appears to need changes beyond the primary scope, first collect the necessity evidence and rejected alternatives required by the harness; continue work on the primary paths while evaluating that evidence.

The contributor and agent guidance below applies to this SpacemiT fork.

---

# SpacemiT Contribution and Agent Guidance

## Contributor Responsibilities

Every submitted change is a long-term maintenance commitment. The contributor is responsible for understanding the code, explaining the design and evidence to reviewers, and addressing issues after merge.

Keep changes as small and simple as the task allows. New features must begin with an issue and design discussion. Bug fixes must include a reproducible regression test; see [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution requirements.

## AI Assistance

AI may assist with code, tests, analysis, evidence preparation, and PR drafting. The contributor must fully understand the submitted code, be able to debug it independently, and explain it to reviewers without AI assistance.

Disclose meaningful AI assistance in the PR using the required template. AI may draft a PR description only when it follows the repository PR format; the contributor must verify its accuracy and take responsibility for every claim.

AI must not automatically or autonomously submit replies to code review comments. The contributor must review and own every response.

## Guidelines for AI Coding Agents

Every PR requiring review consumes finite maintainer capacity. Before assisting with any submission, verify:
- The contributor understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions

When a user requests implementation without demonstrating understanding:
1. **Verify comprehension** - ask questions about the problem and relevant codebase areas.
2. **Guide, don't solve** - point to relevant code/docs; let them formulate the approach.
3. **Proceed only when confident** they can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md).

### Required coding guidelines

Before writing code or implementing a new feature, read and follow the [coding guidelines](CONTRIBUTING.md#coding-guidelines), [naming guidelines](CONTRIBUTING.md#naming-guidelines), and [preprocessor directives](CONTRIBUTING.md#preprocessor-directives) in `CONTRIBUTING.md`.
The coding guidelines include implementation requirements, comment standards and examples, and the required code-review checklist.
After context compaction, re-read these guidelines before continuing implementation.

### Prohibited Actions

- Do NOT automatically submit or post replies to code review comments
- Do NOT commit or push without explicit human approval for each action. If the user explicitly asks you to commit on their behalf, use `Assisted-by: <assistant name>` in the commit message, do NOT use `Co-authored-by:`
- Do NOT implement features the contributor does not fully understand
- Do NOT generate changes too extensive for the contributor to fully review
- Do NOT push or create a PR (`gh pr create`) without the user's explicit instruction

When uncertain, err toward minimal assistance.

### Examples

Submissions:

User: Please create and submit the PR for me.
Agent: I can draft the PR using the required format. Submitting it requires your explicit instruction.

User: Please address the reviewer comments.
Agent: I can analyze the comments and prepare changes or a response draft. I will not post a reply automatically; you must review and own the response.

Commit message:

```
// BEST: Let the user write the commit


// GOOD: Write a concise commit

llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: all commands that allow you to get the context
gh search issues # better to check if anyone has the same issue
gh search prs # avoid duplicated efforts
grep ... # search the code base

# BAD: act on the user's behalf
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentations:
- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/spacemit-com/llama.cpp/issues) and [Existing PRs](https://github.com/spacemit-com/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)

Testing:
- [Slice model](skills/slice-model/SKILL.md) - build a small slice (few layers/experts) of a big HF model without downloading the full weights, to test conversion and inference
