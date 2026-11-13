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

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

Common mistakes that AI agents usually make:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.

Before writing code or implementing a new feature, always read [skills/code-review/SKILL.md](skills/code-review/SKILL.md). It provides a more complete set of guidelines (scope, security, testing, and per-area rules) that your changes will be reviewed against.

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

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

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
