# SpacemiT contributions

These guidelines apply to contributions in `spacemit-com/llama.cpp` and its development forks. Read the [contribution harness](.agents/README.md) and the relevant development, acceptance, and delivery guides alongside this document.

## Scope

The primary source paths are `ggml/src/ggml-spacemit/**` and the maintained multimodal integration in `tools/mtmd/**`, `tools/smt-mtmd/**`, and media-related `tools/server/**` paths. Other `tools/**` paths may be changed when they are part of the same feature; narrow and explain those paths in the PR.

Changes elsewhere in the repository are possible but strongly discouraged. For every such path, the PR must explain why the task cannot be completed without it, which alternatives were considered and why they fail, why the exact change is necessary, and how that path was validated.

The legacy `ggml/src/ggml-cpu/spacemit/**` backend is retired. Do not add backend features or optimizations there; all SpacemiT backend optimization belongs in `ggml/src/ggml-spacemit/**`.

## Before implementation

- Search existing issues and pull requests to avoid duplicate work.
- New features must begin with an issue describing the problem and proposed direction. Discuss the scope with maintainers before implementing the feature.
- Bug fixes must identify a reproducible failure and include a regression test.
- Keep each pull request focused on one feature or fix. Do not combine unrelated cleanup with the required change.
- New CLI or public API additions need a clear justification for why existing mechanisms are insufficient.

## Validation

- Select build and test requirements from the affected paths and behavior using [the acceptance matrix](.agents/acceptance.md). Full repository CI and unrelated backends are not required for every change.
- Run and report the relevant build, correctness, dispatch, platform, and performance checks. A performance claim requires comparable measurements and artifact identity.
- For source changes under `ggml/src/ggml-spacemit/**`, run end-to-end correctness and model-level performance checks on the relevant model or models that exercise the changed path. Operator or kernel tests alone are insufficient; record `NOT RUN` with the reason if the model check is unavailable.
- Record unavailable, skipped, failing, or inconclusive validation explicitly. Do not present missing evidence as a pass.
- Bug-fix regression tests must demonstrate failure before the change and pass after it.

## AI assistance and contributor responsibility

AI may assist with implementation, analysis, testing, evidence preparation, and PR drafting. Contributors must fully understand the submitted code, be able to debug it independently, explain it to reviewers without AI assistance, and take responsibility for its maintenance.

AI may draft a PR description when it follows the required [GitHub PR format](.github/pull_request_template.md) and the detailed [SpacemiT PR template](.agents/templates/pull-request.md). The contributor must review the text and verify every factual claim before submission. Disclose meaningful AI assistance in the PR template.

AI must not automatically submit replies to code review comments. Contributors own and review their responses. An agent may not publish messages, create a PR, push, or merge without the user's explicit instruction.

## Preparing and maintaining a PR

- Explain the problem and trigger, the minimum necessary change, the affected paths, the selected validation, and any remaining limitations.
- Link reviewer-accessible evidence using the validation record and PR templates in `.agents/templates/`.
- Respond to review feedback, update the implementation or evidence when needed, and remain available to maintain the contribution after merge.
- Maintainers decide whether a change fits the architecture and has sufficient evidence; required CI checks and review approvals must pass before merge.

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggml-org/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

# Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

# Code maintenance

- Existing code should have designated collaborators and/or maintainers specified in the [CODEOWNERS](CODEOWNERS) file responsible for:
  - Reviewing and merging related PRs
  - Fixing related bugs
  - Providing developer guidance/support

- When adding or modifying a large piece of code:
  - If you are a collaborator, make sure to add yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for reviewing related PRs
  - If you are a contributor, find an existing collaborator who is willing to review and maintain your code long-term
  - Provide the necessary CI workflow (and hardware) to test your changes (see [ci/README.md](https://github.com/ggml-org/llama.cpp/tree/master/ci))

- New code should follow the guidelines (coding, naming, etc.) outlined in this document. Exceptions are allowed in isolated, backend-specific parts of the code that do not interface directly with the `ggml` interfaces.
  _(NOTE: for legacy reasons, existing code is not required to follow this guideline)_

- For changes in server, please make sure to refer to the [server development documentation](./tools/server/README-dev.md)

# Documentation

- Documentation is a community effort
- When you need to look into the source code to figure out how to use an API consider adding a short summary to the header file for future reference
- When you notice incorrect or outdated documentation, please update it

# Resources

The Github issues, PRs and discussions contain a lot of information that can be useful to get familiar with the codebase. For convenience, some of the more important information is referenced from Github projects:

https://github.com/ggml-org/llama.cpp/projects
