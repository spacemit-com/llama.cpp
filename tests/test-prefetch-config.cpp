// Verify the CLI/environment -> context-parameter configuration path without a model.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static void check(std::initializer_list<const char *> args, bool expected, bool valid = true) {
    common_params params;
    std::vector<std::string> storage{"test-prefetch-config", "-m", "unused.gguf"};
    for (const char * arg : args) {
        storage.emplace_back(arg);
    }
    std::vector<char *> argv;
    for (auto & arg : storage) {
        argv.push_back(&arg[0]);
    }
    CHECK(common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON) == valid);
    if (valid) {
        CHECK(params.prefetch == expected);
        CHECK(common_context_params_to_llama(params).prefetch == expected);
    }
}

int main() {
    const auto saved = common_get_env("LLAMA_ARG_PREFETCH");
    const auto saved_neg = common_get_env("LLAMA_ARG_NO_PREFETCH");
    common_set_env("LLAMA_ARG_PREFETCH", "");
    common_set_env("LLAMA_ARG_NO_PREFETCH", "");
    CHECK(!llama_context_default_params().prefetch);
    check({}, false);
    check({"--prefetch"}, true);
    check({"--no-prefetch"}, false);
    check({"--prefetch", "--no-prefetch"}, false);
    check({"--no-prefetch", "--prefetch"}, true);
    common_set_env("LLAMA_ARG_PREFETCH", "1");
    check({}, true);
    check({"--no-prefetch"}, false);
    common_set_env("LLAMA_ARG_PREFETCH", "0");
    check({}, false);
    check({"--prefetch"}, true);
    common_set_env("LLAMA_ARG_PREFETCH", "invalid");
    check({}, false, false);
    common_set_env("LLAMA_ARG_PREFETCH", saved);
    common_set_env("LLAMA_ARG_NO_PREFETCH", saved_neg);
    std::puts("PASS: prefetch defaults, CLI, environment, overrides, invalid input, context propagation");
}
