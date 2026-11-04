#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct workspace_case {
    const char * name;
    int64_t m;
    int64_t n;
    int64_t k;
    int workers;
    ggml_type type;
};

static const workspace_case cases[] = {
    {"cap",          4096, 14336,  4096, 8, GGML_TYPE_Q4_0},
    {"tcm",             2,  4096, 11008, 1, GGML_TYPE_Q4_0},
    {"cluster-zero",    2,  4096, 11008, 4, GGML_TYPE_Q4_0},
    {"prime",          17,    96, 11104, 3, GGML_TYPE_Q4_0},
    {"single-slice",16385, 16384,  2048, 8, GGML_TYPE_Q4_0},
    {"affine-tcm",      2,  4096, 11008, 1, GGML_TYPE_Q4_1},
    {"q4k-tcm",         2,  4096, 11008, 1, GGML_TYPE_Q4_K},
};

static void put_half(uint8_t * dst, float value) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(value);
    memcpy(dst, &h, sizeof(h));
}

static float activation(size_t pattern, size_t k) {
    return static_cast<float>(static_cast<int>((3 * k + 7 * pattern + 5 * (k / 32)) % 17) - 8) / 16.0f;
}

static void make_weight_row(ggml_type type, size_t pattern, size_t k, uint8_t * dst) {
    const size_t block_k = static_cast<size_t>(ggml_blck_size(type));
    const size_t block_bytes = ggml_type_size(type);
    for (size_t b = 0; b < k / block_k; ++b) {
        uint8_t * block = dst + b * block_bytes;
        const float scale = static_cast<float>(1 + (pattern + b) % 3) / 32.0f;
        put_half(block, scale);
        if (type == GGML_TYPE_Q4_K) {
            put_half(block + 2, scale);
            for (size_t j = 0; j < 4; ++j) {
                const uint8_t sc0 = static_cast<uint8_t>(1 + (pattern + j + b) % 3);
                const uint8_t sc1 = static_cast<uint8_t>(1 + (pattern + j + b + 1) % 3);
                const uint8_t zp0 = static_cast<uint8_t>(2 + (pattern + j + b) % 5);
                const uint8_t zp1 = static_cast<uint8_t>(2 + (pattern + j + b + 2) % 5);
                const uint8_t min1 = sc1 * zp1;
                block[4 + j] = sc0;
                block[8 + j] = static_cast<uint8_t>((sc0 * zp0) | ((min1 >> 4) << 6));
                block[12 + j] = static_cast<uint8_t>(sc1 | ((min1 & 15) << 4));
            }
            for (size_t j = 0; j < 128; ++j) {
                block[16 + j] = static_cast<uint8_t>(((3 * j + pattern + b) % 16) |
                                                   (((5 * j + 2 * pattern + 3 * b) % 16) << 4));
            }
        } else {
            const size_t offset = type == GGML_TYPE_Q4_1 ? 4 : 2;
            if (type == GGML_TYPE_Q4_1) {
                put_half(block + 2, -scale * static_cast<float>(2 + (pattern + b) % 9));
            }
            for (size_t j = 0; j < 16; ++j) {
                block[offset + j] = static_cast<uint8_t>(((3 * j + pattern + b) % 16) |
                                                       (((5 * j + 2 * pattern + 3 * b) % 16) << 4));
            }
        }
    }
}

static bool run_case(const workspace_case & test, ggml_backend_t backend) {
    constexpr size_t a_patterns = 5;
    constexpr size_t b_patterns = 7;
    const size_t m = static_cast<size_t>(test.m);
    const size_t n = static_cast<size_t>(test.n);
    const size_t k = static_cast<size_t>(test.k);
    ggml_context * ctx = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
    GGML_ASSERT(ctx);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, test.type, test.k, test.n);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.k, test.m);
    ggml_tensor * c = ggml_mul_mat(ctx, b, a);
    ggml_set_name(b, "workspace.weight");
    ggml_set_name(a, "workspace.input");
    ggml_set_name(c, "workspace.output");
    ggml_set_input(a);
    ggml_set_output(c);
    GGML_ASSERT(ggml_backend_supports_op(backend, c));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer && b->extra && !a->extra && !c->extra);

    const size_t row_bytes = ggml_row_size(test.type, test.k);
    std::vector<uint8_t> row_templates(b_patterns * row_bytes);
    std::vector<float> decoded(b_patterns * k);
    const ggml_type_traits * traits = ggml_get_type_traits(test.type);
    for (size_t p = 0; p < b_patterns; ++p) {
        make_weight_row(test.type, p, k, row_templates.data() + p * row_bytes);
        traits->to_float(row_templates.data() + p * row_bytes, decoded.data() + p * k, test.k);
    }
    double reference[a_patterns][b_patterns] = {};
    for (size_t ap = 0; ap < a_patterns; ++ap) {
        for (size_t bp = 0; bp < b_patterns; ++bp) {
            for (size_t j = 0; j < k; ++j) {
                reference[ap][bp] += static_cast<double>(activation(ap, j)) * decoded[bp * k + j];
            }
        }
    }
    {
        std::vector<uint8_t> weights(ggml_nbytes(b));
        for (size_t row = 0; row < n; ++row) {
            memcpy(weights.data() + row * row_bytes, row_templates.data() + (row % b_patterns) * row_bytes, row_bytes);
        }
        ggml_backend_tensor_set(b, weights.data(), 0, weights.size());
    }
    constexpr size_t batch_rows = 32;
    std::vector<float> inputs(batch_rows * k);
    for (size_t row = 0; row < m; row += batch_rows) {
        const size_t rows = std::min(batch_rows, m - row);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < k; ++j) {
                inputs[i * k + j] = activation((row + i) % a_patterns, j);
            }
        }
        ggml_backend_tensor_set(a, inputs.data(), row * k * sizeof(float), rows * k * sizeof(float));
    }
    ggml_backend_tensor_memset(c, 0xff, 0, ggml_nbytes(c));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, c);
    printf("RUN %s M=%zu N=%zu K=%zu workers=%d type=%s output_bytes=%zu\n",
           test.name, m, n, k, test.workers, ggml_type_name(test.type), ggml_nbytes(c));
    fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    std::vector<float> output(batch_rows * n);
    size_t errors = 0;
    double max_error = 0;
    for (size_t row = 0; row < m; row += batch_rows) {
        const size_t rows = std::min(batch_rows, m - row);
        ggml_backend_tensor_get(c, output.data(), row * n * sizeof(float), rows * n * sizeof(float));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const double expected = reference[(row + i) % a_patterns][j % b_patterns];
                const double actual = output[i * n + j];
                const double error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual) || error > 0.02 + 0.002 * std::abs(expected)) {
                    if (errors++ < 5) {
                        fprintf(stderr, "m=%zu n=%zu expected=%.9g actual=%.9g\n", row + i, j, expected, actual);
                    }
                }
            }
        }
    }
    printf("%s %s checked=%zu errors=%zu max_abs_error=%.9g compute_seconds=%.6f\n",
           errors == 0 ? "PASS" : "FAIL", test.name, m * n, errors, max_error, seconds);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return errors == 0;
}

int main(int argc, char ** argv) {
    const workspace_case * test = nullptr;
    if (argc == 2) {
        for (const auto & candidate : cases) {
            if (strcmp(argv[1], candidate.name) == 0) {
                test = &candidate;
            }
        }
    }
    if (!test) {
        fprintf(stderr, "usage: env GGML_SPACEMIT_WORKERS=N %s CASE\n", argv[0]);
        for (const auto & candidate : cases) {
            fprintf(stderr, "  %s: M=%lld N=%lld K=%lld workers=%d %s\n", candidate.name,
                    static_cast<long long>(candidate.m), static_cast<long long>(candidate.n),
                    static_cast<long long>(candidate.k), candidate.workers, ggml_type_name(candidate.type));
        }
        return 2;
    }
    const char * workers = getenv("GGML_SPACEMIT_WORKERS");
    char * end = nullptr;
    const long worker_count = workers ? strtol(workers, &end, 10) : 0;
    if (!workers || end == workers || *end != '\0' || worker_count != test->workers) {
        fprintf(stderr, "%s requires GGML_SPACEMIT_WORKERS=%d before process startup (got %s).\n"
                        "usage: env GGML_SPACEMIT_WORKERS=%d %s %s\n",
                test->name, test->workers, workers ? workers : "unset", test->workers, argv[0], test->name);
        return 2;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("SPACEMIT0");
    if (!device) {
        fprintf(stderr, "SKIP: SPACEMIT0 unavailable\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    GGML_ASSERT(backend);
    const bool passed = run_case(*test, backend);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
