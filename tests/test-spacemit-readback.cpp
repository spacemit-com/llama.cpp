#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

struct tensor_buffer {
    ggml_context * ctx;
    ggml_tensor * tensor;
    ggml_backend_buffer_t buffer;

    tensor_buffer(ggml_backend_buffer_type_t buft, ggml_type type, int64_t k, int64_t n, const char * name) {
        ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        GGML_ASSERT(ctx);
        tensor = ggml_new_tensor_2d(ctx, type, k, n);
        ggml_set_name(tensor, name);
        GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, tensor) >= ggml_nbytes(tensor));
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buffer);
        ggml_backend_buffer_clear(buffer, 0xa5);
    }

    ~tensor_buffer() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static bool check_readback(ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
    const size_t size = expected.size();
    std::vector<uint8_t> actual(size);
    ggml_backend_tensor_get(tensor, actual.data(), 0, size);
    const bool whole_ok = actual == expected;
    if (!whole_ok) {
        fprintf(stderr, "%s: whole-tensor read mismatch\n", tensor->name);
    }
    bool unaligned_ok = true;
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<uint8_t> unaligned(size + 2, 0x5a);
        ggml_backend_tensor_get(tensor, unaligned.data() + 1, 0, size);
        if (unaligned.front() != 0x5a || unaligned.back() != 0x5a ||
            memcmp(unaligned.data() + 1, expected.data(), size) != 0) {
            fprintf(stderr, "%s: unaligned read mismatch\n", tensor->name);
            unaligned_ok = false;
        }
    }

    const std::pair<size_t, size_t> ranges[] = {
        {0, 1}, {1, 3}, {19, 37}, {size / 2 - 7, 29}, {size - 257, 257}, {size - 1, 1},
    };
    size_t partial_passed = 0;
    for (const auto & range : ranges) {
        std::vector<uint8_t> part(range.second + 2, 0x5a);
        ggml_backend_tensor_get(tensor, part.data() + 1, range.first, range.second);
        if (part.front() != 0x5a || part.back() != 0x5a ||
            memcmp(part.data() + 1, expected.data() + range.first, range.second) != 0) {
            fprintf(stderr, "%s: partial read mismatch at %zu + %zu\n", tensor->name, range.first, range.second);
        } else {
            ++partial_passed;
        }
    }

    std::vector<uint8_t> strided(3 * 23, 0x5a);
    ggml_backend_tensor_get_2d(tensor, strided.data(), 13, 17, 3, 41, 23);
    size_t strided_passed = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (memcmp(strided.data() + i * 23, expected.data() + 13 + i * 41, 17) != 0 ||
            !std::all_of(strided.begin() + i * 23 + 17, strided.begin() + (i + 1) * 23,
                         [](uint8_t x) { return x == 0x5a; })) {
            fprintf(stderr, "%s: strided read mismatch\n", tensor->name);
        } else {
            ++strided_passed;
        }
    }

    tensor_buffer cpu(ggml_backend_cpu_buffer_type(), tensor->type, tensor->ne[0], tensor->ne[1], "cpu-copy");
    ggml_backend_tensor_copy(tensor, cpu.tensor);
    ggml_backend_tensor_get(cpu.tensor, actual.data(), 0, size);
    const bool copy_ok = actual == expected;
    if (!copy_ok) {
        fprintf(stderr, "%s: CPU copy mismatch\n", tensor->name);
    }
    printf("%s [K=%lld N=%lld]: whole=%s unaligned=%s partial=%zu/6 strided=%zu/3 CPU-copy=%s\n",
           tensor->name, static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
           whole_ok ? "PASS" : "FAIL", tensor->type != GGML_TYPE_F32 ? "N/A" :
           (unaligned_ok ? "PASS" : "FAIL"), partial_passed, strided_passed, copy_ok ? "PASS" : "FAIL");
    return whole_ok && unaligned_ok && partial_passed == 6 && strided_passed == 3 && copy_ok;
}

static bool check_q41(ggml_backend_buffer_type_t buft, int64_t k, int64_t n) {
    tensor_buffer tb(buft, GGML_TYPE_Q4_1, k, n, "q41-readback.weight");
    GGML_ASSERT(tb.tensor->extra);
    const size_t block_bytes = ggml_type_size(GGML_TYPE_Q4_1);
    GGML_ASSERT(block_bytes == 20);
    std::vector<uint8_t> source(ggml_nbytes(tb.tensor));
    std::vector<uint8_t> expected(source.size());
    const int64_t blocks_per_row = k / 32;
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t kb = 0; kb < blocks_per_row; ++kb) {
            const size_t index = static_cast<size_t>(row * blocks_per_row + kb);
            uint8_t * block = source.data() + index * block_bytes;
            const ggml_fp16_t scale = ggml_fp32_to_fp16((1 + index % 7) * 0.125f);
            const ggml_fp16_t minimum = ggml_fp32_to_fp16(-static_cast<float>(3 + index % 11) * ggml_fp16_to_fp32(scale));
            memcpy(block, &scale, sizeof(scale));
            memcpy(block + sizeof(scale), &minimum, sizeof(minimum));
            for (int j = 0; j < 16; ++j) {
                block[4 + j] = static_cast<uint8_t>(((row + 3 * kb + j) % 16) |
                                                   (((5 * row + kb + 7 * j) % 16) << 4));
            }
        }
    }
    expected = source;
    ggml_backend_tensor_set(tb.tensor, source.data(), 0, source.size());
    return check_readback(tb.tensor, expected);
}

static bool check_f32(ggml_backend_buffer_type_t buft, int64_t k, int64_t n, bool router, bool ones) {
    const char * name = router ? "blk.0.ffn_gate_inp.weight" : "plain.weight";
    tensor_buffer tb(buft, GGML_TYPE_F32, k, n, name);
    GGML_ASSERT((tb.tensor->extra != nullptr) == router);
    std::vector<float> source(static_cast<size_t>(k * n));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            source[row * k + col] = ones ? 1.0f :
                static_cast<float>((row * 19 + col * 13) % 127 - 63) * (1 + row % 5) / 17.0f;
        }
    }
    std::vector<float> logical = source;
    if (router) {
        const auto * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
        std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_Q8_0, k));
        for (int64_t row = 0; row < n; ++row) {
            q8->from_float_ref(source.data() + row * k, quantized.data(), k);
            q8->to_float(quantized.data(), logical.data() + row * k, k);
        }
    }
    std::vector<uint8_t> expected(logical.size() * sizeof(float));
    memcpy(expected.data(), logical.data(), expected.size());
    ggml_backend_tensor_set(tb.tensor, source.data(), 0, expected.size());
    return check_readback(tb.tensor, expected);
}

int main(int argc, char ** argv) {
    const char * selection = argc == 1 ? "all" : argv[1];
    const bool all = strcmp(selection, "all") == 0;
    const bool q41 = all || strcmp(selection, "q41") == 0;
    const bool router = all || strcmp(selection, "router") == 0;
    const bool plain = all || strcmp(selection, "plain") == 0;
    if (argc > 2 || !(q41 || router || plain)) {
        fprintf(stderr, "usage: %s [all|q41|router|plain]\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_name("SPACEMIT0");
    if (!device) {
        printf("SKIP: SPACEMIT0 is unavailable\n");
        return 77;
    }
    auto * buft = ggml_backend_dev_buffer_type(device);
    tensor_buffer probe(buft, GGML_TYPE_F32, 32, 32, "probe.ffn_gate_inp.weight");
    if (!probe.tensor->extra) {
        printf("SKIP: IME2 router repack is unavailable\n");
        return 77;
    }

    int failed = 0;
    int total = 0;
    if (q41) {
        failed += !check_q41(buft, 256, 32);
        failed += !check_q41(buft, 64, 64);
        total += 2;
    }
    if (router) {
        failed += !check_f32(buft, 32, 32, true, true);
        failed += !check_f32(buft, 96, 64, true, false);
        total += 2;
    }
    if (plain) {
        failed += !check_f32(buft, 64, 32, false, false);
        total += 1;
    }
    printf("SpacemiT readback (%s): %d/%d passed\n", selection, total - failed, total);
    return failed == 0 ? 0 : 1;
}
