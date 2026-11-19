#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-prefetch-debug.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

struct resolver_state {
    std::atomic<int> calls{0};
    std::vector<float> slot;
};

static const void * resolve(const ggml_tensor *, int64_t expert, void * user) {
    auto & s = *static_cast<resolver_state *>(user);
    s.calls.fetch_add(1);
    return expert == 0 ? s.slot.data() : nullptr;
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    const bool ideal = std::strcmp(argv[1], "enabled") == 0;
    GGML_ASSERT(ggml_prefetch_debug_ideal_enabled() == ideal);
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx);

    // Execute the real CPU graph entry with two workers, including CPU fusion
    // dispatch. Ideal mode must actually bypass the real GEMM.
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_set_f32(w, 2.0f);
    ggml_set_f32(x, 3.0f);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const auto begin = std::chrono::steady_clock::now();
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    for (int64_t i = 0; i < ggml_nelements(y); ++i) {
        GGML_ASSERT(ggml_get_f32_1d(y, int(i)) == (ideal ? 1.0f : 24.0f));
    }
    const char * delay = std::getenv("GGML_PREFETCH_DEBUG_IDEAL_US");
    if (ideal && delay) {
        GGML_ASSERT(elapsed >= std::strtoll(delay, nullptr, 10));
    }

    // Real non-heavy operators are not swallowed by the debug hook.
    ggml_tensor * add = ggml_add(ctx, x, x);
    GGML_ASSERT(!ggml_prefetch_debug_ideal_compute(add, 0, 1));

    // Force deterministic synthetic routing and verify full coverage, valid IDs,
    // distinct experts per token, and unrelated argsorts passing through.
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_f32(scores, 1.0f);
    ggml_tensor * topk = ggml_argsort(ctx, scores, GGML_SORT_ORDER_DESC);
    ggml_set_name(topk, "ffn_moe_argsort-0");
    const bool simulated = ggml_prefetch_debug_ideal_compute(topk, 0, 1);
    GGML_ASSERT(simulated == ideal);
    if (ideal) {
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 4; ++col) {
                GGML_ASSERT(ggml_get_i32_1d(topk, row * 4 + col) == (row + col) % 4);
            }
        }
    }
    ggml_set_name(topk, "unrelated_argsort");
    GGML_ASSERT(!ggml_prefetch_debug_ideal_compute(topk, 0, 1));

    // Expert accesses must still use the resolver, with mmap fallback for misses.
    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 4);
    ggml_set_f32(experts, 2.0f);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 4);
    for (int row = 0; row < 4; ++row) {
        ggml_set_i32_1d(ids, row * 2, row);
        ggml_set_i32_1d(ids, row * 2 + 1, (row + 1) % 4);
    }
    ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 4);
    ggml_set_f32(in, 3.0f);
    ggml_tensor * mmid = ggml_mul_mat_id(ctx, experts, in, ids);
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(experts->data, ggml_nbytes(experts));
    resolver_state state;
    state.slot.resize(size_t(experts->nb[2]) / sizeof(float), 5.0f);
    experts->buffer = buffer;
    ggml_backend_buffer_set_expert_resolver(buffer, resolve, &state);
    if (ideal) {
        std::thread worker([&] { GGML_ASSERT(ggml_prefetch_debug_ideal_compute(mmid, 1, 2)); });
        GGML_ASSERT(ggml_prefetch_debug_ideal_compute(mmid, 0, 2));
        worker.join();
        GGML_ASSERT(state.calls.load() == 4);
        for (int64_t i = 0; i < ggml_nelements(mmid); ++i) {
            GGML_ASSERT(ggml_get_f32_1d(mmid, int(i)) == 1.0f);
        }
    } else {
        GGML_ASSERT(!ggml_prefetch_debug_ideal_compute(mmid, 0, 1));
        GGML_ASSERT(state.calls.load() == 0);
    }
    experts->buffer = nullptr;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("ideal=%d: CPU graph, passthrough, routing, resolver and wait checks passed (%lld us graph)\n",
                int(ideal), (long long) elapsed);
}
