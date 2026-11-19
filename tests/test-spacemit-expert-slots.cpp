// SPDX-License-Identifier: MIT
#include "llama-spacemit-expert-cache.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static void test_type(ggml_type type) {
    ggml_init_params params{2 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx, "ggml context");
    constexpr int K = 256, N = 32, E = 5, M = 3;
    llama_spacemit_expert_cache::banks banks(2);
    for (auto & layer : banks) {
        for (int p = 0; p < 3; ++p) {
            layer.push_back(ggml_new_tensor_3d(ctx, type, K, N, E));
        }
    }
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, M);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, M);
    auto * output = ggml_mul_mat_id(ctx, const_cast<ggml_tensor *>(banks[0][0]), input, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * backend = ggml_backend_cpu_init();
    auto * memory = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(backend && memory, "CPU allocation");
    std::vector<float> raw(K * N * E);
    for (size_t i = 0; i < raw.size(); ++i) { raw[i] = std::sin(float(i) * .013f); }
    for (const auto & layer : banks) {
        for (const auto * tensor : layer) {
            if (type == GGML_TYPE_F32) { std::memcpy(tensor->data, raw.data(), ggml_nbytes(tensor)); }
            else {
                require(ggml_quantize_chunk(type, raw.data(), tensor->data, 0, N * E, K, nullptr) == ggml_nbytes(tensor), "quantize");
            }
        }
    }
    for (int i = 0; i < K * M; ++i) { static_cast<float *>(input->data)[i] = std::cos(float(i) * .017f); }
    const int32_t route[] = {0, 4, 1, 3, 2, 4};
    std::memcpy(ids->data, route, sizeof(route));
    ggml_backend_cpu_set_n_threads(backend, 2);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "baseline compute");
    std::vector<float> baseline(ggml_nelements(output));
    std::memcpy(baseline.data(), output->data, ggml_nbytes(output));
    {
        llama_spacemit_expert_cache zero(banks, 0);
        require(zero.capacity() == 0 && zero.buffer_bytes() == 0, "zero capacity");
        require(ggml_backend_tensor_get_expert(banks[0][0], 0) == banks[0][0]->data, "zero fallback");
    }
    {
        llama_spacemit_expert_cache cache(banks, .65);
        require(cache.total_experts() == 10 && cache.capacity() == 6, "count complete experts once");
        require(cache.buffer_bytes() == 6 * cache.slot_bytes(), "bounded allocation");
        require(cache.slot_bytes() == 3 * banks[0][0]->nb[2], "whole-expert stride");
        require(cache.ranges(0).size() == 9 && cache.ranges(1).size() == 9, "balanced admission");
        for (size_t layer = 0; layer < banks.size(); ++layer) {
            const auto & ranges = cache.ranges(layer);
            for (size_t i = 0; i < ranges.size(); i += 3) {
                const auto & first = ranges[i];
                std::memcpy(first.destination, first.source, first.bytes);
                require(!cache.ready(first.slot), "partial copy must remain unpublished");
                const size_t expert = i / 3;
                require(ggml_backend_tensor_get_expert(banks[layer][0], expert) == first.source, "partial mmap fallback");
                for (size_t p = 1; p < 3; ++p) {
                    const auto & r = ranges[i + p];
                    std::memcpy(r.destination, r.source, r.bytes);
                }
                cache.publish(first.slot);
                require(cache.ready(first.slot), "publication");
                for (size_t p = 0; p < 3; ++p) {
                    require(ggml_backend_tensor_get_expert(banks[layer][p], expert) == ranges[i + p].destination, "published pointer");
                    // Prove graph arithmetic reads the cache, not just its API:
                    // poison admitted source rows after a complete cached copy.
                    std::memset(const_cast<unsigned char *>(ranges[i + p].source), 0, ranges[i + p].bytes);
                }
            }
            const auto * t = banks[layer][0];
            require(ggml_backend_tensor_get_expert(t, 4) == static_cast<const char *>(t->data) + 4 * t->nb[2], "unadmitted fallback");
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "cached compute");
        const auto * actual = static_cast<const float *>(output->data);
        for (size_t i = 0; i < baseline.size(); ++i) {
            require(std::isfinite(actual[i]) && std::fabs(actual[i] - baseline[i]) <= 1e-6f, "cached arithmetic changed");
        }
    }
    require(ggml_backend_tensor_get_expert(banks[0][0], 0) == banks[0][0]->data, "resolver removed on destruction");
    { llama_spacemit_expert_cache full(banks, 1); require(full.capacity() == 10, "full capacity"); }
    ggml_backend_buffer_free(memory);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("PASS %s: bounded buffer, complete publication, actual CPU expert reads and mmap fallback\n", ggml_type_name(type));
}
int main() {
    try {
        ggml_backend_load_all();
        require(llama_spacemit_expert_cache::count_slots(512, .65) == 332, "floor 65 percent");
        require(llama_spacemit_expert_cache::count_slots(512, .5) == 256, "integer boundary");
        require(llama_spacemit_expert_cache::count_slots(7, .01) == 0, "tiny ratio");
        for (double invalid : {-0.1, 1.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            bool rejected = false;
            try { llama_spacemit_expert_cache::count_slots(10, invalid); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "invalid ratio accepted");
        }
        require(llama_spacemit_expert_cache::count_slots(512, std::nextafter(1.0, 0.0)) == 511, "below full capacity");
        test_type(GGML_TYPE_F32);
        test_type(GGML_TYPE_Q4_0);
        test_type(GGML_TYPE_MXFP4);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}