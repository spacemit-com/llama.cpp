#include "ggml-prefetch-debug.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct ideal_config {
    bool enabled;
    int64_t delay_us;
};

const ideal_config & config() {
    static const ideal_config value = [] {
        ideal_config c = { false, 0 };
        const char * enabled = std::getenv("GGML_PREFETCH_DEBUG_IDEAL");
        if (!enabled || std::strcmp(enabled, "0") == 0 || !*enabled) {
            return c;
        }
        GGML_ASSERT(std::strcmp(enabled, "1") == 0 && "GGML_PREFETCH_DEBUG_IDEAL must be 0 or 1");
        c.enabled = true;
        if (const char * delay = std::getenv("GGML_PREFETCH_DEBUG_IDEAL_US")) {
            char * end = nullptr;
            errno = 0;
            const long long parsed = std::strtoll(delay, &end, 10);
            GGML_ASSERT(errno == 0 && end != delay && *end == '\0' && parsed >= 0 && parsed <= 60000000 &&
                        "GGML_PREFETCH_DEBUG_IDEAL_US must be an integer in [0, 60000000]");
            c.delay_us = parsed;
        }
        GGML_LOG_WARN("ggml: PREFETCH DEBUG IDEAL enabled: synthetic outputs and round-robin MoE routing, "
                      "%lld us per heavy op; weight pages still read; inference results are invalid\n",
                      (long long) c.delay_us);
        return c;
    }();
    return value;
}

// Fault in every page, including the final partial page, without doing GEMM.
// This models page residency; it does NOT reproduce full DDR traffic/cache reuse.
void touch_pages(const void * data, size_t size, int ith, int nth) {
    GGML_ASSERT(data && nth > 0);
    const auto * bytes = static_cast<const volatile unsigned char *>(data);
    constexpr size_t stride = 4096;
    for (size_t offset = size_t(ith) * stride; offset < size; offset += size_t(nth) * stride) {
        (void) bytes[offset];
    }
    if (ith == 0 && size > 0) {
        (void) bytes[size - 1];
    }
}

void touch_experts(const ggml_tensor * tensor, int ith, int nth) {
    const ggml_tensor * weights = tensor->src[0];
    const ggml_tensor * ids = tensor->src[2];
    GGML_ASSERT(weights && ids && ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    std::vector<unsigned char> selected(size_t(weights->ne[2]), 0);
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t col = 0; col < ids->ne[0]; ++col) {
            int32_t expert;
            std::memcpy(&expert, static_cast<const char *>(ids->data) + row * ids->nb[1] + col * ids->nb[0], sizeof(expert));
            // CPU/GPU split routing may use negative IDs as disabled entries.
            if (expert < 0) {
                continue;
            }
            GGML_ASSERT(expert < weights->ne[2]);
            selected[size_t(expert)] = 1;
        }
    }
    for (int64_t expert = ith; expert < weights->ne[2]; expert += nth) {
        if (selected[size_t(expert)]) {
            // Resolve at the same boundary as real expert execution: ready slot,
            // otherwise the original mmap. Do not accidentally touch both copies.
            touch_pages(ggml_backend_tensor_get_expert(weights, expert), weights->nb[2], 0, 1);
        }
    }
}

bool synthetic_topk(const ggml_tensor * t) {
    return t->op == GGML_OP_ARGSORT &&
        (std::strncmp(t->name, "ffn_moe_topk", 12) == 0 ||
         std::strncmp(t->name, "ffn_moe_argsort", 15) == 0);
}

} // namespace

bool ggml_prefetch_debug_ideal_enabled(void) {
    return config().enabled;
}

bool ggml_prefetch_debug_ideal_compute(ggml_tensor * tensor, int ith, int nth) {
    const auto & c = config();
    if (!c.enabled) {
        return false;
    }
    const bool topk = synthetic_topk(tensor);
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_LIGHTNING_INDEXER:
            break;
        default:
            if (!topk) {
                return false;
            }
    }
    GGML_ASSERT(ith >= 0 && ith < nth && tensor->data);
    GGML_ASSERT(tensor->type == (topk ? GGML_TYPE_I32 : GGML_TYPE_F32));
    GGML_ASSERT(tensor->nb[0] == sizeof(float));

    if (tensor->op == GGML_OP_MUL_MAT_ID) {
        touch_experts(tensor, ith, nth);
    } else if (tensor->op == GGML_OP_MUL_MAT) {
        touch_pages(tensor->src[0]->data, ggml_nbytes(tensor->src[0]), ith, nth);
    } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_LIGHTNING_INDEXER) {
        // Preserve access to the key/value working set while skipping dot products.
        for (int i = 1; i <= 2; ++i) {
            if (tensor->src[i]) {
                touch_pages(tensor->src[i]->data, ggml_nbytes(tensor->src[i]), ith, nth);
            }
        }
    }

    const int64_t nrows = ggml_nrows(tensor);
    for (int64_t row = ith; row < nrows; row += nth) {
        const int64_t i1 = row % tensor->ne[1];
        const int64_t i2 = row / tensor->ne[1] % tensor->ne[2];
        const int64_t i3 = row / (tensor->ne[1] * tensor->ne[2]);
        auto * out = static_cast<char *>(tensor->data) + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
        if (topk) {
            // A full permutation: top-k views remain valid and enough token rows
            // cover every expert, regardless of skipped router logits/biases.
            auto * ids = reinterpret_cast<int32_t *>(out);
            const int64_t experts = tensor->src[0]->ne[0];
            for (int64_t j = 0; j < tensor->ne[0]; ++j) {
                ids[j] = int32_t((row + j) % experts);
            }
        } else {
            auto * values = reinterpret_cast<float *>(out);
            for (int64_t j = 0; j < tensor->ne[0]; ++j) {
                values[j] = 1.0f;
            }
        }
    }
    // All workers sleep concurrently, followed by the executor's normal barrier.
    // Zero means no intentional wait, not zero total execution time.
    if (!topk && c.delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(c.delay_us));
    }
    return true;
}
