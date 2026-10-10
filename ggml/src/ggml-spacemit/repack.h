#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

inline bool q4_pp_stream_weights_enabled(const ggml_tensor * tensor) {
    constexpr int64_t max_k = 393216 / 32;  // two M8 A tiles in one A100 worker's TCM
    if (tensor == nullptr || tensor->ne[0] <= 0 || tensor->ne[0] > max_k || tensor->ne[1] <= 0 ||
        tensor->ne[1] % 32 != 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return false;
    }
    if (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K) {
        return false;
    }
    return tensor->ne[0] % ggml_blck_size(tensor->type) == 0;
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
