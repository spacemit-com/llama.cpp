#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

inline bool q4_0_pp_weights_enabled(const ggml_tensor * tensor) {
    // The two M8 activation tiles for M16 need 32 bytes of TCM per K element.
    constexpr int64_t max_k = 393216 / 32;
    return tensor->type == GGML_TYPE_Q4_0 && tensor->ne[0] % 32 == 0 && tensor->ne[1] % 32 == 0 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && tensor->ne[0] <= max_k;
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
