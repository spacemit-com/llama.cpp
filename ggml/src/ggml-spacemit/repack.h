#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

// Q5_0 tensor converted exactly to Q8_0 and repacked into the q8_0 32x32 layout (allocation: Q8_0 size)
int repack_q5_0_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);
int repack_f32_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
