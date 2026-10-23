#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

// ---- spine-kernel data types vendored into the tree (MIT, (c) 2025 SpacemiT,
// from build/installed/include/spe_kernel_data_types.h at commit 29955e5);
// the q4_0_f16 path no longer links spine-kernel, so the Q41 repack layout and
// its scalar types live here. ----
struct uint4x2_t {
    uint8_t val{ 0 };

    operator uint8_t() const noexcept { return (val & 0x0F); }

    explicit uint4x2_t(uint8_t v) noexcept { val = v & 0x0F; }
};

using _spe_fp16_ = _Float16;

// Lossless q4_0 repack block: 32 M rows x 32 U4 K values (16 packed bytes per
// row), shared zero point, one f16 scale per row.  Layout must stay
// bit-identical to spine-kernel's Q41Block<32,32,2> -- the vendored ime2
// pack_cast asm and the Q41 GEMV read it directly.
template <size_t, size_t, size_t> struct Q41Block;

template <> struct Q41Block<32, 32, 2> {
    // 32 M rows of 32 unsigned K values, stored as 16 packed bytes per row.
    // K is contiguous: even K is in the low nibble, odd K in the high nibble.
    uint4x2_t data[4 * 8 * 16];

    // spe_pack_cast shares zp[0] from the first input block across the entire call.
    // scale is per M row in each block; the other zp slots are unused by this op.
    uint8_t    zp[32];
    _spe_fp16_ scale[32];
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

// lossless q4_0 -> Q41Block<32,32,2> repack for the f16 path
int repack_q4_0_to_q41(ggml_tensor * t, const void * data, size_t data_size);

// inverse of repack_q4_0_to_q41: writes raw q4_0 blocks (ggml layout) into data
int unpack_q41_to_q4_0(const ggml_tensor * t, void * data, size_t data_size);

// Inverse of the legacy Q4_0 N32/K256 repack, for backend tensor reads.
int unpack_q4_0_32x256(const ggml_tensor * t, void * data, size_t data_size);

// Q5_0 tensor converted exactly to Q8_0 and repacked into the q8_0 32x32 layout (allocation: Q8_0 size)
int repack_q5_0_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
