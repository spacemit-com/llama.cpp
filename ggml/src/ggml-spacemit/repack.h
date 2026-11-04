#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

// ---- spine-kernel data types vendored into the tree (MIT, (c) 2025 SpacemiT,
// from build/installed/include/spe_kernel_data_types.h at commit 29955e5);
// the Q41 f16 path does not link spine-kernel, so the Q41 repack layout and
// its scalar types live here. ----
struct uint4x2_t {
    uint8_t val{ 0 };

    operator uint8_t() const noexcept { return (val & 0x0F); }

    explicit uint4x2_t(uint8_t v) noexcept { val = v & 0x0F; }
};

using _spe_fp16_ = _Float16;

// Original Q41Block layout, unchanged from spine-kernel. All Q4 sources
// use every scale/zp slot per row and K32 block; Q4_0 fills all zps with 8.
// The comments inside the unchanged struct describe the historical shared-zp
// consumer, not the unified per-row semantics used by the current kernels.
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

static_assert(sizeof(Q41Block<32, 32, 2>) == 608, "Q41Block layout changed");

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

// lossless q4_0 -> Q41Block<32,32,2> repack for the f16 path
int repack_q4_0_to_q41(ggml_tensor * t, const void * data, size_t data_size);

// Q4_1 copies source scale bits and nibbles, then rounds -min/scale to
// a saturated uint8 zero point. Q4_K first folds each K32 scale into FP16
// and rounds its offset using that stored scale. These mappings are lossy:
// offsets outside zp=[0,255], fractional zero points, and nonzero offsets
// with zero scale cannot be represented. Zero scale uses zp=0. No nibble
// requantization or replacement metadata is introduced.
int repack_q4_1_to_q41(ggml_tensor * t, const void * data, size_t data_size);
int repack_q4_k_to_q41(ggml_tensor * t, const void * data, size_t data_size);

// Read back in the source type: Q4_0 restores original bytes; Q4_1
// reconstructs min=-zp*scale (rounded to half); Q4_K uses the reference
// quantizer. Readback is not a correctness oracle for original source data.
// Callers pass the source type's byte count.
int unpack_q41_to_q4_0_common(const ggml_tensor * t, void * data, size_t data_size);

// Q5_0 tensor converted exactly to Q8_0 and repacked into the q8_0 32x32 layout (allocation: Q8_0 size)
int repack_q5_0_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);
int repack_f32_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
