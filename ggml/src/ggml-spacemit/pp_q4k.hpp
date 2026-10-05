// SPDX-License-Identifier: MIT
// Q4_K PP layouts and the A100 IME2 M8 kernel.
#pragma once

#include <cstddef>
#include <cstdint>

namespace q4k_pp {

inline constexpr std::size_t kBlockK = 32;
inline constexpr std::size_t kTileN = 32;

// Same layout as the existing block_q4_1x32 weight representation.
struct PackedBlockQ4N32 {
    std::uint16_t d[32];
    std::uint8_t zp[32];
    std::uint8_t qs[512];
};
static_assert(sizeof(PackedBlockQ4N32) == 608);
static_assert(offsetof(PackedBlockQ4N32, qs) == 96);

struct PackedBlockQ8N32 {
    std::uint16_t d[32];
    std::int8_t qs[1024];
};
static_assert(sizeof(PackedBlockQ8N32) == 1088);

// Each N8 quarter stores K0..15 followed by K16..31 for IME2.
struct PackedBlockQ8N32Ime {
    std::uint16_t d[32];
    std::int8_t qs[1024];
};
static_assert(sizeof(PackedBlockQ8N32Ime) == 1088);
static_assert(offsetof(PackedBlockQ8N32Ime, qs) == 64);

// One M8 x K32 activation block; duplicated scales match the dot lanes.
struct PackedABlockM8Ime {
    float scale[64];
    std::int8_t qs[256];
};
static_assert(sizeof(PackedABlockM8Ime) == 512);
static_assert(offsetof(PackedABlockM8Ime, qs) == 256);

void expand_q4n32_to_q8n32(const PackedBlockQ4N32 * source, PackedBlockQ8N32 * destination,
                           std::size_t n, std::size_t k);
void prepack_q8n32_for_ime(const PackedBlockQ8N32 * source, PackedBlockQ8N32Ime * destination,
                           std::size_t n, std::size_t k);
void quantize_a_m8_ime(const float * source, std::size_t k, PackedABlockM8Ime * destination);
void gemm_m8_q8_ab_prepacked(const PackedABlockM8Ime * packed_a, const PackedBlockQ8N32Ime * packed_b,
                             float * output, std::size_t k_blocks, std::size_t ldc);

}  // namespace q4k_pp
