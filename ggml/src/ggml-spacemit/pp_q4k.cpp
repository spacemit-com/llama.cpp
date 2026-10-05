// SPDX-License-Identifier: MIT
// Q4_K zero-point expansion and M8 activation quantization for A100 IME2.
#include "pp_q4k.hpp"

#include <riscv_vector.h>

#include <cstring>
#include <stdexcept>

namespace q4k_pp {

void expand_q4n32_to_q8n32(const PackedBlockQ4N32 * source, PackedBlockQ8N32 * destination,
                           std::size_t n, std::size_t k) {
    const std::size_t blocks = (n / kTileN) * (k / kBlockK);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto & input = source[block];
        auto & output = destination[block];
        std::memcpy(output.d, input.d, sizeof(output.d));
        for (std::size_t col = 0; col < kTileN; ++col) {
            for (std::size_t ki = 0; ki < kBlockK; ++ki) {
                const std::uint8_t byte = input.qs[col * 16 + ki / 2];
                const int q = (ki & 1) == 0 ? byte & 15 : byte >> 4;
                output.qs[col * kBlockK + ki] = static_cast<std::int8_t>(q - input.zp[col]);
            }
        }
    }
}

void prepack_q8n32_for_ime(const PackedBlockQ8N32 * source, PackedBlockQ8N32Ime * destination,
                           std::size_t n, std::size_t k) {
    const std::size_t blocks = (n / kTileN) * (k / kBlockK);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto & input = source[block];
        auto & output = destination[block];
        std::memcpy(output.d, input.d, sizeof(output.d));
        for (std::size_t quarter = 0; quarter < 4; ++quarter) {
            for (std::size_t col = 0; col < 8; ++col) {
                const auto * q = input.qs + (quarter * 8 + col) * kBlockK;
                auto * first = output.qs + quarter * 256 + col * 16;
                std::memcpy(first, q, 16);
                std::memcpy(first + 128, q + 16, 16);
            }
        }
    }
}

void quantize_a_m8_ime(const float * source, std::size_t k, PackedABlockM8Ime * destination) {
    if (__riscv_vlenb() != 128 || k == 0 || k % kBlockK != 0) {
        throw std::invalid_argument("Q8 M8 quantization requires A100 VLEN=1024 and K multiple of 32");
    }
    for (std::size_t offset = 0; offset < k; offset += kBlockK) {
        auto & out = destination[offset / kBlockK];
        for (std::size_t group = 0; group < 2; ++group) {
            const std::size_t vl = __riscv_vsetvl_e32m1(kBlockK);
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            const std::size_t row = group * 4;
            const vfloat32m1_t values0 = __riscv_vle32_v_f32m1(source + (row + 0) * k + offset, vl);
            const vfloat32m1_t values1 = __riscv_vle32_v_f32m1(source + (row + 1) * k + offset, vl);
            const vfloat32m1_t values2 = __riscv_vle32_v_f32m1(source + (row + 2) * k + offset, vl);
            const vfloat32m1_t values3 = __riscv_vle32_v_f32m1(source + (row + 3) * k + offset, vl);
            const auto get_scale = [&](vfloat32m1_t values) {
                const vfloat32m1_t abs_values = __riscv_vfabs_v_f32m1(values, vl);
                const vfloat32m1_t max_vec = __riscv_vfredmax_vs_f32m1_f32m1(abs_values, zero, vl);
                const float max_abs = __riscv_vfmv_f_s_f32m1_f32(max_vec);
                return max_abs / 127.0f;
            };
            const float scale0 = get_scale(values0);
            const float scale1 = get_scale(values1);
            const float scale2 = get_scale(values2);
            const float scale3 = get_scale(values3);
            const float inverse0 = scale0 ? 1.0f / scale0 : 0.0f;
            const float inverse1 = scale1 ? 1.0f / scale1 : 0.0f;
            const float inverse2 = scale2 ? 1.0f / scale2 : 0.0f;
            const float inverse3 = scale3 ? 1.0f / scale3 : 0.0f;
            const std::size_t scale_vl = __riscv_vsetvl_e32m1(8);
            __riscv_vse32_v_f32m1(
                out.scale + group * 32, __riscv_vfmv_v_f_f32m1(scale0, scale_vl), scale_vl);
            __riscv_vse32_v_f32m1(
                out.scale + group * 32 + 8, __riscv_vfmv_v_f_f32m1(scale1, scale_vl), scale_vl);
            __riscv_vse32_v_f32m1(
                out.scale + group * 32 + 16, __riscv_vfmv_v_f_f32m1(scale2, scale_vl), scale_vl);
            __riscv_vse32_v_f32m1(
                out.scale + group * 32 + 24, __riscv_vfmv_v_f_f32m1(scale3, scale_vl), scale_vl);
            const auto store_row = [&](std::size_t r, vfloat32m1_t values, float inverse) {
                const vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(values, inverse, vl);
                const vint16mf2_t q16 = __riscv_vfncvt_x_f_w_i16mf2(scaled, vl);
                const vint8mf4_t q8 = __riscv_vncvt_x_x_w_i8mf4(q16, vl);
                __riscv_vse8_v_i8mf4(out.qs + r * 16, q8, 16);
                const vint8mf4_t high = __riscv_vslidedown_vx_i8mf4(q8, 16, vl);
                __riscv_vse8_v_i8mf4(out.qs + 128 + r * 16, high, 16);
            };
            store_row(row + 0, values0, inverse0);
            store_row(row + 1, values1, inverse1);
            store_row(row + 2, values2, inverse2);
            store_row(row + 3, values3, inverse3);
        }
    }
}

}  // namespace q4k_pp
