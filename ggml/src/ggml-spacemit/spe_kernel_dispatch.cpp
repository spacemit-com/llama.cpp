// Vendored from spine-kernel (MIT, (c) 2026 SpacemiT) at commit 29955e5:
// src/ime2/spe_pack_cast.cc (the three ime2 spe_pack_cast specializations and
// get_spe_pack_cast_function) and src/rvv/spe_reducesum_mxn.cc (the f16->f32
// spe_reducesum_mxn specialization and get_spe_reducesum_mxn_function).
// Only deviation: the generic march/dispatch-table getters are collapsed to
// direct returns of the single matching entry per op -- the q4_0_f16 path
// always builds an A100 march, so the ime2 / rvv gates always pass and the
// scalar fallbacks are dead code.

#include "spe_kernels.h"

// The spe_pack_cast specializations are defined in the owning namespace
// mlir::speir::spekernel (GCC requires explicit specializations in the
// namespace of the primary template); their bodies call the ime2 inner
// kernels with explicit qualification.
namespace mlir::speir::spekernel {

// [M,K] FP32 -> [dst_mc,dst_kc,MB,8] FP16.
// out[mo,ko,mi,ki] = f16(in[mo*MB+mi,ko*8+ki]); offsets/strides are bytes.
template <>
void spe_pack_cast<float, _spe_fp16_, 1, 1, 16, 8>(void *  in,
                                                   int64_t in_offset,
                                                   int64_t in_stride,
                                                   void *  out,
                                                   int64_t out_offset,
                                                   int64_t out_stride,
                                                   int64_t m,
                                                   int64_t k,
                                                   int64_t dst_mc,
                                                   int64_t dst_kc) {
    constexpr int64_t mb = 16;
    constexpr int64_t kb = 8;
    SPE_ASSERT(m >= 0 && k >= 0 && dst_mc >= 0 && dst_kc >= 0);
    if (dst_mc == 0 || dst_kc == 0) {
        return;
    }
    SPE_ASSERT(__riscv_vlenb() == 128);
    SPE_ASSERT(in_stride >= k * static_cast<int64_t>(sizeof(float)));
    SPE_ASSERT(out_stride >= dst_kc * mb * kb * static_cast<int64_t>(sizeof(_spe_fp16_)));
    SPE_ASSERT(out != nullptr);
    SPE_ASSERT(m == 0 || k == 0 || in != nullptr);

    const int64_t valid_m = std::min<int64_t>(m, dst_mc * mb);
    const int64_t valid_k = std::min<int64_t>(k, dst_kc * kb);
    auto *        dst     = reinterpret_cast<int8_t *>(out) + out_offset;
    if (valid_m == 0 || valid_k == 0) {
        ime2::_spe_pack_cast_f16_pad_blocks<mb>(dst, out_stride, dst_mc, dst_kc);
        return;
    }
    auto *        src      = reinterpret_cast<const int8_t *>(in) + in_offset;
    const int64_t mc       = valid_m / mb;
    const int64_t m_remain = valid_m % mb;

    // Complete M blocks, including K tails and K padding.
    if (mc != 0) {
        ime2::_spe_pack_cast_f32_f16_full_blocks<mb>(src, in_stride, dst, out_stride, mc, valid_k, dst_kc);
    }

    // The final partial M block: convert valid rows, then pad missing rows.
    if (m_remain != 0) {
        auto *  src_tail = src + mc * mb * in_stride;
        auto *  dst_tail = reinterpret_cast<_spe_fp16_ *>(dst + mc * out_stride);
        int64_t mi       = 0;
        for (; mi < m_remain; ++mi) {
            ime2::_spe_pack_cast_f32_f16_row<mb>(src_tail + mi * in_stride, dst_tail + mi * kb, valid_k, dst_kc);
        }
        for (; mi < mb; ++mi) {
            ime2::_spe_pack_cast_f16_pad_row<mb>(dst_tail + mi * kb, dst_kc);
        }
    }

    // An oversized destination can also contain complete padding blocks.
    const int64_t written_mc = mc + (m_remain != 0);
    if (written_mc < dst_mc) {
        ime2::_spe_pack_cast_f16_pad_blocks<mb>(dst + written_mc * out_stride, out_stride, dst_mc - written_mc, dst_kc);
    }
}

// [M,K] FP32 -> [dst_mc,dst_kc,MB,8] FP16.
// out[mo,ko,mi,ki] = f16(in[mo*MB+mi,ko*8+ki]); offsets/strides are bytes.
template <>
void spe_pack_cast<float, _spe_fp16_, 1, 1, 32, 8>(void *  in,
                                                   int64_t in_offset,
                                                   int64_t in_stride,
                                                   void *  out,
                                                   int64_t out_offset,
                                                   int64_t out_stride,
                                                   int64_t m,
                                                   int64_t k,
                                                   int64_t dst_mc,
                                                   int64_t dst_kc) {
    constexpr int64_t mb = 32;
    constexpr int64_t kb = 8;
    SPE_ASSERT(m >= 0 && k >= 0 && dst_mc >= 0 && dst_kc >= 0);
    if (dst_mc == 0 || dst_kc == 0) {
        return;
    }
    SPE_ASSERT(__riscv_vlenb() == 128);
    SPE_ASSERT(in_stride >= k * static_cast<int64_t>(sizeof(float)));
    SPE_ASSERT(out_stride >= dst_kc * mb * kb * static_cast<int64_t>(sizeof(_spe_fp16_)));
    SPE_ASSERT(out != nullptr);
    SPE_ASSERT(m == 0 || k == 0 || in != nullptr);

    const int64_t valid_m = std::min<int64_t>(m, dst_mc * mb);
    const int64_t valid_k = std::min<int64_t>(k, dst_kc * kb);
    auto *        dst     = reinterpret_cast<int8_t *>(out) + out_offset;
    if (valid_m == 0 || valid_k == 0) {
        ime2::_spe_pack_cast_f16_pad_blocks<mb>(dst, out_stride, dst_mc, dst_kc);
        return;
    }
    auto *        src      = reinterpret_cast<const int8_t *>(in) + in_offset;
    const int64_t mc       = valid_m / mb;
    const int64_t m_remain = valid_m % mb;

    // Complete M blocks, including K tails and K padding.
    if (mc != 0) {
        ime2::_spe_pack_cast_f32_f16_full_blocks<mb>(src, in_stride, dst, out_stride, mc, valid_k, dst_kc);
    }

    // The final partial M block: convert valid rows, then pad missing rows.
    if (m_remain != 0) {
        auto *  src_tail = src + mc * mb * in_stride;
        auto *  dst_tail = reinterpret_cast<_spe_fp16_ *>(dst + mc * out_stride);
        int64_t mi       = 0;
        for (; mi < m_remain; ++mi) {
            ime2::_spe_pack_cast_f32_f16_row<mb>(src_tail + mi * in_stride, dst_tail + mi * kb, valid_k, dst_kc);
        }
        for (; mi < mb; ++mi) {
            ime2::_spe_pack_cast_f16_pad_row<mb>(dst_tail + mi * kb, dst_kc);
        }
    }

    // An oversized destination can also contain complete padding blocks.
    const int64_t written_mc = mc + (m_remain != 0);
    if (written_mc < dst_mc) {
        ime2::_spe_pack_cast_f16_pad_blocks<mb>(dst + written_mc * out_stride, out_stride, dst_mc - written_mc, dst_kc);
    }
}

// One source struct contains 32x32 U4 values and produces four 32x8 FP16 tiles.
template <>
void spe_pack_cast<Q41Block<32, 32, 2>, _spe_fp16_, 32, 32, 32, 8>(void *  in,
                                                                   int64_t in_offset,
                                                                   int64_t in_stride,
                                                                   void *  out,
                                                                   int64_t out_offset,
                                                                   int64_t out_stride,
                                                                   int64_t src_mc,
                                                                   int64_t src_kc,
                                                                   int64_t dst_mc,
                                                                   int64_t dst_kc) {
    using in_type                     = Q41Block<32, 32, 2>;
    constexpr int64_t tiles_per_block = 32 / 8;
    constexpr int64_t tile_bytes      = 32 * 8 * sizeof(_spe_fp16_);
    constexpr int64_t block_bytes     = sizeof(in_type);
    SPE_ASSERT(src_mc >= 0 && src_kc >= 0 && dst_mc >= 0 && dst_kc >= 0);
    SPE_ASSERT(dst_mc == src_mc && dst_kc % tiles_per_block == 0 && dst_kc / tiles_per_block == src_kc);
    if (src_mc == 0 || src_kc == 0) {
        return;
    }
    SPE_ASSERT(__riscv_vlenb() == 128);
    SPE_ASSERT(in != nullptr && out != nullptr);
    SPE_ASSERT(in_stride >= 0 && in_stride / block_bytes >= src_kc);
    SPE_ASSERT(out_stride >= 0 && out_stride / tile_bytes >= dst_kc);

    const auto *  src        = reinterpret_cast<const uint8_t *>(in) + in_offset;
    auto *        dst        = reinterpret_cast<uint8_t *>(out) + out_offset;
    const int32_t zero_point = src[offsetof(in_type, zp)];
    for (int64_t mo = 0; mo < src_mc; ++mo) {
        ime2::_spe_pack_cast_q41_f16_m32k32(src + mo * in_stride, dst + mo * out_stride, zero_point, src_kc);
    }
}

}  // namespace mlir::speir::spekernel

namespace mlir::speir::spekernel::ime2 {

spe_pack_cast_function_def get_spe_pack_cast_function(const uint64_t         march,
                                                      const std::type_info & in_type,
                                                      const std::type_info & out_type,
                                                      const size_t           src_mb,
                                                      const size_t           src_kb,
                                                      const size_t           dst_mb,
                                                      const size_t           dst_kb) {
    SpineArchInfoDef arch_info;
    arch_info.value = march;
    if (!(arch_info.fields.core_ext_info.fields.ime & SPE_ARCH_IME2) || arch_info.fields.vector_info != 128) {
        return nullptr;
    }

    struct dispatch_entry_t {
        std::type_index            in_type;
        std::type_index            out_type;
        size_t                     src_mb;
        size_t                     src_kb;
        size_t                     dst_mb;
        size_t                     dst_kb;
        spe_pack_cast_function_def func;
    };

    // clang-format off
    static const std::vector<dispatch_entry_t> func_table = {
        { typeid(float), typeid(_spe_fp16_), 1, 1, 16, 8, spe_pack_cast<float, _spe_fp16_, 1, 1, 16, 8> },
        { typeid(float), typeid(_spe_fp16_), 1, 1, 32, 8, spe_pack_cast<float, _spe_fp16_, 1, 1, 32, 8> },
        { typeid(Q41Block<32, 32, 2>), typeid(_spe_fp16_), 32, 32, 32, 8,
          spe_pack_cast<Q41Block<32, 32, 2>, _spe_fp16_, 32, 32, 32, 8> },
    };
    // clang-format on

    const std::type_index in_type_index(in_type);
    const std::type_index out_type_index(out_type);

    for (const auto & entry : func_table) {
        if (entry.in_type == in_type_index && entry.out_type == out_type_index && entry.src_mb == src_mb &&
            entry.src_kb == src_kb && entry.dst_mb == dst_mb && entry.dst_kb == dst_kb) {
            return entry.func;
        }
    }
    return nullptr;
}

}  // namespace mlir::speir::spekernel::ime2

namespace mlir::speir::spekernel {

template <>
void spe_reducesum_mxn<_spe_fp16_, float, 16, 32, 1, 1>(void *  in,
                                                        int64_t lhs_offset,
                                                        int64_t lhs_stride0,
                                                        int64_t lhs_stride1,
                                                        void *  out,
                                                        int64_t out_offset,
                                                        int64_t out_stride,
                                                        int64_t src_mc,
                                                        int64_t src_nc,
                                                        int64_t nth,
                                                        int64_t dst_mc,
                                                        int64_t dst_nc) {
    SPE_ASSERT(src_mc >= 0);
    SPE_ASSERT(src_nc >= 0);
    SPE_ASSERT(dst_mc >= 0);
    SPE_ASSERT(dst_nc >= 0);
    if (dst_mc == 0 || dst_nc == 0) {
        return;
    }
    SPE_ASSERT(nth > 0);
    SPE_ASSERT(in != nullptr);
    SPE_ASSERT(out != nullptr);

    constexpr int64_t block_m       = 16;
    constexpr int64_t block_n       = 32;
    constexpr int64_t thread_stride = block_n * sizeof(_spe_fp16_);
    const int64_t     m_blocks      = dst_mc / block_m + (dst_mc % block_m != 0);
    const int64_t     n_blocks      = dst_nc / block_n + (dst_nc % block_n != 0);
    SPE_ASSERT(src_mc >= m_blocks);
    SPE_ASSERT(src_nc >= n_blocks);

    const auto *  in_raw     = reinterpret_cast<const int8_t *>(in) + lhs_offset;
    auto *        out_raw    = reinterpret_cast<int8_t *>(out) + out_offset;
    const int64_t row_stride = nth * thread_stride;

    const bool use_a100_preload = nth == 8 && __riscv_vlenb() == 128;
    for (int64_t cm = 0; cm < m_blocks; ++cm) {
        const int64_t rows = std::min(block_m, dst_mc - cm * block_m);
        for (int64_t cn = 0; cn < n_blocks; ++cn) {
            const int64_t cols      = std::min(block_n, dst_nc - cn * block_n);
            const auto *  in_block  = in_raw + cm * lhs_stride0 + cn * lhs_stride1;
            auto *        out_block = out_raw + cm * block_m * out_stride + cn * block_n * sizeof(float);
            if (use_a100_preload && rows == block_m && cols == block_n) {
                rvv::_spe_reducesum_mxn_f16_f32_16x32_1x1_a100_nth8(in_block, out_block, out_stride);
                continue;
            }
            for (int64_t i = 0; i < rows; ++i) {
                auto * out_row = out_block + i * out_stride;
                rvv::_spe_reducesum_mxn_f16_f32_16x32_1x1(in_block + i * row_stride, out_row, cols, nth);
            }
        }
    }
}

}  // namespace mlir::speir::spekernel

namespace mlir::speir::spekernel::rvv {

spe_reducesum_mxn_function_def get_spe_reducesum_mxn_function(const uint64_t         march,
                                                              const std::type_info & in_type,
                                                              const std::type_info & out_type,
                                                              const size_t           src_mb,
                                                              const size_t           src_nb,
                                                              const size_t           dst_mb,
                                                              const size_t           dst_nb) {
    SpineArchInfoDef arch_info;
    arch_info.value = march;
    if (!(arch_info.fields.core_ext_info.fields.rvv & 0xFF)) {
        return nullptr;
    }

    struct dispatch_entry_t {
        std::type_index                in_type;
        std::type_index                out_type;
        size_t                         src_mb;
        size_t                         src_nb;
        size_t                         dst_mb;
        size_t                         dst_nb;
        spe_reducesum_mxn_function_def func;
    };

    static const std::vector<dispatch_entry_t> func_table = {
        {typeid(_spe_fp16_), typeid(float), 16, 32, 1, 1, spe_reducesum_mxn<_spe_fp16_, float, 16, 32, 1, 1>},
    };

    const std::type_index in_type_index(in_type);
    const std::type_index out_type_index(out_type);

    for (const auto & entry : func_table) {
        if (entry.in_type == in_type_index && entry.out_type == out_type_index && entry.src_mb == src_mb &&
            entry.src_nb == src_nb && entry.dst_mb == dst_mb && entry.dst_nb == dst_nb) {
            return entry.func;
        }
    }
    return nullptr;
}

}  // namespace mlir::speir::spekernel::rvv
