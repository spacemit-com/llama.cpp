#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_kernels.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "scalar_kernels.h"
#include "spacemit-context.h"
#include "spacemit-env.h"
#include "spacemit-fused.h"
#include "spacemit-opnode.h"
#include "spine_mem_pool.h"
#include "vec.h"

#include "spe_kernels.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <cstdlib>
#include <cstring>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
class tensor_traits : public ggml::spacemit::tensor_traits_base {
    bool work_size(int /* n_threads */, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        return forward_mul_mat(ctx, op);
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        return forward_mul_mat_id(ctx, op);
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        if (ith == 0 && ctx.workspace_size < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(ctx.workspace);

        void *        tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = ctx.shared.size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ctx.sync();

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            // Direct IME-from-DRAM GEMV path, decode only (gemm_m == 1), q4_0 only:
            // q8_0 always loses to the staged TCM path here.
            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256 && NB_COLS == 32) {
                if (gemm_m == 1 && a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                    constexpr int64_t tile_cols = 4 * NB_COLS;
                    for (int64_t ni = (int64_t) ith * tile_cols; ni < gemm_n; ni += tile_cols * nth) {
                        const int64_t nb_real  = std::min(gemm_n - ni, tile_cols);
                        uint8_t *     b_row    = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                        uint8_t *     b_row_zp = block_type_has_zp<BLOC_TYPE>() ? b_row : nullptr;
                        gemm_kernel(b_blk_len, a_row, b_row, b_row_zp, output + ni, 1, nb_real, b_k_blks, gemm_n);
                    }
                    return true;
                }
            }

            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            const int64_t ni0     = (int64_t) ith * NB_COLS;
            const bool    active0 = ni0 < gemm_n;
            const int64_t nb0     = active0 ? std::min(gemm_n - ni0, (int64_t) NB_COLS) : 0;

            if (ith % 2 == 0 && active0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            if (has_pair) {
                spine_barrier_wait(cur_barrier);
            }

            if (ith % 2 != 0 && active0) {
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
            }

            const int64_t base_start = (int64_t) (ith & ~1) * NB_COLS;
            const int64_t lane_off   = (int64_t) (ith & 1) * NB_COLS;
            for (int64_t base = base_start; base < gemm_n; base += NB_COLS * nth) {
                const int64_t ni      = base + lane_off;
                const bool    active  = ni < gemm_n;
                const int64_t nb_real = active ? std::min(gemm_n - ni, (int64_t) NB_COLS) : 0;

                if (has_pair && ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                if (active) {
                    int64_t rows_remaining = gemm_m;
                    float * c_blk          = output + ni;
                    auto *  a_row_cur      = a_row;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                        nb_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_cur += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }

                if (has_pair && ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + NB_COLS * nth;
                if (next_ni < gemm_n) {
                    const int64_t next_nb = std::min(gemm_n - next_ni, (int64_t) NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    next_nb * row_stride_b);
                }
            }
        } else {
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
        return true;
    }

    bool forward_mul_mat_id(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl   = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(ctx.workspace);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        if (ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;
        }

        ctx.sync();

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ctx.shared.data;
        const auto tcm_buffer_size = ctx.shared.size;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                if (has_pair) {
                    spine_barrier_wait(cur_barrier);
                }

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (has_pair && ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }
                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    if (has_pair && ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;
                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (moe_gemm_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (moe_gemm_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }

    void unpack(const ggml_tensor * t, void * data, size_t data_size) const override {
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256 && NB_COLS == 32) {
            GGML_ASSERT(ggml::cpu::riscv64_spacemit::unpack_q4_0_32x256(t, data, data_size) == 0);
        } else {
            ggml::spacemit::tensor_traits_base::unpack(t, data, data_size);
        }
    }
};

class tensor_traits_common : public ggml::spacemit::tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32) {
                    const auto * traits = ggml_get_type_traits_cpu(op->src[0]->type);
                    if (traits && traits->vec_dot) {
                    const size_t row_size = ggml_row_size(traits->vec_dot_type, op->src[1]->ne[0]);
                    const size_t bytes = row_size * op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
                        if (op->src[1]->type != traits->vec_dot_type) {
                            size = bytes;
                            return true;
                        }
                    }
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK +
                                      ggml::spacemit::cache_line_size_f32) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * n_tasks * (DK + 2 * DV + 64);
                    // split-KV decode partials (up to 4 query tokens, FA_SPLIT_KV_MAX_Q)
                    const size_t n_q = (size_t) std::min<int64_t>(op->src[0]->ne[1], 4);
                    decode = MAX(decode, sizeof(float) * (n_q * neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV)));

                    size = MAX(prefill, decode);
                }
                return true;
            case GGML_OP_ROPE:
                {
                    // forward_rope_impl caches sin/cos per thread unless the
                    // small-F32-NEOX legacy path (stack cache) is taken
                    const int  mode    = ggml_get_op_params_i32(op, 2);
                    const bool legacy  = op->src[0]->type == GGML_TYPE_F32 &&
                                        mode == GGML_ROPE_TYPE_NEOX && op->ne[0] <= 128 &&
                                        op->src[0]->nb[0] == sizeof(float);
                    size = legacy ? 0
                                  : (size_t) n_threads *
                                        (op->ne[0] + ggml::spacemit::cache_line_size_f32) * sizeof(float);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    spacemit_kernels::rvv::forward_mul_mat_f16_f32(ctx, op);
                    return true;
                }
                if (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                    spacemit_kernels::scalar::forward_mul_mat(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_rms_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_MUL:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SCALE:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_scale_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_TANH:
                        spacemit_kernels::rvv::forward_unary_tanh_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_GELU:
                        spacemit_kernels::rvv::forward_unary_gelu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SILU:
                        spacemit_kernels::rvv::forward_unary_silu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_EXP:
                        spacemit_kernels::rvv::forward_unary_exp_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SIGMOID:
                        spacemit_kernels::rvv::forward_unary_sigmoid_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_NEG:
                        spacemit_kernels::rvv::forward_unary_neg_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SOFTPLUS:
                        spacemit_kernels::rvv::forward_unary_softplus_f32(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SOFT_MAX:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_soft_max_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_GLU:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    switch (ggml_get_glu_op(op)) {
                        case GGML_GLU_OP_GEGLU:
                            spacemit_kernels::rvv::forward_glu_geglu_f32(ctx, op);
                            return true;
                        case GGML_GLU_OP_SWIGLU:
                            spacemit_kernels::rvv::forward_glu_swiglu_f32(ctx, op);
                            return true;
                        default:
                            break;
                    }
                }
                return false;
            case GGML_OP_FLASH_ATTN_EXT:
                // Use the RVV implementation for its supported F32-Q/F16-KV
                // shapes; retain the CPU-equivalent implementation for all
                // other layouts and dtypes.
                if (forward_flash_attn_ext_f16(ctx, op)) {
                    return true;
                }
                spacemit_kernels::scalar::forward_flash_attn_ext(ctx, op);
                return true;
            case GGML_OP_ROPE:
                spacemit_kernels::rvv::forward_rope(ctx, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == src0->type && ggml_is_contiguous(op) &&
                        ggml_nelements(op) == ggml_nelements(src0) &&
                        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)) {
                        if (op->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cont_general<float>(ctx, op);
                        } else {
                            spacemit_kernels::rvv::forward_cont_general<_Float16>(ctx, op);
                        }
                        return true;
                    }
                    return false;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (ggml_nelements(op) == 0) {
                        return true;
                    }
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_strided_f32(ctx, op);
                        return true;
                    }
                    // row-contiguous F32 copy with arbitrary outer strides (GDN rollback snapshots)
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float) &&
                        op->ne[0] == src0->ne[0] && op->ne[1] == src0->ne[1] &&
                        op->ne[2] == src0->ne[2] && op->ne[3] == src0->ne[3]) {
                        spacemit_kernels::rvv::forward_cpy_f32_row_strided(ctx, op);
                        return true;
                    }
                    if (ggml_is_contiguous(op) && ggml_is_contiguous(src0) &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        if (op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cpy_f32_to_f16(ctx, op);
                            return true;
                        }
                        if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16) {
                            spacemit_kernels::rvv::forward_cpy_f16_to_f32(ctx, op);
                            return true;
                        }
                    }
                    return false;
                }
            case GGML_OP_SET_ROWS:
                spacemit_kernels::rvv::forward_set_rows(ctx, op);
                return true;
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(ctx, op);
                    } else {
                        return false;
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    if (spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, op, 0)) {
                        return true;
                    }
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            // Scalar ops dispatched to scalar_kernels
            case GGML_OP_L2_NORM:
                spacemit_kernels::scalar::forward_l2_norm_f32(ctx, op);
                return true;
            case GGML_OP_FILL:
                spacemit_kernels::scalar::forward_fill_f32(ctx, op);
                return true;
            case GGML_OP_CUMSUM:
                spacemit_kernels::scalar::forward_cumsum_f32(ctx, op);
                return true;
            case GGML_OP_PAD:
                spacemit_kernels::scalar::forward_pad_f32(ctx, op);
                return true;
            case GGML_OP_TRI:
                spacemit_kernels::scalar::forward_tri_f32(ctx, op);
                return true;
            case GGML_OP_DIAG:
                spacemit_kernels::scalar::forward_diag_f32(ctx, op);
                return true;
            case GGML_OP_SET:
                spacemit_kernels::scalar::forward_set_f32(ctx, op);
                return true;
            case GGML_OP_SOLVE_TRI:
                spacemit_kernels::scalar::forward_solve_tri_f32(ctx, op);
                return true;
            case GGML_OP_GATED_DELTA_NET:
                spacemit_kernels::rvv::forward_gated_delta_net(ctx, op);
                return true;
            case GGML_OP_ARGSORT:
                spacemit_kernels::scalar::forward_argsort_f32(ctx, op);
                return true;
            case GGML_OP_CLAMP:
                spacemit_kernels::scalar::forward_clamp_f32(ctx, op);
                return true;
            case GGML_OP_SSM_CONV:
                if (!spacemit_kernels::rvv::forward_ssm_conv_f32(ctx, op)) {
                    spacemit_kernels::scalar::forward_ssm_conv_f32(ctx, op);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward_fused(ggml::spacemit::context & ctx, const spacemit_opnode & node) const override {
        switch (node.opcode) {
            case SPACEMIT_OP_RMS_NORM_MUL:
                // out = rms_norm(x) * w in one pass; node.node = RMS_NORM, node.fused[0] = MUL
                spacemit_kernels::rvv::forward_rms_norm_mul_f32(ctx, node.node, node.fused[0]);
                return true;
            default:
                return false;
        }
    }

    bool forward_flash_attn_ext_f16(ggml::spacemit::context & ctx, ggml_tensor * dst) const {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 256 && DV > 0 && DV <= 256);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            return false;
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = ctx.ith;
        const int nth = ctx.nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = neq1 >= Q_TILE_SZ;

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        // decode / MTP verification (1-4 query tokens): split the KV range across threads so each K/V row is
        // read once, then merge the partials (ported from ggml-cpu/spacemit)
        const int n_split = (!use_tiled && ctx.shared.data != nullptr) ?
                                spacemit_kernels::rvv::flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(dst, nth) :
                                0;
        if (n_split > 0) {
            float * partials = (float *) ctx.workspace;
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(
                ctx, dst, n_split, partials, ctx.shared.data, ctx.shared.size);
            ctx.sync();
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(ctx, dst, n_split, partials);
            return true;
        }

        int64_t nchunk = nth;

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            }

            current_chunk += nth;
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Dense-only Q41 f16 experiment. Expert tensors retain the legacy traits.
// Keep repack selection and allocation sizing behind the same gate.
// GGML_SPACEMIT_Q41_F16 defaults to ON; setting it to 0 turns the path off.
static bool q4_0_f16_repack_enabled(const ggml_tensor * cur) {
#if defined(RISCV64_SPACEMIT_IME2)
    if (!global_spine_env_info.use_ime2 || cur->ne[2] != 1 || cur->ne[3] != 1 || cur->ne[1] % 32 != 0) {
        return false;
    }
    const char * env = getenv("GGML_SPACEMIT_Q41_F16");
    return env == nullptr || atoi(env) != 0;
#else
    GGML_UNUSED(cur);
    return false;
#endif
}

// Auto routing table for the q4_0_f16 path, decided by the 192-cell
// (K x gemm_m x gemm_n) x (split-M/K/N) perf sweep on k3-dev-008
// (2026-09-30, /tmp results archived in codex-memory):
//   K >= 2048                -> split-K  (won all 48 cells, +30-90% over 2nd)
//   K <= 1024, M <= 256      -> split-N  (won all 16 cells at M=128)
//   K <= 1024, M > 256       -> split-M  (won all 12 cells at M >= 512)
// split-K degenerates at K=1024 because each worker only owns K/8=128
// columns and the reduce dominates; split-N's per-worker whole-A pack kills
// it once M grows past a few hundred rows.  The M=256 crossover between
// split-N and split-M is interpolated (128 vs 512 measured), and decode
// (M=1) never reaches this path anyway.
constexpr size_t Q4_0_F16_SPLIT_K_MIN     = 2048;  // K at/above -> split-K
constexpr size_t Q4_0_F16_SPLIT_N_M_MAX   = 256;   // small-M ceiling for split-N

// Split-N schedule: workers own round-robin N32 tiles; each tile walks the
// whole K in TCM-sized chunks -- B is Q41Block->f16 dequanted once per
// (tile, chunk) and kept in TCM while M streams through the remaining A
// budget in m_fit-row groups, and the f32 kernel's accumulate epilogue
// streams C straight to the output, so there are no partials and no reduce
// phase.  Takes the auto seat for K < 2048 && M <= Q4_0_F16_SPLIT_N_M_MAX
// (won every sweep cell there, e.g. 1210 vs 797 GFLOPS at m=128,n=512,k=1024
// against split-K).  At larger M the per-worker whole-A pack (nth-fold
// duplicated A traffic) loses to split-M/K, and at M=1 (decode) it measured
// 2.3x below split-K.  The cluster hybrid block below only ever sees
// K >= 2048: at K < 2048 its partials + reduce + chunk handshake cost more
// than they save (Qwen3-0.6B qkv/o/gateup K=1024 shapes measured -28..-33%
// vs split-N, while the same shapes' only K>=2048 op, down_proj, gained +4%).
static bool q4_0_f16_split_n(const size_t K, const size_t M) {
    return K < Q4_0_F16_SPLIT_K_MIN && M <= Q4_0_F16_SPLIT_N_M_MAX;
}

// q4_0 weights repacked losslessly into spine-kernel's Q41Block<32,32,2>
// (per-row scale bit-exact, shared zp=8, nibbles identical to q4_0 qs).
// compute_forward ports the f16 split-K schedule: each worker owns one K
// slice, A is f32->f16 pack_cast into TCM, B is Q41Block->f16 pack_cast into
// TCM, the IME2 m16n32k8 kernel accumulates f16 partials in the workspace,
// and spe_reducesum_mxn merges the slices into the f32 output.  The split-M
// schedule (ported from llama.cpp-f16try's M-major probe) instead gives each
// worker a round-robin M block, walks whole K through TCM-sized K chunks and
// writes f32 straight to the output via the accumulate epilogue.  The
// split-N schedule gives each worker round-robin N32 tiles; each tile
// dequants B once (1x, like split-K) and streams the whole-K GEMM straight
// to the f32 output with no partials and no reduce.  Auto routing is
// prefill-only (decode stays on the legacy i8i4 kernel); within the f16
// path the sweep table picks split-K / split-N / split-M / cluster
// hybrid by shape.
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
class tensor_traits_f16 : public ggml::spacemit::tensor_traits_base {
    static_assert(std::is_same_v<BLOC_TYPE, Q41Block<32, 32, 2>>, "f16 traits expect a Q41Block weight layout");
    static constexpr size_t F16_M_TILE = 16;
    static constexpr size_t F16_N_TILE = 32;

    // Per-block geometry: one BLOC_TYPE covers NB_COLS rows x INTER_SIZE
    // elements of K.
    static constexpr size_t KB = (size_t) INTER_SIZE;   // K elements per block
    static constexpr size_t NB = (size_t) NB_COLS;      // rows per block
    static constexpr size_t KB_BYTES   = sizeof(BLOC_TYPE);
    static constexpr size_t ROW_PITCH  = KB_BYTES / NB;  // bytes per row per block

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const size_t N        = static_cast<size_t>(op->src[0]->ne[1]);
        const size_t M        = static_cast<size_t>(op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3]);
        const size_t M_pad    = (M + F16_M_TILE - 1) & ~size_t{ F16_M_TILE - 1 };
        const size_t nc_count = (N + F16_N_TILE - 1) / F16_N_TILE;
        // Split-K partials are f16 in [M/16][nc][16][nth][32]; round the base
        // up to 64B at runtime, +64B slack covers the worst-case shift.
        const size_t partial_size = M_pad * nc_count * static_cast<size_t>(n_threads) * F16_N_TILE * sizeof(_Float16);
        // Cap covers the sweep grid's worst split-K cell (gemm_m=4096 x
        // gemm_n=6144 -> 384 MiB); over-cap shapes would silently fall back
        // to the legacy repack kernel with an undersized workspace.
        constexpr size_t partial_cap = size_t{ 512 } << 20;
        // Decode (M == 1) instead uses the Q41 i8i4 GEMV: the i8-quantized A
        // row lives at [K/KB] x 38B in the workspace (see compute_forward).
        const size_t K         = static_cast<size_t>(op->src[1]->ne[0]);
        const size_t gemv_size = ((K + KB - 1) / KB) * 38;
        size_t       need      = partial_size <= partial_cap ? partial_size + 64 : 0;
        need                   = std::max(need, GGML_PAD(gemv_size, sizeof(int64_t)));
        size                   = need;
        return true;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        // Batch-flattened M (same as forward_mul_mat's gemm_m): src1 may be
        // 3-D, rows are contiguous with stride src1->nb[1] bytes.
        const size_t K             = static_cast<size_t>(src1->ne[0]);
        const size_t M             = static_cast<size_t>(src1->ne[1] * src1->ne[2] * src1->ne[3]);

        const size_t N             = static_cast<size_t>(src0->ne[1]);
        const size_t a_src_stride  = src1->nb[1];  // bytes
        const size_t a_src_stride0 = src1->nb[0];  // bytes
        GGML_ASSERT(a_src_stride0 == sizeof(float));
        GGML_ASSERT(dst->nb[0] == sizeof(float));

        const int ith = ctx.ith;
        const int nth = ctx.nth;

        // Shape tracer for the q4_0_f16 path: one line per MUL_MAT op per
        // worker-0 invocation (prefill prints M=batch, decode M=1).  Guarded
        // by a static env probe so the hot path stays branch-light.
        if (ith == 0 && getenv("GGML_SPACEMIT_Q41_TRACE") != nullptr) {
            fprintf(stderr, "q41_mul_mat: N=%zu M=%zu K=%zu (%s)\n", N, M, K, M == 1 ? "decode" : "prefill");
        }

        const size_t out_row_stride = dst->nb[1];  // bytes

        constexpr size_t m_tile = F16_M_TILE;
        constexpr size_t n_tile = F16_N_TILE;
        const size_t     mt_count = (M + m_tile - 1) / m_tile;
        const size_t     nt_count = (N + n_tile - 1) / n_tile;

        void *      tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = ctx.shared.size;
        if (tcm_buffer == nullptr || tcm_buffer_size <= 0) {
            GGML_ABORT("q4_0_f16: no shared TCM buffer");
        }
        const size_t tcm_size = static_cast<size_t>(tcm_buffer_size);

        // Hand-built march for the vendored spe dispatch getters (verbatim
        // from spine-kernel, defined in ime2_kernels.cpp): ime2 pack_cast
        // requires ime bit 1 + vector_info==128, rvv reducesum requires a
        // nonzero rvv byte.  Matches spine_arch_info.cc's A100 descriptor.
        SpineArchInfoDef march{};
        march.fields.core_arch_id = SpineCoreArchID::CoreArchA100;
        march.fields.matrix_info  = 128;
        march.fields.vector_info  = 128;
        march.fields.core_ext_info.fields.rvv = (1 << 0);
        march.fields.core_ext_info.fields.ime = (1 << 1);

        // A: f32 -> f16 [K/8][16][8]; B: Q41Block -> f16 [K/8][32][8];
        // reduce: f16 partials -> f32.
        const auto pack_func =
            mlir::speir::spekernel::ime2::get_spe_pack_cast_function(march.value, typeid(float), typeid(_spe_fp16_), 1, 1, 16, 8);
        const auto q41_func = mlir::speir::spekernel::ime2::get_spe_pack_cast_function(
            march.value, typeid(BLOC_TYPE), typeid(_spe_fp16_), NB, KB, NB, 8);
        const auto reduce_func = mlir::speir::spekernel::rvv::get_spe_reducesum_mxn_function(
            march.value, typeid(_spe_fp16_), typeid(float), 16, 32, 1, 1);
        if (!pack_func || !q41_func || !reduce_func) {
            GGML_ABORT("q4_0_f16: spe dispatch getter returned null (pack=%d q41=%d reduce=%d)", pack_func ? 1 : 0,
                       q41_func ? 1 : 0, reduce_func ? 1 : 0);
        }

        const auto * a_src = reinterpret_cast<const uint8_t *>(src1->data);
        const auto * b_src = reinterpret_cast<const uint8_t *>(src0->data);
        auto *       out   = reinterpret_cast<uint8_t *>(dst->data);

        // Schedule dispatch: split-N (K < 2048 with M <= 256: one worker
        // per N32 tile, whole K, straight f32 output, 1x B dequant, no
        // reduce), split-M (K < 2048 with M > 256: workers own M blocks,
        // whole K, no reduce, but per-block B dequant repeats mb_count-fold),
        // cluster hybrid (K >= 2048: cluster-pair K slices with adaptive B
        // slots, partials + tile-split reduce; per-thread split-K below is
        // the infeasible-shape fallback and the down_proj route).

        // Decode (M == 1): legacy i8i4 K32 GEMV over the Q41Block layout.
        // A is i8-quantized once (K32 blocks split across workers, same
        // contract as forward_mul_mat's gemm_m==1 branch) into the workspace,
        // then each worker runs round-robin N32 tiles straight from DRAM --
        // no TCM staging, no partials.  The Q41-adapted kernel
        // (gemm_kernel_i8i4_m1_q41) differs from the legacy one only in the
        // B field offsets/stride (data @ block+0, scale follows the data field
        // blocks); A quantization (quantize_a_row_i8) is reused verbatim.
        // Takes precedence over every f16 schedule so
        // decode always stays on the i8i4 route (f16 GEMM is prefill-only).
        if (M == 1 && K % KB == 0) {
            constexpr size_t a_blk_len      = 32;
            const size_t     a_blk_stride   = spacemit_kernels::q8_blk_size(a_blk_len, true);
            const size_t     a_k_blks       = K / KB;
            const size_t     k_blks         = K / KB;
            const size_t     gemv_ws_needed = a_k_blks * a_blk_stride;
            if (ctx.workspace_size >= gemv_ws_needed) {
                auto * quant_a = reinterpret_cast<uint8_t *>(ctx.workspace);

                int task_per_thread = spacemit_kernels::div_round_up((int) a_k_blks, nth);
                int a_blk_start     = ith * task_per_thread;
                int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
                if (a_blk_start < a_blk_end) {
                    spacemit_kernels::rvv::quantize_a_row_i8(
                        a_blk_len, reinterpret_cast<const float *>(a_src) + a_blk_start * a_blk_len,
                        (a_blk_end - a_blk_start) * a_blk_len, quant_a + a_blk_start * a_blk_stride);
                }

                ctx.sync();

                // N32 tile pitch in the Q41 repack: one block covers
                // NB rows x K32, so rows advance at ROW_PITCH bytes per row per K block.
                const size_t b_row_stride = k_blks * ROW_PITCH;
                constexpr size_t tile_cols = 32;
                for (int64_t ni = (int64_t) ith * tile_cols; ni < (int64_t) N;
                     ni += (int64_t) tile_cols * nth) {
                    const int64_t nb_real = std::min((int64_t) N - ni, (int64_t) tile_cols);
                    const uint8_t * b_row = b_src + ni * b_row_stride;
                    auto *          c_row = reinterpret_cast<float *>(out) + ni;
                    spacemit_kernels::ime2::gemm_kernel_i8i4_m1_q41(KB, quant_a, b_row, nullptr, c_row, 1,
                                                                    (size_t) nb_real, k_blks, N);
                }
                return true;
            }
        }

        if (q4_0_f16_split_n(K, M) && K % KB == 0) {
            // A-resident variant: shrink the K chunk until the whole
            // (padded) M rows of A plus one B tile fit TCM together, so A is
            // packed once per chunk and reused by every N tile the worker
            // owns while B tiles rotate through the remaining slot.  A pack
            // and B dequant both stay 1x per worker; C goes straight to the
            // f32 output with the accumulate epilogue across chunks.
            const size_t m_pad  = (M + m_tile - 1) & ~size_t{ m_tile - 1 };
            const size_t kb_res = (tcm_size / ((m_pad + n_tile) * sizeof(_Float16))) & ~size_t{ 31 };
            const size_t kb_a   = std::min(K, kb_res);
            if (kb_a >= 32 && (m_pad + n_tile) * kb_a * sizeof(_Float16) <= tcm_size) {
                auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_tile = a_tile + m_pad * kb_a;
                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                const size_t m_tiles = (M + m_tile - 1) / m_tile;
                for (size_t k0 = 0; k0 < K; k0 += kb_a) {
                    const size_t kk = std::min(kb_a, K - k0);
                    // f32 -> f16 [kk/8][16][8] for every M16 row group of
                    // this chunk; the ime2 kernel zero-pads M < 16 rows.
                    for (size_t mt = 0; mt < m_tiles; ++mt) {
                        const size_t tm0 = mt * m_tile;
                        pack_func((void *) (a_src + tm0 * a_src_stride),
                                  static_cast<int64_t>(k0 * sizeof(float)),               // in_offset bytes
                                  static_cast<int64_t>(a_src_stride),                       // in_stride bytes
                                  (void *) (a_tile + mt * m_tile * kb_a), 0,                // out_offset
                                  static_cast<int64_t>(kb_a * m_tile * sizeof(_Float16)),   // out_stride bytes
                                  static_cast<int64_t>(std::min(m_tile, M - tm0)),          // mc
                                  static_cast<int64_t>(kk),                                 // kc
                                  1, static_cast<int64_t>(kk / 8));                         // dst_mc, dst_kc
                    }
                    for (size_t nt_i = (size_t) ith; nt_i < nt_count; nt_i += (size_t) nth) {
                        const size_t n0 = nt_i * n_tile;
                        const size_t nc = std::min(n_tile, N - n0);
                        // Q41Block -> f16 dequant of this worker's N32 tile
                        // for the chunk's K range; output [kk/8][32][8],
                        // out_stride = dst_kc * 512B.
                        q41_func((void *) (b_src + nt_i * b_row_stride + (k0 / KB) * KB_BYTES),
                                 0,                                   // in_offset
                                 static_cast<int64_t>(b_row_stride),   // in_stride bytes
                                 (void *) b_tile, 0,
                                 static_cast<int64_t>(kk / 8) * 512,   // out_stride bytes
                                 1, static_cast<int64_t>(kk / KB),     // src_mc, src_kc
                                 1, static_cast<int64_t>(kk / 8));     // dst_mc, dst_kc
                        for (size_t mt = 0; mt < m_tiles; ++mt) {
                            const size_t tm0    = mt * m_tile;
                            float *     c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                            // C written straight to the f32 output;
                            // accumulate seeds from C on k0 > 0 chunks.
                            spacemit_kernels::ime2::gemm_kernel_f16f16f32_m16_vfwmadot(
                                kk, b_tile, a_tile + mt * m_tile * kb_a, c_tile,
                                std::min(m_tile, M - tm0), nc, N, k0 != 0);
                        }
                    }
                }
                return true;
            }
            // Group-streaming fallback for very tall M (A alone cannot fit
            // TCM even at kb = 32): per owned N tile, B is dequanted per
            // (tile, chunk) and kept in TCM while M streams through the A
            // slot in m_fit-row groups.  A is repacked once per owned tile
            // -- the tall-M trade for keeping every B dequant at 1x.
            const size_t bytes_per_k  = (m_tile + n_tile) * sizeof(_Float16);
            const size_t kb           = std::min(K, (tcm_size / bytes_per_k) & ~size_t{ 31 });
            const size_t b_tile_bytes = n_tile * kb * sizeof(_Float16);
            const size_t a_budget     = tcm_size > b_tile_bytes ? tcm_size - b_tile_bytes : 0;
            const size_t m_fit        = kb >= 32 ? (a_budget / (kb * sizeof(_Float16))) & ~size_t{ 15 } : 0;
            if (m_fit >= m_tile) {
                auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_tile = a_tile + m_fit * kb;
                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                for (size_t nt_i = (size_t) ith; nt_i < nt_count; nt_i += (size_t) nth) {
                    const size_t n0 = nt_i * n_tile;
                    const size_t nc = std::min(n_tile, N - n0);
                    for (size_t k0 = 0; k0 < K; k0 += kb) {
                        const size_t kk = std::min(kb, K - k0);
                        // This worker's N32 tile of B for the chunk's K
                        // range: Q41Block -> f16 [kk/8][32][8], dequanted
                        // once per (tile, chunk) and kept in TCM across all
                        // M groups below -- B work stays 1x regardless of
                        // how many groups stream against it.
                        q41_func((void *) (b_src + nt_i * b_row_stride + (k0 / KB) * KB_BYTES),
                                 0,                                   // in_offset
                                 static_cast<int64_t>(b_row_stride),   // in_stride bytes
                                 (void *) b_tile, 0,
                                 static_cast<int64_t>(kk / 8) * 512,   // out_stride bytes
                                 1, static_cast<int64_t>(kk / KB),     // src_mc, src_kc
                                 1, static_cast<int64_t>(kk / 8));     // dst_mc, dst_kc
                        // M streams through the single A slot in m_fit-row
                        // groups; each group is f32->f16 packed for this
                        // chunk's K range and consumed immediately, so A is
                        // repacked once per owned tile -- the tall-M trade
                        // for keeping every B dequant at 1x.
                        for (size_t m0 = 0; m0 < M; m0 += m_fit) {
                            const size_t m_count = std::min(m_fit, M - m0);
                            const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = m0 + mt * m_tile;
                                // f32 -> f16 [kk/8][16][8] into this chunk's
                                // A slot; the ime2 kernel zero-pads M < 16
                                // rows.
                                pack_func((void *) (a_src + tm0 * a_src_stride),
                                          static_cast<int64_t>(k0 * sizeof(float)),               // in_offset bytes
                                          static_cast<int64_t>(a_src_stride),                       // in_stride bytes
                                          (void *) (a_tile + mt * m_tile * kb), 0,                  // out_offset
                                          static_cast<int64_t>(kb * m_tile * sizeof(_Float16)),     // out_stride bytes
                                          static_cast<int64_t>(std::min(m_tile, M - tm0)),          // mc
                                          static_cast<int64_t>(kk),                                 // kc
                                          1, static_cast<int64_t>(kk / 8));                         // dst_mc, dst_kc
                                float * c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                                // C written straight to the f32 output;
                                // accumulate seeds from C on k0 > 0 chunks.
                                spacemit_kernels::ime2::gemm_kernel_f16f16f32_m16_vfwmadot(
                                    kk, b_tile, a_tile + mt * m_tile * kb, c_tile,
                                    std::min(m_tile, M - tm0), nc, N, k0 != 0);
                            }
                        }
                    }
                }
                return true;
            }
        }

        // Split-M schedule (ported from llama.cpp-f16try's M-major probe):
        // workers own round-robin M blocks; each block walks the whole K in
        // TCM-sized chunks -- A is f32->f16 packed per chunk and reused across
        // every N tile, B is Q41Block->f16 dequanted per N32 tile per chunk,
        // and the f32 kernel's accumulate epilogue streams C straight to the
        // output, so there are no partials and no reduce phase.  Selected by
        // the auto dispatch for K < 2048 with M > Q4_0_F16_SPLIT_N_M_MAX;
        // at small M the mb_count clamp degenerates to nth-fold B dequant,
        // which split-N handles instead.  Shapes whose TCM cannot hold one B
        // tile beside an M16 slab fall through to split-K below.
        if (K < Q4_0_F16_SPLIT_K_MIN && M > Q4_0_F16_SPLIT_N_M_MAX && K % KB == 0) {
            const size_t bytes_per_k  = (m_tile + n_tile) * sizeof(_Float16);
            const size_t kb           = std::min(K, (tcm_size / bytes_per_k) & ~size_t{ 31 });
            const size_t b_tile_bytes = n_tile * kb * sizeof(_Float16);
            const size_t a_budget     = tcm_size > b_tile_bytes ? tcm_size - b_tile_bytes : 0;
            const size_t m_fit        = kb >= 32 ? (a_budget / (kb * sizeof(_Float16))) & ~size_t{ 15 } : 0;
            if (m_fit >= m_tile) {
                // Fewest blocks that fit (ceil(M/m_fit)), but never fewer than
                // nth, so every worker owns at least one block and stays busy.
                size_t mb_count = (M + m_fit - 1) / m_fit;
                if (mb_count < (size_t) nth) {
                    mb_count = (size_t) nth;
                }
                size_t m_block = 0;
                do {
                    // Round the per-block rows up to 16; if that overshoots M
                    // the last block would be empty, so retry with fewer
                    // blocks (keeps mb < M for every issued block).
                    m_block = (((M + mb_count - 1) / mb_count) + 15) & ~size_t{ 15 };
                    if ((mb_count - 1) * m_block < M) {
                        break;
                    }
                    --mb_count;
                } while (mb_count > 1);
                if (m_block <= m_fit && m_block >= m_tile) {
                    auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                    auto * b_tile = a_tile + m_block * kb;
                    const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                    for (size_t mb_i = (size_t) ith; mb_i < mb_count; mb_i += (size_t) nth) {
                        const size_t mb      = mb_i * m_block;
                        const size_t m_count = std::min(m_block, M - mb);
                        const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                        for (size_t k0 = 0; k0 < K; k0 += kb) {
                            const size_t kk = std::min(kb, K - k0);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = mb + mt * m_tile;
                                // f32 -> f16 [kk/8][16][8] into this chunk's A
                                // slot; the ime2 kernel zero-pads M < 16 rows.
                                pack_func((void *) (a_src + tm0 * a_src_stride),
                                          static_cast<int64_t>(k0 * sizeof(float)),               // in_offset bytes
                                          static_cast<int64_t>(a_src_stride),                       // in_stride bytes
                                          (void *) (a_tile + mt * m_tile * kb), 0,                  // out_offset
                                          static_cast<int64_t>(kb * m_tile * sizeof(_Float16)),     // out_stride bytes
                                          static_cast<int64_t>(std::min(m_tile, M - tm0)),          // mc
                                          static_cast<int64_t>(kk),                                 // kc
                                          1, static_cast<int64_t>(kk / 8));                         // dst_mc, dst_kc
                            }
                            for (size_t nt = 0; nt < nt_count; ++nt) {
                                const size_t n0 = nt * n_tile;
                                const size_t nc = std::min(n_tile, N - n0);
                                // Q41Block -> f16 dequant of this chunk's K
                                // range for the [n0, n0+32) rows: output
                                // [kk/8][32][8], out_stride = dst_kc * 512B.
                                q41_func((void *) (b_src + nt * b_row_stride + (k0 / KB) * KB_BYTES),
                                         0,                                   // in_offset
                                         static_cast<int64_t>(b_row_stride),   // in_stride bytes
                                         (void *) b_tile, 0,
                                         static_cast<int64_t>(kk / 8) * 512,   // out_stride bytes
                                         1, static_cast<int64_t>(kk / KB),     // src_mc, src_kc
                                         1, static_cast<int64_t>(kk / 8));     // dst_mc, dst_kc
                                for (size_t mt = 0; mt < m_tiles; ++mt) {
                                    const size_t tm0   = mb + mt * m_tile;
                                    float *       c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                                    // C written straight to the f32 output;
                                    // accumulate seeds from C on k0 > 0 chunks.
                                    spacemit_kernels::ime2::gemm_kernel_f16f16f32_m16_vfwmadot(
                                        kk, b_tile, a_tile + mt * m_tile * kb, c_tile,
                                        std::min(m_tile, M - tm0), nc, N, k0 != 0);
                                }
                            }
                        }
                    }
                    return true;
                }
            }
        }

        // Cluster-pair hybrid with adaptive B slots: K is cut into g =
        // nth/2 slices, one per 2-core cluster (workers 2i and 2i+1 sit on
        // one cluster whose two TCM blocks are mutually accessible at full
        // speed).  Each thread packs half the M rows of its cluster's slice
        // and dequants its own N32 tile per 64-column chunk; each thread
        // runs its A against both its own and the sibling's B tile
        // (cross-read straight from the sibling's TCM), writing to distinct
        // partial slots of the cluster's slice.  packA/dequant/GEMM totals
        // match split-K while partials and reduce traffic drop from nth to g
        // slices.  Cross-thread ordering uses ready/done flags in TCM,
        // exchanged at chunk granularity (~5us cross-core visibility).
        // The second B slot costs n_tile rows of A in TCM, which can push a
        // shape over a pass boundary and force an extra full-B re-dequant
        // pass (K=9728: m_fit 48 -> 16, passes 2 -> 4; the 4B model
        // regressed pp128 -5.8% / pp512 -11.8% from this), so the slot
        // count is picked per shape: two slots only when that does not
        // increase the M pass count and there is more than one N chunk to
        // pipeline, one slot otherwise.  With one slot the in-loop refill
        // overwrites the slot the sibling just read, so the done wait
        // tightens to the full handshake (sib_done >= seq instead of
        // seq-1).  Shapes with K < 2048 never reach this block: the
        // split-N / split-M auto seats above claim them first (see
        // q4_0_f16_split_n), so this block only handles the K >= 2048
        // region where the cluster pair beats split-K.
        if ((nth % 2) == 0 && nth <= 16 && K % KB == 0) {
            const size_t g  = (size_t) nth / 2;
            const size_t ks = K / g;
            const size_t b_slot = n_tile * ks;                    // f16 per B slot
            const size_t n_chunk = n_tile * 2;                    // 64 columns per cluster chunk
            const size_t chunks  = (N + n_chunk - 1) / n_chunk;
            const auto m_fit_for = [&](size_t slots) {
                const size_t b_bytes  = slots * b_slot * sizeof(_Float16);
                const size_t a_budget = tcm_size > b_bytes + 64 ? tcm_size - b_bytes - 64 : 0;
                return (a_budget / (ks * sizeof(_Float16))) & ~size_t{ 15 };
            };
            const auto n_passes = [&](size_t m_fit) {
                const size_t m_cap = m_fit * 2;                   // cluster M rows per pass
                return (M + m_cap - 1) / m_cap;
            };
            const size_t m_fit1 = m_fit_for(1);
            const size_t m_fit2 = m_fit_for(2);
            const size_t slots =
                chunks > 1 && m_fit2 >= m_tile && n_passes(m_fit2) == n_passes(m_fit1) ? 2 : 1;
            const size_t m_fit = slots == 2 ? m_fit2 : m_fit1;
            // Shape routing to the per-thread split-K below.
            // Two gates, both required:
            // 1. Cluster A-budget collapse: m_fit1 < 128 (one thread cannot
            //    hold a 128-row batch).  Outside this region the cluster wins
            //    even at an M-pass deficit -- the halved partials (g vs nth
            //    slices) and chunk-rotated B access dominate the extra
            //    re-dequant (grid-sweep measured at K=4096, m_fit1=144:
            //    gm=1024 4 vs 3 passes +3..12%, gm=4096 15 vs 12 -2..+8%;
            //    K=3072 gm=4096 10 vs 9 +7..15%; K=2048 gm=4096 7 vs 6
            //    +12..20%) -- so the pass comparison must not run there.
            //    Inside it (K=6144: m_fit1=80, slots collapse to 1; down_proj
            //    K=9728: 48) every extra M pass re-dequants all of B with no
            //    overlap and the deficit bites (K=6144 gm>=512 measured
            //    -3..-15% vs split-K).
            // 2. Split-K needs strictly fewer M passes at this shape; at
            //    equal passes the cluster keeps its partials advantage
            //    (small M: 1 pass either way -> stay clustered).
            bool route_split_k = false;
            if (m_fit1 < 128 && K % (KB * (size_t) nth) == 0) {
                const size_t ksg  = (size_t) K / (size_t) nth;
                const size_t bg   = n_tile * ksg * sizeof(_Float16);
                const size_t mfg  = tcm_size > bg
                                        ? ((tcm_size - bg) / (ksg * sizeof(_Float16))) & ~size_t{ 15 }
                                        : 0;
                if (mfg >= m_tile) {
                    const size_t mbg = std::min(std::max(M, m_tile), mfg);
                    route_split_k    = (M + mbg - 1) / mbg < n_passes(m_fit);
                }
            }
            // Per-thread TCM: [m_fit x ks A][slots x 32 x ks B][64B flags].
            const size_t slice_stride = mt_count * nt_count * m_tile * n_tile;
            if (!route_split_k && K % (KB * g) == 0 && m_fit >= m_tile &&
                g * slice_stride * sizeof(_Float16) + 128 <= ctx.workspace_size) {
                const size_t group    = (size_t) ith / 2;
                const size_t intra    = (size_t) ith % 2;
                const size_t sib      = (size_t) ith ^ 1;
                const size_t slice_k0 = group * ks;
                auto * partials_h = reinterpret_cast<_Float16 *>(
                    (reinterpret_cast<uintptr_t>(ctx.workspace) + 63) & ~uintptr_t{ 63 });
                uint64_t * bases = reinterpret_cast<uint64_t *>(partials_h + g * slice_stride);
                auto * a_blk = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_blk = a_blk + m_fit * ks;             // slot 0; slot 1 at +b_slot
                volatile uint64_t * f_ready =
                    reinterpret_cast<volatile uint64_t *>(reinterpret_cast<uint8_t *>(tcm_buffer) + tcm_size - 64);
                volatile uint64_t * f_done = f_ready + 1;

                bases[ith] = (uint64_t) tcm_buffer;
                *f_ready   = 0;
                *f_done    = 0;
                ctx.sync();
                const auto * sib_base = reinterpret_cast<const uint8_t *>(bases[sib]);
                const auto * sib_b_blk =
                    reinterpret_cast<const _Float16 *>(sib_base + m_fit * ks * sizeof(_Float16));
                volatile uint64_t * sib_ready =
                    reinterpret_cast<volatile uint64_t *>(const_cast<uint8_t *>(sib_base) + tcm_size - 64);
                volatile uint64_t * sib_done = sib_ready + 1;

                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                const size_t m_cap   = m_fit * 2;                    // cluster M rows per pass

                // Dequant this thread's N32 tile of visit position pos into
                // B slot pos%slots.  Visit order (j+group)%chunks is shared
                // by both threads of the cluster, so the slot index matches.
                auto dequant_chunk = [&](size_t pos) {
                    const size_t chunk = (pos + group) % chunks;
                    const size_t nt    = (chunk * n_chunk) / n_tile + intra;
                    if (nt >= nt_count) {
                        return;
                    }
                    q41_func((void *) (b_src + nt * b_row_stride +
                                           (slice_k0 / KB) * KB_BYTES),
                             0,                                            // in_offset
                             static_cast<int64_t>(b_row_stride),            // in_stride bytes
                             (void *) (b_blk + (pos % slots) * b_slot), 0,
                             static_cast<int64_t>((ks / 8) * 512),          // out_stride bytes
                             1, static_cast<int64_t>(ks / KB),              // src_mc, src_kc
                             1, static_cast<int64_t>(ks / 8));              // dst_mc, dst_kc
                };

                uint64_t seq = 0;   // global visit-position counter, monotonically increasing
                for (size_t mb = 0; mb < M; mb += m_cap) {
                    const size_t rows   = std::min(m_cap, M - mb);
                    // Split this pass's rows between the two threads,
                    // rounded to M16 so pack tiles stay whole.
                    size_t       r0     = std::min(std::min(((rows + 31) / 32) * 16, rows), m_fit);
                    const size_t row0   = mb + (intra == 0 ? 0 : r0);
                    const size_t rows_t = intra == 0 ? r0 : rows - r0;
                    const size_t m_tiles = (rows_t + m_tile - 1) / m_tile;
                    // Prologue: position 0 into slot 0 and publish it before
                    // packA so a lagging sibling is unblocked as early as
                    // possible.  Protected by the previous pass's tail wait.
                    dequant_chunk(0);
                    __sync_synchronize();
                    *f_ready = seq + 1;
                    for (size_t mt = 0; mt < m_tiles; ++mt) {
                        const size_t tm0 = row0 + mt * m_tile;
                        pack_func((void *) (a_src + tm0 * a_src_stride),
                                  static_cast<int64_t>(slice_k0 * sizeof(float)),          // in_offset bytes
                                  static_cast<int64_t>(a_src_stride),                      // in_stride bytes
                                  (void *) (a_blk + mt * m_tile * ks), 0,                  // out_offset
                                  static_cast<int64_t>(ks * m_tile * sizeof(_Float16)),    // out_stride bytes
                                  static_cast<int64_t>(std::min(m_tile, rows_t - mt * m_tile)), // mc
                                  static_cast<int64_t>(ks),                                // kc
                                  1, static_cast<int64_t>(ks / 8));                        // dst_mc, dst_kc
                    }
                    for (size_t j = 0; j < chunks; ++j) {
                        const size_t chunk = (j + group) % chunks;
                        const size_t n0    = chunk * n_chunk;
                        const size_t nt    = n0 / n_tile + intra;       // own tile this chunk
                        const size_t nt_s  = n0 / n_tile + (1 - intra); // sibling's tile
                        ++seq;
                        // Local half: own A x own B (slot j%slots, packed by
                        // the prologue or the previous iteration's refill).
                        if (nt < nt_count) {
                            const size_t nc = std::min(n_tile, N - nt * n_tile);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = row0 + mt * m_tile;
                                _Float16 *     c_blk =
                                    partials_h + ((tm0 / m_tile) * nt_count + nt) * m_tile * g * n_tile +
                                    group * n_tile;
                                spacemit_kernels::ime2::gemm_kernel_f16f16f16_m16_vfwmadot(
                                    ks, b_blk + (j % slots) * b_slot, a_blk + mt * m_tile * ks, c_blk,
                                    std::min(m_tile, rows_t - mt * m_tile), nc, g * n_tile, false);
                            }
                        }
                        // Cross half: own A x sibling's B (its slot j%slots).
                        while (*sib_ready < seq) ;
                        __sync_synchronize();
                        if (nt_s < nt_count) {
                            const size_t nc = std::min(n_tile, N - nt_s * n_tile);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = row0 + mt * m_tile;
                                _Float16 *     c_blk =
                                    partials_h + ((tm0 / m_tile) * nt_count + nt_s) * m_tile * g * n_tile +
                                    group * n_tile;
                                spacemit_kernels::ime2::gemm_kernel_f16f16f16_m16_vfwmadot(
                                    ks, sib_b_blk + (j % slots) * b_slot, a_blk + mt * m_tile * ks, c_blk,
                                    std::min(m_tile, rows_t - mt * m_tile), nc, g * n_tile, false);
                            }
                        }
                        __sync_synchronize();
                        *f_done = seq;
                        // Refill the slot the next position needs.  With two
                        // slots that slot last served position seq-1 (reader
                        // confirmed done >= seq-1); with one slot it just
                        // served position seq (full handshake).
                        if (j + 1 < chunks) {
                            while (*sib_done < seq - (slots - 1)) ;
                            dequant_chunk(j + 1);
                            __sync_synchronize();
                            *f_ready = seq + 1;
                        }
                    }
                    // Full handshake: the next pass's prologue overwrites
                    // slot 0, whose last reader finished at done == seq.
                    while (*sib_done < seq) ;
                }

                // Reduce: tile-split block-cyclic
                // across ALL nth workers over the g slices.
                ctx.sync();
                constexpr size_t reduce_chunk = 4;  // tiles per chunk
                const size_t     tile_count   = mt_count * nt_count;
                const size_t     n_rchunks    = (tile_count + reduce_chunk - 1) / reduce_chunk;
                for (size_t c = (size_t) ith; c < n_rchunks; c += (size_t) nth) {
                    const size_t t0 = c * reduce_chunk;
                    const size_t t1 = std::min(tile_count, t0 + reduce_chunk);
                    for (size_t ti = t0; ti < t1; ++ti) {
                        const size_t     mt      = ti / nt_count;
                        const size_t     nt      = ti % nt_count;
                        const _Float16 * tile    = partials_h + ti * m_tile * g * n_tile;
                        const size_t     count_m = std::min(m_tile, M - mt * m_tile);
                        const size_t     count_n = std::min(n_tile, N - nt * n_tile);
                        auto *           out_tile =
                            reinterpret_cast<float *>(out + (mt * m_tile) * out_row_stride) + nt * n_tile;
                        reduce_func((void *) tile, 0, 0, 0,
                                    (void *) out_tile, 0,
                                    static_cast<int64_t>(out_row_stride),  // out row stride bytes
                                    1, 1, static_cast<int64_t>(g),         // src_mc, src_nc, nth
                                    static_cast<int64_t>(count_m), static_cast<int64_t>(count_n));
                    }
                }
                return true;
            }
        }

        // Pick the K-slice count: prefer one slice per worker, fall back
        // until slices start on a Q41 32-element block boundary (the B
        // dequant indexes whole blocks), the [M16 tiles * ks] A block
        // plus the [N32 * ks] B tile fit TCM and the partials fit the
        // workspace.
        size_t s_count = 0;
        size_t ks      = 0;
        for (size_t s = static_cast<size_t>(nth); s >= 1; --s) {
            if (K % (KB * s) != 0) {
                continue;
            }
            const size_t cand_ks  = K / s;
            const size_t b_bytes  = n_tile * cand_ks * sizeof(_Float16);
            const size_t a_budget = tcm_size > b_bytes ? tcm_size - b_bytes : 0;
            const size_t m_fit    = (a_budget / (cand_ks * sizeof(_Float16))) & ~size_t{ 15 };
            if (m_fit < m_tile) {
                continue;
            }
            const size_t slice_stride = mt_count * nt_count * m_tile * n_tile;
            const size_t partial_size = s * slice_stride * sizeof(_Float16);
            if (partial_size + 64 > ctx.workspace_size) {
                continue;
            }
            s_count = s;
            ks      = cand_ks;
            break;
        }
        if (s_count == 0) {
            GGML_ABORT("q4_0_f16: no K-slice count fits TCM/workspace (K=%zu nth=%d tcm=%zu wsize=%zu)", K, nth,
                       tcm_size, ctx.workspace_size);
        }

        const size_t a_budget = tcm_size - n_tile * ks * sizeof(_Float16);
        const size_t m_fit    = (a_budget / (ks * sizeof(_Float16))) & ~size_t{ 15 };
        const size_t m_block  = std::min(std::max(M, m_tile), m_fit);
        const size_t nc_count = nt_count;

        // Round the workspace base up to 64B: the f16f16f16 kernel's 32-f16
        // row stores must each land in a single DDR bank.
        auto * partials =
            reinterpret_cast<_Float16 *>((reinterpret_cast<uintptr_t>(ctx.workspace) + 63) & ~uintptr_t{ 63 });
        auto * a_blk    = reinterpret_cast<_Float16 *>(tcm_buffer);
        auto * b_blk    = a_blk + m_block * ks;
        const size_t slice_k0 = static_cast<size_t>(ith) * ks;

        // GEMM phase: only workers owning a K slice participate; the rest
        // fall through to the sync.
        if (static_cast<size_t>(ith) < s_count) {
            const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
            for (size_t m0 = 0; m0 < M; m0 += m_block) {
                const size_t m_count = std::min(m_block, M - m0);
                const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                for (size_t mt = 0; mt < m_tiles; ++mt) {
                    const size_t tm0 = m0 + mt * m_tile;
                    // f32 -> f16 [ks/8][16][8]; the ime2 kernel zero-pads
                    // M < 16 rows.
                    pack_func((void *) (a_src + tm0 * a_src_stride),
                              static_cast<int64_t>(slice_k0 * sizeof(float)),          // in_offset bytes
                              static_cast<int64_t>(a_src_stride),                      // in_stride bytes
                              (void *) (a_blk + mt * m_tile * ks), 0,                  // out_offset
                              static_cast<int64_t>(ks * m_tile * sizeof(_Float16)),    // out_stride bytes
                              static_cast<int64_t>(std::min(m_tile, M - tm0)),         // mc
                              static_cast<int64_t>(ks),                                // kc
                              1, static_cast<int64_t>(ks / 8));                        // dst_mc, dst_kc
                }
                // Diagonal skew: worker s rotates its N-tile traversal by s so
                // lockstep workers write tiles spread along a diagonal of the
                // [mb][nc] grid instead of converging on one tile block.
                const size_t skew = static_cast<size_t>(ith);
                for (size_t j = 0; j < nc_count; ++j) {
                    const size_t nt = (j + skew) % nc_count;
                    const size_t n0 = nt * n_tile;
                    const size_t nc = std::min(n_tile, N - n0);
                    // Q41Block -> f16 dequant of this worker's K slice for
                    // the [n0, n0+32) rows; IME2 and scalar kernels share the
                    // contract dst_mc=src_mc, dst_kc=4*src_kc, out_stride =
                    // dst_kc * 512B tile; output [ks/8][32][8].
                    q41_func((void *) (b_src + nt * b_row_stride + (slice_k0 / KB) * KB_BYTES),
                             0,                                            // in_offset
                             static_cast<int64_t>(b_row_stride),            // in_stride bytes
                             (void *) b_blk, 0,
                             static_cast<int64_t>(ks / 8) * 512,            // out_stride bytes
                             1, static_cast<int64_t>(ks / KB),              // src_mc, src_kc
                             1, static_cast<int64_t>(ks / 8));              // dst_mc, dst_kc
                    for (size_t i = 0; i < m_tiles; ++i) {
                        const size_t mt  = (i + skew) % m_tiles;
                        const size_t tm0 = m0 + mt * m_tile;
                        // [M/16][nc][16][s_count][32] f16 partials: row r of
                        // slice ith at r*64B*s_count + ith*64B, one DDR bank
                        // per row store; ldc = s_count*32 f16 rows.
                        _Float16 * c_blk = partials +
                                           ((tm0 / m_tile) * nc_count + nt) * m_tile * s_count * n_tile +
                                           static_cast<size_t>(ith) * n_tile;
                        spacemit_kernels::ime2::gemm_kernel_f16f16f16_m16_vfwmadot(
                            ks, b_blk, a_blk + mt * m_tile * ks, c_blk, std::min(m_tile, M - tm0), nc,
                            s_count * n_tile, false);
                    }
                }
            }
        }

        ctx.sync();

        // Tile split reduce, block-cyclic across the K-slice workers: at any
        // instant the workers' chunks are adjacent, so the combined stream
        // over partials is one sequential sweep instead of eight strided
        // streams aliasing the DDR bank hash.
        constexpr size_t reduce_chunk = 4;  // tiles per chunk (32KB)
        const size_t     tile_count   = mt_count * nt_count;
        const size_t     n_chunks     = (tile_count + reduce_chunk - 1) / reduce_chunk;
        if (static_cast<size_t>(ith) < s_count) {
            for (size_t c = static_cast<size_t>(ith); c < n_chunks; c += s_count) {
                const size_t t0 = c * reduce_chunk;
                const size_t t1 = std::min(tile_count, t0 + reduce_chunk);
                for (size_t t = t0; t < t1; ++t) {
                    const size_t      mt      = t / nt_count;
                    const size_t      nt      = t % nt_count;
                    const _Float16 *  tile    = partials + t * m_tile * s_count * n_tile;
                    const size_t      count_m = std::min(m_tile, M - mt * m_tile);
                    const size_t      count_n = std::min(n_tile, N - nt * n_tile);
                    auto *            out_tile = reinterpret_cast<float *>(out + (mt * m_tile) * out_row_stride) +
                                                nt * n_tile;
                    // rvv reducesum_mxn expects src rows at 64B*nth stride
                    // (one 32-f16 row per worker), which is exactly the
                    // partials layout above.
                    reduce_func((void *) tile, 0, 0, 0,
                                (void *) out_tile, 0,
                                static_cast<int64_t>(out_row_stride),  // out row stride bytes
                                1, 1, static_cast<int64_t>(s_count),   // src_mc, src_nc, nth
                                static_cast<int64_t>(count_m), static_cast<int64_t>(count_n));  // dst_mc, dst_nc
                }
            }
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s (type %s) with q4_0_q41block_f16\n", __func__, t->name, ggml_type_name(t->type));
        // The BLOC_TYPE layout is only produced from q4_0 weights today;
        // other source types would add their own repack here.
        return ggml::cpu::riscv64_spacemit::repack_q4_0_to_q41(t, data, data_size);
    }

    void unpack(const ggml_tensor * t, void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: unpack tensor %s from q4_0_q41block_f16\n", __func__, t->name);
        auto ok = ggml::cpu::riscv64_spacemit::unpack_q41_to_q4_0(t, data, data_size);
        GGML_ASSERT(ok == 0);
    }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
static const tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
static const tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
static const tensor_traits_f16<Q41Block<32, 32, 2>, 32, 32> q4_0_f16_impl;

// Q5_0 weights converted exactly to Q8_0 at load time and run on the Q8_0 IME path
class tensor_traits_q5_0_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_q5_0_to_q8_0_32(t, data, data_size);
    }
};

static const tensor_traits_q5_0_as_q8_0 q5_0_as_q8_0_32x32;
// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (ggml::cpu::riscv64_spacemit::q4_0_f16_repack_enabled(cur)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_f16_impl;
                }

                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_MXFP4:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // exact conversion to Q8_0 at load time, run on the Q8_0 IME path
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_as_q8_0_32x32;
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

size_t ggml_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
            break;
        case GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            if (ggml_spacemit_get_optimal_repack_type(tensor) == &ggml::cpu::riscv64_spacemit::q5_0_as_q8_0_32x32) {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(block_q8_0), 32);
            } else {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            }
            break;
        case GGML_TYPE_Q4_0:
            // Gated f16 path stores Q41Block<32,32,2> (608B) per 32 rows x K32
            // window = 19 bytes per row per K32 block, padded to 32 rows.
            nbytes = ggml::cpu::riscv64_spacemit::q4_0_f16_repack_enabled(tensor)
                         ? remap_block_nbytes(sizeof(block_q4_0), 19, 32)
                         : add_strided_nbytes(row_nbytes, 1, 1);
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}

bool ggml_spacemit_compute_opnode(ggml::spacemit::context & ctx, const spacemit_opnode & node) {
    // Single-op opnodes with a special opcode: the two halves of the concat -> ssm_conv fusion.
    // They run at their original graph positions (see try_fuse_node); the allocator plans buffer
    // reuse on that order, so they must not be merged into one opnode at the concat position.
    switch (node.opcode) {
        case SPACEMIT_OP_CONCAT_STATE:
            // only the trailing columns read back through views (the conv state update)
            if (!spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, node.node,
                                                                           (int64_t) node.kernel_params[0])) {
                GGML_ABORT("fused concat state update: unsupported shape");
            }
            return true;
        case SPACEMIT_OP_SSM_CONV_CONCAT:
            // the concat only materialized its trailing columns; read its inputs directly
            if (!spacemit_kernels::rvv::forward_ssm_conv_concat_f32(ctx, node.node)) {
                GGML_ABORT("fused ssm_conv: unsupported shape");
            }
            return true;
        default:
            break;
    }
    // Fused opnodes dispatch through the fused-op table; single-op opnodes keep
    // the existing traits path.
    if (!node.fused.empty()) {
        return ggml_spacemit_compute_fused(ctx, node);
    }
    return ggml_spacemit_compute_forward(ctx, node.node);
}

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor * op) {
    if (op == nullptr) {
        return nullptr;
    }

    const auto * common = &ggml::cpu::riscv64_spacemit::rvv_impl;
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 2 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                if (traits) return traits;
                const auto * repack = ggml_spacemit_get_optimal_repack_type(op->src[0]);
                if (repack) return repack;
                // F16 dense weight: same condition as compute_forward
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    return common;
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 3 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                return traits ? traits : ggml_spacemit_get_optimal_repack_type(op->src[0]);
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                return common;
            }
            break;
        case GGML_OP_SCALE:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) &&
                op->nb[0] == ggml_type_size(op->type) && ggml_can_repeat(op->src[1], op->src[0]) &&
                ggml_are_same_shape(op->src[0], op)) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        case GGML_OP_ROPE:
            if (op->src[0] && op->src[1]) {
                float freq_base;
                memcpy(&freq_base,  op->op_params + 5, sizeof(float));
                const int n_dims = ggml_get_op_params_i32(op, 1);
                const int mode   = ggml_get_op_params_i32(op, 2);
                const bool src2_ok = op->src[2] == nullptr ||
                    (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]));
                const bool mode_ok = (mode == GGML_ROPE_TYPE_NORMAL ||
                                      mode == GGML_ROPE_TYPE_NEOX   ||
                                      mode == GGML_ROPE_TYPE_MROPE  ||
                                      mode == GGML_ROPE_TYPE_IMROPE);
                const bool c1 = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                const bool c2 = op->type == op->src[0]->type;
                const bool c3 = op->src[1]->type == GGML_TYPE_I32;
                const bool c4 = src2_ok;
                const bool c5 = op->src[0]->nb[0] == ggml_type_size(op->src[0]->type);
                const bool c6 = op->nb[0] == ggml_type_size(op->type);
                const bool c7 = freq_base > 0.0f;
                const bool c8 = n_dims > 0 && n_dims <= op->ne[0] && n_dims <= 512 && n_dims % 2 == 0;
                const bool c9 = mode_ok;
                if (c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {
                    return common;
                }
            }
            break;
        case GGML_OP_CONT:
            if (op->src[0] && op->type == op->src[0]->type) {
                // permute case
                if (op->nb[0] != op->src[0]->nb[0] && op->nb[0] == op->src[0]->nb[1] &&
                    op->ne[3] * op->ne[2] * op->nb[2] == op->src[0]->ne[3] * op->src[0]->ne[2] * op->src[0]->nb[2]) {
                    return common;
                }
                // general stride-aware copy for F32/F16
                if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op) &&
                    ggml_nelements(op) == ggml_nelements(op->src[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_CPY:
            if (op->src[0] && ggml_nelements(op) == 0) {
                return common;
            }
            if (op->src[0] && op->type == op->src[0]->type && op->nb[0] == op->src[0]->nb[1] &&
                op->src[0]->nb[0] != op->src[0]->nb[1] && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            // row-contiguous F32 copy with arbitrary outer strides (GDN rollback-snapshot copies)
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                op->nb[0] == sizeof(float) && op->src[0]->nb[0] == sizeof(float) &&
                op->ne[0] == op->src[0]->ne[0] && op->ne[1] == op->src[0]->ne[1] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3]) {
                return common;
            }
            if (op->src[0] &&
                ((op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32) ||
                 (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F16)) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            break;
        case GGML_OP_SET_ROWS:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64) &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[2]->type == op->type &&
                ggml_are_same_shape(op, op->src[2]) && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[2]->nb[0] == ggml_type_size(op->src[2]->type) && op->nb[0] == ggml_type_size(op->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) && op->ne[0] == op->src[0]->ne[0] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3] &&
                op->src[0]->ne[2] % op->src[1]->ne[1] == 0 && op->src[0]->ne[3] % op->src[1]->ne[2] == 0) {
                return common;
            }
            break;
        case GGML_OP_REPEAT:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                const bool rows_equal = ggml_nrows(op->src[0]) == ggml_nrows(op);
                const bool n0_matches = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];
                if ((rows_equal && n0_matches) ||
                    (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_SUM_ROWS:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_GET_ROWS:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->type == op->src[0]->type &&
                op->src[1]->type == GGML_TYPE_I32 && op->ne[0] == op->src[0]->ne[0] &&
                op->src[0]->ne[2] == op->src[1]->ne[1] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                ggml_nrows(op) == ggml_nelements(op->src[1])) {
                return common;
            }
            break;
        case GGML_OP_CONCAT:
            if (op->src[0] && op->src[1] && ggml_get_op_params_i32(op, 0) == 0 &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->nb[0] == ggml_type_size(op->type) &&
                op->nb[1] == ggml_type_size(op->type) * (op->src[0]->ne[0] + op->src[1]->ne[0])) {
                return common;
            }
            break;
        case GGML_OP_UNARY:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                (ggml_get_unary_op(op) == GGML_UNARY_OP_TANH    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_GELU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SILU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_EXP     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_NEG     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS)) {
                return common;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float)) {
                return common;
            }
            break;
        case GGML_OP_GLU:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                if (ggml_get_glu_op(op) == GGML_GLU_OP_GEGLU) {
                    return common;
                }
                if (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_is_contiguous_1(op->src[0]) &&
                    op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float)) {
                    const int32_t swapped = ggml_get_op_params_i32(op, 1);
                    if (op->src[1]) {
                        if (swapped == 0 && op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(op->src[1]) &&
                            ggml_are_same_shape(op->src[0], op->src[1]) && ggml_are_same_shape(op, op->src[0])) {
                            return common;
                        }
                    } else if (op->ne[0] == op->src[0]->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(op->src[0])) {
                        return common;
                    }
                }
            }
            break;
        default:
            break;
    }

    // Scalar ops: route to common so compute_forward dispatches to scalar_kernels
    switch (op->op) {
        case GGML_OP_L2_NORM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_FILL:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16))
                return common;
            break;
        case GGML_OP_CUMSUM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_PAD:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_TRI:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]))
                return common;
            break;
        case GGML_OP_DIAG:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SET:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SOLVE_TRI:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_GATED_DELTA_NET:
            if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2) {
                return common;
            }
            break;
        case GGML_OP_SSM_CONV:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_ARGSORT:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(int32_t))
                return common;
            break;
        case GGML_OP_CLAMP:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && ggml_are_same_shape(op, op->src[0]))
                return common;
            break;
        case GGML_OP_MUL_MAT:
            // Generic CPU dot fallback for dense or unsupported quantized
            // layouts which do not have a repacked Spacemit trait.
            if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        default:
            break;
    }

    return nullptr;
}

int ggml_riscv64_spacemit_repack_tensor(ggml_tensor * tensor, const void * data, size_t size) {
    auto traits = (ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        return traits->repack(tensor, data, size);
    }
    memcpy(tensor->data, data, size);
    return 0;
}

void ggml_riscv64_spacemit_unpack_tensor(const ggml_tensor * tensor, void * data, size_t size) {
    auto traits = (const ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        traits->unpack(tensor, data, size);
        return;
    }
    memcpy(data, tensor->data, size);
}

bool ggml_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->work_size(n_threads, op, *size);
}

bool ggml_spacemit_compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->compute_forward(ctx, op);
}
