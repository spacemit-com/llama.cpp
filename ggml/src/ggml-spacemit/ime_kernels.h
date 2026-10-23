#pragma once

#include <cassert>
#include <cstddef>
#include <functional>

namespace spacemit_kernels {

#define BLOCK_QNK_LEN 256

template <int N> struct nrow_block_q2_k {
    // [4bit scale + 4bit zp] * N * 16
    uint8_t  scales[N * BLOCK_QNK_LEN / 16];
    // [b0, b16, b32, b48] [b1, b17, b33, b49] ... [b15, b31, b47, b63]
    // [b64, b80, b96, b112] ...[b79, b95, b111, b127]
    // [b128, b144, b160, b176] ...[b143, b159, b175, b191]
    // [b192, b208, b224, b240] ...[b207, b223, b239, b255]
    uint8_t  qs[N * BLOCK_QNK_LEN / 4];
    uint16_t scales16[N];
    uint16_t zeros16[N];
};

template <int N> struct nrow_block_q3_k {
    // [8bit scale] * N * 16
    int8_t   scales[N * 16];
    // [b0, b1, b2, b3, b4, b5, b6, b7] ... [b248, b249, b250, b251, b252, b253, b254, b255]
    uint8_t  hmask[N * BLOCK_QNK_LEN / 8];
    // [b0, b16, b32, b48] [b1, b17, b33, b49] ... [b15, b31, b47, b63]
    // [b64, b80, b96, b112] ...[b79, b95, b111, b127]
    // [b128, b144, b160, b176] ...[b143, b159, b175, b191]
    // [b192, b208, b224, b240] ...[b207, b223, b239, b255]
    uint8_t  qs[N * BLOCK_QNK_LEN / 4];
    uint16_t scales16[N];
};

template <int N> struct nrow_block_mxfp4 {
    uint8_t e[N];
    uint8_t qh[4 * N];
    uint8_t qs[16 * N];
};

template <int N> struct __attribute__((packed)) nrow_block_q5_1 {
    uint16_t scales16[N];
    uint8_t  zp[N];
    // n0 [bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7] ....
    uint8_t  qh[4 * N];
    // n0 [b0, b1], [b2, b3] ....  [b30, b31]
    // n1 [b0, b1], [b2, b3] ....  [b30, b31]
    uint8_t  qs[16 * N];
};

static_assert(sizeof(nrow_block_q5_1<1>) == sizeof(uint8_t) + 22, "wrong nrow_block_q5_1 block size/padding");

template <int N> struct __attribute__((packed)) nrow_block_q5_0 {
    uint16_t scales16[N];
    // n0 [bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7] ....
    uint8_t  qh[4 * N];
    // n0 [b0, b1], [b2, b3] ....  [b30, b31]
    // n1 [b0, b1], [b2, b3] ....  [b30, b31]
    uint8_t  qs[16 * N];
};

static_assert(sizeof(nrow_block_q5_0<1>) == 22, "wrong nrow_block_q5_0 block size/padding");

using gemm_kernel_quantize_def = std::function<
    size_t(size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t)>;

using moe_gemm_kernel_quantize_def = std::function<
    size_t(size_t, const uint8_t **, const uint8_t *, const uint8_t *, float **, size_t, size_t, size_t, size_t)>;

namespace ime1 {
size_t gemm_kernel_i8i4(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc);

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

}  // namespace ime1

namespace ime2 {
size_t gemm_kernel_i8i2k(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc);

size_t gemm_kernel_i8i3k(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc);

size_t gemm_kernel_i8i4(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc);

size_t gemm_kernel_i8i4_hp(size_t          blk_len,
                           const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float *         c_ptr,
                           size_t          count_m,
                           size_t          count_n,
                           size_t          k_blks,
                           size_t          ldc);

// M1 i8i4 GEMV over spine-kernel's Q41Block<32,32,2> weight layout (the
// q4_0_f16 repack): B data sits at block+0 and the per-row fp16 scales at
// block+544 (608B blocks, 19B per-row pitch) instead of the legacy
// block_q4_0x32 [scale@+0 | data@+64] (576B, 18B pitch).  zp=8 is implicit.
size_t gemm_kernel_i8i4_m1_q41(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc);

size_t moe_m2_gemm_kernel_i8i4(size_t           blk_len,
                               const uint8_t ** quant_a_ptr,
                               const uint8_t *  quant_b_data,
                               const uint8_t *  quant_b_zp,
                               float **         c_ptr,
                               size_t           count_m,
                               size_t           count_n,
                               size_t           k_blks,
                               size_t           ldc);

size_t gemm_kernel_i8i8(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc);

size_t gemm_kernel_i8mxfp4(size_t          blk_len,
                           const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float *         c_ptr,
                           size_t          count_m,
                           size_t          count_n,
                           size_t          k_blks,
                           size_t          ldc);

size_t moe_m2_gemm_kernel_i8mxfp4(size_t           blk_len,
                                  const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_m,
                                  size_t           count_n,
                                  size_t           k_blks,
                                  size_t           ldc);

size_t gemm_kernel_i8i5(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc);

size_t moe_m2_gemm_kernel_i8i5(size_t           blk_len,
                               const uint8_t ** quant_a_ptr,
                               const uint8_t *  quant_b_data,
                               const uint8_t *  quant_b_zp,
                               float **         c_ptr,
                               size_t           count_m,
                               size_t           count_n,
                               size_t           k_blks,
                               size_t           ldc);
// f16 GEMM inner-kernel wrapper around spine-kernel's _spe_hpgemm_impl_m16n32k8_none.
// a/b are the native [K/8][M16|N32][K8] f16 layouts, gemm_k is the K extent,
// ldc counts f16 rows of the output tile.
void gemm_kernel_f16f16f16_m16_vfwmadot(size_t           gemm_k,
                                        const _Float16 * b_f16_k8,
                                        _Float16 *       a_f16_k8,
                                        _Float16 *       c_ptr,
                                        size_t           count_m,
                                        size_t           count_n,
                                        size_t           ldc,
                                        bool             accumulate);

// f16xf16 -> f32 macro tile with a K-accumulation epilogue, used by the
// M-split schedule (whole K loop inside one call, C written straight to the
// f32 output).  a/b are the same native [K/8][M16|N32][K8] f16 layouts;
// accumulate=true seeds the accumulators from c_ptr's rows first, ldc counts
// f32 rows of the output tile.
void gemm_kernel_f16f16f32_m16_vfwmadot(size_t           gemm_k,
                                        const _Float16 * b_f16_k8,
                                        _Float16 *       a_f16_k8,
                                        float *          c_ptr,
                                        size_t           count_m,
                                        size_t           count_n,
                                        size_t           ldc,
                                        bool             accumulate = false);

}  // namespace ime2
}  // namespace spacemit_kernels
