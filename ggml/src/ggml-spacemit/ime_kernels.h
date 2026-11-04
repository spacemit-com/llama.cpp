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

// Matmul interface suite.
//
// Entry names carry the interface facts only: the A/B payload bit widths at
// the interface boundary (i8i4 = int8 A payload x 4-bit B payload, both
// widened value-identically inside the kernel; i8i5 = 4-bit low + 1-bit high
// B mask merged inside; i8i2k/i8i3k = 2/3-bit k-quant B) and the repacked B
// granule (k32 = N32xK32 blocks, k256 = N32xK256 superblocks).  Layout
// details below that (B byte order, A scale/sum placement) are properties of
// the pack ops that produced the buffers, not of the entry name.
//
//   a_pack         how the A rows handed to an entry were packed, relative to
//                 the entry's family: r1/r4/r8 = row-block width, r4_al = the
//                 family's 4-row group layout in its aligned (SoA + alignment
//                 padding) form.  The caller passes the a_pack matching the
//                 pack op it used; the entry picks its leaf kernel from it.
//
//   family entry   gemm_<widths>_<granule>(a_pack, ..., count_m, ...) — takes
//                  count_m A rows, returns the row count consumed.
//
//   rows entry     gemm_<widths>_<granule>_rows(a_pack, ..., const uint8_t **
//                  a, ..., float ** c, ...) — routed rows arrive as a pointer
//                  array because each row lands in its own destination row.
//
//   leaf kernel    gemm_kernel_<family>_m<N> — one concrete register-blocking
//                  per row count; internal to ime2_kernels.cpp.  A leaf takes
//                  only the buffers, count_n, k_blks and (for m > 1) ldc: the
//                  row count, block length and zp presence are implied by the
//                  entry that dispatches to it.  (Exception: the f16 tile
//                  kernels and the Q41 GEMV leaf below are declared here
//                  because the f16 tensor traits in ime.cpp calls them
//                  directly.)
//
// A pack ops (rvv / ime1): quantize_a_<bits>_r<rows>[_<layout>] — e.g.
// quantize_a_i8_r4_hp packs 4 rows into the k256 hp family group layout.
// B repack ops: repack_<src>_<dst>_<granule> (repack.cpp).
enum class a_pack : int {
    r1,     // one row per call, natural row layout
    r4,     // 4-row group, the family's natural group layout
    r8,     // 8-row group (k256 hp aligned family only)
    r4_al,  // 4-row group, aligned (SoA) layout
};

using gemm_kernel_quantize_def = std::function<
    size_t(a_pack, size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t)>;

using gemm_rows_def = std::function<
    size_t(a_pack, size_t, const uint8_t **, const uint8_t *, const uint8_t *, float **, size_t, size_t, size_t, size_t)>;

namespace ime1 {
size_t gemm_i8i4_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

void quantize_a_i8_r1(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_i8_r4(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

}  // namespace ime1

namespace ime2 {
size_t gemm_i8i2k_k256(a_pack          ap,
                       size_t          blk_len,
                       const uint8_t * quant_a_ptr,
                       const uint8_t * quant_b_data,
                       const uint8_t * quant_b_zp,
                       float *         c_ptr,
                       size_t          count_m,
                       size_t          count_n,
                       size_t          k_blks,
                       size_t          ldc);

size_t gemm_i8i3k_k256(a_pack          ap,
                       size_t          blk_len,
                       const uint8_t * quant_a_ptr,
                       const uint8_t * quant_b_data,
                       const uint8_t * quant_b_zp,
                       float *         c_ptr,
                       size_t          count_m,
                       size_t          count_n,
                       size_t          k_blks,
                       size_t          ldc);

size_t gemm_i8i4_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

// Q4 family k256 entry: N32xK256 superblock B, serving the MoE expert
// tensors' r8/r4_al/r1 A groups (the dense Q4 MUL_MAT runs on the f16
// traits instead).  The zero-point pointer selects the B encoding:
// nullptr = Q4_0's offset-nibble encoding (the repack twists qs ^ 0x88),
// where the vmadot/vmadotus pair folds the fixed zero point into the i4
// dot product and skips it as a separate term; non-null = Q4_1 / Q4_K's
// bare-nibble encoding with a dedicated zero-point area (repacked from
// Q4_1 or Q4_K), where the vmadotsu/vmadotu dot pair runs on natural
// unsigned B nibbles and the zero-point correction is applied as an
// explicit term.
size_t gemm_i8i4_k256(a_pack          ap,
                      size_t          blk_len,
                      const uint8_t * quant_a_ptr,
                      const uint8_t * quant_b_data,
                      const uint8_t * quant_b_zp,
                      float *         c_ptr,
                      size_t          count_m,
                      size_t          count_n,
                      size_t          k_blks,
                      size_t          ldc);

size_t gemm_i8i8_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

size_t gemm_i8i5_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

size_t gemm_i8i4_k32_rows(a_pack           ap,
                         size_t           blk_len,
                         const uint8_t ** quant_a_ptr,
                         const uint8_t *  quant_b_data,
                         const uint8_t *  quant_b_zp,
                         float **         c_ptr,
                         size_t           count_m,
                         size_t           count_n,
                         size_t           k_blks,
                         size_t           ldc);

size_t gemm_i8i5_k32_rows(a_pack           ap,
                         size_t           blk_len,
                         const uint8_t ** quant_a_ptr,
                         const uint8_t *  quant_b_data,
                         const uint8_t *  quant_b_zp,
                         float **         c_ptr,
                         size_t           count_m,
                         size_t           count_n,
                         size_t           k_blks,
                         size_t           ldc);

// ---- f16 matmul kernels (dense Q4 MUL_MAT on the f16 compute path).  These
// are tile kernels called directly by the f16 tensor traits in ime.cpp, not
// family entries: count_m/count_n are the real extents of one M16xN32 tile
// (tail masking), not a row loop count. ----

// M1 i8i4 GEMV over Q41Block<32,32,2> (608B) for all three Q4 sources.
// Each row/K32 block has an explicit uint8 zero point (Q4_0: always 8).
// INT32 dot/correction is followed by FP32 scaling and accumulation.
void gemm_kernel_i8i4_m1_q41(const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           float *         c_ptr,
                           size_t          count_n,
                           size_t          k_blks);

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
