#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
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

template <int N> struct nrow_block_q6_k_body {
    int8_t  scales[2 * N];
    uint8_t qh[8 * N];
    uint8_t ql[16 * N];
};

template <int N> struct nrow_super_q6_k {
    uint16_t                scales16[N];
    nrow_block_q6_k_body<N> sub[8];
};

static_assert(sizeof(nrow_block_q6_k_body<1>) == 26, "wrong q6_k dedup body size");
static_assert(sizeof(nrow_super_q6_k<1>) == 210, "wrong q6_k dedup super-tile size");
static_assert(sizeof(nrow_block_q6_k_body<32>) == 832, "wrong q6_k dedup N32 body size");
static_assert(sizeof(nrow_super_q6_k<32>) == 6720, "wrong q6_k dedup N32 super-tile size");

template <int N> struct nrow_block_mxfp4 {
    uint8_t e[N];
    uint8_t qh[4 * N];
    uint8_t qs[16 * N];
};
// q4_K with the fp16 scale de-duplicated ACROSS the K256 super-block.
//
// repack_q4_k_to_q4_1_32_bl computes, per (row, K32 sub-block j),
//     d1 = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src.d) * sc_j)
// where src.d is ONE fp16 per (row, K256) and sc_j is a 6-bit value from
// get_scale_min_k4(j, ...).  So the eight stored fp16 d1 values of a K256 are
// fully determined by one fp16 plus eight 6-bit integers, yet all eight are
// materialised: 8 x 64 B = 512 B per (32 cols x K256).
//
// Storing d_orig once and sc per sub-block, and rebuilding d1 in the kernel,
// costs 64 + 8*32 = 320 B instead of 512 B.  Combined with the nibble-packed zp
// this makes the super-tile 4544 B instead of 4736 B (zpn) / 4864 B (legacy),
// i.e. 4.4375 bits/weight -- BELOW the 4.5 of the GGUF's own block_q4_K, which
// is legitimate because zp needs only 4 bits where q4_K's min uses 6, and canon
// is defined by the d*(q-zp) approximation, not by exact q4_K semantics.
//
// Bit-exactness of the in-kernel rebuild is not assumed: it was proven by
// exhaustive comparison over all 65536 fp16 patterns x all sc in [0,63]
// (4,194,304 cases, 0 mismatches) on an A100 core -- see plan 9.49 item 4.
// d leads the super-tile so the DRAM first-touch order stays strictly
// sequential, per the design rule established in plan 9.48.
template <int N> struct __attribute__((packed)) q4_k_ds_sub {
    uint8_t sc[N];
    uint8_t zp4[N / 2];
    uint8_t qs[16 * N];
};

template <int N> struct __attribute__((packed)) nrow_super_q4_k_ds {
    uint16_t         d[N];
    q4_k_ds_sub<N>   sub[8];
};

static_assert(sizeof(q4_k_ds_sub<32>) == 560, "wrong q4_k ds sub-block size");
static_assert(sizeof(nrow_super_q4_k_ds<32>) == 4544, "wrong q4_k ds super-tile size");

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
// Q4 PP weights are expanded to signed Q8 once at repack time. Each block
// contains N output columns and one K slice in the IME2 dot layout.
template <int K, int N> struct q4_pp_weight_block {
    static_assert(K > 0 && N > 0);
    uint16_t d[N];
    int8_t   qs[K * N];
};

// One M x K activation tile; duplicated scales match the IME2 dot lanes.
template <int K, int M> struct q4_pp_activation_block {
    static_assert(K > 0 && K % 4 == 0 && M > 0);
    float  scale[(K / 4) * M];
    int8_t qs[K * M];
};

using q4_pp_block_b = q4_pp_weight_block<32, 32>;
using q4_pp_block_a = q4_pp_activation_block<32, 8>;

static_assert(sizeof(q4_pp_block_b) == 1088, "wrong Q4 PP B block size");
static_assert(offsetof(q4_pp_block_b, qs) == 64, "wrong Q4 PP B layout");
static_assert(sizeof(q4_pp_block_a) == 512, "wrong Q4 PP A block size");
static_assert(offsetof(q4_pp_block_a, qs) == 256, "wrong Q4 PP A layout");

void quantize_a_m8_ime(const float * source, size_t k, q4_pp_block_a * destination);
void gemm_m8_q8_ab_prepacked(const q4_pp_block_a * packed_a,
                             const q4_pp_block_b * packed_b,
                             float *                 output,
                             size_t                  k_blocks,
                             size_t                  ldc);

size_t gemm_kernel_i8i6k(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc);

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

size_t gemm_kernel_i8i4_legacy(size_t          blk_len,
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
}  // namespace ime2
}  // namespace spacemit_kernels
