#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "scalar_kernels.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "vec.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace spacemit_kernels::scalar {

static void norm_f16_impl(ggml::spacemit::context & ctx, ggml_tensor * op, bool rms) {
    const ggml_tensor * src = op->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_are_same_shape(src, op) && src->nb[0] == sizeof(ggml_fp16_t));
    const int64_t n0 = src->ne[0], nr = src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t dr = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t r0 = dr * ctx.ith, r1 = std::min(r0 + dr, nr);
    float eps = ggml_get_op_params_f32(op, 0);
    for (int64_t ir = r0; ir < r1; ++ir) {
        const int64_t i3 = ir / (src->ne[2] * src->ne[1]);
        const int64_t i2 = (ir - i3 * src->ne[2] * src->ne[1]) / src->ne[1];
        const int64_t i1 = ir - i3 * src->ne[2] * src->ne[1] - i2 * src->ne[1];
        const auto * x = (const ggml_fp16_t *) ((const char *) src->data + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]);
        auto * y = (ggml_fp16_t *) ((char *) op->data + i1 * op->nb[1] + i2 * op->nb[2] + i3 * op->nb[3]);
        // Match ggml-cpu's two-pass normalization: calculate the row
        // statistics once, then apply the scale to every element.  Keeping
        // the mean outside the element loop is important for the relatively
        // wide F16 rows used by Gemma (the previous implementation was O(n0^2)).
        float mean = 0.0f;
        for (int64_t j = 0; j < n0; ++j) {
            const float v = ggml_fp16_to_fp32(x[j]);
            mean += rms ? v * v : v;
        }
        mean /= n0;
        const float row_mean = mean;
        float inv_std;
        if (rms) {
            inv_std = 1.0f / sqrtf(mean + eps);
        } else {
            float var = 0.0f;
            for (int64_t j = 0; j < n0; ++j) {
                const float v = ggml_fp16_to_fp32(x[j]) - row_mean;
                var += v * v;
            }
            inv_std = 1.0f / sqrtf(var / n0 + eps);
        }
        for (int64_t j = 0; j < n0; ++j) {
            float v = ggml_fp16_to_fp32(x[j]);
            if (!rms) v -= row_mean;
            y[j] = ggml_fp32_to_fp16(v * inv_std);
        }
    }
}

void forward_norm_f16(ggml::spacemit::context & ctx, ggml_tensor * op) { norm_f16_impl(ctx, op, false); }
void forward_rms_norm_f16(ggml::spacemit::context & ctx, ggml_tensor * op) { norm_f16_impl(ctx, op, true); }

// ── L2 NORM ──────────────────────────────────────────────────────────────────
void forward_l2_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));
    if (eps < 0.0f) eps = 0.0f;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb0  = op->nb[0],   nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr    = ne01 * ne02 * ne03;
    const int64_t dr    = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0   = dr * ctx.ith;
    const int64_t ir1   = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float *       y = (float *)((char *)op->data + i01*nb1 + i02*nb2 + i03*nb3);

        double sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) sum += (double)(x[i] * x[i]);
        const float scale = 1.0f / fmaxf(sqrtf((float)sum), eps);
        for (int64_t i = 0; i < ne00; ++i) y[i] = x[i] * scale;
    }
}

// ── FILL ─────────────────────────────────────────────────────────────────────
void forward_fill_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const float c = ggml_get_op_params_f32(op, 0);

    const int64_t ne0 = op->ne[0], ne1 = op->ne[1], ne2 = op->ne[2], ne3 = op->ne[3];
    const size_t  nb1 = op->nb[1], nb2 = op->nb[2], nb3 = op->nb[3];

    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
        float * dst_row = (float *)((char *)op->data + i3*nb3 + i2*nb2 + i1*nb1);
        for (int64_t i = 0; i < ne0; ++i) dst_row[i] = c;
    }
}

// ── CUMSUM ────────────────────────────────────────────────────────────────────
void forward_cumsum_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr  = ne01 * ne02 * ne03;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03*ne02*ne01) / ne01;
        const int64_t i01 = ir - i03*ne02*ne01 - i02*ne01;

        const float * src_row = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float *       dst_row = (float *)((char *)op->data + i01*nb1 + i02*nb2 + i03*nb3);

        float acc = 0.0f;
        for (int64_t i = 0; i < ne00; ++i) { acc += src_row[i]; dst_row[i] = acc; }
    }
}

// ── PAD ───────────────────────────────────────────────────────────────────────
void forward_pad_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t ne0 = op->ne[0], ne1 = op->ne[1], ne2 = op->ne[2], ne3 = op->ne[3];
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3*ne2*ne1) / ne1;
        const int64_t i1 = ir - i3*ne2*ne1 - i2*ne1;

        float * dst_row = (float *)((char *)op->data + i3*nb3 + i2*nb2 + i1*nb1);

        if (i3 < ne03 && i2 < ne02 && i1 < ne01) {
            const float * src_row = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
            const int64_t copy_n = std::min(ne0, ne00);
            for (int64_t i = 0; i < copy_n; ++i) dst_row[i] = src_row[i];
            for (int64_t i = copy_n; i < ne0; ++i) dst_row[i] = 0.0f;
        } else {
            for (int64_t i = 0; i < ne0; ++i) dst_row[i] = 0.0f;
        }
    }
}

// ── TRI ───────────────────────────────────────────────────────────────────────
void forward_tri_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(op, 0);

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3*ne2*ne1) / ne1;
        const int64_t i1 = ir - i3*ne2*ne1 - i2*ne1;  // row index

        const float * src_row = (const float *)src0->data + (i3*ne2*ne1 + i2*ne1 + i1) * ne0;
        float *       dst_row = (float *)op->data          + (i3*ne2*ne1 + i2*ne1 + i1) * ne0;

        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            bool keep;
            switch (ttype) {
                case GGML_TRI_TYPE_LOWER:      keep = i0 <  i1; break;
                case GGML_TRI_TYPE_LOWER_DIAG: keep = i0 <= i1; break;
                case GGML_TRI_TYPE_UPPER:      keep = i0 >  i1; break;
                case GGML_TRI_TYPE_UPPER_DIAG: keep = i0 >= i1; break;
                default: keep = false;
            }
            dst_row[i0] = keep ? src_row[i0] : 0.0f;
        }
    }
}

// ── DIAG ──────────────────────────────────────────────────────────────────────
void forward_diag_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    // src0 is 1D vector [n], dst is 2D [n,n]
    const int64_t n   = src0->ne[0];
    const int64_t dr  = (n + ctx.nth - 1) / ctx.nth;
    const int64_t i0  = dr * ctx.ith, i1 = std::min(i0 + dr, n);

    // zero the output first (only thread 0 to avoid races)
    if (ctx.ith == 0) {
        memset(op->data, 0, ggml_nbytes(op));
    }
    ctx.sync();

    const float * src = (const float *)src0->data;
    for (int64_t i = i0; i < i1; ++i) {
        float * dst_row = (float *)((char *)op->data + i * op->nb[1]);
        dst_row[i] = src[i];
    }
}

// ── SET ───────────────────────────────────────────────────────────────────────
void forward_set_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];  // destination tensor data (will be copied to output)
    const ggml_tensor * src1 = op->src[1];  // source of values to set

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    // op params: nb1, nb2, nb3, offset
    const size_t  nb1    = ((const int32_t *)op->op_params)[0];
    const size_t  nb2    = ((const int32_t *)op->op_params)[1];
    const size_t  nb3    = ((const int32_t *)op->op_params)[2];
    const size_t  offset = ((const int32_t *)op->op_params)[3];

    // copy src0 to dst (thread 0 only for simplicity)
    if (ctx.ith == 0) {
        if (op->data != src0->data) {
            memcpy(op->data, src0->data, ggml_nbytes(src0));
        }
    }
    ctx.sync();

    // now overlay src1 at the offset
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t  nb10 = src1->nb[0];

    const int64_t nr  = ne11 * ne12 * ne13;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne12 * ne11);
        const int64_t i2 = (ir - i3*ne12*ne11) / ne11;
        const int64_t i1 = ir - i3*ne12*ne11 - i2*ne11;

        const float * src_row = (const float *)((const char *)src1->data + i1*src1->nb[1] + i2*src1->nb[2] + i3*src1->nb[3]);
        float *       dst_row = (float *)((char *)op->data + offset + i1*nb1 + i2*nb2 + i3*nb3);

        for (int64_t i = 0; i < ne10; ++i) {
            dst_row[i] = src_row[i];
        }
    }
}

// Generic CPU MUL_MAT fallback. This follows ggml_compute_forward_mul_mat:
// source-1 is quantized to the source-0 dot type in the shared spert
// workspace, then the CPU type-trait dot kernel computes tiled output chunks.
void forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS;

    const auto * traits = ggml_get_type_traits_cpu(src0->type);
    GGML_ASSERT(traits && traits->vec_dot);
    const ggml_type vec_dot_type = traits->vec_dot_type;
    // The RHS must be converted to the type expected by vec_dot.  For
    // quantized weights this is normally Q8_0 (not src0's quant format).
    const auto * dot_traits = ggml_get_type_traits_cpu(vec_dot_type);
    GGML_ASSERT(dot_traits && dot_traits->from_float);
    const ggml_from_float_t from_float = dot_traits->from_float;
    const int64_t vec_dot_num_rows = traits->nrows;

    GGML_ASSERT(ne0 == ne01 && ne1 == ne11 && ne2 == ne12 && ne3 == ne13);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type) && nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb0 == sizeof(float) && nb0 <= nb1 && nb1 <= nb2 && nb2 <= nb3);

    const int ith = (int) ctx.ith;
    const int nth = (int) ctx.nth;
    const bool src1_cont = ggml_is_contiguous(src1);
    const size_t nbw0 = ggml_type_size(vec_dot_type);
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const size_t nbw1 = row_size;
    const size_t nbw2 = nbw1 * ne11;
    const size_t nbw3 = nbw2 * ne12;
    const size_t needed = (src1->type == vec_dot_type) ? 0 : ne13 * nbw3;
    GGML_ASSERT(ctx.workspace_size >= needed);

    char * wdata = (char *) ctx.workspace;
    if (src1->type != vec_dot_type) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    const size_t bs = ggml_blck_size(vec_dot_type);
                    const int64_t start = (ith * (ne10 / bs)) / nth;
                    const int64_t end = ((ith + 1) * (ne10 / bs)) / nth;
                    from_float((const float *) ((const char *) src1->data + i13 * nb13 + i12 * nb12 +
                                                i11 * nb11 + start * bs * nb10),
                               (void *) (wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1 + start * nbw0),
                               (end - start) * bs);
                }
            }
        }
    }
    ctx.sync();

    const int64_t nr0 = ne0;
    const int64_t nr1 = ne1 * ne2 * ne3;
    int chunk_size = (nr0 == 1 || nr1 == 1) ? 64 : 16;
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        nchunk0 = nr0 > nr1 ? nth : 1;
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    for (int64_t current = ith; current < nchunk0 * nchunk1; current += nth) {
        const int64_t ith0 = current % nchunk0;
        const int64_t ith1 = current / nchunk0;
        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = std::min(ir0_start + dr0, nr0);
        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = std::min(ir1_start + dr1, nr1);
        if (ir0_start >= ir0_end || ir1_start >= ir1_end) continue;

        int64_t rows_per_dot = vec_dot_num_rows;
        if ((nr0 % 2) || (ne11 % 2) || ((ir0_end - ir0_start) % 2) || ((ir1_end - ir1_start) % 2)) {
            rows_per_dot = 1;
        }
        const int64_t blck0 = 16;
        const int64_t blck1 = 16;
        float tmp[32];
        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck1) {
            for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck0) {
                for (int64_t row = iir1; row < iir1 + blck1 && row < ir1_end; row += rows_per_dot) {
                    const int64_t i13 = row / (ne12 * ne1);
                    const int64_t i12 = (row - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = row - i13 * ne12 * ne1 - i12 * ne1;
                    const int64_t i03 = i13 / r3;
                    const int64_t i02 = i12 / r2;
                    const char * src0_row = (const char *) src0->data + i02 * nb02 + i03 * nb03;
                    const void * wbase = src1->type == vec_dot_type ? src1->data : wdata;
                    const char * src1_col = (const char *) wbase +
                        (src1_cont || src1->type != vec_dot_type
                             ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                             : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                    float * dst_col = (float *) ((char *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);
                    for (int64_t col = iir0; col < iir0 + blck0 && col < ir0_end; col += rows_per_dot) {
                        traits->vec_dot(ne00, &tmp[col - iir0], rows_per_dot > 1 ? 16 : 0,
                                       src0_row + col * nb01, rows_per_dot > 1 ? nb01 : 0,
                                       src1_col, rows_per_dot > 1 ? src1_col_stride : 0, (int) rows_per_dot);
                    }
                    for (int cn = 0; cn < rows_per_dot; ++cn) {
                        memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + cn * 16,
                               (std::min(iir0 + blck0, ir0_end) - iir0) * sizeof(float));
                    }
                }
            }
        }
    }
}

// Reference online-softmax FlashAttention fallback.  This mirrors the CPU
// implementation but uses the spert context and supports F16/F32 Q, K and V.
void forward_flash_attn_ext(ggml::spacemit::context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(dst->op_params[3] == GGML_PREC_DEFAULT || dst->op_params[3] == GGML_PREC_F32);
    GGML_ASSERT((q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
                (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
                (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16));
    GGML_ASSERT(q->ne[0] == k->ne[0] && dst->ne[0] == v->ne[0]);

    const int64_t DK = k->ne[0], DV = v->ne[0], N = q->ne[1];
    const int64_t nr = q->ne[1] * q->ne[2] * q->ne[3];
    const int64_t dr = (nr + (int64_t) ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    float scale = ggml_get_op_params_f32(dst, 0);
    float max_bias = ggml_get_op_params_f32(dst, 1);
    float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t) q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2((double) n_head));
    const float m0 = powf(2.0f, -max_bias / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const int64_t rk2 = q->ne[2] / k->ne[2], rk3 = q->ne[3] / k->ne[3];
    const int64_t rv2 = q->ne[2] / v->ne[2], rv3 = q->ne[3] / v->ne[3];

    const size_t qsz = ggml_type_size(q->type), ksz = ggml_type_size(k->type), vsz = ggml_type_size(v->type);
    const size_t qnb1 = q->nb[1], qnb2 = q->nb[2], qnb3 = q->nb[3];
    const size_t knb1 = k->nb[1], knb2 = k->nb[2], knb3 = k->nb[3];
    const size_t vnb1 = v->nb[1], vnb2 = v->nb[2], vnb3 = v->nb[3];
    const size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];
    GGML_ASSERT(q->nb[0] == qsz && k->nb[0] == ksz && v->nb[0] == vsz);

    // This is the CPU backend's one-chunk algorithm, using the type-specific
    // vec_dot and conversion routines.  The previous implementation did the
    // K/Q and V accumulation element-by-element, which made DK=256 models
    // (Gemma/Qwen3.5) fall back to a several-orders-slower path.
    const ggml_type k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    const ggml_from_float_t q_to_vec_dot = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    const ggml_vec_dot_t kq_vec_dot = ggml_get_type_traits_cpu(k->type)->vec_dot;
    const ggml_to_float_t v_to_float = ggml_get_type_traits(v->type)->to_float;
    GGML_ASSERT(q_to_vec_dot && kq_vec_dot);
    GGML_ASSERT(v->type == GGML_TYPE_F32 || v_to_float);

    const int64_t stride = DK + 2 * DV + ggml::spacemit::cache_line_size_f32;
    GGML_ASSERT(ctx.workspace_size >= (size_t) ctx.nth * stride * sizeof(float));

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iq3 = ir / (q->ne[2] * q->ne[1]);
        const int64_t iq2 = (ir - iq3 * q->ne[2] * q->ne[1]) / q->ne[1];
        const int64_t iq1 = ir - iq3 * q->ne[2] * q->ne[1] - iq2 * q->ne[1];
        const uint32_t h = (uint32_t) iq2;
        const float slope = max_bias > 0.0f ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1)) : 1.0f;
        const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
        const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;
        const char * qdata = (const char *) q->data + iq1 * qnb1 + iq2 * qnb2 + iq3 * qnb3;
        const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]) : nullptr;
        // dst shape is [DV, n_heads, n_tokens, batch] (permute 0,2,1,3).
        float * out = (float *) ((char *) dst->data + iq2 * dnb1 + iq1 * dnb2 + iq3 * dnb3);
        float * VKQ32 = (float *) ctx.workspace + ctx.ith * stride;
        float * V32 = VKQ32 + DV;
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + DV);
        ggml_fp16_t * Q_q = (ggml_fp16_t *) (VKQ32 + 2 * DV);
        if (v->type == GGML_TYPE_F16) memset(VKQ16, 0, DV * sizeof(ggml_fp16_t));
        else memset(VKQ32, 0, DV * sizeof(float));
        GGML_ASSERT(q->type == GGML_TYPE_F32);
        q_to_vec_dot((const float *) qdata, Q_q, DK);
        float S = 0.0f, M = -INFINITY;
        for (int64_t ic = 0; ic < k->ne[1]; ++ic) {
            const float mv = mp ? slope * ggml_fp16_to_fp32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) continue;
            const char * kd = (const char *) k->data + ic * knb1 + ik2 * knb2 + ik3 * knb3;
            const char * vd = (const char *) v->data + ic * vnb1 + iv2 * vnb2 + iv3 * vnb3;
            float dot = 0.0f;
            kq_vec_dot((int) DK, &dot, 0, kd, 0, Q_q, 0, 1);
            float s = dot * scale; if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s); s += mv;
            const float Mold = M; float ms = 1.0f, vs = 1.0f;
            if (v->type == GGML_TYPE_F16) {
                if (s > M) { M = s; ms = expf(Mold - M); ggml_vec_scale_f16(DV, VKQ16, ms); }
                else vs = expf(s - M);
                ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) vd, vs);
            } else {
                if (s > M) { M = s; ms = expf(Mold - M); ggml_vec_scale_f32(DV, VKQ32, ms); }
                else vs = expf(s - M);
                if (v_to_float) { v_to_float(vd, V32, DV); ggml_vec_mad_f32(DV, VKQ32, V32, vs); }
                else ggml_vec_mad_f32(DV, VKQ32, (const float *) vd, vs);
            }
            S = S * ms + vs;
        }
        if (v->type == GGML_TYPE_F16) for (int64_t d = 0; d < DV; ++d) VKQ32[d] = ggml_fp16_to_fp32(VKQ16[d]);
        if (sinks) { const float ss = ((const float *) sinks->data)[h]; const float ms = ss > M ? expf(M - ss) : 1.0f; const float vs = ss > M ? 1.0f : expf(ss - M); if (ss > M) { M = ss; ggml_vec_scale_f32(DV, VKQ32, ms); } S = S * ms + vs; }
        const float inv = S == 0.0f ? 0.0f : 1.0f / S;
        ggml_vec_scale_f32(DV, VKQ32, inv);
        memcpy(out, VKQ32, DV * sizeof(float));
    }
    (void) N; (void) dnb3;
}

// ── SOLVE_TRI ─────────────────────────────────────────────────────────────────
// Forward/backward triangular solve — scalar fallback
void forward_solve_tri_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    // Only thread 0 executes; others skip and wait
    if (ctx.ith != 0) return;

    const ggml_tensor * src0 = op->src[0];  // triangular matrix A
    const ggml_tensor * src1 = op->src[1];  // right-hand side B

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int upper    = ggml_get_op_params_i32(op, 0);  // 1=upper, 0=lower
    const int transpose_A = ggml_get_op_params_i32(op, 1);

    const int64_t n = src0->ne[0];  // matrix size
    const int64_t nrhs = src1->ne[1];

    // copy src1 → dst first
    memcpy(op->data, src1->data, ggml_nbytes(src1));

    float * X = (float *)op->data;
    const float * A = (const float *)src0->data;

    // Simple triangular solve (forward substitution for lower, back for upper)
    for (int64_t j = 0; j < nrhs; ++j) {
        if (!upper && !transpose_A) {
            // lower triangular, no transpose
            for (int64_t i = 0; i < n; ++i) {
                float s = X[j * n + i];
                for (int64_t k = 0; k < i; ++k) s -= A[i * n + k] * X[j * n + k];
                X[j * n + i] = s / A[i * n + i];
            }
        } else {
            // upper triangular, no transpose (back substitution)
            for (int64_t i = n - 1; i >= 0; --i) {
                float s = X[j * n + i];
                for (int64_t k = i + 1; k < n; ++k) s -= A[i * n + k] * X[j * n + k];
                X[j * n + i] = s / A[i * n + i];
            }
        }
    }
}

// ── SSM CONV ──────────────────────────────────────────────────────────────────
// Ported from ggml_compute_forward_ssm_conv_f32, parallelised over d_inner rows.
void forward_ssm_conv_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0]; // conv_x  {d_conv-1+n_t, d_inner, n_seqs}
    const ggml_tensor * src1 = op->src[1]; // weight  {d_conv, d_inner}

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int nc  = (int)src1->ne[0]; // d_conv
    const int ncs = (int)src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = (int)src0->ne[1]; // d_inner
    const int n_t = (int)op->ne[1];   // tokens per sequence
    const int n_s = (int)op->ne[2];   // number of sequences

    GGML_ASSERT(op->ne[0] == nr);

    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min((int64_t)nr, ir0 + dr);

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *)((const char *)src0->data
                + ir0 * src0->nb[1] + i2 * src0->nb[0] + i3 * src0->nb[2]);
            const float * c = (const float *)((const char *)src1->data
                + ir0 * src1->nb[1]);
            float * x = (float *)((char *)op->data
                + ir0 * op->nb[0] + i2 * op->nb[1] + i3 * op->nb[2]);

            const int ir = (int)(ir1 - ir0);
            for (int i1 = 0; i1 < ir; ++i1) {
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1 * ncs] * c[i0 + i1 * nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

// ── GATED DELTA NET ───────────────────────────────────────────────────────────
// Ported from ggml_compute_forward_gated_delta_net_one_chunk.
// Parallelised over heads × sequences (ir = head_index + seq * H).
void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op) {
    ggml_tensor * src_q     = op->src[0];
    ggml_tensor * src_k     = op->src[1];
    ggml_tensor * src_v     = op->src[2];
    ggml_tensor * src_g     = op->src[3];
    ggml_tensor * src_beta  = op->src[4];
    ggml_tensor * src_state = op->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    const int64_t K = ggml_get_op_params_i32(op, 0);
    GGML_ASSERT(K >= 1);

    const int64_t nr  = H * n_seqs;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min(nr, ir0 + dr);

    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;

    float * attn_out_base  = (float *)op->data;
    float * state_out_base = (float *)op->data + attn_score_elems;
    const float * state_in_base = (const float *)src_state->data;

    const float scale = 1.0f / sqrtf((float)S_v);
    const bool kda = (src_g->ne[0] == S_v);

    // scratch buffer for delta (S_v floats) and optional state_work (S_v*S_v floats)
    std::vector<float> scratch((size_t)(S_v + (K > 1 ? S_v * S_v : 0)));
    float * delta      = scratch.data();
    float * state_work = K > 1 ? (delta + S_v) : nullptr;

    // local tensor nb helpers
    const size_t nbq1 = src_q->nb[1], nbq2 = src_q->nb[2], nbq3 = src_q->nb[3];
    const size_t nbk1 = src_k->nb[1], nbk2 = src_k->nb[2], nbk3 = src_k->nb[3];
    const size_t nbv1 = src_v->nb[1], nbv2 = src_v->nb[2], nbv3 = src_v->nb[3];
    const size_t nbg1 = src_g->nb[1], nbg2 = src_g->nb[2], nbg3 = src_g->nb[3];
    const size_t nbb1 = src_beta->nb[1], nbb2 = src_beta->nb[2], nbb3 = src_beta->nb[3];
    const int64_t neq1 = src_q->ne[1], neq3 = src_q->ne[3];
    const int64_t nek1 = src_k->ne[1], nek3 = src_k->ne[3];
    const int64_t nev3 = src_v->ne[3];
    const int64_t rq3  = nev3 / neq3;
    const int64_t rk3  = nev3 / nek3;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;   // head index
        const int64_t iv3 = ir / H;   // sequence index

        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, (size_t)(S_v * S_v) * sizeof(float));

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *)((const char *)src_q->data + iq3*nbq3 + t*nbq2 + iq1*nbq1);
            const float * k_d = (const float *)((const char *)src_k->data + ik3*nbk3 + t*nbk2 + ik1*nbk1);
            const float * v_d = (const float *)((const char *)src_v->data + iv3*nbv3 + t*nbv2 + iv1*nbv1);
            const float   beta_val = *(const float *)((const char *)src_beta->data + iv3*nbb3 + t*nbb2 + iv1*nbb1);
            const float * g_d      =  (const float *)((const char *)src_g->data    + iv3*nbg3 + t*nbg2 + iv1*nbg1);

            if (kda) {
                for (int64_t i = 0; i < S_v; ++i) delta[i] = expf(g_d[i]);
                for (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32((int) S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } else {
                float eg = expf(g_d[0]);
                ggml_vec_scale_f32((int) (S_v * S_v), s_out, eg);
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32((int) S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            for (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32((int) S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32((int) S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state_o = state_out_base + target_slot * state_size_per_snap
                                         + (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, (size_t)(S_v * S_v) * sizeof(float));
                }
            }
        }
    }
}

} // namespace spacemit_kernels::scalar
