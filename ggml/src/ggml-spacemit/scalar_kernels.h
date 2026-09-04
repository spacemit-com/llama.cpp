#pragma once

#include "spacemit-context.h"
#include "ggml.h"

namespace spacemit_kernels::scalar {

void forward_l2_norm_f32   (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_fill_f32      (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_cumsum_f32    (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_pad_f32       (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_tri_f32       (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_diag_f32      (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_set_f32       (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_mul_mat       (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_flash_attn_ext(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_norm_f16       (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_rms_norm_f16   (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_solve_tri_f32 (ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_ssm_conv_f32  (ggml::spacemit::context & ctx, ggml_tensor * op);

} // namespace spacemit_kernels::scalar
