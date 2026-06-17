#include "smt-vision-preprocess.h"

#include "stb/stb_image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static bool contains_icase(const std::string & text, const std::string & pattern) {
    return to_lower_ascii(text).find(to_lower_ascii(pattern)) != std::string::npos;
}

struct ep_preproc_spec {
    int32_t target_w                    = 0;
    int32_t target_h                    = 0;
    bool    normalize_to_01             = false;
    bool    quantize_to_u8_after_resize = false;
    bool    apply_rescale_and_normalize = false;
    bool    qwen2vl_patch_flatten       = false;
};

static ep_preproc_spec resolve_preproc_spec(const std::string & architecture) {
    // Qwen2VL ONNX exported from model.visual expects the processor's flattened
    // patch tensor: [grid_h * grid_w, 3 * temporal_patch_size * 14 * 14].
    if (contains_icase(architecture, "qwen2vl") || contains_icase(architecture, "qwen2_vl")) {
        return { /* target_w */ 0, /* target_h */ 0, /* normalize_to_01 */ false, /* quantize */ true,
                 /* apply */ true, /* qwen2vl_patch_flatten */ true };
    }

    // Qwen3VL SMT ONNX keeps internal (x - 127.5) / 127.5 preprocessing.
    if (contains_icase(architecture, "qwen3vl")) {
        return { /* target_w */ 768, /* target_h */ 768, /* normalize_to_01 */ false, /* quantize */ true,
                 /* apply */ false };
    }

    // FastVLM (LlavaQwen2ForCausalLM) expects 0..1 normalized CHW float32.
    if (contains_icase(architecture, "llavaqwen2forcausallm") || contains_icase(architecture, "llavaqwen2") ||
        contains_icase(architecture, "fastvlm")) {
        return { /* target_w */ 512, /* target_h */ 512, /* normalize_to_01 */ true, /* quantize */ true,
                 /* apply */ false };
    }

    if (contains_icase(architecture, "paddleocr")) {
        return { /* target_w */ 0, /* target_h */ 0, /* normalize_to_01 */ false, /* quantize */ true,
                 /* apply */ true };
    }

    return {};
}

static ep_preproc_spec resolve_preproc_spec(const std::string & architecture,
                                            int32_t             input_width,
                                            int32_t             input_height) {
    auto spec = resolve_preproc_spec(architecture);
    if (input_width > 0 && input_height > 0) {
        spec.target_w = input_width;
        spec.target_h = input_height;
    }
    return spec;
}

struct linear_contrib {
    std::vector<int32_t> idx;
    std::vector<float>   w;
};

static std::vector<linear_contrib> precompute_linear_contrib(int32_t in_size, int32_t out_size) {
    if (in_size <= 0 || out_size <= 0) {
        throw std::runtime_error("Invalid resize dimensions");
    }

    const float scale        = (float) in_size / (float) out_size;
    const float scale_factor = scale > 1.0f ? scale : 1.0f;
    const float support      = scale > 1.0f ? scale : 1.0f;

    std::vector<linear_contrib> table((size_t) out_size);
    for (int32_t o = 0; o < out_size; ++o) {
        const float   center = ((float) o + 0.5f) * scale - 0.5f;
        const int32_t left   = (int32_t) std::ceil(center - support);
        const int32_t right  = (int32_t) std::floor(center + support);

        auto & c = table[(size_t) o];
        c.idx.reserve((size_t) std::max(0, right - left + 1));
        c.w.reserve((size_t) std::max(0, right - left + 1));

        float w_sum = 0.0f;
        for (int32_t i = left; i <= right; ++i) {
            const float x = ((float) i - center) / scale_factor;
            const float w = std::max(0.0f, 1.0f - std::fabs(x));
            c.idx.push_back(std::clamp(i, 0, in_size - 1));
            c.w.push_back(w);
            w_sum += w;
        }

        if (w_sum > 0.0f) {
            for (auto & w : c.w) {
                w /= w_sum;
            }
        }
    }

    return table;
}

static std::vector<uint8_t> resize_rgb_u8_antialias(const uint8_t * src,
                                                    int32_t         src_w,
                                                    int32_t         src_h,
                                                    int32_t         dst_w,
                                                    int32_t         dst_h,
                                                    bool            quantize_u8) {
    if (src == nullptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("Invalid image dimensions");
    }

    const auto contrib_x = precompute_linear_contrib(src_w, dst_w);
    const auto contrib_y = precompute_linear_contrib(src_h, dst_h);

    std::vector<float> tmp((size_t) src_h * (size_t) dst_w * 3, 0.0f);
    for (int32_t y = 0; y < src_h; ++y) {
        for (int32_t x = 0; x < dst_w; ++x) {
            const auto & c = contrib_x[(size_t) x];
            for (size_t k = 0; k < c.idx.size(); ++k) {
                const int32_t sx    = c.idx[k];
                const float   w     = c.w[k];
                const size_t  s_idx = ((size_t) y * (size_t) src_w + (size_t) sx) * 3;
                const size_t  d_idx = ((size_t) y * (size_t) dst_w + (size_t) x) * 3;
                tmp[d_idx + 0] += (float) src[s_idx + 0] * w;
                tmp[d_idx + 1] += (float) src[s_idx + 1] * w;
                tmp[d_idx + 2] += (float) src[s_idx + 2] * w;
            }
        }
    }

    std::vector<uint8_t> out((size_t) dst_h * (size_t) dst_w * 3, 0);
    for (int32_t y = 0; y < dst_h; ++y) {
        const auto & c = contrib_y[(size_t) y];
        for (int32_t x = 0; x < dst_w; ++x) {
            float acc[3] = { 0.0f, 0.0f, 0.0f };
            for (size_t k = 0; k < c.idx.size(); ++k) {
                const int32_t sy    = c.idx[k];
                const float   w     = c.w[k];
                const size_t  s_idx = ((size_t) sy * (size_t) dst_w + (size_t) x) * 3;
                acc[0] += tmp[s_idx + 0] * w;
                acc[1] += tmp[s_idx + 1] * w;
                acc[2] += tmp[s_idx + 2] * w;
            }

            const size_t d_idx = ((size_t) y * (size_t) dst_w + (size_t) x) * 3;
            for (int c_id = 0; c_id < 3; ++c_id) {
                float v                    = quantize_u8 ? std::round(acc[c_id]) : acc[c_id];
                v                          = std::clamp(v, 0.0f, 255.0f);
                out[d_idx + (size_t) c_id] = (uint8_t) v;
            }
        }
    }

    return out;
}


static std::vector<uint8_t> rgba_u8_to_rgb_u8_white(const uint8_t * src, int32_t w, int32_t h) {
    if (src == nullptr || w <= 0 || h <= 0) {
        throw std::runtime_error("Invalid RGBA image dimensions");
    }

    std::vector<uint8_t> out((size_t) w * (size_t) h * 3u, 255);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t src_idx = ((size_t) y * (size_t) w + (size_t) x) * 4u;
            const size_t dst_idx = ((size_t) y * (size_t) w + (size_t) x) * 3u;
            const uint32_t a = src[src_idx + 3];
            for (int32_t c = 0; c < 3; ++c) {
                const uint32_t v = (uint32_t) src[src_idx + (size_t) c];
                out[dst_idx + (size_t) c] = (uint8_t) ((v * a + 255u * (255u - a) + 127u) / 255u);
            }
        }
    }
    return out;
}

static std::vector<uint8_t> resize_rgb_u8_pillow_bicubic(const std::vector<uint8_t> & src,
                                                         int32_t                      src_w,
                                                         int32_t                      src_h,
                                                         int32_t                      dst_w,
                                                         int32_t                      dst_h) {
    if (src.size() != (size_t) src_w * (size_t) src_h * 3u || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("Invalid Pillow bicubic resize dimensions");
    }
    if (src_w == dst_w && src_h == dst_h) {
        return src;
    }

    constexpr int    precision_bits = 32 - 8 - 2;
    constexpr double filter_support = 2.0;

    auto bicubic_filter = [](double x) -> double {
        constexpr double a = -0.5;
        if (x < 0.0) {
            x = -x;
        }
        if (x < 1.0) {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        }
        if (x < 2.0) {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        }
        return 0.0;
    };

    auto clip8 = [](int32_t value) -> uint8_t {
        if (value < 0) {
            return 0;
        }
        if (value > 255) {
            return 255;
        }
        return (uint8_t) value;
    };

    auto precompute_weights = [&](int32_t in_size, int32_t out_size,
                                  std::vector<int32_t> & bounds,
                                  std::vector<int32_t> & weights) -> int32_t {
        const double scale = (double) in_size / (double) out_size;
        const double filterscale = std::max(1.0, scale);
        const double support = filter_support * filterscale;
        const int32_t ksize = (int32_t) std::ceil(support) * 2 + 1;
        const double ss = 1.0 / filterscale;
        const double fxp_scale = std::ldexp(1.0, precision_bits);

        bounds.resize((size_t) out_size * 2u);
        weights.assign((size_t) out_size * (size_t) ksize, 0);

        for (int32_t out = 0; out < out_size; ++out) {
            const double center = ((double) out + 0.5) * scale;
            int32_t xmin = (int32_t) (center - support + 0.5);
            int32_t xmax = (int32_t) (center + support + 0.5);
            xmin = std::max(0, xmin);
            xmax = std::min(in_size, xmax);
            const int32_t count = xmax - xmin;

            bounds[(size_t) out * 2u + 0u] = xmin;
            bounds[(size_t) out * 2u + 1u] = count;

            double weight_sum = 0.0;
            std::vector<double> tmp((size_t) ksize, 0.0);
            for (int32_t k = 0; k < count; ++k) {
                const double w = bicubic_filter(((double) k + (double) xmin - center + 0.5) * ss);
                tmp[(size_t) k] = w;
                weight_sum += w;
            }
            if (weight_sum != 0.0) {
                for (int32_t k = 0; k < count; ++k) {
                    tmp[(size_t) k] /= weight_sum;
                }
            }
            for (int32_t k = 0; k < ksize; ++k) {
                double v = tmp[(size_t) k] * fxp_scale;
                v += tmp[(size_t) k] < 0.0 ? -0.5 : 0.5;
                v = std::round(v);
                v = std::clamp(v, (double) std::numeric_limits<int32_t>::min(), (double) std::numeric_limits<int32_t>::max());
                weights[(size_t) out * (size_t) ksize + (size_t) k] = (int32_t) v;
            }
        }
        return ksize;
    };

    auto resample_horizontal = [&](const std::vector<uint8_t> & input,
                                   std::vector<uint8_t> &       output,
                                   int32_t                      in_w,
                                   int32_t                      in_h,
                                   int32_t                      out_w,
                                   int32_t                      ksize,
                                   const std::vector<int32_t> & bounds,
                                   const std::vector<int32_t> & weights) {
        output.resize((size_t) out_w * (size_t) in_h * 3u);
        for (int32_t y = 0; y < in_h; ++y) {
            for (int32_t x = 0; x < out_w; ++x) {
                const int32_t xmin = bounds[(size_t) x * 2u + 0u];
                const int32_t count = bounds[(size_t) x * 2u + 1u];
                int32_t acc[3] = { 1 << (precision_bits - 1), 1 << (precision_bits - 1), 1 << (precision_bits - 1) };
                for (int32_t k = 0; k < count; ++k) {
                    const size_t src_idx = ((size_t) y * (size_t) in_w + (size_t) (xmin + k)) * 3u;
                    const int32_t w = weights[(size_t) x * (size_t) ksize + (size_t) k];
                    acc[0] += (int32_t) input[src_idx + 0u] * w;
                    acc[1] += (int32_t) input[src_idx + 1u] * w;
                    acc[2] += (int32_t) input[src_idx + 2u] * w;
                }
                const size_t dst_idx = ((size_t) y * (size_t) out_w + (size_t) x) * 3u;
                output[dst_idx + 0u] = clip8(acc[0] >> precision_bits);
                output[dst_idx + 1u] = clip8(acc[1] >> precision_bits);
                output[dst_idx + 2u] = clip8(acc[2] >> precision_bits);
            }
        }
    };

    auto resample_vertical = [&](const std::vector<uint8_t> & input,
                                 std::vector<uint8_t> &       output,
                                 int32_t                      in_w,
                                 int32_t                      in_h,
                                 int32_t                      out_h,
                                 int32_t                      ksize,
                                 const std::vector<int32_t> & bounds,
                                 const std::vector<int32_t> & weights) {
        output.resize((size_t) in_w * (size_t) out_h * 3u);
        for (int32_t y = 0; y < out_h; ++y) {
            const int32_t ymin = bounds[(size_t) y * 2u + 0u];
            const int32_t count = bounds[(size_t) y * 2u + 1u];
            for (int32_t x = 0; x < in_w; ++x) {
                int32_t acc[3] = { 1 << (precision_bits - 1), 1 << (precision_bits - 1), 1 << (precision_bits - 1) };
                for (int32_t k = 0; k < count; ++k) {
                    const size_t src_idx = ((size_t) (ymin + k) * (size_t) in_w + (size_t) x) * 3u;
                    const int32_t w = weights[(size_t) y * (size_t) ksize + (size_t) k];
                    acc[0] += (int32_t) input[src_idx + 0u] * w;
                    acc[1] += (int32_t) input[src_idx + 1u] * w;
                    acc[2] += (int32_t) input[src_idx + 2u] * w;
                }
                const size_t dst_idx = ((size_t) y * (size_t) in_w + (size_t) x) * 3u;
                output[dst_idx + 0u] = clip8(acc[0] >> precision_bits);
                output[dst_idx + 1u] = clip8(acc[1] >> precision_bits);
                output[dst_idx + 2u] = clip8(acc[2] >> precision_bits);
            }
        }
    };

    std::vector<int32_t> bounds_x;
    std::vector<int32_t> bounds_y;
    std::vector<int32_t> weights_x;
    std::vector<int32_t> weights_y;
    const bool need_x = src_w != dst_w;
    const bool need_y = src_h != dst_h;
    const int32_t ksize_x = need_x ? precompute_weights(src_w, dst_w, bounds_x, weights_x) : 0;
    const int32_t ksize_y = need_y ? precompute_weights(src_h, dst_h, bounds_y, weights_y) : 0;

    if (need_x && need_y) {
        std::vector<uint8_t> tmp;
        resample_horizontal(src, tmp, src_w, src_h, dst_w, ksize_x, bounds_x, weights_x);
        std::vector<uint8_t> out;
        resample_vertical(tmp, out, dst_w, src_h, dst_h, ksize_y, bounds_y, weights_y);
        return out;
    }
    if (need_x) {
        std::vector<uint8_t> out;
        resample_horizontal(src, out, src_w, src_h, dst_w, ksize_x, bounds_x, weights_x);
        return out;
    }
    std::vector<uint8_t> out;
    resample_vertical(src, out, src_w, src_h, dst_h, ksize_y, bounds_y, weights_y);
    return out;
}

static std::vector<float> rgb_u8_to_chw_f32(const std::vector<uint8_t> & src,
                                            int32_t                      w,
                                            int32_t                      h,
                                            bool                         normalize_to_01) {
    const size_t plane = (size_t) w * (size_t) h;
    if (src.size() != plane * 3) {
        throw std::runtime_error("Invalid RGB tensor size");
    }

    std::vector<float> out(plane * 3, 0.0f);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t src_idx = ((size_t) y * (size_t) w + (size_t) x) * 3;
            const size_t dst_idx = (size_t) y * (size_t) w + (size_t) x;
            for (int32_t c = 0; c < 3; ++c) {
                float v = (float) src[src_idx + (size_t) c];
                if (normalize_to_01) {
                    v /= 255.0f;
                }
                out[(size_t) c * plane + dst_idx] = v;
            }
        }
    }
    return out;
}

static std::vector<float> rgb_u8_to_chw_f32_with_config(const std::vector<uint8_t> &         src,
                                                        int32_t                              w,
                                                        int32_t                              h,
                                                        const smt_vision_preprocess_config & config) {
    const size_t plane = (size_t) w * (size_t) h;
    if (src.size() != plane * 3) {
        throw std::runtime_error("Invalid RGB tensor size");
    }

    std::vector<float> out(plane * 3, 0.0f);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t src_idx = ((size_t) y * (size_t) w + (size_t) x) * 3;
            const size_t dst_idx = (size_t) y * (size_t) w + (size_t) x;
            for (int32_t c = 0; c < 3; ++c) {
                const float denom                 = config.image_std[c] == 0.0f ? 1.0f : config.image_std[c];
                float       v                     = (float) src[src_idx + (size_t) c];
                v                                 = v * config.rescale_factor;
                v                                 = (v - config.image_mean[c]) / denom;
                out[(size_t) c * plane + dst_idx] = v;
            }
        }
    }
    return out;
}

static std::vector<float> rgb_u8_to_qwen2vl_patch_f32_with_config(
    const std::vector<uint8_t> &         src,
    int32_t                              w,
    int32_t                              h,
    const smt_vision_preprocess_config & config) {
    constexpr int32_t patch_size          = 14;
    constexpr int32_t temporal_patch_size = 2;
    constexpr int32_t merge_size          = 2;
    constexpr int32_t channels            = 3;

    if (w <= 0 || h <= 0 || w % patch_size != 0 || h % patch_size != 0) {
        throw std::runtime_error("Qwen2VL image dimensions must be positive multiples of patch_size");
    }
    if ((w / patch_size) % merge_size != 0 || (h / patch_size) % merge_size != 0) {
        throw std::runtime_error("Qwen2VL image grid must be divisible by merge_size");
    }

    const size_t plane = (size_t) w * (size_t) h;
    if (src.size() != plane * channels) {
        throw std::runtime_error("Invalid RGB tensor size");
    }

    std::vector<float> chw(plane * channels, 0.0f);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t src_idx = ((size_t) y * (size_t) w + (size_t) x) * channels;
            const size_t dst_idx = (size_t) y * (size_t) w + (size_t) x;
            for (int32_t c = 0; c < channels; ++c) {
                const float denom = config.image_std[c] == 0.0f ? 1.0f : config.image_std[c];
                float       v     = (float) src[src_idx + (size_t) c];
                v                 = v * config.rescale_factor;
                v                 = (v - config.image_mean[c]) / denom;
                chw[(size_t) c * plane + dst_idx] = v;
            }
        }
    }

    const int32_t grid_h       = h / patch_size;
    const int32_t grid_w       = w / patch_size;
    const int32_t grid_h_group = grid_h / merge_size;
    const int32_t grid_w_group = grid_w / merge_size;
    const size_t  row_size     = (size_t) channels * temporal_patch_size * patch_size * patch_size;

    std::vector<float> out((size_t) grid_h * (size_t) grid_w * row_size, 0.0f);
    size_t             row = 0;
    for (int32_t ghg = 0; ghg < grid_h_group; ++ghg) {
        for (int32_t gwg = 0; gwg < grid_w_group; ++gwg) {
            for (int32_t mh = 0; mh < merge_size; ++mh) {
                for (int32_t mw = 0; mw < merge_size; ++mw) {
                    size_t col = 0;
                    for (int32_t c = 0; c < channels; ++c) {
                        for (int32_t t = 0; t < temporal_patch_size; ++t) {
                            (void) t;
                            for (int32_t py = 0; py < patch_size; ++py) {
                                const int32_t y = (ghg * merge_size + mh) * patch_size + py;
                                for (int32_t px = 0; px < patch_size; ++px) {
                                    const int32_t x = (gwg * merge_size + mw) * patch_size + px;
                                    out[row * row_size + col] =
                                        chw[(size_t) c * plane + (size_t) y * (size_t) w + (size_t) x];
                                    ++col;
                                }
                            }
                        }
                    }
                    ++row;
                }
            }
        }
    }
    return out;
}

static std::vector<uint8_t> pack_f32_bytes(const std::vector<float> & values) {
    if (values.empty()) {
        return {};
    }

    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("float tensor is too large");
    }

    std::vector<uint8_t> out(values.size() * sizeof(float), 0);
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

}  // namespace

smt_vision_preprocess_result smt_vision_preprocess_if_image(const std::vector<uint8_t> &         input,
                                                            const std::string &                  architecture,
                                                            int32_t                              input_width,
                                                            int32_t                              input_height,
                                                            const smt_vision_preprocess_config * config) {
    smt_vision_preprocess_result out;
    if (input.empty()) {
        return out;
    }

    if (input.size() > (size_t) std::numeric_limits<int>::max()) {
        return out;
    }

    int       src_w = 0, src_h = 0, src_c = 0;
    uint8_t * pixels = stbi_load_from_memory(input.data(), (int) input.size(), &src_w, &src_h, &src_c,
                                             /* desired_channels */ 3);
    if (pixels == nullptr) {
        return out;
    }

    try {
        const ep_preproc_spec spec = resolve_preproc_spec(architecture, input_width, input_height);
        if (spec.target_w <= 0 || spec.target_h <= 0) {
            stbi_image_free(pixels);
            throw std::runtime_error("SMT image preprocessing for architecture '" + architecture +
                                     "' is not configured yet; please provide preprocessed .bin");
        }

        const auto resized_u8 = resize_rgb_u8_antialias(pixels, src_w, src_h, spec.target_w, spec.target_h,
                                                        spec.quantize_to_u8_after_resize);
        const auto preprocess_config = config != nullptr ? *config : smt_vision_preprocess_config();
        const auto f32 =
            spec.qwen2vl_patch_flatten ?
                rgb_u8_to_qwen2vl_patch_f32_with_config(resized_u8, spec.target_w, spec.target_h, preprocess_config) :
            spec.apply_rescale_and_normalize ?
                rgb_u8_to_chw_f32_with_config(resized_u8, spec.target_w, spec.target_h, preprocess_config) :
                rgb_u8_to_chw_f32(resized_u8, spec.target_w, spec.target_h, spec.normalize_to_01);
        stbi_image_free(pixels);

        out.was_image       = true;
        out.target_w        = spec.target_w;
        out.target_h        = spec.target_h;
        out.normalize_to_01 = spec.normalize_to_01;
        out.tensor_bytes    = pack_f32_bytes(f32);
        return out;

    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
}


smt_lingbot_map_preprocess_result smt_lingbot_map_preprocess_images(
        const std::vector<std::vector<uint8_t>> & images,
        int32_t                                   target_w,
        int32_t                                   target_h,
        int32_t                                   patch_size,
        const float                               mean[3],
        const float                               std_values[3]) {
    if (images.empty()) {
        throw std::invalid_argument("LingBot-MAP preprocessing requires at least one image");
    }
    if (target_w <= 0 || target_h <= 0 || patch_size <= 0) {
        throw std::invalid_argument("Invalid LingBot-MAP preprocessing dimensions");
    }

    smt_lingbot_map_preprocess_result out;
    out.target_w = target_w;
    out.target_h = target_h;
    out.n_images = (int32_t) images.size();
    out.tensor_nchw.resize((size_t) out.n_images * 3u * (size_t) target_h * (size_t) target_w);
    out.resized_heights.reserve(images.size());

    const size_t image_plane = (size_t) target_h * (size_t) target_w;
    for (size_t i = 0; i < images.size(); ++i) {
        const auto & input = images[i];
        if (input.empty() || input.size() > (size_t) std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Invalid LingBot-MAP image payload");
        }

        int       src_w = 0, src_h = 0, src_c = 0;
        uint8_t * pixels = stbi_load_from_memory(input.data(), (int) input.size(), &src_w, &src_h, &src_c,
                                                 /* desired_channels */ 4);
        if (pixels == nullptr || src_w <= 0 || src_h <= 0) {
            if (pixels != nullptr) {
                stbi_image_free(pixels);
            }
            throw std::invalid_argument("LingBot-MAP input is not a supported image");
        }

        try {
            int32_t resized_h = (int32_t) std::round(((double) src_h * (double) target_w / (double) src_w) /
                                                     (double) patch_size) * patch_size;
            resized_h = std::max(patch_size, resized_h);
            out.resized_heights.push_back(resized_h);

            const auto rgb = rgba_u8_to_rgb_u8_white(pixels, src_w, src_h);
            stbi_image_free(pixels);
            pixels = nullptr;

            const auto resized = resize_rgb_u8_pillow_bicubic(rgb, src_w, src_h, target_w, resized_h);

            const int32_t crop_y = resized_h > target_h ? (resized_h - target_h) / 2 : 0;
            const int32_t pad_y  = resized_h < target_h ? (target_h - resized_h) / 2 : 0;

            for (int32_t c = 0; c < 3; ++c) {
                const float denom = std_values[c] == 0.0f ? 1.0f : std_values[c];
                float * dst = out.tensor_nchw.data() + ((i * 3u + (size_t) c) * image_plane);
                for (int32_t y = 0; y < target_h; ++y) {
                    const int32_t src_y = y + crop_y - pad_y;
                    for (int32_t x = 0; x < target_w; ++x) {
                        uint8_t v = 255;
                        if (src_y >= 0 && src_y < resized_h) {
                            v = resized[((size_t) src_y * (size_t) target_w + (size_t) x) * 3u + (size_t) c];
                        }
                        dst[(size_t) y * (size_t) target_w + (size_t) x] = (((float) v / 255.0f) - mean[c]) / denom;
                    }
                }
            }
        } catch (...) {
            if (pixels != nullptr) {
                stbi_image_free(pixels);
            }
            throw;
        }
    }

    return out;
}
