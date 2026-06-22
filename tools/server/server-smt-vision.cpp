#include "server-smt-vision.h"

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(LLAMA_SERVER_SMT_VISION)
#    include "../mtmd/smt-audio-wrapper.h"
#    include "../mtmd/smt-vision-preprocess.h"
#    include "../mtmd/smt-vision-wrapper.h"
#    include "../mtmd/lingbot-map-wrapper.h"
#endif

#if defined(LLAMA_SERVER_SMT_VISION)
namespace onnxruntime {
extern const OrtApi * g_ort;
}
#endif


struct lingbot_map_postprocess_result {
    std::vector<int64_t> pose_encoding_shape;
    std::vector<int64_t> extrinsic_shape;
    std::vector<int64_t> intrinsic_shape;
    std::vector<int64_t> world_points_shape;
    std::vector<int64_t> world_points_conf_shape;
    std::vector<float> pose_encoding_sample;
    std::vector<float> extrinsic_first;
    std::vector<float> intrinsic_first;
    std::vector<float> world_points_sample;
    std::string world_points_path;
    int64_t point_count = 0;
    int64_t world_points_bytes = 0;
    int32_t sample_count = 0;
    double depth_min = 0.0;
    double depth_max = 0.0;
    double depth_mean = 0.0;
    double depth_conf_min = 0.0;
    double depth_conf_max = 0.0;
    double depth_conf_mean = 0.0;
    std::string pose_source;
};

struct lingbot_map_onnx_context {
    Ort::Env            env{ ORT_LOGGING_LEVEL_WARNING, "lingbot-map" };
    Ort::Session        vision_session{ nullptr };
    Ort::Session        depth_session{ nullptr };

    std::vector<std::string>  vision_input_names;
    std::vector<std::string>  vision_output_names;
    std::vector<const char *> vision_input_names_raw;
    std::vector<const char *> vision_output_names_raw;

    std::vector<std::string>  depth_input_names;
    std::vector<std::string>  depth_output_names;
    std::vector<const char *> depth_input_names_raw;
    std::vector<const char *> depth_output_names_raw;

    std::vector<int64_t> vision_input_shape;
    std::vector<std::vector<int64_t>> depth_input_shapes;
    std::vector<std::vector<int64_t>> depth_output_shapes;
    int32_t              vision_input_h = 0;
    int32_t              vision_input_w = 0;
};

#if defined(_WIN32)
#    include <io.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <unistd.h>
#    include <dlfcn.h>
#endif

struct server_smt_vision_context {
#if defined(LLAMA_SERVER_SMT_VISION)
    std::unique_ptr<smt_vision_context>      smt_vision;
    std::unique_ptr<smt_audio_context>       smt_audio;
    std::unique_ptr<lingbot_map_context>     lingbot_map;
    std::unique_ptr<lingbot_map_onnx_context> lingbot_onnx;
#endif
    std::mutex               mu;
    int32_t                  hidden_size   = 0;
    bool                     use_mrope_pos = false;
    std::vector<llama_token> tok_img_beg;
    std::vector<llama_token> tok_img_end;
    std::vector<llama_token> tok_audio_beg;
    std::vector<llama_token> tok_audio_end;
    std::string              architecture;
    std::string              config_dir;
};


static int64_t lingbot_current_rss_mb() {
#if defined(_WIN32)
    return -1;
#else
    std::ifstream statm("/proc/self/statm");
    int64_t pages_total = 0;
    int64_t pages_rss = 0;
    if (!(statm >> pages_total >> pages_rss)) {
        return -1;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return -1;
    }
    return (pages_rss * (int64_t) page_size) / (1024 * 1024);
#endif
}


static int64_t lingbot_elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

static void lingbot_log_rss(const char * stage) {
    const int64_t rss_mb = lingbot_current_rss_mb();
    if (rss_mb >= 0) {
        std::cerr << "[LingBot-MAP][mem] " << stage << " rss=" << rss_mb << " MiB\n";
    }
}


static std::pair<std::string, std::string> lingbot_make_world_points_paths(const std::string & config_dir) {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::path root = config_dir.empty() ? fs::temp_directory_path(ec) : fs::path(config_dir);
    if (ec) {
        throw std::runtime_error("failed to resolve temporary directory for LingBot-MAP point cloud output");
    }

    fs::path out_dir = root / "lingbot_map_outputs";
    fs::create_directories(out_dir, ec);
    if (ec) {
        throw std::runtime_error("failed to create LingBot-MAP point cloud output directory: " + out_dir.string());
    }

    static std::atomic<uint64_t> counter{ 0 };
    const uint64_t stamp = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    const uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    const std::string file_name = "world_points_" + std::to_string(stamp) + "_" + std::to_string(seq) + ".f32.bin";
    const fs::path relative_path = fs::path("lingbot_map_outputs") / file_name;
    const fs::path write_path = fs::absolute(root / relative_path);
    return { write_path.string(), relative_path.generic_string() };
}

static std::vector<const char *> lingbot_make_name_ptrs(const std::vector<std::string> & names) {
    std::vector<const char *> ptrs;
    ptrs.reserve(names.size());
    for (const auto & name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

static std::vector<int64_t> lingbot_get_io_shape(Ort::Session & session, bool inputs, size_t index) {
    Ort::TypeInfo type_info = inputs ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

static std::vector<std::string> lingbot_get_io_names(Ort::Session & session, bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto allocated = inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(allocated.get());
    }
    return names;
}


static bool lingbot_init_spacemit_execution_provider(
        Ort::SessionOptions & options,
        const std::unordered_map<std::string, std::string> & provider_options,
        std::string & error_message) {
    std::vector<const char *> keys;
    std::vector<const char *> values;
    keys.reserve(provider_options.size());
    values.reserve(provider_options.size());
    for (const auto & entry : provider_options) {
        keys.push_back(entry.first.c_str());
        values.push_back(entry.second.c_str());
    }

#if defined(_WIN32)
    GGML_UNUSED(options);
    GGML_UNUSED(keys);
    GGML_UNUSED(values);
    error_message = "Spacemit EP dynamic initialization is not implemented on Windows";
    return false;
#else
    void * handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) {
        error_message = std::string("failed to load libspacemit_ep.so: ") + dlerror();
        return false;
    }

    auto * ep_init = reinterpret_cast<OrtStatus * (*) (OrtSessionOptions *, const char * const *, const char * const *, size_t)>(
            dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (!ep_init) {
        error_message = std::string("failed to find OrtSessionOptionsSpaceMITEnvInit: ") + dlerror();
        return false;
    }

    if (OrtStatus * status = ep_init(options, keys.data(), values.data(), keys.size())) {
        error_message = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(status);
        return false;
    }
    return true;
#endif
}

static void lingbot_append_spacemit_ep(Ort::SessionOptions & session_options,
                                       const char * session_name,
                                       const lingbot_map_config & cfg) {
    std::unordered_map<std::string, std::string> provider_options = cfg.ep_config;
    if (provider_options.find("SPACEMIT_EP_INTRA_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = "4";
    }
    if (provider_options.find("SPACEMIT_EP_INTER_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
    }

    std::string error_message;
    if (!lingbot_init_spacemit_execution_provider(session_options, provider_options, error_message)) {
        throw std::runtime_error(std::string("[LingBot-MAP] failed to initialize Spacemit EP for ") + session_name + ": " + error_message);
    }

    std::cerr << "[LingBot-MAP] Spacemit EP enabled for " << session_name << " (";
    for (const auto & pair : provider_options) {
        std::cerr << ", " << pair.first << "=" << pair.second;
    }
    std::cerr << ")\n";
}

static std::unique_ptr<lingbot_map_onnx_context> create_lingbot_map_onnx_context(const lingbot_map_config & cfg) {
    auto ctx = std::make_unique<lingbot_map_onnx_context>();
    Ort::SessionOptions vision_options;
    Ort::SessionOptions depth_options;
    vision_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    depth_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    lingbot_append_spacemit_ep(vision_options, "vit_encoder", cfg);
    lingbot_append_spacemit_ep(depth_options, "dpt_head", cfg);

    ctx->vision_session = Ort::Session(ctx->env, cfg.vision_model_path.c_str(), vision_options);
    ctx->depth_session = Ort::Session(ctx->env, cfg.depth_model_path.c_str(), depth_options);

    ctx->vision_input_names = lingbot_get_io_names(ctx->vision_session, true);
    ctx->vision_output_names = lingbot_get_io_names(ctx->vision_session, false);
    ctx->depth_input_names = lingbot_get_io_names(ctx->depth_session, true);
    ctx->depth_output_names = lingbot_get_io_names(ctx->depth_session, false);

    ctx->vision_input_shape = lingbot_get_io_shape(ctx->vision_session, true, 0);
    ctx->depth_input_shapes.reserve(ctx->depth_input_names.size());
    for (size_t i = 0; i < ctx->depth_input_names.size(); ++i) {
        ctx->depth_input_shapes.push_back(lingbot_get_io_shape(ctx->depth_session, true, i));
    }
    ctx->depth_output_shapes.reserve(ctx->depth_output_names.size());
    for (size_t i = 0; i < ctx->depth_output_names.size(); ++i) {
        ctx->depth_output_shapes.push_back(lingbot_get_io_shape(ctx->depth_session, false, i));
    }

    if (ctx->vision_input_shape.size() == 5) {
        if (ctx->vision_input_shape[3] > 0) {
            ctx->vision_input_h = (int32_t) ctx->vision_input_shape[3];
        }
        if (ctx->vision_input_shape[4] > 0) {
            ctx->vision_input_w = (int32_t) ctx->vision_input_shape[4];
        }
    }

    if (ctx->vision_input_names.empty() || ctx->vision_output_names.empty()) {
        throw std::runtime_error("LingBot-MAP ViT ONNX session has empty IO signature");
    }
    if (ctx->depth_input_names.empty() || ctx->depth_output_names.empty()) {
        throw std::runtime_error("LingBot-MAP DPT ONNX session has empty IO signature");
    }

    ctx->vision_input_names_raw = lingbot_make_name_ptrs(ctx->vision_input_names);
    ctx->vision_output_names_raw = lingbot_make_name_ptrs(ctx->vision_output_names);
    ctx->depth_input_names_raw = lingbot_make_name_ptrs(ctx->depth_input_names);
    ctx->depth_output_names_raw = lingbot_make_name_ptrs(ctx->depth_output_names);
    return ctx;
}


static int64_t lingbot_numel(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        return 0;
    }
    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

static std::vector<int64_t> lingbot_make_depth_input_shape(
        const std::vector<int64_t> & onnx_shape,
        int32_t                      n_frames,
        int32_t                      tokens_per_frame,
        int32_t                      camera_hidden_size) {
    if (onnx_shape.size() != 4) {
        throw std::runtime_error("LingBot-MAP DPT input must be rank-4 [1, frames, tokens, hidden]");
    }
    std::vector<int64_t> shape = onnx_shape;
    const int64_t expected[4] = { 1, (int64_t) n_frames, (int64_t) tokens_per_frame, (int64_t) camera_hidden_size };
    for (size_t i = 0; i < 4; ++i) {
        if (shape[i] < 0) {
            shape[i] = expected[i];
        }
        if (shape[i] != expected[i]) {
            throw std::runtime_error("LingBot-MAP DPT input shape does not match aggregator selected output boundary");
        }
    }
    return shape;
}

static void lingbot_validate_depth_outputs(const std::vector<Ort::Value> & outputs) {
    if (outputs.size() != 2 || !outputs[0].IsTensor() || !outputs[1].IsTensor()) {
        throw std::runtime_error("LingBot-MAP DPT ONNX must return depth and depth_conf tensors");
    }
    const auto depth_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto conf_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (depth_shape.size() != 5 || conf_shape.size() != 4) {
        throw std::runtime_error("LingBot-MAP DPT ONNX returned unexpected output ranks");
    }
    if (depth_shape[0] != conf_shape[0] || depth_shape[1] != conf_shape[1] ||
        depth_shape[2] != conf_shape[2] || depth_shape[3] != conf_shape[3] || depth_shape[4] != 1) {
        throw std::runtime_error("LingBot-MAP DPT depth/depth_conf output shapes are inconsistent");
    }
}



static void lingbot_quat_xyzw_to_mat(const float * q, float r[9]) {
    const double x = q[0];
    const double y = q[1];
    const double z = q[2];
    const double w = q[3];
    double denom = x*x + y*y + z*z + w*w;
    if (denom <= 1e-12) {
        denom = 1.0;
    }
    const double two_s = 2.0 / denom;
    r[0] = (float) (1.0 - two_s * (y*y + z*z));
    r[1] = (float) (two_s * (x*y - z*w));
    r[2] = (float) (two_s * (x*z + y*w));
    r[3] = (float) (two_s * (x*y + z*w));
    r[4] = (float) (1.0 - two_s * (x*x + z*z));
    r[5] = (float) (two_s * (y*z - x*w));
    r[6] = (float) (two_s * (x*z - y*w));
    r[7] = (float) (two_s * (y*z + x*w));
    r[8] = (float) (1.0 - two_s * (x*x + y*y));
}

static lingbot_map_postprocess_result lingbot_postprocess_reconstruction(
        const float *                       pose_encoding,
        const std::string &                 pose_source,
        const float *                       depth,
        const float *                       depth_conf,
        const std::vector<int64_t> &        depth_shape,
        const std::vector<int64_t> &        depth_conf_shape,
        int32_t                             n_frames,
        const std::string &                 world_points_write_path,
        const std::string &                 world_points_response_path) {
    if (pose_encoding == nullptr || depth == nullptr || depth_conf == nullptr) {
        throw std::runtime_error("LingBot-MAP postprocess requires pose, depth and depth_conf data");
    }
    if (depth_shape.size() != 5 || depth_conf_shape.size() != 4 || depth_shape[0] != 1 ||
        depth_shape[1] != n_frames || depth_shape[4] != 1 || depth_conf_shape[0] != 1 ||
        depth_conf_shape[1] != n_frames || depth_conf_shape[2] != depth_shape[2] ||
        depth_conf_shape[3] != depth_shape[3]) {
        throw std::runtime_error("LingBot-MAP postprocess received incompatible depth output shapes");
    }

    const int64_t h = depth_shape[2];
    const int64_t w = depth_shape[3];
    const int64_t point_count = (int64_t) n_frames * h * w;
    if (point_count <= 0) {
        throw std::runtime_error("LingBot-MAP postprocess requires non-empty depth outputs");
    }

    lingbot_map_postprocess_result result;
    result.pose_source = pose_source;
    result.pose_encoding_shape = { 1, n_frames, 9 };
    result.extrinsic_shape = { 1, n_frames, 3, 4 };
    result.intrinsic_shape = { 1, n_frames, 3, 3 };
    result.world_points_shape = { 1, n_frames, h, w, 3 };
    result.world_points_conf_shape = { 1, n_frames, h, w };
    result.point_count = point_count;
    result.world_points_path = world_points_response_path;

    std::ofstream world_points_file;
    if (!world_points_write_path.empty()) {
        world_points_file.open(world_points_write_path, std::ios::binary | std::ios::trunc);
        if (!world_points_file.is_open()) {
            throw std::runtime_error("failed to open LingBot-MAP world points output: " + world_points_write_path);
        }
    }

    std::vector<float> extrinsics_w2c((size_t) n_frames * 12, 0.0f);
    std::vector<float> extrinsics_c2w((size_t) n_frames * 12, 0.0f);
    std::vector<float> intrinsics((size_t) n_frames * 9, 0.0f);
    std::vector<float> c2w_rot((size_t) n_frames * 9, 0.0f);
    std::vector<float> c2w_trans((size_t) n_frames * 3, 0.0f);

    for (int32_t f = 0; f < n_frames; ++f) {
        const float * p = pose_encoding + (size_t) f * 9;
        float r[9];
        lingbot_quat_xyzw_to_mat(p + 3, r);

        float * e = extrinsics_w2c.data() + (size_t) f * 12;
        e[0] = r[0]; e[1] = r[1]; e[2] = r[2]; e[3] = p[0];
        e[4] = r[3]; e[5] = r[4]; e[6] = r[5]; e[7] = p[1];
        e[8] = r[6]; e[9] = r[7]; e[10] = r[8]; e[11] = p[2];

        float fov_h = p[7];
        float fov_w = p[8];
        if (fov_h <= 1e-6f) {
            fov_h = 1.0471975511965977f;
        }
        if (fov_w <= 1e-6f) {
            fov_w = 1.0471975511965977f;
        }
        const float fy = (float) ((double) h / 2.0 / std::tan((double) fov_h / 2.0));
        const float fx = (float) ((double) w / 2.0 / std::tan((double) fov_w / 2.0));
        float * k = intrinsics.data() + (size_t) f * 9;
        k[0] = fx;
        k[4] = fy;
        k[2] = (float) w / 2.0f;
        k[5] = (float) h / 2.0f;
        k[8] = 1.0f;

        float * cr = c2w_rot.data() + (size_t) f * 9;
        cr[0] = r[0]; cr[1] = r[3]; cr[2] = r[6];
        cr[3] = r[1]; cr[4] = r[4]; cr[5] = r[7];
        cr[6] = r[2]; cr[7] = r[5]; cr[8] = r[8];

        float * ct = c2w_trans.data() + (size_t) f * 3;
        ct[0] = -(cr[0] * p[0] + cr[1] * p[1] + cr[2] * p[2]);
        ct[1] = -(cr[3] * p[0] + cr[4] * p[1] + cr[5] * p[2]);
        ct[2] = -(cr[6] * p[0] + cr[7] * p[1] + cr[8] * p[2]);

        float * c2w = extrinsics_c2w.data() + (size_t) f * 12;
        c2w[0] = cr[0]; c2w[1] = cr[1]; c2w[2] = cr[2]; c2w[3] = ct[0];
        c2w[4] = cr[3]; c2w[5] = cr[4]; c2w[6] = cr[5]; c2w[7] = ct[1];
        c2w[8] = cr[6]; c2w[9] = cr[7]; c2w[10] = cr[8]; c2w[11] = ct[2];
    }

    result.pose_encoding_sample.assign(pose_encoding, pose_encoding + std::min<int64_t>((int64_t) n_frames * 9, 9));
    result.extrinsic_first.assign(extrinsics_c2w.begin(), extrinsics_c2w.begin() + std::min<size_t>(extrinsics_c2w.size(), 12));
    result.intrinsic_first.assign(intrinsics.begin(), intrinsics.begin() + std::min<size_t>(intrinsics.size(), 9));

    const int32_t sample_limit = 64;
    const int64_t sample_stride = std::max<int64_t>(1, point_count / sample_limit);
    result.world_points_sample.reserve((size_t) sample_limit * 3);

    double depth_sum = 0.0;
    double conf_sum = 0.0;
    result.depth_min = depth[0];
    result.depth_max = depth[0];
    result.depth_conf_min = depth_conf[0];
    result.depth_conf_max = depth_conf[0];

    int32_t sample_count = 0;
    for (int32_t f = 0; f < n_frames; ++f) {
        const float * k = intrinsics.data() + (size_t) f * 9;
        const float * cr = c2w_rot.data() + (size_t) f * 9;
        const float * ct = c2w_trans.data() + (size_t) f * 3;
        const float fx_cur = k[0];
        const float fy_cur = k[4];
        const float cx = k[2];
        const float cy = k[5];
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                const int64_t idx = ((int64_t) f * h + y) * w + x;
                const float d = depth[idx];
                const float c = depth_conf[idx];
                result.depth_min = std::min(result.depth_min, (double) d);
                result.depth_max = std::max(result.depth_max, (double) d);
                result.depth_conf_min = std::min(result.depth_conf_min, (double) c);
                result.depth_conf_max = std::max(result.depth_conf_max, (double) c);
                depth_sum += d;
                conf_sum += c;

                const float cam_x = ((float) x - cx) * d / fx_cur;
                const float cam_y = ((float) y - cy) * d / fy_cur;
                const float cam_z = d;
                const float world_xyz[3] = {
                    cr[0] * cam_x + cr[1] * cam_y + cr[2] * cam_z + ct[0],
                    cr[3] * cam_x + cr[4] * cam_y + cr[5] * cam_z + ct[1],
                    cr[6] * cam_x + cr[7] * cam_y + cr[8] * cam_z + ct[2],
                };

                if (world_points_file.is_open()) {
                    world_points_file.write(reinterpret_cast<const char *>(world_xyz), sizeof(world_xyz));
                    if (!world_points_file) {
                        throw std::runtime_error("failed to write LingBot-MAP world points output: " + world_points_write_path);
                    }
                }

                if (idx % sample_stride == 0 && sample_count < sample_limit) {
                    result.world_points_sample.insert(result.world_points_sample.end(), world_xyz, world_xyz + 3);
                    sample_count += 1;
                }
            }
        }
    }
    if (world_points_file.is_open()) {
        world_points_file.close();
        result.world_points_bytes = point_count * 3 * (int64_t) sizeof(float);
    }
    result.depth_mean = depth_sum / (double) point_count;
    result.depth_conf_mean = conf_sum / (double) point_count;
    result.sample_count = sample_count;
    return result;
}

bool server_smt_vision_config_is_lingbot_map(const std::string & config_dir) {
    if (config_dir.empty()) {
        return false;
    }

    const std::string config_path = config_dir + "/config.json";
    std::ifstream     file(config_path);
    if (!file.is_open()) {
        return false;
    }

    try {
        nlohmann::json config = nlohmann::json::parse(file);
        if (!config.contains("architectures")) {
            return false;
        }
        const auto & arch = config.at("architectures");
        if (arch.is_array()) {
            for (const auto & value : arch) {
                if (value.is_string() && value.get<std::string>() == "LingBotMapFor3DReconstruction") {
                    return true;
                }
            }
        } else if (arch.is_string()) {
            return arch.get<std::string>() == "LingBotMapFor3DReconstruction";
        }
    } catch (...) {
        return false;
    }
    return false;
}

static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t       hash      = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static bool contains_icase(const std::string & text, const std::string & pattern) {
    return to_lower_ascii(text).find(to_lower_ascii(pattern)) != std::string::npos;
}

static bool arch_requires_mrope(const std::string & arch_name) {
    return contains_icase(arch_name, "qwen2vl") || contains_icase(arch_name, "qwen2_5_vl") ||
           contains_icase(arch_name, "qwen3vl") || contains_icase(arch_name, "glm4v") ||
           contains_icase(arch_name, "paddleocr");
}

static bool arch_is_qwen3asr(const std::string & arch_name) {
    return contains_icase(arch_name, "qwen3asr");
}

static std::pair<int32_t, int32_t> infer_image_grid_xy(int32_t n_tokens) {
    if (n_tokens <= 0) {
        return { 0, 0 };
    }

    int32_t       best_y = 1;
    int32_t       best_x = n_tokens;
    const int32_t root   = (int32_t) std::sqrt((double) n_tokens);
    for (int32_t y = root; y >= 1; --y) {
        if (n_tokens % y == 0) {
            best_y = y;
            best_x = n_tokens / y;
            break;
        }
    }
    return { best_x, best_y };
}

static std::vector<llama_token> tokenize_exact_special(llama_context * lctx, const std::string & token_text) {
    auto toks = common_tokenize(lctx, token_text, /* add_special */ false, /* parse_special */ true);
    if (toks.size() != 1) {
        return {};
    }
    if (common_token_to_piece(lctx, toks[0]) != token_text) {
        return {};
    }
    return toks;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> detect_image_boundary_tokens_native(
    llama_context *     lctx,
    const std::string & arch_name) {
    if (contains_icase(arch_name, "qwen2vl") || contains_icase(arch_name, "qwen2_5_vl") ||
        contains_icase(arch_name, "qwen3vl") || contains_icase(arch_name, "youtuvl")) {
        return { tokenize_exact_special(lctx, "<|vision_start|>"), tokenize_exact_special(lctx, "<|vision_end|>") };
    }
    if (contains_icase(arch_name, "llama4") || contains_icase(arch_name, "lfm2")) {
        return { tokenize_exact_special(lctx, "<|image_start|>"), tokenize_exact_special(lctx, "<|image_end|>") };
    }
    if (contains_icase(arch_name, "gemma3")) {
        return { tokenize_exact_special(lctx, "<start_of_image>"), tokenize_exact_special(lctx, "<end_of_image>") };
    }
    if (contains_icase(arch_name, "internvl")) {
        return { tokenize_exact_special(lctx, "<img>"), tokenize_exact_special(lctx, "</img>") };
    }
    if (contains_icase(arch_name, "glm4v")) {
        return { tokenize_exact_special(lctx, "<|begin_of_image|>"), tokenize_exact_special(lctx, "<|end_of_image|>") };
    }
    if (contains_icase(arch_name, "paddleocr")) {
        return { tokenize_exact_special(lctx, "<|IMAGE_START|>"), tokenize_exact_special(lctx, "<|IMAGE_END|>") };
    }
    if (contains_icase(arch_name, "lightonocr")) {
        return { tokenize_exact_special(lctx, "<|im_start|>"), tokenize_exact_special(lctx, "<|im_end|>") };
    }
    return {};
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> detect_image_boundary_tokens_auto(
    llama_context * lctx) {
    static const std::array<std::pair<const char *, const char *>, 8> candidates = {
        {
         { "<|vision_start|>", "<|vision_end|>" },
         { "<|image_start|>", "<|image_end|>" },
         { "<start_of_image>", "<end_of_image>" },
         { "<img>", "</img>" },
         { "<|begin_of_image|>", "<|end_of_image|>" },
         { "<|IMAGE_START|>", "<|IMAGE_END|>" },
         { "<|im_start|>", "<|im_end|>" },
         { "<image>", "</image>" },
         }
    };

    for (const auto & candidate : candidates) {
        auto beg = tokenize_exact_special(lctx, candidate.first);
        auto end = tokenize_exact_special(lctx, candidate.second);
        if (!beg.empty() && !end.empty()) {
            return { std::move(beg), std::move(end) };
        }
    }
    return {};
}

enum class smt_image_boundary_mode {
    native,
    auto_detect,
    none,
};

static smt_image_boundary_mode smt_image_boundary_mode_from_env() {
    const char * env = std::getenv("MTMD_SMT_IMAGE_BOUNDARY");
    if (env == nullptr || env[0] == '\0') {
        return smt_image_boundary_mode::native;
    }

    const std::string value = to_lower_ascii(env);
    if (value == "native") {
        return smt_image_boundary_mode::native;
    }
    if (value == "auto" || value == "detect") {
        return smt_image_boundary_mode::auto_detect;
    }
    if (value == "none" || value == "off" || value == "0") {
        return smt_image_boundary_mode::none;
    }

    LOG_WRN("[server-smt] unknown MTMD_SMT_IMAGE_BOUNDARY='%s', fallback to 'native'\n", env);
    return smt_image_boundary_mode::native;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> resolve_image_boundary_tokens(
    llama_context *     lctx,
    const std::string & arch_name) {
    const auto mode = smt_image_boundary_mode_from_env();
    if (mode == smt_image_boundary_mode::none) {
        return {};
    }
    if (mode == smt_image_boundary_mode::auto_detect) {
        return detect_image_boundary_tokens_auto(lctx);
    }
    return detect_image_boundary_tokens_native(lctx, arch_name);
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> resolve_audio_boundary_tokens(
    llama_context *     lctx,
    const std::string & arch_name) {
    if (!arch_is_qwen3asr(arch_name)) {
        return {};
    }
    return { tokenize_exact_special(lctx, "<|audio_start|>"), tokenize_exact_special(lctx, "<|audio_end|>") };
}

static bool looks_like_audio_file(const std::vector<uint8_t> & data) {
    if (data.size() < 12) {
        return false;
    }

    const char * buf    = reinterpret_cast<const char *>(data.data());
    const bool   is_wav = std::memcmp(buf, "RIFF", 4) == 0 && std::memcmp(buf + 8, "WAVE", 4) == 0;
    const bool   is_mp3 =
        data.size() >= 3 && (std::memcmp(buf, "ID3", 3) == 0 || (static_cast<unsigned char>(buf[0]) == 0xFF &&
                                                                 (static_cast<unsigned char>(buf[1]) & 0xE0) == 0xE0));
    const bool is_flac = std::memcmp(buf, "fLaC", 4) == 0;
    return is_wav || is_mp3 || is_flac;
}

static std::string write_temp_bin_file(const std::vector<uint8_t> & data) {
#if defined(_WIN32)
    char temp_path[MAX_PATH] = { 0 };
    char temp_file[MAX_PATH] = { 0 };

    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("failed to get temp path");
    }
    if (GetTempFileNameA(temp_path, "lse", 0, temp_file) == 0) {
        throw std::runtime_error("failed to create temp file");
    }

    FILE * fp = std::fopen(temp_file, "wb");
    if (!fp) {
        throw std::runtime_error("failed to open temp file");
    }
    const size_t n = std::fwrite(data.data(), 1, data.size(), fp);
    std::fclose(fp);
    if (n != data.size()) {
        std::remove(temp_file);
        throw std::runtime_error("failed to write temp file");
    }
    return std::string(temp_file);
#else
    char      tmpl[] = "/tmp/llama-server-smt-XXXXXX";
    const int fd     = mkstemp(tmpl);
    if (fd < 0) {
        throw std::runtime_error("failed to create temp file");
    }

    size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = write(fd, data.data() + written, data.size() - written);
        if (n <= 0) {
            close(fd);
            std::remove(tmpl);
            throw std::runtime_error("failed to write temp file");
        }
        written += (size_t) n;
    }
    close(fd);
    return std::string(tmpl);
#endif
}

static int decode_tokens(llama_context *                  lctx,
                         const std::vector<llama_token> & tokens,
                         llama_pos &                      n_past,
                         int32_t                          seq_id,
                         int32_t                          n_batch,
                         bool                             logits_last) {
    if (tokens.empty()) {
        return 0;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    size_t      i     = 0;
    while (i < tokens.size()) {
        batch.n_tokens = 0;
        for (; i < tokens.size() && batch.n_tokens < n_batch; ++i) {
            const int32_t j    = batch.n_tokens;
            batch.token[j]     = tokens[i];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j]    = false;
            batch.n_tokens++;
        }

        if (logits_last && i == tokens.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        n_past += batch.n_tokens;
    }

    llama_batch_free(batch);
    return 0;
}

static int decode_embd(llama_context * lctx,
                       const float *   embd,
                       int32_t         n_tokens,
                       int32_t         n_embd,
                       llama_pos &     n_past,
                       int32_t         seq_id,
                       int32_t         n_batch,
                       bool            logits_last,
                       bool            use_mrope_pos,
                       int32_t         nx,
                       int32_t         ny) {
    const int n_pos_per_embd = use_mrope_pos ? 4 : 1;

    std::vector<llama_pos>      pos((size_t) n_tokens * n_pos_per_embd);
    std::vector<int32_t>        n_seq_id(n_tokens);
    std::vector<llama_seq_id>   seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = seq_id;
        seq_ids[i]  = &seq_id_0[i];
    }

    if (use_mrope_pos) {
        if (nx > 0 && ny > 0 && nx * ny == n_tokens) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int i                    = y * nx + x;
                    pos[(size_t) i]                = n_past;
                    pos[(size_t) i + n_tokens]     = n_past + y;
                    pos[(size_t) i + 2 * n_tokens] = n_past + x;
                    pos[(size_t) i + 3 * n_tokens] = 0;
                }
            }
        } else {
            for (int i = 0; i < n_tokens; ++i) {
                pos[(size_t) i]                = n_past + i;
                pos[(size_t) i + n_tokens]     = n_past + i;
                pos[(size_t) i + 2 * n_tokens] = n_past + i;
                pos[(size_t) i + 3 * n_tokens] = 0;
            }
        }
    } else {
        for (int i = 0; i < n_tokens; ++i) {
            pos[(size_t) i] = n_past + i;
        }
    }

    int processed = 0;
    while (processed < n_tokens) {
        const int  batch_size    = std::min(n_batch, n_tokens - processed);
        const bool is_last_batch = processed + batch_size >= n_tokens;

        for (int i = 0; i < batch_size; ++i) {
            if (!use_mrope_pos) {
                pos[processed + i] = n_past + processed + i;
            }
            n_seq_id[processed + i] = 1;
            logits[processed + i]   = (logits_last && is_last_batch && i == batch_size - 1);
        }

        llama_pos *            pos_ptr = nullptr;
        std::vector<llama_pos> pos_view;
        if (use_mrope_pos) {
            pos_view.reserve((size_t) batch_size * n_pos_per_embd);
            for (int d = 0; d < n_pos_per_embd; ++d) {
                const size_t src_idx = (size_t) d * n_tokens + processed;
                pos_view.insert(pos_view.end(), pos.data() + src_idx, pos.data() + src_idx + batch_size);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + processed;
        }

        llama_batch batch = {
            /* n_tokens  */ batch_size,
            /* token     */ nullptr,
            /* embd      */ const_cast<float *>(embd + (size_t) processed * n_embd),
            /* pos       */ pos_ptr,
            /* n_seq_id  */ n_seq_id.data() + processed,
            /* seq_id    */ seq_ids.data() + processed,
            /* logits    */ logits.data() + processed,
        };

        if (llama_decode(lctx, batch) != 0) {
            return 1;
        }
        processed += batch_size;
    }

    if (use_mrope_pos) {
        n_past += std::max(nx, ny);
    } else {
        n_past += n_tokens;
    }

    return 0;
}

server_smt_vision_context * server_smt_vision_init(llama_context * lctx, const std::string & config_dir, bool warmup) {
#if defined(LLAMA_SERVER_SMT_VISION)
    auto        ctx = std::make_unique<server_smt_vision_context>();
    ctx->config_dir = config_dir;
    std::string primary_architecture;

    if (server_smt_vision_config_is_lingbot_map(config_dir)) {
        GGML_UNUSED(lctx);
        GGML_UNUSED(warmup);
        onnxruntime::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ctx->lingbot_map = lingbot_map_context::create(config_dir);
        ctx->lingbot_onnx = create_lingbot_map_onnx_context(ctx->lingbot_map->config());
        ctx->architecture = ctx->lingbot_map->architecture();
        LOG_INF("[server-smt] loaded LingBot-MAP model from '%s', architecture=%s, tensors=%" PRId64 "\n",
                config_dir.c_str(), ctx->architecture.c_str(), ctx->lingbot_map->tensor_count());
        LOG_INF("[server-smt] loaded LingBot-MAP ONNX sessions: vit_inputs=%zu, vit_outputs=%zu, dpt_inputs=%zu, dpt_outputs=%zu\n",
                ctx->lingbot_onnx->vision_input_names.size(), ctx->lingbot_onnx->vision_output_names.size(),
                ctx->lingbot_onnx->depth_input_names.size(), ctx->lingbot_onnx->depth_output_names.size());
        return ctx.release();
    }

    try {
        ctx->smt_vision      = smt_vision_context::create(config_dir, warmup);
        ctx->hidden_size     = (int32_t) ctx->smt_vision->hidden_size();
        primary_architecture = ctx->smt_vision->architecture();
        auto boundaries      = resolve_image_boundary_tokens(lctx, primary_architecture);
        ctx->tok_img_beg     = std::move(boundaries.first);
        ctx->tok_img_end     = std::move(boundaries.second);
    } catch (const std::exception & e) {
        LOG_WRN("[server-smt] failed to initialize SMT vision backend from '%s': %s\n", config_dir.c_str(), e.what());
    }

    try {
        ctx->smt_audio = smt_audio_context::create(config_dir, warmup);
        if (ctx->hidden_size == 0) {
            ctx->hidden_size = (int32_t) ctx->smt_audio->hidden_size();
        } else if (ctx->hidden_size != ctx->smt_audio->hidden_size()) {
            throw std::runtime_error("SMT image/audio hidden size mismatch");
        }
        if (primary_architecture.empty()) {
            primary_architecture = ctx->smt_audio->architecture();
        }
        auto audio_boundaries = resolve_audio_boundary_tokens(lctx, ctx->smt_audio->architecture());
        ctx->tok_audio_beg    = std::move(audio_boundaries.first);
        ctx->tok_audio_end    = std::move(audio_boundaries.second);
    } catch (const std::exception & e) {
        LOG_WRN("[server-smt] failed to initialize SMT audio backend from '%s': %s\n", config_dir.c_str(), e.what());
    }

    if (!ctx->smt_vision && !ctx->smt_audio) {
        throw std::runtime_error("Neither SMT vision nor SMT audio backend is available");
    }

    ctx->architecture  = primary_architecture;
    ctx->use_mrope_pos = arch_requires_mrope(ctx->architecture);

    return ctx.release();
#else
    GGML_UNUSED(lctx);
    GGML_UNUSED(config_dir);
    GGML_UNUSED(warmup);
    throw std::runtime_error("SMT media backend is not compiled. Rebuild with LLAMA_SERVER_SMT_VISION=ON.");
#endif
}

void server_smt_vision_free(server_smt_vision_context * ctx) {
    delete ctx;
}

bool server_smt_vision_supports_image(const server_smt_vision_context * ctx) {
    return ctx != nullptr
#if defined(LLAMA_SERVER_SMT_VISION)
           && ctx->smt_vision != nullptr
#endif
        ;
}

bool server_smt_vision_supports_audio(const server_smt_vision_context * ctx) {
    return ctx != nullptr
#if defined(LLAMA_SERVER_SMT_VISION)
           && ctx->smt_audio != nullptr
#endif
        ;
}

bool server_smt_vision_supports_prompt_embeddings(const server_smt_vision_context * ctx) {
    return ctx != nullptr
#if defined(LLAMA_SERVER_SMT_VISION)
           && (ctx->smt_vision != nullptr || ctx->smt_audio != nullptr)
#endif
        ;
}

bool server_smt_vision_is_lingbot_map(const server_smt_vision_context * ctx) {
    return ctx != nullptr
#if defined(LLAMA_SERVER_SMT_VISION)
           && ctx->lingbot_map != nullptr
#endif
        ;
}

server_smt_lingbot_map_reconstruct_result server_smt_vision_lingbot_map_reconstruct(
        server_smt_vision_context * ctx,
        const std::vector<std::vector<uint8_t>> & images,
        const server_smt_lingbot_map_reconstruct_options & options) {
#if defined(LLAMA_SERVER_SMT_VISION)
    if (ctx == nullptr || ctx->lingbot_map == nullptr) {
        throw std::runtime_error("SMT context does not contain a LingBot-MAP model");
    }
    if (images.empty()) {
        throw std::invalid_argument("LingBot-MAP reconstruction requires at least one image");
    }
    if (options.max_frames > 0 && (int32_t) images.size() > options.max_frames) {
        throw std::invalid_argument("LingBot-MAP reconstruction request exceeds max_frames");
    }

    std::lock_guard<std::mutex> lock(ctx->mu);
    lingbot_log_rss("request_start");

    const auto & cfg = ctx->lingbot_map->config();
    if (ctx->lingbot_onnx == nullptr) {
        throw std::runtime_error("LingBot-MAP ONNX sessions are not loaded");
    }

    const int32_t input_w = ctx->lingbot_onnx->vision_input_w > 0 ? ctx->lingbot_onnx->vision_input_w : cfg.image_size;
    const int32_t input_h = ctx->lingbot_onnx->vision_input_h > 0 ? ctx->lingbot_onnx->vision_input_h : cfg.image_size;

    auto preproc = smt_lingbot_map_preprocess_images(images, input_w, input_h, cfg.patch_size,
                                                     cfg.image_mean, cfg.image_std);
    lingbot_log_rss("after_preprocess");

    std::vector<int64_t> input_shape = { 1, (int64_t) images.size(), 3, input_h, input_w };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preproc.tensor_nchw.data(),
                                                              preproc.tensor_nchw.size(),
                                                              input_shape.data(), input_shape.size());

    auto stage_start = std::chrono::steady_clock::now();
    auto vit_outputs = ctx->lingbot_onnx->vision_session.Run(
            Ort::RunOptions{ nullptr },
            ctx->lingbot_onnx->vision_input_names_raw.data(),
            &input_tensor,
            1,
            ctx->lingbot_onnx->vision_output_names_raw.data(),
            ctx->lingbot_onnx->vision_output_names_raw.size());
    std::cerr << "[LingBot-MAP][time] vit_onnx_ms=" << lingbot_elapsed_ms(stage_start) << "\n";
    lingbot_log_rss("after_vit_onnx");

    if (vit_outputs.empty() || !vit_outputs[0].IsTensor()) {
        throw std::runtime_error("LingBot-MAP ViT ONNX did not return a tensor output");
    }

    auto tensor_info = vit_outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> vision_output_shape = tensor_info.GetShape();
    const int64_t vision_output_float_count = (int64_t) tensor_info.GetElementCount();
    if (vision_output_shape.size() != 3) {
        throw std::runtime_error("LingBot-MAP ViT output must be rank-3 [frames, tokens, hidden]");
    }
    if (vision_output_shape[0] != (int64_t) images.size()) {
        throw std::runtime_error("LingBot-MAP ViT output frame count does not match input image count");
    }
    if (vision_output_shape[2] != cfg.hidden_size) {
        throw std::runtime_error("LingBot-MAP ViT output hidden size does not match config hidden_size");
    }
    if (vision_output_shape[1] <= 0) {
        throw std::runtime_error("LingBot-MAP ViT output token count must be positive");
    }

    const float * vit_output_data = vit_outputs[0].GetTensorData<float>();
    const auto aggregator_input = ctx->lingbot_map->build_aggregator_input(
            vit_output_data,
            (int32_t) vision_output_shape[0],
            (int32_t) vision_output_shape[1],
            (int32_t) vision_output_shape[2],
            input_h,
            input_w,
            /* num_frame_for_scale */ 1);
    lingbot_log_rss("after_aggregator_input");
    stage_start = std::chrono::steady_clock::now();
    const auto runtime = ctx->lingbot_map->run_aggregator_camera_head(aggregator_input, /* prefer_smt */ true);
    std::cerr << "[LingBot-MAP][time] aggregator_camera_total_ms=" << lingbot_elapsed_ms(stage_start) << "\n";
    lingbot_log_rss("after_aggregator_camera_ggml");

    if (ctx->lingbot_onnx->depth_input_names.size() != runtime.selected_output_shapes.size() ||
        runtime.selected_output_shapes.size() != runtime.selected_outputs.size()) {
        throw std::runtime_error("LingBot-MAP DPT input count does not match aggregator runtime selected output count");
    }

    std::vector<std::vector<int64_t>> depth_input_shapes;
    depth_input_shapes.reserve(ctx->lingbot_onnx->depth_input_names.size());
    int64_t depth_input_float_count = 0;
    for (size_t i = 0; i < ctx->lingbot_onnx->depth_input_names.size(); ++i) {
        const auto & selected_shape = runtime.selected_output_shapes[i];
        if (selected_shape.size() != 4 || selected_shape[0] != cfg.camera_hidden_size ||
            selected_shape[1] != aggregator_input.tokens_per_frame || selected_shape[2] != aggregator_input.n_frames) {
            throw std::runtime_error("LingBot-MAP aggregator selected output shape is not compatible with DPT input");
        }
        const auto input_shape = lingbot_make_depth_input_shape(ctx->lingbot_onnx->depth_input_shapes[i],
                                                                aggregator_input.n_frames,
                                                                aggregator_input.tokens_per_frame,
                                                                cfg.camera_hidden_size);
        depth_input_float_count += lingbot_numel(input_shape);
        depth_input_shapes.push_back(input_shape);
    }

    std::vector<float> depth_input_storage((size_t) depth_input_float_count, 0.0f);
    std::vector<Ort::Value> depth_input_tensors;
    depth_input_tensors.reserve(depth_input_shapes.size());
    size_t depth_input_offset = 0;
    for (size_t i = 0; i < depth_input_shapes.size(); ++i) {
        const auto & shape = depth_input_shapes[i];
        const int64_t n_elem = lingbot_numel(shape);
        if ((size_t) n_elem != runtime.selected_outputs[i].size()) {
            throw std::runtime_error("LingBot-MAP runtime selected output size does not match DPT input shape");
        }
        std::copy(runtime.selected_outputs[i].begin(), runtime.selected_outputs[i].end(),
                  depth_input_storage.begin() + (std::ptrdiff_t) depth_input_offset);
        depth_input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info,
                                                                      depth_input_storage.data() + depth_input_offset,
                                                                      (size_t) n_elem,
                                                                      shape.data(), shape.size()));
        depth_input_offset += (size_t) n_elem;
    }
    lingbot_log_rss("after_dpt_input_pack");

    stage_start = std::chrono::steady_clock::now();
    auto depth_outputs = ctx->lingbot_onnx->depth_session.Run(
            Ort::RunOptions{ nullptr },
            ctx->lingbot_onnx->depth_input_names_raw.data(),
            depth_input_tensors.data(),
            depth_input_tensors.size(),
            ctx->lingbot_onnx->depth_output_names_raw.data(),
            ctx->lingbot_onnx->depth_output_names_raw.size());
    std::cerr << "[LingBot-MAP][time] dpt_onnx_ms=" << lingbot_elapsed_ms(stage_start) << "\n";
    lingbot_log_rss("after_dpt_onnx");
    lingbot_validate_depth_outputs(depth_outputs);

    std::vector<std::vector<int64_t>> depth_output_shapes;
    std::vector<int64_t> depth_output_float_counts;
    depth_output_shapes.reserve(depth_outputs.size());
    depth_output_float_counts.reserve(depth_outputs.size());
    for (const auto & output : depth_outputs) {
        auto output_info = output.GetTensorTypeAndShapeInfo();
        depth_output_shapes.push_back(output_info.GetShape());
        depth_output_float_counts.push_back((int64_t) output_info.GetElementCount());
    }

    if (runtime.pose_encoding.size() != (size_t) aggregator_input.n_frames * 9) {
        throw std::runtime_error("LingBot-MAP runtime pose output shape does not match frame count");
    }

    const bool save_point_cloud = options.output_point_cloud && cfg.output_point_cloud;
    const auto world_points_paths = save_point_cloud ? lingbot_make_world_points_paths(ctx->config_dir) : std::pair<std::string, std::string>{};
    const auto postprocess = lingbot_postprocess_reconstruction(
            runtime.pose_encoding.data(),
            "camera_head_ggml_runtime",
            depth_outputs[0].GetTensorData<float>(),
            depth_outputs[1].GetTensorData<float>(),
            depth_output_shapes[0],
            depth_output_shapes[1],
            aggregator_input.n_frames,
            world_points_paths.first,
            world_points_paths.second);
    lingbot_log_rss("after_postprocess");

    server_smt_lingbot_map_reconstruct_result result;
    result.architecture = ctx->lingbot_map->architecture();
    result.message = "LingBot-MAP ViT ONNX inference completed; aggregator/camera_head GGML runtime ran on SMT; DPT ONNX ran; postprocess completed";
    result.stages = {
        "config_loaded",
        "images_preprocessed",
        "vit_onnx_ran",
        "aggregator_input_prepared",
        "aggregator_camera_head_ggml_runtime_ran",
        "depth_onnx_ran",
        "postprocess_completed",
    };
    if (!postprocess.world_points_path.empty()) {
        result.stages.push_back("point_cloud_bin_saved");
    }
    result.tensor_count = ctx->lingbot_map->tensor_count();
    result.n_images = (int32_t) images.size();
    result.image_size = cfg.image_size;
    result.patch_size = cfg.patch_size;
    result.hidden_size = cfg.hidden_size;
    result.camera_hidden_size = cfg.camera_hidden_size;
    result.preprocess_width = preproc.target_w;
    result.preprocess_height = preproc.target_h;
    result.vision_input_float_count = (int64_t) preproc.tensor_nchw.size();
    result.vision_output_float_count = vision_output_float_count;
    result.vision_output_frames = (int32_t) vision_output_shape[0];
    result.vision_output_tokens = (int32_t) vision_output_shape[1];
    result.vision_output_hidden = (int32_t) vision_output_shape[2];
    result.aggregator_tokens_per_frame = aggregator_input.tokens_per_frame;
    result.aggregator_patch_start_idx = aggregator_input.patch_start_idx;
    result.aggregator_patch_tokens = aggregator_input.patch_tokens;
    result.aggregator_vit_prefix_tokens = aggregator_input.vit_prefix_tokens;
    result.aggregator_graph_nodes = runtime.graph_nodes;
    result.aggregator_graph_selected_outputs = runtime.selected_output_count;
    result.aggregator_graph_frame_blocks = runtime.frame_block_count;
    result.aggregator_graph_global_blocks = runtime.global_block_count;
    result.aggregator_graph_tokens_per_frame = runtime.tokens_per_frame;
    result.aggregator_graph_patch_start_idx = runtime.patch_start_idx;
    result.aggregator_graph_selected_output_shapes = runtime.selected_output_shapes;
    result.aggregator_selected_layers = runtime.selected_layers;
    result.camera_head_graph_nodes = runtime.graph_nodes;
    result.camera_head_trunk_blocks = runtime.camera_trunk_block_count;
    result.camera_head_iterations = runtime.camera_iteration_count;
    result.camera_head_pose_dim = runtime.camera_pose_dim;
    result.camera_head_input_shape = runtime.camera_head_input_shape;
    result.camera_head_final_pose_shape = runtime.camera_head_final_pose_shape;
    result.camera_head_iteration_pose_shapes = runtime.camera_head_iteration_pose_shapes;
    result.ggml_runtime_graph_nodes = runtime.graph_nodes;
    result.ggml_runtime_backend = runtime.backend_name;
    result.ggml_runtime_buffer_type = runtime.buffer_type_name;
    result.depth_onnx_input_count = (int32_t) ctx->lingbot_onnx->depth_input_names.size();
    result.depth_onnx_output_count = (int32_t) depth_outputs.size();
    result.depth_onnx_input_float_count = depth_input_float_count;
    result.depth_input_source = "aggregator_ggml_runtime_selected_outputs";
    result.depth_input_names = ctx->lingbot_onnx->depth_input_names;
    result.depth_output_names = ctx->lingbot_onnx->depth_output_names;
    result.depth_input_shapes = std::move(depth_input_shapes);
    result.depth_output_shapes = std::move(depth_output_shapes);
    result.depth_output_float_counts = std::move(depth_output_float_counts);
    result.pose_output_source = postprocess.pose_source;
    result.pose_encoding_shape = postprocess.pose_encoding_shape;
    result.extrinsic_shape = postprocess.extrinsic_shape;
    result.intrinsic_shape = postprocess.intrinsic_shape;
    result.world_points_shape = postprocess.world_points_shape;
    result.world_points_conf_shape = postprocess.world_points_conf_shape;
    result.pose_encoding_sample = postprocess.pose_encoding_sample;
    result.extrinsic_first = postprocess.extrinsic_first;
    result.intrinsic_first = postprocess.intrinsic_first;
    result.world_points_sample = postprocess.world_points_sample;
    result.world_points_path = postprocess.world_points_path;
    result.world_points_bytes = postprocess.world_points_bytes;
    result.postprocess_point_count = postprocess.point_count;
    result.postprocess_sample_count = postprocess.sample_count;
    result.depth_min = postprocess.depth_min;
    result.depth_max = postprocess.depth_max;
    result.depth_mean = postprocess.depth_mean;
    result.depth_conf_min = postprocess.depth_conf_min;
    result.depth_conf_max = postprocess.depth_conf_max;
    result.depth_conf_mean = postprocess.depth_conf_mean;
    result.vision_input_shape = std::move(input_shape);
    result.vision_output_shape = std::move(vision_output_shape);
    result.resized_heights = std::move(preproc.resized_heights);
    result.output_pose = options.output_pose && cfg.output_pose;
    result.output_depth = options.output_depth && cfg.output_depth;
    result.output_point_cloud = options.output_point_cloud && cfg.output_point_cloud;
    result.onnx_sessions_loaded = true;
    result.inference_ready = true;
    return result;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(images);
    GGML_UNUSED(options);
    throw std::runtime_error("SMT media backend is not compiled");
#endif
}

server_smt_image_chunk server_smt_vision_encode_image_bin(server_smt_vision_context *  ctx,
                                                          const std::vector<uint8_t> & data) {
    if (ctx == nullptr) {
        throw std::runtime_error("SMT context is null");
    }

#if defined(LLAMA_SERVER_SMT_VISION)
    std::lock_guard<std::mutex> lock(ctx->mu);

    if (ctx->smt_vision == nullptr) {
        throw std::runtime_error("SMT vision backend is not initialized");
    }

    std::vector<uint8_t> smt_input = data;
    auto                 preproc =
        smt_vision_preprocess_if_image(data, ctx->architecture, ctx->smt_vision ? ctx->smt_vision->input_width() : 0,
                                       ctx->smt_vision ? ctx->smt_vision->input_height() : 0,
                                       ctx->smt_vision ? &ctx->smt_vision->preprocess_config() : nullptr);
    if (preproc.was_image) {
        smt_input = std::move(preproc.tensor_bytes);
    }

    const std::string tmp_file = write_temp_bin_file(smt_input);

    server_smt_image_chunk out;
    out.type = server_smt_media_type::image;
    try {
        const int64_t t0 = ggml_time_us();
        out.embd         = ctx->smt_vision->encode_image(tmp_file);
        out.t_encode_ms  = (ggml_time_us() - t0) / 1e3;
        std::remove(tmp_file.c_str());
    } catch (...) {
        std::remove(tmp_file.c_str());
        throw;
    }

    if (ctx->hidden_size <= 0 || out.embd.empty() || out.embd.size() % (size_t) ctx->hidden_size != 0) {
        throw std::runtime_error("Invalid SMT embedding shape");
    }

    const int32_t n_image_tokens = (int32_t) (out.embd.size() / (size_t) ctx->hidden_size);
    auto          grid           = infer_image_grid_xy(n_image_tokens);

    const int32_t n_pos_img = ctx->use_mrope_pos ? std::max(grid.first, grid.second) : n_image_tokens;
    out.n_tokens            = (int32_t) ctx->tok_img_beg.size() + n_image_tokens + (int32_t) ctx->tok_img_end.size();
    out.n_pos               = (int32_t) ctx->tok_img_beg.size() + n_pos_img + (int32_t) ctx->tok_img_end.size();
    out.grid_nx             = grid.first;
    out.grid_ny             = grid.second;
    out.id                  = fnv_hash(data.data(), data.size());

    return out;
#else
    GGML_UNUSED(data);
    throw std::runtime_error("SMT media backend is not compiled. Rebuild with LLAMA_SERVER_SMT_VISION=ON.");
#endif
}

server_smt_image_chunk server_smt_vision_encode_media_bin(server_smt_vision_context *  ctx,
                                                          const std::vector<uint8_t> & data) {
#if defined(LLAMA_SERVER_SMT_VISION)
    if (looks_like_audio_file(data)) {
        if (ctx == nullptr || ctx->smt_audio == nullptr) {
            throw std::runtime_error("SMT audio backend is not initialized");
        }

        std::lock_guard<std::mutex> lock(ctx->mu);
        const std::string           tmp_file = write_temp_bin_file(data);

        server_smt_image_chunk out;
        out.type = server_smt_media_type::audio;
        try {
            const int64_t t0 = ggml_time_us();
            out.embd         = ctx->smt_audio->encode_audio(tmp_file);
            out.t_encode_ms  = (ggml_time_us() - t0) / 1e3;
            std::remove(tmp_file.c_str());
        } catch (...) {
            std::remove(tmp_file.c_str());
            throw;
        }

        if (ctx->hidden_size <= 0 || out.embd.empty() || out.embd.size() % (size_t) ctx->hidden_size != 0) {
            throw std::runtime_error("Invalid SMT audio embedding shape");
        }

        const int32_t n_audio_tokens = (int32_t) (out.embd.size() / (size_t) ctx->hidden_size);
        out.n_tokens = (int32_t) ctx->tok_audio_beg.size() + n_audio_tokens + (int32_t) ctx->tok_audio_end.size();
        out.n_pos    = out.n_tokens;
        out.grid_nx  = n_audio_tokens;
        out.grid_ny  = 1;
        out.id       = std::string("audio:") + fnv_hash(data.data(), data.size());
        return out;
    }
#endif

    return server_smt_vision_encode_image_bin(ctx, data);
}

int32_t server_smt_vision_decode_chunk(llama_context *                   lctx,
                                       const server_smt_vision_context * ctx,
                                       const server_smt_image_chunk &    chunk,
                                       llama_pos &                       n_past,
                                       int32_t                           seq_id,
                                       int32_t                           n_batch,
                                       bool                              logits_last) {
    if (ctx == nullptr) {
        return -1;
    }

    const int32_t n_embd_tokens = (int32_t) (chunk.embd.size() / (size_t) ctx->hidden_size);
    if (n_embd_tokens <= 0) {
        return -1;
    }

    const std::vector<llama_token> * tok_beg       = &ctx->tok_img_beg;
    const std::vector<llama_token> * tok_end       = &ctx->tok_img_end;
    bool                             use_mrope_pos = ctx->use_mrope_pos;
    int32_t                          grid_nx       = chunk.grid_nx;
    int32_t                          grid_ny       = chunk.grid_ny;

    if (chunk.type == server_smt_media_type::audio) {
        tok_beg       = &ctx->tok_audio_beg;
        tok_end       = &ctx->tok_audio_end;
        use_mrope_pos = false;
        grid_nx       = n_embd_tokens;
        grid_ny       = 1;
    }

    if (!tok_beg->empty()) {
        if (decode_tokens(lctx, *tok_beg, n_past, seq_id, n_batch, false) != 0) {
            return -1;
        }
    }

    const bool logits_on_embd = logits_last && tok_end->empty();
    if (decode_embd(lctx, chunk.embd.data(), n_embd_tokens, ctx->hidden_size, n_past, seq_id, n_batch, logits_on_embd,
                    use_mrope_pos, grid_nx, grid_ny) != 0) {
        return -1;
    }

    if (!tok_end->empty()) {
        if (decode_tokens(lctx, *tok_end, n_past, seq_id, n_batch, logits_last) != 0) {
            return -1;
        }
    }

    return 0;
}
