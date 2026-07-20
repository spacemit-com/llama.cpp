#pragma once

#include "llama.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

enum class server_smt_media_type : uint8_t {
    image = 0,
    audio = 1,
};

struct server_smt_image_chunk {
    std::string id;
    server_smt_media_type type = server_smt_media_type::image;
    std::vector<float> embd;

    int32_t n_embd_tokens = 0;
    int32_t embd_width = 0;
    int32_t n_tokens = 0;
    int32_t n_pos = 0;
    int32_t grid_nx = 0;
    int32_t grid_ny = 0;

    double t_encode_ms = 0.0; // encoder wall-clock time in ms
    double t_image_decode_ms = 0.0; // image embedding decode wall-clock time in ms
};

struct server_smt_vision_context;

struct server_smt_lingbot_map_reconstruct_options {
    bool output_pose = true;
    bool output_depth = true;
    bool output_point_cloud = true;
    int32_t max_frames = -1;
};

struct server_smt_lingbot_map_reconstruct_result {
    std::string architecture;
    std::string message;
    std::vector<std::string> stages;

    int64_t tensor_count = 0;
    int32_t n_images = 0;
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t hidden_size = 0;
    int32_t camera_hidden_size = 0;
    int32_t preprocess_width = 0;
    int32_t preprocess_height = 0;
    int64_t vision_input_float_count = 0;
    int64_t vision_output_float_count = 0;
    int32_t vision_output_frames = 0;
    int32_t vision_output_tokens = 0;
    int32_t vision_output_hidden = 0;
    int32_t aggregator_tokens_per_frame = 0;
    int32_t aggregator_patch_start_idx = 0;
    int32_t aggregator_patch_tokens = 0;
    int32_t aggregator_vit_prefix_tokens = 0;
    int32_t aggregator_probe_graph_nodes = 0;
    int32_t aggregator_global_probe_graph_nodes = 0;
    int32_t aggregator_global_probe_input_tokens = 0;
    int32_t aggregator_full_probe_graph_nodes = 0;
    int32_t aggregator_full_probe_selected_outputs = 0;
    int32_t aggregator_full_probe_frame_blocks = 0;
    int32_t aggregator_full_probe_global_blocks = 0;
    int32_t aggregator_graph_nodes = 0;
    int32_t aggregator_graph_selected_outputs = 0;
    int32_t aggregator_graph_frame_blocks = 0;
    int32_t aggregator_graph_global_blocks = 0;
    int32_t aggregator_graph_tokens_per_frame = 0;
    int32_t aggregator_graph_patch_start_idx = 0;
    int32_t camera_head_graph_nodes = 0;
    int32_t camera_head_trunk_blocks = 0;
    int32_t camera_head_iterations = 0;
    int32_t camera_head_pose_dim = 0;
    int32_t ggml_runtime_graph_nodes = 0;
    int32_t depth_onnx_input_count = 0;
    int32_t depth_onnx_output_count = 0;
    int64_t depth_onnx_input_float_count = 0;
    int64_t postprocess_point_count = 0;
    int64_t world_points_bytes = 0;
    int32_t postprocess_sample_count = 0;
    double depth_min = 0.0;
    double depth_max = 0.0;
    double depth_mean = 0.0;
    double depth_conf_min = 0.0;
    double depth_conf_max = 0.0;
    double depth_conf_mean = 0.0;
    std::string depth_input_source;
    std::string pose_output_source;
    std::string ggml_runtime_backend;
    std::string ggml_runtime_buffer_type;
    std::string world_points_path;
    std::vector<int32_t> aggregator_probe_qkv_shape;
    std::vector<int32_t> aggregator_probe_output_shape;
    std::vector<int32_t> aggregator_global_probe_qkv_shape;
    std::vector<int32_t> aggregator_global_probe_output_shape;
    std::vector<int32_t> aggregator_full_probe_final_frame_shape;
    std::vector<int32_t> aggregator_full_probe_final_global_shape;
    std::vector<int32_t> aggregator_graph_final_frame_shape;
    std::vector<int32_t> aggregator_graph_final_global_shape;
    std::vector<std::vector<int32_t>> aggregator_graph_selected_output_shapes;
    std::vector<int32_t> aggregator_selected_layers;
    std::vector<int32_t> camera_head_input_shape;
    std::vector<int32_t> camera_head_final_pose_shape;
    std::vector<std::vector<int32_t>> camera_head_iteration_pose_shapes;
    std::vector<std::string> depth_input_names;
    std::vector<std::string> depth_output_names;
    std::vector<std::vector<int64_t>> depth_input_shapes;
    std::vector<std::vector<int64_t>> depth_output_shapes;
    std::vector<int64_t> depth_output_float_counts;
    std::vector<int64_t> pose_encoding_shape;
    std::vector<int64_t> extrinsic_shape;
    std::vector<int64_t> intrinsic_shape;
    std::vector<int64_t> world_points_shape;
    std::vector<int64_t> world_points_conf_shape;
    std::vector<float> pose_encoding_sample;
    std::vector<float> extrinsic_first;
    std::vector<float> intrinsic_first;
    std::vector<float> world_points_sample;
    std::vector<int64_t> vision_input_shape;
    std::vector<int64_t> vision_output_shape;
    std::vector<int32_t> resized_heights;

    bool output_pose = true;
    bool output_depth = true;
    bool output_point_cloud = true;
    bool onnx_sessions_loaded = false;
    bool inference_ready = false;
};

#if defined(LLAMA_SERVER_SMT_VISION)
bool server_smt_vision_config_is_lingbot_map(const std::string & config_dir);

server_smt_vision_context * server_smt_vision_init(
        llama_context * lctx,
        const std::string & config_dir,
        bool warmup = true);

void server_smt_vision_free(server_smt_vision_context * ctx);

bool server_smt_vision_supports_image(const server_smt_vision_context * ctx);
bool server_smt_vision_supports_audio(const server_smt_vision_context * ctx);
bool server_smt_vision_supports_prompt_embeddings(const server_smt_vision_context * ctx);
bool server_smt_vision_is_lingbot_map(const server_smt_vision_context * ctx);

server_smt_lingbot_map_reconstruct_result server_smt_vision_lingbot_map_reconstruct(
        server_smt_vision_context * ctx,
        const std::vector<std::vector<uint8_t>> & images,
        const server_smt_lingbot_map_reconstruct_options & options);

server_smt_image_chunk server_smt_vision_encode_media_bin(
        server_smt_vision_context * ctx,
        const std::vector<uint8_t> & data);

server_smt_image_chunk server_smt_vision_encode_image_bin(
        server_smt_vision_context * ctx,
        const std::vector<uint8_t> & data);

int32_t server_smt_vision_decode_chunk(
        llama_context * lctx,
        const server_smt_vision_context * ctx,
        const server_smt_image_chunk & chunk,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last);
#else
inline bool server_smt_vision_config_is_lingbot_map(const std::string & /* config_dir */) {
    return false;
}

inline server_smt_vision_context * server_smt_vision_init(
        llama_context * /* lctx */,
        const std::string & /* config_dir */,
        bool /* warmup */ = true) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline void server_smt_vision_free(server_smt_vision_context * /* ctx */) {
}

inline bool server_smt_vision_supports_image(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline bool server_smt_vision_supports_audio(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline bool server_smt_vision_supports_prompt_embeddings(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline bool server_smt_vision_is_lingbot_map(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline server_smt_lingbot_map_reconstruct_result server_smt_vision_lingbot_map_reconstruct(
        server_smt_vision_context * /* ctx */,
        const std::vector<std::vector<uint8_t>> & /* images */,
        const server_smt_lingbot_map_reconstruct_options & /* options */) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline server_smt_image_chunk server_smt_vision_encode_media_bin(
        server_smt_vision_context * /* ctx */,
        const std::vector<uint8_t> & /* data */) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline server_smt_image_chunk server_smt_vision_encode_image_bin(
        server_smt_vision_context * /* ctx */,
        const std::vector<uint8_t> & /* data */) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline int32_t server_smt_vision_decode_chunk(
        llama_context * /* lctx */,
        const server_smt_vision_context * /* ctx */,
        const server_smt_image_chunk & /* chunk */,
        llama_pos & /* n_past */,
        int32_t /* seq_id */,
        int32_t /* n_batch */,
        bool /* logits_last */) {
    return -1;
}
#endif
