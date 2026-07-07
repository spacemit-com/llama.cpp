#include "smt-audio-wrapper.h"

#include "ggml-profile.h"
#include "mtmd-audio.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#ifndef _WIN32
#    include <dirent.h>
#    include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

namespace onnxruntime {
extern const OrtApi * g_ort;
}

namespace {

constexpr int32_t      k_gemma4_default_warmup_feature_frames = 135;
constexpr const char * k_gemma4_audio_architecture            = "Gemma4Audio";

struct smt_audio_config {
    std::vector<std::string>                     architectures;
    std::string                                  encoder_model_path;
    std::string                                  frontend_model_path;
    std::string                                  backend_model_path;
    std::unordered_map<std::string, std::string> ep_config;
    int64_t                                      d_model          = 0;
    int64_t                                      hidden_size      = 0;
    int32_t                                      num_mel_bins     = 128;
    int32_t                                      sample_rate      = 16000;
    int32_t                                      n_fft            = 400;
    int32_t                                      window_len       = 400;
    int32_t                                      hop_len          = 160;
    int32_t                                      intra_thread_num = 4;
    int32_t                                      inter_thread_num = 1;
    int32_t                                      lfr_m            = 0;
    int32_t                                      lfr_n            = 0;
    int32_t                                      feature_frames   = 0;
};

struct gemma4_encoder_input_shape {
    int32_t feature_frames = 0;
    int32_t num_mel_bins   = 0;

    bool dynamic_feature_frames() const {
        return feature_frames <= 0;
    }
};

struct gemma4_encoder_session {
    explicit gemma4_encoder_session(Ort::Session && session_in) :
        session(std::move(session_in)) {}

    std::string model_path;
    Ort::Session session{ nullptr };
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char *> input_names_raw;
    std::vector<const char *> output_names_raw;
    gemma4_encoder_input_shape input_shape;
};

static bool arch_is_gemma4_audio(const std::string & arch_name) {
    return arch_name == k_gemma4_audio_architecture;
}

static bool uses_gemma4_single_encoder(const smt_audio_config & config, const std::string & arch_name) {
    return !config.encoder_model_path.empty() && arch_is_gemma4_audio(arch_name);
}

static std::string read_file_to_string(const std::string & path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static size_t find_closing_brace(const std::string & text, size_t start_pos) {
    int depth = 0;
    for (size_t index = start_pos; index < text.size(); ++index) {
        if (text[index] == '{') {
            ++depth;
        } else if (text[index] == '}') {
            --depth;
            if (depth == 0) {
                return index;
            }
        }
    }
    return std::string::npos;
}

static std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

static std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir == ".") {
        return name;
    }
    if (dir.back() == '/' || dir.back() == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static std::string extract_string_value(const std::string & text, const std::string & key) {
    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return {};
    }

    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return {};
    }

    const size_t first_quote = text.find('"', colon_pos + 1);
    if (first_quote == std::string::npos) {
        return {};
    }

    const size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return {};
    }

    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

static int64_t extract_int64_value(const std::string & text, const std::string & key, int64_t default_value) {
    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return default_value;
    }

    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return default_value;
    }

    size_t value_start = colon_pos + 1;
    while (value_start < text.size() && std::isspace(static_cast<unsigned char>(text[value_start]))) {
        ++value_start;
    }

    size_t value_end = value_start;
    if (value_end < text.size() && (text[value_end] == '-' || text[value_end] == '+')) {
        ++value_end;
    }
    while (value_end < text.size() && std::isdigit(static_cast<unsigned char>(text[value_end]))) {
        ++value_end;
    }

    if (value_end == value_start) {
        return default_value;
    }

    try {
        return std::stoll(text.substr(value_start, value_end - value_start));
    } catch (...) {
        return default_value;
    }
}

static std::unordered_map<std::string, std::string> extract_string_map(const std::string & text,
                                                                       const std::string & key) {
    std::unordered_map<std::string, std::string> values;

    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }

    const size_t brace_start = text.find('{', key_pos + marker.size());
    const size_t brace_end   = find_closing_brace(text, brace_start);
    if (brace_start == std::string::npos || brace_end == std::string::npos || brace_end <= brace_start) {
        return values;
    }

    std::string content = text.substr(brace_start + 1, brace_end - brace_start - 1);
    size_t      pos     = 0;
    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }

        // Find key
        if (content[pos] != '"') {
            break;
        }
        const size_t key_start = pos + 1;
        const size_t key_end   = content.find('"', key_start);
        if (key_end == std::string::npos) {
            break;
        }
        std::string map_key = content.substr(key_start, key_end - key_start);
        pos                 = key_end + 1;

        // Skip :
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] != ':') {
            break;
        }
        ++pos;

        // Skip whitespace
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }

        // Find value
        if (content[pos] != '"') {
            break;
        }
        const size_t value_start = pos + 1;
        const size_t value_end   = content.find('"', value_start);
        if (value_end == std::string::npos) {
            break;
        }
        std::string map_value = content.substr(value_start, value_end - value_start);
        pos                   = value_end + 1;

        values[map_key] = map_value;

        // Skip comma or end
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
        }
    }

    return values;
}

static std::vector<std::string> extract_string_array(const std::string & text, const std::string & key) {
    std::vector<std::string> values;

    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }

    const size_t bracket_start = text.find('[', key_pos + marker.size());
    const size_t bracket_end   = text.find(']', bracket_start == std::string::npos ? key_pos : bracket_start + 1);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos || bracket_end <= bracket_start) {
        return values;
    }

    std::string content = text.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    size_t      pos     = 0;
    while (pos < content.size()) {
        const size_t first_quote = content.find('"', pos);
        if (first_quote == std::string::npos) {
            break;
        }
        const size_t second_quote = content.find('"', first_quote + 1);
        if (second_quote == std::string::npos) {
            break;
        }
        values.push_back(content.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }

    return values;
}

static std::string normalize_path(const std::string & base_dir, const std::string & path) {
    const std::string trimmed = trim_ascii(path);
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed.front() == '/') {
        return trimmed;
    }
    return base_dir + "/" + trimmed;
}

static bool contains_legacy_spacemit_ep_config(const std::string & text) {
    return text.find("\"spacemit_ep_intra_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_inter_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_intra_thread_affinity\"") != std::string::npos;
}

static void warn_legacy_spacemit_ep_config_if_needed(const std::string & text, const char * section_name) {
    if (!contains_legacy_spacemit_ep_config(text)) {
        return;
    }

    std::cerr << "[SMT][audio] warning: detected deprecated legacy Spacemit EP config keys";
    if (section_name != nullptr && section_name[0] != '\0') {
        std::cerr << " in " << section_name;
    }
    std::cerr << "; this style will be removed in a future release. "
              << "Please migrate to the `ep_config` format.\n";
}

static void apply_legacy_spacemit_ep_config(const std::string & text, smt_audio_config & config) {
    config.intra_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_intra_thread_num", config.intra_thread_num);
    config.inter_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_inter_thread_num", config.inter_thread_num);

    const std::string affinity = extract_string_value(text, "spacemit_ep_intra_thread_affinity");
    if (!affinity.empty() && config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
    }

    if (config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(config.intra_thread_num);
    }
    if (config.ep_config.find("SPACEMIT_EP_INTER_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(config.inter_thread_num);
    }
}

static bool parse_audio_config_block(const std::string & config_dir,
                                     const std::string & content,
                                     smt_audio_config &  config) {
    const size_t audio_start = content.find("\"audio_model\":");
    if (audio_start == std::string::npos) {
        return false;
    }

    const size_t audio_block_start = content.find('{', audio_start);
    const size_t audio_block_end   = find_closing_brace(content, audio_block_start);
    if (audio_block_start == std::string::npos || audio_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'audio_model' block.\n";
        return false;
    }

    const std::string audio_block = content.substr(audio_block_start, audio_block_end - audio_block_start + 1);
    warn_legacy_spacemit_ep_config_if_needed(audio_block, "audio_model");
    config.encoder_model_path = normalize_path(config_dir, extract_string_value(audio_block, "encoder_model_path"));
    if (config.encoder_model_path.empty()) {
        config.encoder_model_path = normalize_path(config_dir, extract_string_value(audio_block, "encoder_path"));
    }
    if (config.encoder_model_path.empty()) {
        config.encoder_model_path = normalize_path(config_dir, extract_string_value(audio_block, "onnx_model_path"));
    }
    config.frontend_model_path = normalize_path(config_dir, extract_string_value(audio_block, "frontend_model_path"));
    if (config.frontend_model_path.empty()) {
        config.frontend_model_path = normalize_path(config_dir, extract_string_value(audio_block, "frontend_path"));
    }
    config.backend_model_path = normalize_path(config_dir, extract_string_value(audio_block, "backend_model_path"));
    if (config.backend_model_path.empty()) {
        config.backend_model_path = normalize_path(config_dir, extract_string_value(audio_block, "backend_path"));
    }

    config.d_model     = extract_int64_value(audio_block, "d_model", config.d_model);
    config.hidden_size = extract_int64_value(audio_block, "hidden_size", config.hidden_size);
    if (config.hidden_size <= 0) {
        config.hidden_size = extract_int64_value(audio_block, "output_dim", config.hidden_size);
    }
    config.num_mel_bins = (int32_t) extract_int64_value(audio_block, "num_mel_bins", config.num_mel_bins);
    config.sample_rate  = (int32_t) extract_int64_value(audio_block, "sample_rate", config.sample_rate);
    config.n_fft        = (int32_t) extract_int64_value(audio_block, "n_fft", config.n_fft);
    config.window_len   = (int32_t) extract_int64_value(audio_block, "window_len", config.window_len);
    config.hop_len      = (int32_t) extract_int64_value(audio_block, "hop_len", config.hop_len);
    config.lfr_m        = (int32_t) extract_int64_value(audio_block, "lfr_m", config.lfr_m);
    config.lfr_n        = (int32_t) extract_int64_value(audio_block, "lfr_n", config.lfr_n);
    config.feature_frames = (int32_t) extract_int64_value(audio_block, "feature_frames", config.feature_frames);
    config.ep_config    = extract_string_map(audio_block, "ep_config");
    apply_legacy_spacemit_ep_config(audio_block, config);
    config.architectures = extract_string_array(content, "architectures");

    return true;
}

static std::string find_split_metadata_file(const std::string & config_dir) {
    DIR * dir = opendir(config_dir.c_str());
    if (dir == nullptr) {
        return {};
    }

    std::string found;
    while (const dirent * entry = readdir(dir)) {
        if (entry->d_type != DT_REG && entry->d_type != DT_UNKNOWN) {
            continue;
        }
        const std::string name = entry->d_name;
        if (name.size() >= strlen("-encoder-split-metadata.json") &&
            name.compare(name.size() - strlen("-encoder-split-metadata.json"), strlen("-encoder-split-metadata.json"),
                         "-encoder-split-metadata.json") == 0) {
            found = config_dir + "/" + name;
            break;
        }
    }

    closedir(dir);
    return found;
}

static bool load_smt_audio_config_from_metadata(const std::string & config_dir,
                                                const std::string & metadata_path,
                                                smt_audio_config &  config) {
    const std::string content = read_file_to_string(metadata_path);
    if (content.empty()) {
        return false;
    }

    config.frontend_model_path = normalize_path(config_dir, extract_string_value(content, "frontend_onnx"));
    config.backend_model_path  = normalize_path(config_dir, extract_string_value(content, "backend_onnx"));
    config.num_mel_bins        = (int32_t) extract_int64_value(content, "num_mel_bins", config.num_mel_bins);
    config.d_model             = extract_int64_value(content, "d_model", config.d_model);
    config.hidden_size         = extract_int64_value(content, "output_dim", config.hidden_size);
    if (config.architectures.empty()) {
        config.architectures = { "Qwen3ASRForConditionalGeneration" };
    }

    return !config.frontend_model_path.empty() && !config.backend_model_path.empty() && config.d_model > 0 &&
           config.hidden_size > 0;
}

static bool load_smt_audio_config(const std::string & config_dir, smt_audio_config & config) {
    const std::string config_path    = config_dir + "/config.json";
    const std::string config_content = read_file_to_string(config_path);
    if (!config_content.empty() && parse_audio_config_block(config_dir, config_content, config)) {
        warn_legacy_spacemit_ep_config_if_needed(config_content, "top-level config");
        if (config.hidden_size <= 0) {
            const size_t text_start = config_content.find("\"text_model\":");
            if (text_start != std::string::npos) {
                const size_t text_block_start = config_content.find('{', text_start);
                const size_t text_block_end   = find_closing_brace(config_content, text_block_start);
                if (text_block_start != std::string::npos && text_block_end != std::string::npos) {
                    const std::string text_block =
                        config_content.substr(text_block_start, text_block_end - text_block_start + 1);
                    config.hidden_size = extract_int64_value(text_block, "hidden_size", config.hidden_size);
                }
            }
        }
        // 从顶层配置读取 ep_config（如果 audio_model 块中没有设置）
        auto top_ep_config = extract_string_map(config_content, "ep_config");
        for (const auto & pair : top_ep_config) {
            if (config.ep_config.find(pair.first) == config.ep_config.end()) {
                config.ep_config[pair.first] = pair.second;
            }
        }
        apply_legacy_spacemit_ep_config(config_content, config);
        // 从 ep_config 提取线程数设置
        if (config.ep_config.count("SPACEMIT_EP_INTRA_THREAD_NUM")) {
            config.intra_thread_num = std::stoi(config.ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"]);
        }
        if (config.ep_config.count("SPACEMIT_EP_INTER_THREAD_NUM")) {
            config.inter_thread_num = std::stoi(config.ep_config["SPACEMIT_EP_INTER_THREAD_NUM"]);
        }
        if (!config.architectures.empty() && config.hidden_size > 0) {
            return true;
        }
    }

    const std::string metadata_path = find_split_metadata_file(config_dir);
    if (metadata_path.empty()) {
        return false;
    }

    return load_smt_audio_config_from_metadata(config_dir, metadata_path, config);
}

static int floor_div(int value, int divisor) {
    int quotient  = value / divisor;
    int remainder = value % divisor;
    if (remainder != 0 && ((remainder > 0) != (divisor > 0))) {
        --quotient;
    }
    return quotient;
}

static int get_feat_extract_output_lengths(int input_lengths) {
    const int input_lengths_leave = input_lengths % 100;
    const int feat_lengths        = floor_div(input_lengths_leave - 1, 2) + 1;
    const int output_lengths = floor_div(floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 + (input_lengths / 100) * 13;
    return output_lengths;
}

static bool decode_audio_file(const std::string & path, int target_sample_rate, std::vector<float> & pcmf32_mono) {
    const int         channels       = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sample_rate);
    ma_decoder        decoder;

    if (ma_decoder_init_file(path.c_str(), &decoder_config, &decoder) != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize((size_t) frame_count);
    ma_uint64 frames_read = 0;
    if (ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize((size_t) frames_read);
    ma_decoder_uninit(&decoder);
    return !pcmf32_mono.empty();
}

static bool init_spacemit_execution_provider(Ort::SessionOptions &                                options,
                                             const std::unordered_map<std::string, std::string> & provider_options,
                                             std::string &                                        error_message) {
    std::vector<const char *> keys;
    std::vector<const char *> values;
    keys.reserve(provider_options.size());
    values.reserve(provider_options.size());
    for (const auto & entry : provider_options) {
        keys.push_back(entry.first.c_str());
        values.push_back(entry.second.c_str());
    }

    void * handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) {
        error_message = std::string("failed to load libspacemit_ep.so: ") + dlerror();
        return false;
    }

    auto * ep_init =
        reinterpret_cast<OrtStatus * (*) (OrtSessionOptions *, const char * const *, const char * const *, size_t)>(
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
}

static void append_optional_spacemit_ep(Ort::SessionOptions &    session_options,
                                        const char *             session_name,
                                        const smt_audio_config & config) {
    std::unordered_map<std::string, std::string> provider_options = config.ep_config;

    // Add defaults if not specified in ep_config
    if (provider_options.find("SPACEMIT_EP_INTRA_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(config.intra_thread_num);
    }
    if (provider_options.find("SPACEMIT_EP_INTER_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(config.inter_thread_num);
    }
    std::string error_message;
    if (!init_spacemit_execution_provider(session_options, provider_options, error_message)) {
        throw std::runtime_error(std::string("[SMT][audio] failed to initialize Spacemit EP for ") + session_name +
                                 ": " + error_message);
    }
    std::cerr << "[SMT][audio] Spacemit EP enabled for " << session_name << " (";
    for (const auto & pair : provider_options) {
        std::cerr << ", " << pair.first << "=" << pair.second;
    }
    std::cerr << ")\n";
}

static bool has_spacemit_ep_affinity(const smt_audio_config & config) {
    auto it = config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY");
    return it != config.ep_config.end() && !trim_ascii(it->second).empty();
}

static std::vector<const char *> make_name_ptrs(const std::vector<std::string> & names) {
    std::vector<const char *> ptrs;
    ptrs.reserve(names.size());
    for (const auto & name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

static std::vector<std::string> get_io_names(Ort::Session & session, bool inputs) {
    std::vector<std::string>         names;
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t                     count = inputs ? session.GetInputCount() : session.GetOutputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto allocated =
            inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(allocated.get());
    }
    return names;
}

static gemma4_encoder_input_shape get_gemma4_encoder_input_shape(const Ort::Session & session) {
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        const auto input_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        const auto shape      = input_info.GetShape();
        if (shape.size() != 3) {
            continue;
        }

        gemma4_encoder_input_shape result;
        result.feature_frames = shape[1] > 0 ? (int32_t) shape[1] : 0;
        result.num_mel_bins   = shape[2] > 0 ? (int32_t) shape[2] : 0;
        return result;
    }

    return {};
}

static Ort::Value make_tensor_f32(const std::vector<int64_t> & shape, std::vector<float> & data) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
}

static Ort::Value make_tensor_bool(const std::vector<int64_t> & shape, std::vector<uint8_t> & data) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor(memory_info, data.data(), data.size() * sizeof(uint8_t), shape.data(), shape.size(),
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

}  // namespace

struct smt_audio_context::impl {
    smt_audio_config    config;
    Ort::Env            env{ ORT_LOGGING_LEVEL_WARNING, "smt-audio" };
    Ort::SessionOptions frontend_options;
    Ort::SessionOptions backend_options;
    Ort::Session        frontend_session{ nullptr };
    Ort::Session        backend_session{ nullptr };

    bool                                   warmup_encoder_sessions = false;
    std::mutex                             encoder_session_mutex;
    std::string                            encoder_session_model_path;
    std::unique_ptr<gemma4_encoder_session> encoder_session;

    std::vector<std::string>  frontend_input_names;
    std::vector<std::string>  frontend_output_names;
    std::vector<const char *> frontend_input_names_raw;
    std::vector<const char *> frontend_output_names_raw;

    std::vector<std::string>  backend_input_names;
    std::vector<std::string>  backend_output_names;
    std::vector<const char *> backend_input_names_raw;
    std::vector<const char *> backend_output_names_raw;

    std::string arch_name;

    void warmup_gemma4_encoder_session(gemma4_encoder_session & encoder) const {
        std::cerr << "[SMT][audio] warmup encoder ONNX session";
        if (!arch_name.empty()) {
            std::cerr << " for " << arch_name;
        }
        std::cerr << ": " << encoder.model_path << "\n";

        const int32_t warmup_frames =
            encoder.input_shape.dynamic_feature_frames() ?
                (config.feature_frames > 0 ? config.feature_frames : k_gemma4_default_warmup_feature_frames) :
                encoder.input_shape.feature_frames;
        std::vector<float>         feature_data((size_t) warmup_frames * config.num_mel_bins, 0.0f);
        std::vector<uint8_t>       feature_mask((size_t) warmup_frames, 1);
        const std::vector<int64_t> feature_shape = { 1, warmup_frames, config.num_mel_bins };
        const std::vector<int64_t> mask_shape    = { 1, warmup_frames };
        auto                       feature_tensor = make_tensor_f32(feature_shape, feature_data);
        auto                       mask_tensor    = make_tensor_bool(mask_shape, feature_mask);
        std::array<Ort::Value, 2>  inputs         = { std::move(feature_tensor), std::move(mask_tensor) };
        (void) encoder.session.Run(Ort::RunOptions{ nullptr }, encoder.input_names_raw.data(), inputs.data(),
                                   inputs.size(), encoder.output_names_raw.data(), encoder.output_names_raw.size());
    }

    gemma4_encoder_session & get_gemma4_encoder_session(const std::string & model_path) {
        if (encoder_session && encoder_session_model_path == model_path) {
            return *encoder_session;
        }
        encoder_session.reset();
        encoder_session_model_path.clear();

        Ort::SessionOptions encoder_options;
        encoder_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const bool ep_affinity_is_configured = has_spacemit_ep_affinity(config);
        if (!ep_affinity_is_configured) {
            encoder_options.SetIntraOpNumThreads(config.intra_thread_num);
            encoder_options.SetInterOpNumThreads(config.inter_thread_num);
        }

        std::cerr << "[SMT][audio] using ORT CPU for Gemma4 encoder: " << model_path << "\n";
        auto encoder = std::make_unique<gemma4_encoder_session>(Ort::Session(env, model_path.c_str(), encoder_options));
        encoder->model_path       = model_path;
        encoder->input_names      = get_io_names(encoder->session, true);
        encoder->output_names     = get_io_names(encoder->session, false);
        encoder->input_names_raw  = make_name_ptrs(encoder->input_names);
        encoder->output_names_raw = make_name_ptrs(encoder->output_names);
        encoder->input_shape      = get_gemma4_encoder_input_shape(encoder->session);

        if (encoder->input_names_raw.size() != 2 || encoder->output_names_raw.size() != 2) {
            throw std::runtime_error("Unexpected SMT audio single-encoder ONNX IO signature");
        }
        if (encoder->input_shape.num_mel_bins <= 0) {
            encoder->input_shape.num_mel_bins = config.num_mel_bins;
        }
        if (encoder->input_shape.feature_frames <= 0 && config.feature_frames > 0) {
            encoder->input_shape.feature_frames = config.feature_frames;
        }
        if (encoder->input_shape.num_mel_bins > 0 && encoder->input_shape.num_mel_bins != config.num_mel_bins) {
            throw std::runtime_error("Gemma4 audio encoder num_mel_bins mismatch: config " +
                                     std::to_string(config.num_mel_bins) + ", ONNX " +
                                     std::to_string(encoder->input_shape.num_mel_bins));
        }

        if (warmup_encoder_sessions) {
            warmup_gemma4_encoder_session(*encoder);
        }

        encoder_session_model_path = model_path;
        encoder_session            = std::move(encoder);
        return *encoder_session;
    }
};

smt_audio_context::~smt_audio_context() = default;

std::unique_ptr<smt_audio_context> smt_audio_context::create(const std::string & config_dir, bool warmup) {
    auto ctx    = std::unique_ptr<smt_audio_context>(new smt_audio_context());
    ctx->pimpl_ = std::make_unique<impl>();
    auto & d    = *ctx->pimpl_;

    if (!load_smt_audio_config(config_dir, d.config)) {
        throw std::runtime_error("Failed to load SMT audio config from: " + config_dir);
    }

    if (!d.config.architectures.empty()) {
        d.arch_name = d.config.architectures[0];
    } else {
        d.arch_name = "Qwen3ASRForConditionalGeneration";
    }

    const bool gemma4_single_encoder = uses_gemma4_single_encoder(d.config, d.arch_name);
    if (!d.config.encoder_model_path.empty() && !gemma4_single_encoder) {
        throw std::runtime_error("SMT audio encoder_model_path is currently supported only for Gemma4 audio models");
    }

    if (gemma4_single_encoder) {
        if (d.config.hidden_size <= 0) {
            throw std::runtime_error("Invalid Gemma4 audio single-encoder hidden_size");
        }
    } else {
        if (d.config.frontend_model_path.empty() || d.config.backend_model_path.empty()) {
            throw std::runtime_error("Missing SMT audio frontend/backend model path");
        }
        if (d.config.d_model <= 0 || d.config.hidden_size <= 0) {
            throw std::runtime_error("Invalid SMT audio model dimensions");
        }
    }

    onnxruntime::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    d.frontend_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    d.backend_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const bool ep_affinity_is_configured = has_spacemit_ep_affinity(d.config);
    if (!ep_affinity_is_configured) {
        d.frontend_options.SetIntraOpNumThreads(d.config.intra_thread_num);
        d.backend_options.SetIntraOpNumThreads(d.config.intra_thread_num);
        d.frontend_options.SetInterOpNumThreads(d.config.inter_thread_num);
        d.backend_options.SetInterOpNumThreads(d.config.inter_thread_num);
    } else {
        std::cerr << "[SMT][audio] detected SPACEMIT_EP_INTRA_THREAD_AFFINITY, skip ORT session thread pinning"
                  << " to avoid conflicting with EP-managed affinity\n";
    }

    if (gemma4_single_encoder) {
        d.warmup_encoder_sessions = warmup;
    } else {
        append_optional_spacemit_ep(d.frontend_options, "frontend", d.config);
        append_optional_spacemit_ep(d.backend_options, "backend", d.config);

        d.frontend_session = Ort::Session(d.env, d.config.frontend_model_path.c_str(), d.frontend_options);
        d.backend_session  = Ort::Session(d.env, d.config.backend_model_path.c_str(), d.backend_options);

        d.frontend_input_names      = get_io_names(d.frontend_session, true);
        d.frontend_output_names     = get_io_names(d.frontend_session, false);
        d.frontend_input_names_raw  = make_name_ptrs(d.frontend_input_names);
        d.frontend_output_names_raw = make_name_ptrs(d.frontend_output_names);

        d.backend_input_names      = get_io_names(d.backend_session, true);
        d.backend_output_names     = get_io_names(d.backend_session, false);
        d.backend_input_names_raw  = make_name_ptrs(d.backend_input_names);
        d.backend_output_names_raw = make_name_ptrs(d.backend_output_names);

        if (d.frontend_input_names_raw.size() != 1 || d.frontend_output_names_raw.size() != 1 ||
            (d.backend_input_names_raw.size() != 1 && d.backend_input_names_raw.size() != 2) ||
            d.backend_output_names_raw.size() != 1) {
            throw std::runtime_error("Unexpected SMT audio ONNX IO signature");
        }

        if (warmup) {
            std::cerr << "[SMT][audio] warmup ONNX sessions";
            if (!d.arch_name.empty()) {
                std::cerr << " for " << d.arch_name;
            }
            std::cerr << "\n";

            int                warmup_t_out;
            std::vector<float> warmup_hidden;

            if (d.config.lfr_m > 0) {
                const int warmup_frames = 10;
                const int feat_dim      = d.config.num_mel_bins * d.config.lfr_m;

                std::vector<float>         frontend_input_data((size_t) warmup_frames * feat_dim, 0.0f);
                const std::vector<int64_t> frontend_input_shape = { 1, warmup_frames, (int64_t) feat_dim };
                auto                       frontend_input       = make_tensor_f32(frontend_input_shape, frontend_input_data);

                std::cerr << "[SMT][audio] warmup frontend ONNX session (FunASR): " << d.config.frontend_model_path << "\n";
                auto frontend_outputs = d.frontend_session.Run(Ort::RunOptions{ nullptr }, d.frontend_input_names_raw.data(),
                                                               &frontend_input, 1, d.frontend_output_names_raw.data(), 1);
                if (frontend_outputs.empty()) {
                    throw std::runtime_error("SMT audio warmup frontend returned no outputs");
                }
                const auto frontend_output_info = frontend_outputs[0].GetTensorTypeAndShapeInfo();
                if (frontend_output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    throw std::runtime_error("SMT audio warmup frontend output must be float32");
                }
                auto shape    = frontend_output_info.GetShape();
                warmup_t_out  = (int) shape[1];
                warmup_hidden.resize((size_t) warmup_t_out * (size_t) d.config.d_model, 0.0f);
                const float * frontend_output = frontend_outputs[0].GetTensorData<float>();
                std::memcpy(warmup_hidden.data(), frontend_output, warmup_hidden.size() * sizeof(float));
            } else {
                const int chunk_frames = 100;
                const int chunk_tokens = 13;
                warmup_t_out           = chunk_tokens;

                std::vector<float>         frontend_input_data((size_t) d.config.num_mel_bins * chunk_frames, 0.0f);
                const std::vector<int64_t> frontend_input_shape = { 1, d.config.num_mel_bins, chunk_frames };
                auto                       frontend_input       = make_tensor_f32(frontend_input_shape, frontend_input_data);

                std::cerr << "[SMT][audio] warmup frontend ONNX session: " << d.config.frontend_model_path << "\n";
                auto frontend_outputs = d.frontend_session.Run(Ort::RunOptions{ nullptr }, d.frontend_input_names_raw.data(),
                                                               &frontend_input, 1, d.frontend_output_names_raw.data(), 1);

                warmup_hidden.resize((size_t) warmup_t_out * (size_t) d.config.d_model, 0.0f);
                if (frontend_outputs.empty()) {
                    throw std::runtime_error("SMT audio warmup frontend returned no outputs");
                }
                const auto frontend_output_info = frontend_outputs[0].GetTensorTypeAndShapeInfo();
                if (frontend_output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    throw std::runtime_error("SMT audio warmup frontend output must be float32");
                }
                const int64_t frontend_output_elems = frontend_output_info.GetElementCount();
                if (frontend_output_elems < 0 || (size_t) frontend_output_elems < warmup_hidden.size()) {
                    throw std::runtime_error("SMT audio warmup frontend output is smaller than expected");
                }
                const float * frontend_output = frontend_outputs[0].GetTensorData<float>();
                std::memcpy(warmup_hidden.data(), frontend_output, warmup_hidden.size() * sizeof(float));
            }

            const std::vector<int64_t> backend_hidden_shape = { 1, warmup_t_out, d.config.d_model };
            auto                       hidden_tensor        = make_tensor_f32(backend_hidden_shape, warmup_hidden);

            std::cerr << "[SMT][audio] warmup backend ONNX session: " << d.config.backend_model_path << "\n";
            if (d.backend_input_names_raw.size() == 2) {
                std::vector<float>         attention_mask((size_t) warmup_t_out * (size_t) warmup_t_out, 0.0f);
                const std::vector<int64_t> backend_mask_shape = { 1, 1, warmup_t_out, warmup_t_out };
                auto                       mask_tensor        = make_tensor_f32(backend_mask_shape, attention_mask);
                std::array<Ort::Value, 2>  backend_inputs     = { std::move(hidden_tensor), std::move(mask_tensor) };
                (void) d.backend_session.Run(Ort::RunOptions{ nullptr }, d.backend_input_names_raw.data(),
                                             backend_inputs.data(), backend_inputs.size(),
                                             d.backend_output_names_raw.data(), 1);
            } else {
                (void) d.backend_session.Run(Ort::RunOptions{ nullptr }, d.backend_input_names_raw.data(),
                                             &hidden_tensor, 1, d.backend_output_names_raw.data(), 1);
            }
        }
    }

    return ctx;
}

std::vector<float> smt_audio_context::encode_audio(const std::string & audio_path) {
    auto & d = *pimpl_;

    ggml_trace_log_begin("encode_audio", "Audio", NULL);

    std::vector<float> samples;
    ggml_trace_log_begin("decode_audio_file", "Audio", NULL);
    if (!decode_audio_file(audio_path, d.config.sample_rate, samples)) {
        ggml_trace_log_end("decode_audio_file", "Audio", NULL);
        ggml_trace_log_end("encode_audio", "Audio", NULL);
        ggml_profile_flush_tls();
        throw std::runtime_error("failed to decode audio file: " + audio_path);
    }
    ggml_trace_log_end("decode_audio_file", "Audio", NULL);

    if (uses_gemma4_single_encoder(d.config, d.arch_name)) {
        std::vector<float> features;
        int                n_feature_frames = 0;
        ggml_trace_log_begin("compute_gemma4_features", "Audio", NULL);
        if (!mtmd_audio_compute_gemma4_features(samples.data(), samples.size(), d.config.sample_rate,
                                                d.config.num_mel_bins, d.config.n_fft, d.config.window_len,
                                                d.config.hop_len, features, n_feature_frames)) {
            ggml_trace_log_end("compute_gemma4_features", "Audio", NULL);
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("failed to compute Gemma4 audio features");
        }
        ggml_trace_log_end("compute_gemma4_features", "Audio", NULL);

        const std::string & encoder_model_path = d.config.encoder_model_path;
        if (encoder_model_path.empty()) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("Gemma4 audio encoder_model_path is empty");
        }

        std::lock_guard<std::mutex> encoder_lock(d.encoder_session_mutex);
        auto &                      encoder = d.get_gemma4_encoder_session(encoder_model_path);

        const int32_t encoder_feature_frames =
            encoder.input_shape.dynamic_feature_frames() ? n_feature_frames : encoder.input_shape.feature_frames;
        if (encoder_feature_frames <= 0) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("Gemma4 audio feature length must be positive");
        }

        if (!encoder.input_shape.dynamic_feature_frames() && n_feature_frames > encoder_feature_frames) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("Gemma4 audio feature length " + std::to_string(n_feature_frames) +
                                     " exceeds ONNX static feature_frames " +
                                     std::to_string(encoder_feature_frames) + " for " + encoder.model_path);
        }

        std::vector<float> feature_data((size_t) encoder_feature_frames * d.config.num_mel_bins, 0.0f);
        for (int frame = 0; frame < n_feature_frames; ++frame) {
            std::memcpy(feature_data.data() + (size_t) frame * d.config.num_mel_bins,
                        features.data() + (size_t) frame * d.config.num_mel_bins,
                        (size_t) d.config.num_mel_bins * sizeof(float));
        }
        std::vector<uint8_t> feature_mask((size_t) encoder_feature_frames, 0);
        std::fill(feature_mask.begin(), feature_mask.begin() + n_feature_frames, (uint8_t) 1);

        const std::vector<int64_t> feature_shape = { 1, encoder_feature_frames, d.config.num_mel_bins };
        const std::vector<int64_t> mask_shape    = { 1, encoder_feature_frames };
        auto                       feature_tensor = make_tensor_f32(feature_shape, feature_data);
        auto                       mask_tensor    = make_tensor_bool(mask_shape, feature_mask);
        std::array<Ort::Value, 2>  inputs         = { std::move(feature_tensor), std::move(mask_tensor) };

        ggml_trace_log_begin("encoder_session_run", "Audio", NULL);
        auto outputs = encoder.session.Run(Ort::RunOptions{ nullptr }, encoder.input_names_raw.data(),
                                           inputs.data(), inputs.size(), encoder.output_names_raw.data(),
                                           encoder.output_names_raw.size());
        ggml_trace_log_end("encoder_session_run", "Audio", NULL);

        if (outputs.size() < 2) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("Gemma4 audio encoder returned fewer than 2 outputs");
        }

        const auto embd_info = outputs[0].GetTensorTypeAndShapeInfo();
        if (embd_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("Gemma4 audio_embeddings output must be float32");
        }
        const int64_t embd_elems = embd_info.GetElementCount();
        if (embd_elems <= 0 || embd_elems % d.config.hidden_size != 0) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("invalid Gemma4 audio_embeddings shape");
        }
        const int n_audio_tokens = (int) (embd_elems / d.config.hidden_size);

        const auto mask_info = outputs[1].GetTensorTypeAndShapeInfo();
        if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
            mask_info.GetElementCount() < (size_t) n_audio_tokens) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("invalid Gemma4 audio_token_mask shape");
        }

        const float * embd = outputs[0].GetTensorData<float>();
        const bool *  mask = outputs[1].GetTensorData<bool>();
        std::vector<float> audio_embd;
        audio_embd.reserve((size_t) n_audio_tokens * (size_t) d.config.hidden_size);
        for (int token = 0; token < n_audio_tokens; ++token) {
            if (!mask[token]) {
                continue;
            }
            const float * src = embd + (size_t) token * (size_t) d.config.hidden_size;
            audio_embd.insert(audio_embd.end(), src, src + d.config.hidden_size);
        }

        ggml_trace_log_end("encode_audio", "Audio", NULL);
        ggml_profile_flush_tls();
        return audio_embd;
    }

    int                t_out;
    std::vector<float> hidden_states;

    if (d.config.lfr_m > 0) {
        // FunASR path: kaldi fbank -> LFR -> frontend -> backend
        std::vector<float> fbank_features;
        int                n_fbank_frames = 0;
        ggml_trace_log_begin("compute_kaldi_fbank", "Audio", NULL);
        if (!mtmd_audio_compute_kaldi_fbank(samples.data(), samples.size(), d.config.sample_rate, d.config.num_mel_bins,
                                 d.config.window_len, d.config.hop_len, 0.97f, fbank_features, n_fbank_frames)) {
            ggml_trace_log_end("compute_kaldi_fbank", "Audio", NULL);
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("failed to compute kaldi fbank features");
        }
        ggml_trace_log_end("compute_kaldi_fbank", "Audio", NULL);

        std::vector<float> lfr_features;
        int                n_lfr_frames = 0;
        if (!mtmd_audio_compute_lfr(fbank_features, n_fbank_frames, d.config.num_mel_bins, d.config.lfr_m, d.config.lfr_n,
                         lfr_features, n_lfr_frames)) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("failed to compute LFR features");
        }

        // Per-frame mean subtraction for ONNX numerical stability.
        // LayerNorm is shift-invariant: LN(x + c) = LN(x), so subtracting the
        // per-frame mean does not change the model output, but prevents catastrophic
        // cancellation in the ONNX decomposed variance computation (E[x²] - E[x]²)
        // when input values have large magnitude but small variance.
        const int feat_dim = d.config.num_mel_bins * d.config.lfr_m;
        for (int i = 0; i < n_lfr_frames; i++) {
            float * frame = lfr_features.data() + (size_t) i * feat_dim;
            float   sum   = 0.0f;
            for (int j = 0; j < feat_dim; j++) {
                sum += frame[j];
            }
            float mean = sum / (float) feat_dim;
            for (int j = 0; j < feat_dim; j++) {
                frame[j] -= mean;
            }
        }
        const std::vector<int64_t> frontend_input_shape = { 1, (int64_t) n_lfr_frames, (int64_t) feat_dim };
        auto                       frontend_input       = make_tensor_f32(frontend_input_shape, lfr_features);

        ggml_trace_log_begin("frontend_session_run", "Audio", NULL);
        auto frontend_outputs = d.frontend_session.Run(Ort::RunOptions{ nullptr }, d.frontend_input_names_raw.data(),
                                                       &frontend_input, 1, d.frontend_output_names_raw.data(), 1);
        ggml_trace_log_end("frontend_session_run", "Audio", NULL);

        if (frontend_outputs.empty()) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("FunASR frontend returned no outputs");
        }

        const auto frontend_shape = frontend_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        t_out                     = (int) frontend_shape[1];
        hidden_states.resize((size_t) t_out * (size_t) d.config.d_model);
        std::memcpy(hidden_states.data(), frontend_outputs[0].GetTensorData<float>(),
                    hidden_states.size() * sizeof(float));

    } else {
        // Qwen3ASR path: mel spectrogram -> chunk -> frontend -> backend
        mtmd_audio_mel mel;
        ggml_trace_log_begin("compute_log_mel_spectrogram", "Audio", NULL);
        if (!mtmd_audio_compute_log_mel_spectrogram(samples.data(), samples.size(), 4, d.config.num_mel_bins,
                                                    d.config.n_fft, d.config.window_len, d.config.hop_len,
                                                    d.config.sample_rate, true, 0.0f, false, false, mel)) {
            ggml_trace_log_end("compute_log_mel_spectrogram", "Audio", NULL);
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("failed to compute Qwen3-ASR mel spectrogram");
        }
        ggml_trace_log_end("compute_log_mel_spectrogram", "Audio", NULL);

        if (mel.n_len <= 0 || mel.n_mel != d.config.num_mel_bins) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("invalid mel spectrogram shape");
        }

        const int frames        = mel.n_len;
        const int chunk_frames  = 100;
        const int chunk_tokens  = 13;
        const int padded_frames = ((frames + chunk_frames - 1) / chunk_frames) * chunk_frames;
        const int n_chunks      = padded_frames / chunk_frames;

        hidden_states.resize((size_t) n_chunks * chunk_tokens * (size_t) d.config.d_model);
        std::vector<float>         chunk_input((size_t) d.config.num_mel_bins * chunk_frames, 0.0f);
        const std::vector<int64_t> frontend_input_shape = { 1, d.config.num_mel_bins, chunk_frames };

        ggml_trace_log_begin("frontend_session_run", "Audio", NULL);
        for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            std::fill(chunk_input.begin(), chunk_input.end(), 0.0f);
            const int frame_offset = chunk_idx * chunk_frames;
            const int copy_frames  = std::min(chunk_frames, frames - frame_offset);
            if (copy_frames > 0) {
                for (int mel_idx = 0; mel_idx < mel.n_mel; ++mel_idx) {
                    const float * src = mel.data.data() + (size_t) mel_idx * mel.n_len + frame_offset;
                    float *       dst = chunk_input.data() + (size_t) mel_idx * chunk_frames;
                    std::memcpy(dst, src, (size_t) copy_frames * sizeof(float));
                }
            }

            auto    frontend_input   = make_tensor_f32(frontend_input_shape, chunk_input);
            auto    frontend_outputs = d.frontend_session.Run(Ort::RunOptions{ nullptr }, d.frontend_input_names_raw.data(),
                                                              &frontend_input, 1, d.frontend_output_names_raw.data(), 1);
            float * chunk_out        = frontend_outputs[0].GetTensorMutableData<float>();
            std::memcpy(hidden_states.data() + (size_t) chunk_idx * chunk_tokens * (size_t) d.config.d_model, chunk_out,
                        (size_t) chunk_tokens * (size_t) d.config.d_model * sizeof(float));
        }
        ggml_trace_log_end("frontend_session_run", "Audio", NULL);

        t_out = get_feat_extract_output_lengths(frames);
        if (t_out <= 0 || t_out > n_chunks * chunk_tokens) {
            ggml_trace_log_end("encode_audio", "Audio", NULL);
            ggml_profile_flush_tls();
            throw std::runtime_error("invalid split-encoder output length");
        }

        hidden_states.resize((size_t) t_out * (size_t) d.config.d_model);
    }

    // Common backend path
    const std::vector<int64_t> backend_hidden_shape = { 1, t_out, d.config.d_model };
    auto                       hidden_tensor        = make_tensor_f32(backend_hidden_shape, hidden_states);

    ggml_trace_log_begin("backend_session_run", "Audio", NULL);
    std::vector<Ort::Value> backend_outputs;
    if (d.backend_input_names_raw.size() == 2) {
        // Backend expects hidden_states + attention_mask
        std::vector<float>         attention_mask((size_t) t_out * (size_t) t_out, 0.0f);
        const std::vector<int64_t> backend_mask_shape = { 1, 1, t_out, t_out };
        auto                       mask_tensor        = make_tensor_f32(backend_mask_shape, attention_mask);
        std::array<Ort::Value, 2>  inputs             = { std::move(hidden_tensor), std::move(mask_tensor) };
        backend_outputs = d.backend_session.Run(Ort::RunOptions{ nullptr }, d.backend_input_names_raw.data(),
                                                inputs.data(), inputs.size(), d.backend_output_names_raw.data(), 1);
    } else {
        // Backend only expects hidden_states (attention_mask pruned by ONNX exporter)
        backend_outputs = d.backend_session.Run(Ort::RunOptions{ nullptr }, d.backend_input_names_raw.data(),
                                                &hidden_tensor, 1, d.backend_output_names_raw.data(), 1);
    }
    ggml_trace_log_end("backend_session_run", "Audio", NULL);

    float *            output = backend_outputs[0].GetTensorMutableData<float>();
    std::vector<float> audio_embd((size_t) t_out * (size_t) d.config.hidden_size);
    std::memcpy(audio_embd.data(), output, audio_embd.size() * sizeof(float));

    ggml_trace_log_end("encode_audio", "Audio", NULL);
    ggml_profile_flush_tls();
    return audio_embd;
}

int64_t smt_audio_context::hidden_size() const {
    return pimpl_->config.hidden_size;
}

const std::string & smt_audio_context::architecture() const {
    return pimpl_->arch_name;
}
