// SMT vision wrapper for llama.cpp SpacemiT integration.

#include "smt-vision-wrapper.h"

#include "ggml-profile.h"
#include "onnxruntime_cxx_api.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#if defined(_WIN32)
#    include <io.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <unistd.h>
#endif

namespace onnxruntime {
const OrtApi * g_ort = NULL;
}  // namespace onnxruntime

namespace {

static int get_ep_thread_num(const std::unordered_map<std::string, std::string> & ep_config,
                             const std::string &                                  key,
                             int                                                  default_value) {
    auto it = ep_config.find(key);
    if (it == ep_config.end() || it->second.empty()) {
        return default_value;
    }
    return std::stoi(it->second);
}

static bool has_spacemit_ep_affinity(const std::unordered_map<std::string, std::string> & ep_config) {
    auto it = ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY");
    return it != ep_config.end() && !it->second.empty();
}

static std::unordered_map<std::string, std::string> make_provider_options(
    const std::unordered_map<std::string, std::string> & ep_config) {
    std::unordered_map<std::string, std::string> provider_options = ep_config;
    if (provider_options.find("SPACEMIT_EP_INTRA_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = "1";
    }
    if (provider_options.find("SPACEMIT_EP_INTER_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
    }
    return provider_options;
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

static Ort::Value make_tensor_f32(const std::vector<int64_t> & shape, std::vector<float> & data) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
}

class smt_ort_vision_engine {
public:
    smt_ort_vision_engine(std::string model_path, std::unordered_map<std::string, std::string> ep_config) :
        model_path_(std::move(model_path)),
        ep_config_(std::move(ep_config)),
        env_(ORT_LOGGING_LEVEL_WARNING, "smt-vision") {}

    Ort::Session & create_session() {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const int intra_thread_num = get_ep_thread_num(ep_config_, "SPACEMIT_EP_INTRA_THREAD_NUM", 1);
        const int inter_thread_num = get_ep_thread_num(ep_config_, "SPACEMIT_EP_INTER_THREAD_NUM", 1);
        if (!has_spacemit_ep_affinity(ep_config_)) {
            session_options_.SetIntraOpNumThreads(intra_thread_num);
            session_options_.SetInterOpNumThreads(inter_thread_num);
        } else {
            std::cerr << "[SMT][vision] detected SPACEMIT_EP_INTRA_THREAD_AFFINITY, skip ORT session thread pinning"
                      << " to avoid conflicting with EP-managed affinity\n";
        }

        provider_options_ = make_provider_options(ep_config_);
        std::string error_message;
        if (!init_spacemit_execution_provider(session_options_, provider_options_, error_message)) {
            throw std::runtime_error("[SMT][vision] failed to initialize Spacemit EP: " + error_message);
        }

        std::cerr << "[SMT][vision] Spacemit EP enabled (";
        for (const auto & pair : provider_options_) {
            std::cerr << ", " << pair.first << "=" << pair.second;
        }
        std::cerr << ")\n";

        session_          = Ort::Session(env_, model_path_.c_str(), session_options_);
        input_names_      = get_io_names(session_, true);
        output_names_     = get_io_names(session_, false);
        input_names_raw_  = make_name_ptrs(input_names_);
        output_names_raw_ = make_name_ptrs(output_names_);

        if (input_names_raw_.size() != 1 || output_names_raw_.size() != 1) {
            throw std::runtime_error("Unexpected SMT vision ONNX IO signature");
        }

        return session_;
    }

    Ort::Value & set_input_tensor(const std::string & input_binary_path) {
        auto                      type_info   = session_.GetInputTypeInfo(0);
        auto                      tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> input_shape = tensor_info.GetShape();

        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("SMT vision expects float32 input tensor");
        }

        size_t input_size = 1;
        for (const int64_t dim : input_shape) {
            if (dim <= 0) {
                throw std::runtime_error("SMT vision input tensor must have static positive shape");
            }
            if (input_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::runtime_error("SMT vision input tensor is too large");
            }
            input_size *= static_cast<size_t>(dim);
        }

        input_data_.resize(input_size);
        std::ifstream file(input_binary_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open SMT vision input binary: " + input_binary_path);
        }

        const std::streamoff actual_bytes   = file.tellg();
        const size_t         expected_bytes = input_data_.size() * sizeof(float);
        if (actual_bytes < 0 || static_cast<size_t>(actual_bytes) != expected_bytes) {
            throw std::runtime_error("SMT vision input binary size mismatch: expected " +
                                     std::to_string(expected_bytes) + ", actual " +
                                     std::to_string(actual_bytes < 0 ? 0 : static_cast<size_t>(actual_bytes)));
        }

        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(input_data_.data()), static_cast<std::streamsize>(expected_bytes));
        if (!file) {
            throw std::runtime_error("failed to read SMT vision input binary: " + input_binary_path);
        }

        input_tensor_ = make_tensor_f32(input_shape, input_data_);
        return input_tensor_;
    }

    Ort::Value make_zero_input_tensor() {
        auto                      type_info   = session_.GetInputTypeInfo(0);
        auto                      tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> input_shape = tensor_info.GetShape();

        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("SMT vision warmup expects float32 input tensor");
        }

        size_t input_size = 1;
        for (const int64_t dim : input_shape) {
            if (dim <= 0) {
                throw std::runtime_error("SMT vision warmup requires a static positive input shape");
            }
            if (input_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::runtime_error("SMT vision warmup input tensor is too large");
            }
            input_size *= static_cast<size_t>(dim);
        }

        warmup_data_.assign(input_size, 0.0f);
        return make_tensor_f32(input_shape, warmup_data_);
    }

    std::vector<float> run_session(Ort::Value & input_tensor) {
        std::vector<Ort::Value> output_tensors =
            session_.Run(Ort::RunOptions{ nullptr }, input_names_raw_.data(), &input_tensor, input_names_raw_.size(),
                         output_names_raw_.data(), output_names_raw_.size());

        if (output_tensors.empty()) {
            throw std::runtime_error("SMT vision ONNX returned no outputs");
        }

        Ort::Value & output      = output_tensors[0];
        auto         tensor_info = output.GetTensorTypeAndShapeInfo();
        auto         shape       = tensor_info.GetShape();
        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Expected float32 output from SMT vision model");
        }

        if (shape.size() == 3 && shape[0] == 1) {
            shape = { shape[1], shape[2] };
        }
        if (shape.size() != 2) {
            throw std::runtime_error("Unexpected output shape from SMT vision encoder");
        }

        const size_t total_elements = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
        const float * data          = output.GetTensorData<float>();
        return std::vector<float>(data, data + total_elements);
    }

private:
    std::string                                  model_path_;
    std::unordered_map<std::string, std::string> ep_config_;
    std::unordered_map<std::string, std::string> provider_options_;
    Ort::Env                                     env_;
    Ort::SessionOptions                          session_options_;
    Ort::Session                                 session_{ nullptr };
    std::vector<std::string>                     input_names_;
    std::vector<std::string>                     output_names_;
    std::vector<const char *>                    input_names_raw_;
    std::vector<const char *>                    output_names_raw_;
    std::vector<float>                           input_data_;
    std::vector<float>                           warmup_data_;
    Ort::Value                                   input_tensor_{ nullptr };
};

struct smt_vision_config {
    std::vector<std::string>                     architectures;
    std::string                                  vision_model_path;
    std::unordered_map<std::string, std::string> ep_config;
    int64_t                                      hidden_size  = 0;
    int32_t                                      input_width  = 0;
    int32_t                                      input_height = 0;
    smt_vision_preprocess_config                 preprocess_config;
};

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

static double extract_double_value(const std::string & text, const std::string & key, double default_value) {
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
    bool seen_dot = false;
    while (value_end < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[value_end]);
        if (std::isdigit(ch)) {
            ++value_end;
            continue;
        }
        if (ch == '.' && !seen_dot) {
            seen_dot = true;
            ++value_end;
            continue;
        }
        break;
    }

    if (value_end == value_start) {
        return default_value;
    }

    try {
        return std::stod(text.substr(value_start, value_end - value_start));
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

static std::vector<double> extract_number_array(const std::string & text, const std::string & key) {
    std::vector<double> values;

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
        while (pos < content.size() &&
               (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }
        size_t end      = pos;
        bool   seen_dot = false;
        if (content[end] == '-' || content[end] == '+') {
            ++end;
        }
        while (end < content.size()) {
            const unsigned char ch = static_cast<unsigned char>(content[end]);
            if (std::isdigit(ch)) {
                ++end;
                continue;
            }
            if (ch == '.' && !seen_dot) {
                seen_dot = true;
                ++end;
                continue;
            }
            break;
        }
        if (end > pos) {
            try {
                values.push_back(std::stod(content.substr(pos, end - pos)));
            } catch (...) {
                values.clear();
                return values;
            }
        }
        pos = end + 1;
    }

    return values;
}

static std::string normalize_path(const std::string & base_dir, const std::string & path) {
    const std::string trimmed = trim_ascii(path);
    if (trimmed.empty()) {
        return {};
    }
    if (!trimmed.empty() && trimmed.front() == '/') {
        return trimmed;
    }
    return base_dir + "/" + trimmed;
}

static std::string canonicalize_vision_architecture(std::string arch) {
    const std::string trimmed = trim_ascii(arch);
    if (trimmed == "Qwen3_5ForConditionalGeneration") {
        return "Qwen3VL";
    }
    return trimmed;
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

    std::cerr << "[SMT][vision] warning: detected deprecated legacy Spacemit EP config keys";
    if (section_name != nullptr && section_name[0] != '\0') {
        std::cerr << " in " << section_name;
    }
    std::cerr << "; this style will be removed in a future release. "
              << "Please migrate to the `ep_config` format.\n";
}

static void apply_legacy_spacemit_ep_config(const std::string & text, smt_vision_config & config) {
    const int32_t intra_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_intra_thread_num",
                                      config.ep_config.count("SPACEMIT_EP_INTRA_THREAD_NUM") ?
                                          std::stoll(config.ep_config.at("SPACEMIT_EP_INTRA_THREAD_NUM")) :
                                          4);
    const int32_t inter_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_inter_thread_num",
                                      config.ep_config.count("SPACEMIT_EP_INTER_THREAD_NUM") ?
                                          std::stoll(config.ep_config.at("SPACEMIT_EP_INTER_THREAD_NUM")) :
                                          1);

    if (config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(intra_thread_num);
    }
    if (config.ep_config.find("SPACEMIT_EP_INTER_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(inter_thread_num);
    }

    const std::string affinity = extract_string_value(text, "spacemit_ep_intra_thread_affinity");
    if (!affinity.empty() && config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
    }
}

static bool load_smt_vision_config(const std::string & config_dir, smt_vision_config & config) {
    const std::string config_path = config_dir + "/config.json";
    const std::string content     = read_file_to_string(config_path);
    if (content.empty()) {
        std::cerr << "Error: Failed to read config file: " << config_path << "\n";
        return false;
    }

    const size_t vision_start = content.find("\"vision_model\":");
    if (vision_start == std::string::npos) {
        return false;
    }
    const size_t vision_block_start = content.find('{', vision_start);
    const size_t vision_block_end   = find_closing_brace(content, vision_block_start);
    if (vision_block_start == std::string::npos || vision_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'vision_model' block.\n";
        return false;
    }
    const std::string vision_block = content.substr(vision_block_start, vision_block_end - vision_block_start + 1);

    const size_t text_start = content.find("\"text_model\":");
    if (text_start == std::string::npos) {
        std::cerr << "Error: 'text_model' section not found.\n";
        return false;
    }
    const size_t text_block_start = content.find('{', text_start);
    const size_t text_block_end   = find_closing_brace(content, text_block_start);
    if (text_block_start == std::string::npos || text_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'text_model' block.\n";
        return false;
    }
    const std::string text_block = content.substr(text_block_start, text_block_end - text_block_start + 1);

    std::string  preprocess_block;
    const size_t preprocess_start = content.find("\"vision_preprocess\":");
    if (preprocess_start != std::string::npos) {
        const size_t preprocess_block_start = content.find('{', preprocess_start);
        const size_t preprocess_block_end   = find_closing_brace(content, preprocess_block_start);
        if (preprocess_block_start == std::string::npos || preprocess_block_end == std::string::npos) {
            std::cerr << "Error: Invalid 'vision_preprocess' block.\n";
            return false;
        }
        preprocess_block = content.substr(preprocess_block_start, preprocess_block_end - preprocess_block_start + 1);
    }

    warn_legacy_spacemit_ep_config_if_needed(vision_block, "vision_model");
    warn_legacy_spacemit_ep_config_if_needed(content, "top-level config");

    config.vision_model_path = normalize_path(config_dir, extract_string_value(vision_block, "model_path"));
    config.hidden_size       = extract_int64_value(text_block, "hidden_size", 0);
    config.architectures     = extract_string_array(content, "architectures");
    const int64_t input_size = extract_int64_value(vision_block, "input_size", 0);
    config.input_width       = (int32_t) extract_int64_value(vision_block, "input_width", input_size);
    config.input_height      = (int32_t) extract_int64_value(vision_block, "input_height", input_size);
    config.ep_config         = extract_string_map(vision_block, "ep_config");
    if (!preprocess_block.empty()) {
        config.preprocess_config.rescale_factor = (float) extract_double_value(preprocess_block, "rescale_factor", 1.0);
        const auto image_mean                   = extract_number_array(preprocess_block, "image_mean");
        const auto image_std                    = extract_number_array(preprocess_block, "image_std");
        if (image_mean.size() == 3 && image_std.size() == 3) {
            for (size_t i = 0; i < 3; ++i) {
                config.preprocess_config.image_mean[i] = (float) image_mean[i];
                config.preprocess_config.image_std[i]  = (float) image_std[i];
            }
            config.preprocess_config.has_normalize_config = true;
        }
    }
    apply_legacy_spacemit_ep_config(vision_block, config);

    // 从顶层配置读取 ep_config（如果 vision_model 块中没有设置）
    auto top_ep_config = extract_string_map(content, "ep_config");
    for (const auto & pair : top_ep_config) {
        if (config.ep_config.find(pair.first) == config.ep_config.end()) {
            config.ep_config[pair.first] = pair.second;
        }
    }
    apply_legacy_spacemit_ep_config(content, config);

    if (config.vision_model_path.empty()) {
        std::cerr << "Error: Missing required key 'vision_model.model_path'.\n";
        return false;
    }
    if (config.hidden_size <= 0) {
        std::cerr << "Error: Missing or invalid required key 'text_model.hidden_size'.\n";
        return false;
    }
    if (config.architectures.empty()) {
        std::cerr << "Error: Missing required key 'architectures'.\n";
        return false;
    }
    if ((config.input_width < 0 || config.input_height < 0) ||
        ((config.input_width == 0) != (config.input_height == 0))) {
        std::cerr << "Error: vision_model.input_width/input_height must both be set and positive.\n";
        return false;
    }

    return true;
}

}  // namespace

struct smt_vision_context::impl {
    smt_vision_config                      config;
    std::unique_ptr<smt_ort_vision_engine> vision_engine;
    std::string                            arch_name;
};

namespace {

static void warmup_vision_engine(smt_ort_vision_engine & vision_engine, const std::string & arch_name) {
    std::cerr << "[SMT][vision] warmup ONNX session";
    if (!arch_name.empty()) {
        std::cerr << " for " << arch_name;
    }
    std::cerr << "\n";

    Ort::Value input_tensor = vision_engine.make_zero_input_tensor();
    (void) vision_engine.run_session(input_tensor);
}

}  // namespace

smt_vision_context::~smt_vision_context() = default;

std::unique_ptr<smt_vision_context> smt_vision_context::create(const std::string & config_dir, bool warmup) {
    auto ctx    = std::unique_ptr<smt_vision_context>(new smt_vision_context());
    ctx->pimpl_ = std::make_unique<impl>();
    auto & d    = *ctx->pimpl_;

    // 1. Load config from directory
    if (!load_smt_vision_config(config_dir, d.config)) {
        throw std::runtime_error("Failed to load SMT config from: " + config_dir);
    }

    if (!d.config.architectures.empty()) {
        d.arch_name = canonicalize_vision_architecture(d.config.architectures[0]);
    }

    // 2. Initialize ORT API
    onnxruntime::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 3. Create vision engine and session
    d.vision_engine = std::make_unique<smt_ort_vision_engine>(d.config.vision_model_path, d.config.ep_config);
    (void) d.vision_engine->create_session();
    if (warmup) {
        warmup_vision_engine(*d.vision_engine, d.arch_name);
    }

    return ctx;
}

std::vector<float> smt_vision_context::encode_image(const std::string & binary_path) {
    auto & d = *pimpl_;

    ggml_trace_log_begin("encode_image", "Vision", NULL);

    ggml_trace_log_begin("set_input_tensor", "Vision", NULL);
    Ort::Value & input_tensor = d.vision_engine->set_input_tensor(binary_path);
    ggml_trace_log_end("set_input_tensor", "Vision", NULL);

    ggml_trace_log_begin("vision_session_run", "Vision", NULL);
    std::vector<float> result = d.vision_engine->run_session(input_tensor);
    ggml_trace_log_end("vision_session_run", "Vision", NULL);

    ggml_trace_log_end("encode_image", "Vision", NULL);
    ggml_profile_flush_tls();
    return result;
}

int64_t smt_vision_context::hidden_size() const {
    return pimpl_->config.hidden_size;
}

int32_t smt_vision_context::input_width() const {
    return pimpl_->config.input_width;
}

int32_t smt_vision_context::input_height() const {
    return pimpl_->config.input_height;
}

int64_t smt_vision_context::vocab_size() const {
    return 0;
}

const std::string & smt_vision_context::token_embedding_path() const {
    static const std::string empty;
    return empty;
}

const std::string & smt_vision_context::architecture() const {
    return pimpl_->arch_name;
}

const smt_vision_preprocess_config & smt_vision_context::preprocess_config() const {
    return pimpl_->config.preprocess_config;
}
