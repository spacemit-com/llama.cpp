#include "qwen3_tts_runtime.h"

#include "qwen3_tts_gguf.h"
#include "qwen3_tts_protocol.h"

#include "llama.h"
#include "onnxruntime_cxx_api.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

namespace qwen3_tts {
namespace {

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

constexpr int hidden_size = 1024;
constexpr int code_groups = 16;
constexpr int code_vocab = 2048;
constexpr int codec_bucket = 50;
constexpr int codec_context = 25;
constexpr int samples_per_frame = 1920;

constexpr int text_tts_pad = 151671;
constexpr int text_tts_bos = 151672;
constexpr int text_tts_eos = 151673;
constexpr int codec_pad = 2148;
constexpr int codec_bos = 2149;
constexpr int codec_nothink = 2155;
constexpr int codec_think_bos = 2156;
constexpr int codec_think_eos = 2157;

struct runtime_config {
    std::string tokenizer_model;
    std::string text_embedding_model;
    std::string codec_decoder_model;
    std::string talker_model;
    std::string code_predictor_model;
    std::string aux_model;
    std::string default_speaker;
    std::unordered_map<std::string, std::string> ep_config;
    int sample_rate = 24000;
    int max_frames = 160;
    int max_prefill = 128;
    int segment_pause_ms = 200;
    int frontend_threads = 2;
    int codec_threads = 4;
    int talker_threads = 4;
};

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Qwen3-TTS config: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string resolve_path(const std::filesystem::path & root, const std::string & value) {
    if (value.empty()) {
        return {};
    }
    const std::filesystem::path path(value);
    return path.is_absolute() ? path.string() : (root / path).lexically_normal().string();
}

std::string json_string_value(const json & object, const char * key) {
    if (!object.contains(key) || !object.at(key).is_string() || object.at(key).get<std::string>().empty()) {
        throw std::runtime_error(std::string("missing Qwen3-TTS config key: tts_model.") + key);
    }
    return object.at(key).get<std::string>();
}

int json_int_value(const json & object, const char * key, int fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_number_integer()) {
        throw std::runtime_error(std::string("invalid Qwen3-TTS integer config: tts_model.") + key);
    }
    return object.at(key).get<int>();
}

runtime_config load_config(const std::string & config_dir) {
    const std::filesystem::path root(config_dir);
    const json document = json::parse(read_text_file(root / "config.json"));
    if (!document.contains("architectures") || !document.at("architectures").is_array()) {
        throw std::runtime_error("Qwen3-TTS config requires an architectures array");
    }
    bool architecture_matches = false;
    for (const auto & item : document.at("architectures")) {
        architecture_matches = architecture_matches ||
            (item.is_string() && item.get<std::string>() == "Qwen3TTSForConditionalGeneration");
    }
    if (!architecture_matches || !document.contains("tts_model") || !document.at("tts_model").is_object()) {
        throw std::runtime_error("SMT config is not a Qwen3-TTS model bundle");
    }

    const json & model = document.at("tts_model");
    runtime_config config;
    config.tokenizer_model = resolve_path(root, json_string_value(model, "tokenizer_model_path"));
    config.text_embedding_model = resolve_path(root, json_string_value(model, "text_embedding_model_path"));
    config.codec_decoder_model = resolve_path(root, json_string_value(model, "codec_decoder_model_path"));
    config.talker_model = resolve_path(root, json_string_value(model, "talker_model_path"));
    config.code_predictor_model = resolve_path(root, json_string_value(model, "code_predictor_model_path"));
    config.aux_model = resolve_path(root, json_string_value(model, "aux_model_path"));
    if (model.contains("speaker_file") && model.at("speaker_file").is_string()) {
        config.default_speaker = resolve_path(root, model.at("speaker_file").get<std::string>());
    }
    config.sample_rate = json_int_value(model, "sample_rate", config.sample_rate);
    config.max_frames = json_int_value(model, "max_frames", config.max_frames);
    config.max_prefill = json_int_value(model, "max_prefill", config.max_prefill);
    config.segment_pause_ms = json_int_value(model, "segment_pause_ms", config.segment_pause_ms);
    config.frontend_threads = json_int_value(model, "frontend_threads", config.frontend_threads);
    config.codec_threads = json_int_value(model, "codec_threads", config.codec_threads);
    config.talker_threads = json_int_value(model, "talker_threads", config.talker_threads);
    if (config.sample_rate != 24000 || config.max_frames <= 0 || config.max_prefill <= 0 ||
        config.segment_pause_ms < 0 || config.frontend_threads <= 0 || config.codec_threads <= 0 ||
        config.talker_threads <= 0) {
        throw std::runtime_error("invalid Qwen3-TTS runtime limits");
    }
    if (model.contains("ep_config")) {
        if (!model.at("ep_config").is_object()) {
            throw std::runtime_error("tts_model.ep_config must be an object");
        }
        for (const auto & [key, value] : model.at("ep_config").items()) {
            if (value.is_string()) {
                config.ep_config[key] = value.get<std::string>();
            } else if (value.is_number_integer()) {
                config.ep_config[key] = std::to_string(value.get<int64_t>());
            } else if (value.is_boolean()) {
                config.ep_config[key] = value.get<bool>() ? "1" : "0";
            } else {
                throw std::runtime_error("Qwen3-TTS EP option values must be scalar");
            }
        }
    }
    config.ep_config.try_emplace("SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(config.codec_threads));
    config.ep_config.try_emplace("SPACEMIT_EP_INTER_THREAD_NUM", "1");

    for (const std::string * path : {&config.tokenizer_model, &config.text_embedding_model,
                                     &config.codec_decoder_model, &config.talker_model,
                                     &config.code_predictor_model, &config.aux_model}) {
        if (!std::filesystem::is_regular_file(*path)) {
            throw std::runtime_error("missing Qwen3-TTS model file: " + *path);
        }
    }
    return config;
}

std::vector<float> read_speaker(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open Qwen3-TTS speaker file: " + path);
    }
    std::vector<float> speaker(hidden_size);
    input.read(reinterpret_cast<char *>(speaker.data()), static_cast<std::streamsize>(speaker.size() * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Qwen3-TTS speaker file must be raw float32[1024]");
    }
    return speaker;
}

class shared_library {
  public:
    shared_library() = default;
    ~shared_library() {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    void * open(const char * name) {
        if (handle_ == nullptr) {
            handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (handle_ == nullptr) {
                throw std::runtime_error(std::string("failed to load ") + name + ": " + dlerror());
            }
        }
        return handle_;
    }

  private:
    void * handle_ = nullptr;
};

void append_spacemit_ep(Ort::SessionOptions & options,
                        shared_library & library,
                        const std::unordered_map<std::string, std::string> & provider_options) {
    using init_fn = OrtStatus * (ORT_API_CALL *)(OrtSessionOptions *, const char * const *,
                                                 const char * const *, size_t);
    void * handle = library.open("libspacemit_ep.so");
    auto * initialize = reinterpret_cast<init_fn>(dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (initialize == nullptr) {
        throw std::runtime_error(std::string("failed to resolve OrtSessionOptionsSpaceMITEnvInit: ") + dlerror());
    }
    std::vector<const char *> keys;
    std::vector<const char *> values;
    keys.reserve(provider_options.size());
    values.reserve(provider_options.size());
    for (const auto & [key, value] : provider_options) {
        keys.push_back(key.c_str());
        values.push_back(value.c_str());
    }
    if (OrtStatus * status = initialize(options, keys.data(), values.data(), keys.size())) {
        const std::string error = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(status);
        throw std::runtime_error("failed to initialize SpaceMIT EP: " + error);
    }
}

void set_socket_timeout(int fd, int seconds) {
    timeval timeout{};
    timeout.tv_sec = seconds;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw std::runtime_error("failed to set Qwen3-TTS process timeout");
    }
}

class talker_process {
  public:
    talker_process(const std::string & executable, const runtime_config & config) {
        int sockets[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
            throw std::runtime_error("failed to create Qwen3-TTS talker socket");
        }
        const pid_t child = fork();
        if (child < 0) {
            close(sockets[0]);
            close(sockets[1]);
            throw std::runtime_error("failed to fork Qwen3-TTS talker");
        }
        if (child == 0) {
            close(sockets[0]);
            const pid_t parent = getppid();
            if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent) {
                _exit(125);
            }
            if (sockets[1] != 3) {
                if (dup2(sockets[1], 3) < 0) {
                    _exit(126);
                }
                close(sockets[1]);
            }
            const std::string max_prefill = std::to_string(config.max_prefill);
            const std::string max_frames = std::to_string(config.max_frames);
            const std::string threads = std::to_string(config.talker_threads);
            execl(executable.c_str(), executable.c_str(), "3", config.talker_model.c_str(),
                  config.code_predictor_model.c_str(), config.aux_model.c_str(), max_prefill.c_str(),
                  max_frames.c_str(), threads.c_str(), static_cast<char *>(nullptr));
            _exit(127);
        }
        close(sockets[1]);
        pid_ = child;
        fd_ = sockets[0];
        set_socket_timeout(fd_, 240);

        protocol::message message;
        if (!protocol::receive(fd_, message)) {
            throw std::runtime_error("Qwen3-TTS talker exited during startup");
        }
        if (message.type == protocol::message_type::error) {
            throw std::runtime_error(std::string(message.payload.begin(), message.payload.end()));
        }
        if (message.type != protocol::message_type::ready) {
            throw std::runtime_error("Qwen3-TTS talker did not send ready");
        }
    }

    talker_process(const talker_process &) = delete;
    talker_process & operator=(const talker_process &) = delete;

    ~talker_process() {
        if (fd_ >= 0) {
            try {
                protocol::send(fd_, protocol::message_type::shutdown);
            } catch (...) {
            }
            close(fd_);
            fd_ = -1;
        }
        if (pid_ > 0) {
            for (int i = 0; i < 100; ++i) {
                if (waitpid(pid_, nullptr, WNOHANG) == pid_) {
                    pid_ = -1;
                    break;
                }
                usleep(20000);
            }
            if (pid_ > 0) {
                kill(pid_, SIGTERM);
                waitpid(pid_, nullptr, 0);
                pid_ = -1;
            }
        }
    }

    std::vector<std::array<int32_t, code_groups>> generate(
            const std::vector<float> & prefill,
            const std::vector<float> & trailing,
            const std::array<float, hidden_size> & pad,
            uint32_t max_frames) {
        const uint32_t n_prefill = static_cast<uint32_t>(prefill.size() / hidden_size);
        const uint32_t n_trailing = static_cast<uint32_t>(trailing.size() / hidden_size);
        std::vector<uint8_t> payload;
        payload.reserve(12 + (prefill.size() + trailing.size() + pad.size()) * sizeof(float));
        protocol::append_u32(payload, n_prefill);
        protocol::append_u32(payload, n_trailing);
        protocol::append_u32(payload, max_frames);
        protocol::append_bytes(payload, prefill.data(), prefill.size() * sizeof(float));
        protocol::append_bytes(payload, trailing.data(), trailing.size() * sizeof(float));
        protocol::append_bytes(payload, pad.data(), pad.size() * sizeof(float));
        protocol::send(fd_, protocol::message_type::talker_job, payload);

        std::vector<std::array<int32_t, code_groups>> frames;
        protocol::message message;
        while (protocol::receive(fd_, message)) {
            if (message.type == protocol::message_type::error) {
                throw std::runtime_error(std::string(message.payload.begin(), message.payload.end()));
            }
            if (message.type == protocol::message_type::talker_frame) {
                protocol::reader reader(message.payload);
                std::array<int32_t, code_groups> frame{};
                for (auto & code : frame) {
                    code = static_cast<int32_t>(reader.u32());
                }
                if (reader.remaining() != 0) {
                    throw std::runtime_error("invalid Qwen3-TTS code frame");
                }
                frames.push_back(frame);
                continue;
            }
            if (message.type == protocol::message_type::talker_done) {
                protocol::reader reader(message.payload);
                const uint32_t reported_frames = reader.u32();
                const bool hit_limit = reader.u32() != 0;
                if (reader.remaining() != 0 || reported_frames != frames.size()) {
                    throw std::runtime_error("invalid Qwen3-TTS talker completion");
                }
                if (hit_limit) {
                    throw std::runtime_error("Qwen3-TTS generation reached the frame limit without EOS");
                }
                return frames;
            }
            throw std::runtime_error("unexpected Qwen3-TTS talker response");
        }
        throw std::runtime_error("Qwen3-TTS talker exited during generation");
    }

  private:
    pid_t pid_ = -1;
    int fd_ = -1;
};

size_t utf8_char_size(unsigned char first) {
    if ((first & 0x80U) == 0) return 1;
    if ((first & 0xE0U) == 0xC0U) return 2;
    if ((first & 0xF0U) == 0xE0U) return 3;
    if ((first & 0xF8U) == 0xF0U) return 4;
    return 1;
}

size_t utf8_length(const std::string & text) {
    size_t count = 0;
    for (size_t i = 0; i < text.size();) {
        i += std::min(utf8_char_size(static_cast<unsigned char>(text[i])), text.size() - i);
        ++count;
    }
    return count;
}

uint32_t utf8_codepoint(const std::string & text, size_t offset, size_t size) {
    const auto b0 = static_cast<unsigned char>(text[offset]);
    if (size == 1) return b0;
    if (size == 2 && offset + 1 < text.size()) {
        return (static_cast<uint32_t>(b0 & 0x1FU) << 6) |
               (static_cast<unsigned char>(text[offset + 1]) & 0x3FU);
    }
    if (size == 3 && offset + 2 < text.size()) {
        return (static_cast<uint32_t>(b0 & 0x0FU) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3FU) << 6) |
               (static_cast<unsigned char>(text[offset + 2]) & 0x3FU);
    }
    if (size == 4 && offset + 3 < text.size()) {
        return (static_cast<uint32_t>(b0 & 0x07U) << 18) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3FU) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 2]) & 0x3FU) << 6) |
               (static_cast<unsigned char>(text[offset + 3]) & 0x3FU);
    }
    return b0;
}

bool contains_cjk(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        const size_t size = std::min(utf8_char_size(static_cast<unsigned char>(text[i])), text.size() - i);
        const uint32_t codepoint = utf8_codepoint(text, i, size);
        if ((codepoint >= 0x3400 && codepoint <= 0x9FFF) ||
            (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
            (codepoint >= 0x20000 && codepoint <= 0x2EBEF)) {
            return true;
        }
        i += size;
    }
    return false;
}

std::string trim(std::string text) {
    const auto whitespace = [](unsigned char value) {
        return value == ' ' || value == '\t' || value == '\r' || value == '\n';
    };
    while (!text.empty() && whitespace(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
    while (!text.empty() && whitespace(static_cast<unsigned char>(text.back()))) text.pop_back();
    return text;
}

bool ends_with(const std::string & text, const std::string & suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool in_set(const std::vector<std::string> & values, const std::string & value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::vector<std::string> split_on_punctuation(const std::string & text,
                                               const std::vector<std::string> & punctuation,
                                               size_t minimum_chars) {
    std::vector<std::string> result;
    std::string current;
    for (size_t i = 0; i < text.size();) {
        const size_t size = std::min(utf8_char_size(static_cast<unsigned char>(text[i])), text.size() - i);
        const std::string character = text.substr(i, size);
        current += character;
        i += size;
        if (in_set(punctuation, character) && utf8_length(current) >= minimum_chars) {
            std::string segment = trim(current);
            if (!segment.empty()) result.push_back(std::move(segment));
            current.clear();
        }
    }
    current = trim(current);
    if (!current.empty()) result.push_back(std::move(current));
    return result;
}

std::vector<std::string> split_by_length(const std::string & text, size_t maximum_chars) {
    std::vector<std::string> result;
    std::string current;
    size_t characters = 0;
    for (size_t i = 0; i < text.size();) {
        const size_t size = std::min(utf8_char_size(static_cast<unsigned char>(text[i])), text.size() - i);
        current.append(text, i, size);
        i += size;
        ++characters;
        if (characters >= maximum_chars) {
            std::string segment = trim(current);
            if (!segment.empty()) result.push_back(std::move(segment));
            current.clear();
            characters = 0;
        }
    }
    current = trim(current);
    if (!current.empty()) result.push_back(std::move(current));
    return result;
}

std::string ensure_sentence_end(std::string segment) {
    segment = trim(std::move(segment));
    if (segment.empty()) return segment;
    const std::vector<std::string> endings = {
        "\xE3\x80\x82", ".", "\xEF\xBC\x81", "\xEF\xBC\x9F", "!", "?", "\xEF\xBC\x9B", ";"};
    for (const auto & ending : endings) {
        if (ends_with(segment, ending)) return segment;
    }
    segment += contains_cjk(segment) ? "\xE3\x80\x82" : ".";
    return segment;
}

std::vector<std::string> split_text(const std::string & text) {
    const size_t maximum_chars = contains_cjk(text) ? 48 : 96;
    const std::vector<std::string> strong = {
        "\xE3\x80\x82", ".", "\xEF\xBC\x81", "\xEF\xBC\x9F", "!", "?", "\xEF\xBC\x9B", ";"};
    const std::vector<std::string> weak = {
        "\xEF\xBC\x8C", ",", "\xE3\x80\x81", "\xEF\xBC\x9A", ":"};
    std::vector<std::string> result;
    for (const auto & sentence : split_on_punctuation(text, strong, 1)) {
        if (utf8_length(sentence) <= maximum_chars) {
            result.push_back(ensure_sentence_end(sentence));
            continue;
        }
        auto parts = split_on_punctuation(sentence, weak, 24);
        if (parts.size() <= 1) parts = split_by_length(sentence, maximum_chars);
        for (const auto & part : parts) {
            if (utf8_length(part) <= maximum_chars) {
                result.push_back(ensure_sentence_end(part));
            } else {
                for (const auto & item : split_by_length(part, maximum_chars)) {
                    result.push_back(ensure_sentence_end(item));
                }
            }
        }
    }
    if (result.empty()) {
        const std::string segment = ensure_sentence_end(text);
        if (!segment.empty()) result.push_back(segment);
    }
    return result;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, true);
    if (count >= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
    if (count < 0) {
        throw std::runtime_error("failed to tokenize Qwen3-TTS text");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string make_prompt(const std::string & text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

struct frontend_input {
    std::vector<float> prefill;
    std::vector<float> trailing;
    std::array<float, hidden_size> pad{};
};

void append_vector(std::vector<float> & destination, const float * source) {
    destination.insert(destination.end(), source, source + hidden_size);
}

void append_sum(std::vector<float> & destination, const float * left, const float * right) {
    const size_t offset = destination.size();
    destination.resize(offset + hidden_size);
    for (int i = 0; i < hidden_size; ++i) destination[offset + i] = left[i] + right[i];
}

void put_wav_u16(std::vector<uint8_t> & out, size_t offset, uint16_t value) {
    protocol::put_u16(out.data() + offset, value);
}

void put_wav_u32(std::vector<uint8_t> & out, size_t offset, uint32_t value) {
    protocol::put_u32(out.data() + offset, value);
}

std::vector<uint8_t> make_wav(const std::vector<int16_t> & samples, int sample_rate) {
    const uint64_t pcm_bytes_u64 = samples.size() * sizeof(int16_t);
    if (pcm_bytes_u64 > protocol::max_payload_size - 44U) {
        throw std::runtime_error("Qwen3-TTS WAV output is too large");
    }
    const uint32_t pcm_bytes = static_cast<uint32_t>(pcm_bytes_u64);
    std::vector<uint8_t> wav(static_cast<size_t>(44) + pcm_bytes);
    std::memcpy(wav.data(), "RIFF", 4);
    put_wav_u32(wav, 4, 36 + pcm_bytes);
    std::memcpy(wav.data() + 8, "WAVEfmt ", 8);
    put_wav_u32(wav, 16, 16);
    put_wav_u16(wav, 20, 1);
    put_wav_u16(wav, 22, 1);
    put_wav_u32(wav, 24, static_cast<uint32_t>(sample_rate));
    put_wav_u32(wav, 28, static_cast<uint32_t>(sample_rate * 2));
    put_wav_u16(wav, 32, 2);
    put_wav_u16(wav, 34, 16);
    std::memcpy(wav.data() + 36, "data", 4);
    put_wav_u32(wav, 40, pcm_bytes);
    std::memcpy(wav.data() + 44, samples.data(), pcm_bytes);
    return wav;
}

}  // namespace

struct runtime::impl {
    impl(std::string config_dir, std::string speaker_file, const std::string & talker_executable) :
        config(load_config(config_dir)),
        ort_env(ORT_LOGGING_LEVEL_WARNING, "qwen3-tts") {
        if (speaker_file.empty()) {
            speaker_file = config.default_speaker;
        } else if (!std::filesystem::path(speaker_file).is_absolute()) {
            speaker_file = resolve_path(config_dir, speaker_file);
        }
        if (speaker_file.empty()) {
            throw std::runtime_error("Qwen3-TTS requires --tts-speaker-file or tts_model.speaker_file");
        }
        speaker = read_speaker(speaker_file);
        aux = std::make_unique<mapped_gguf>(config.aux_model);
        codec_embedding = static_cast<const float *>(aux->tensor(
            "q3tts.codec_embedding.weight", GGML_TYPE_F32,
            3072ULL * hidden_size * sizeof(float)));

        llama_model_params tokenizer_params = llama_model_default_params();
        tokenizer_params.vocab_only = true;
        tokenizer_model.reset(llama_model_load_from_file(config.tokenizer_model.c_str(), tokenizer_params));
        if (!tokenizer_model) {
            throw std::runtime_error("failed to load Qwen3-TTS tokenizer GGUF");
        }
        vocab = llama_model_get_vocab(tokenizer_model.get());

        Ort::SessionOptions text_options;
        text_options.SetIntraOpNumThreads(config.frontend_threads);
        text_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        text_session = std::make_unique<Ort::Session>(ort_env, config.text_embedding_model.c_str(), text_options);
        text_input_name = session_name(*text_session, true);
        text_output_name = session_name(*text_session, false);
        tts_embeddings = embed({text_tts_bos, text_tts_eos, text_tts_pad});
        if (tts_embeddings.size() != 3ULL * hidden_size) {
            throw std::runtime_error("unexpected Qwen3-TTS control embedding shape");
        }

        Ort::SessionOptions codec_options;
        codec_options.SetIntraOpNumThreads(config.codec_threads);
        append_spacemit_ep(codec_options, ep_library, config.ep_config);
        codec_session = std::make_unique<Ort::Session>(ort_env, config.codec_decoder_model.c_str(), codec_options);
        codec_input_name = session_name(*codec_session, true);
        codec_output_name = session_name(*codec_session, false);
        std::vector<std::array<int32_t, code_groups>> warmup(codec_bucket);
        (void) decode_codec_chunk(warmup, 0);

        talker = std::make_unique<talker_process>(talker_executable, config);
    }

    struct model_deleter { void operator()(llama_model * model) const { llama_model_free(model); } };

    static std::string session_name(Ort::Session & session, bool input) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto name = input ? session.GetInputNameAllocated(0, allocator) :
                            session.GetOutputNameAllocated(0, allocator);
        return name.get();
    }

    std::vector<float> embed(const std::vector<int64_t> & ids) {
        std::array<int64_t, 2> shape{1, static_cast<int64_t>(ids.size())};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<int64_t>(memory, const_cast<int64_t *>(ids.data()), ids.size(),
                                                       shape.data(), shape.size());
        const char * input_names[] = {text_input_name.c_str()};
        const char * output_names[] = {text_output_name.c_str()};
        auto outputs = text_session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count != ids.size() * static_cast<size_t>(hidden_size)) {
            throw std::runtime_error("unexpected Qwen3-TTS text embedding shape");
        }
        const float * data = outputs[0].GetTensorData<float>();
        return std::vector<float>(data, data + count);
    }

    frontend_input make_frontend(const std::string & text) {
        const auto prompt_tokens = tokenize(vocab, make_prompt(text));
        if (prompt_tokens.size() < 9) {
            throw std::runtime_error("Qwen3-TTS text prompt is too short");
        }
        std::vector<int64_t> ids(prompt_tokens.begin(), prompt_tokens.end());
        const std::vector<float> text_embeddings = embed(ids);
        const float * bos = tts_embeddings.data();
        const float * eos = tts_embeddings.data() + hidden_size;
        const float * pad = tts_embeddings.data() + 2 * hidden_size;
        auto codec = [&](int id) -> const float * {
            return codec_embedding + static_cast<size_t>(id) * hidden_size;
        };

        const std::array<const float *, 6> controls = {
            codec(codec_nothink), codec(codec_think_bos), codec(codec_think_eos),
            speaker.data(), codec(codec_pad), codec(codec_bos)};

        frontend_input result;
        std::copy(pad, pad + hidden_size, result.pad.begin());
        for (int i = 0; i < 3; ++i) {
            append_vector(result.prefill, text_embeddings.data() + static_cast<size_t>(i) * hidden_size);
        }
        for (size_t i = 0; i + 1 < controls.size(); ++i) {
            append_sum(result.prefill, i + 2 == controls.size() ? bos : pad, controls[i]);
        }
        append_sum(result.prefill, text_embeddings.data() + 3ULL * hidden_size, controls.back());
        if (ids.size() > 9) {
            for (size_t i = 4; i < ids.size() - 5; ++i) {
                append_vector(result.trailing, text_embeddings.data() + i * hidden_size);
            }
        }
        append_vector(result.trailing, eos);
        if (result.prefill.size() / hidden_size > static_cast<size_t>(config.max_prefill)) {
            throw std::runtime_error("Qwen3-TTS segment exceeds max_prefill");
        }
        return result;
    }

    std::vector<float> decode_codec_chunk(
            const std::vector<std::array<int32_t, code_groups>> & codes,
            int context_frames) {
        if (codes.empty() || codes.size() > codec_bucket || context_frames < 0 ||
            context_frames > static_cast<int>(codes.size())) {
            throw std::runtime_error("invalid Qwen3-TTS codec chunk");
        }
        std::vector<int64_t> input_data(code_groups * codec_bucket, 0);
        for (size_t frame = 0; frame < codes.size(); ++frame) {
            for (int group = 0; group < code_groups; ++group) {
                input_data[static_cast<size_t>(group) * codec_bucket + frame] = codes[frame][group];
            }
        }
        std::array<int64_t, 3> shape{1, code_groups, codec_bucket};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<int64_t>(memory, input_data.data(), input_data.size(),
                                                       shape.data(), shape.size());
        const char * input_names[] = {codec_input_name.c_str()};
        const char * output_names[] = {codec_output_name.c_str()};
        auto outputs = codec_session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        const size_t output_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const size_t begin = static_cast<size_t>(context_frames) * samples_per_frame;
        const size_t end = codes.size() * samples_per_frame;
        if (begin > end || end > output_count) {
            throw std::runtime_error("unexpected Qwen3-TTS codec output shape");
        }
        const float * output = outputs[0].GetTensorData<float>();
        return std::vector<float>(output + begin, output + end);
    }

    std::vector<float> decode_codec(const std::vector<std::array<int32_t, code_groups>> & frames) {
        std::vector<float> audio;
        size_t completed = 0;
        while (completed < frames.size()) {
            const size_t new_frames = std::min<size_t>(codec_bucket, frames.size() - completed);
            const size_t context = std::min({completed, static_cast<size_t>(codec_context),
                                             static_cast<size_t>(codec_bucket) - new_frames});
            std::vector<std::array<int32_t, code_groups>> chunk(
                frames.begin() + static_cast<std::ptrdiff_t>(completed - context),
                frames.begin() + static_cast<std::ptrdiff_t>(completed + new_frames));
            auto decoded = decode_codec_chunk(chunk, static_cast<int>(context));
            audio.insert(audio.end(), decoded.begin(), decoded.end());
            completed += new_frames;
        }
        return audio;
    }

    synthesis_result synthesize(const std::string & text) {
        const auto start = clock_type::now();
        const auto segments = split_text(text);
        if (segments.empty()) {
            throw std::invalid_argument("Qwen3-TTS input is empty after segmentation");
        }
        std::vector<int16_t> pcm;
        const size_t pause_samples = static_cast<size_t>(config.sample_rate) * config.segment_pause_ms / 1000;
        for (size_t index = 0; index < segments.size(); ++index) {
            const frontend_input input = make_frontend(segments[index]);
            const auto codes = talker->generate(input.prefill, input.trailing, input.pad, config.max_frames);
            const auto audio = decode_codec(codes);
            if (audio.empty()) {
                throw std::runtime_error("Qwen3-TTS codec produced empty audio");
            }
            if (index > 0) pcm.insert(pcm.end(), pause_samples, 0);
            pcm.reserve(pcm.size() + audio.size());
            for (float sample : audio) {
                const float clipped = std::clamp(sample, -1.0f, 1.0f);
                pcm.push_back(static_cast<int16_t>(std::lrint(clipped * 32767.0f)));
            }
        }
        synthesis_result result;
        result.wav = make_wav(pcm, config.sample_rate);
        result.stats.segments = static_cast<uint32_t>(segments.size());
        result.stats.sample_rate = static_cast<uint32_t>(config.sample_rate);
        result.stats.samples = pcm.size();
        result.stats.wall_seconds = std::chrono::duration<double>(clock_type::now() - start).count();
        return result;
    }

    runtime_config config;
    shared_library ep_library;
    Ort::Env ort_env;
    std::unique_ptr<Ort::Session> text_session;
    std::unique_ptr<Ort::Session> codec_session;
    std::string text_input_name;
    std::string text_output_name;
    std::string codec_input_name;
    std::string codec_output_name;
    std::unique_ptr<mapped_gguf> aux;
    const float * codec_embedding = nullptr;
    std::vector<float> speaker;
    std::vector<float> tts_embeddings;
    std::unique_ptr<llama_model, model_deleter> tokenizer_model;
    const llama_vocab * vocab = nullptr;
    std::unique_ptr<talker_process> talker;
};

runtime::runtime(std::string config_dir, std::string speaker_file, std::string talker_executable) :
    pimpl_(std::make_unique<impl>(std::move(config_dir), std::move(speaker_file), talker_executable)) {}

runtime::~runtime() = default;

synthesis_result runtime::synthesize(const std::string & text) {
    return pimpl_->synthesize(text);
}

}  // namespace qwen3_tts
