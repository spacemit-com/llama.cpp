#include "server-speech-qwen3-tts.h"

#include "qwen3_tts_protocol.h"
#include "server-speech-backend.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::ordered_json;

namespace {

constexpr const char * backend_name = "qwen3-tts";
constexpr size_t max_input_bytes = 1024 * 1024;

bool config_has_qwen3_tts_architecture(const std::filesystem::path & config_path) {
    try {
        std::ifstream input(config_path);
        if (!input) {
            return false;
        }
        const json document = json::parse(input);
        if (!document.contains("architectures") || !document.at("architectures").is_array()) {
            return false;
        }
        for (const auto & architecture : document.at("architectures")) {
            if (architecture.is_string() && architecture.get<std::string>() == "Qwen3TTSForConditionalGeneration") {
                return true;
            }
        }
    } catch (...) {
    }
    return false;
}

std::filesystem::path executable_directory() {
    std::vector<char> path(4096);
    const ssize_t size = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (size <= 0) {
        throw std::runtime_error("failed to locate Qwen3-TTS private executables");
    }
    path[static_cast<size_t>(size)] = '\0';
    return std::filesystem::path(path.data()).parent_path();
}

std::string require_private_executable(const std::filesystem::path & path) {
    if (access(path.c_str(), X_OK) != 0) {
        throw std::runtime_error("missing private Qwen3-TTS executable: " + path.string());
    }
    return path.string();
}

void set_socket_timeout(int fd) {
    timeval timeout{};
    timeout.tv_sec = 240;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw std::runtime_error("failed to set Qwen3-TTS runner timeout");
    }
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

std::vector<std::pair<std::string, std::string>> parse_hotwords(const json & body) {
    const json * value = nullptr;
    if (body.contains("hotwords")) {
        value = &body.at("hotwords");
    } else if (body.contains("lexicon")) {
        value = &body.at("lexicon");
    }
    if (value == nullptr) {
        return {};
    }

    std::vector<std::pair<std::string, std::string>> result;
    if (value->is_object()) {
        for (const auto & item : value->items()) {
            if (!item.key().empty() && item.value().is_string()) {
                result.emplace_back(item.key(), item.value().get<std::string>());
            }
        }
        return result;
    }
    if (value->is_array()) {
        for (const auto & item : *value) {
            if (!item.is_object()) {
                throw std::invalid_argument("hotwords array entries must be objects");
            }
            const auto from = item.find("from");
            const auto to = item.find("to");
            if (from == item.end() || to == item.end() || !from->is_string() || !to->is_string()) {
                throw std::invalid_argument("hotwords entries require string from and to fields");
            }
            result.emplace_back(from->get<std::string>(), to->get<std::string>());
        }
        return result;
    }
    throw std::invalid_argument("hotwords must be an object or array");
}

std::string request_text(const json & body) {
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    const auto input = body.find("input");
    if (input == body.end() || !input->is_string() || input->get_ref<const std::string &>().empty()) {
        throw std::invalid_argument("input must be a non-empty string");
    }
    if (input->get_ref<const std::string &>().size() > max_input_bytes) {
        throw std::invalid_argument("input is too large");
    }
    if (const auto format = body.find("response_format"); format != body.end() &&
        (!format->is_string() || format->get<std::string>() != "wav")) {
        throw std::invalid_argument("response_format must be wav");
    }
    if (const auto speed = body.find("speed"); speed != body.end() &&
        (!speed->is_number() || !std::isfinite(speed->get<double>()) || std::abs(speed->get<double>() - 1.0) > 1e-9)) {
        throw std::invalid_argument("speed must be 1.0");
    }
    for (const char * field : {"model", "voice"}) {
        if (const auto value = body.find(field); value != body.end() && !value->is_string()) {
            throw std::invalid_argument(std::string(field) + " must be a string");
        }
    }

    std::string text = input->get<std::string>();
    for (const auto & [from, to] : parse_hotwords(body)) {
        replace_all(text, from, to);
    }
    if (text.empty() || text.size() > max_input_bytes) {
        throw std::invalid_argument("input is invalid after hotword replacement");
    }
    return text;
}

class runner_process {
  public:
    runner_process(const std::string & config_dir, const std::string & speaker_file) {
        std::signal(SIGPIPE, SIG_IGN);
        const std::filesystem::path bin_dir = executable_directory();
        const std::string runner = require_private_executable(bin_dir / "llama-qwen3-tts-runner");
        const std::string talker = require_private_executable(bin_dir / "llama-qwen3-tts-talker");

        int sockets[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
            throw std::runtime_error("failed to create Qwen3-TTS runner socket");
        }
        const pid_t child = fork();
        if (child < 0) {
            close(sockets[0]);
            close(sockets[1]);
            throw std::runtime_error("failed to fork Qwen3-TTS runner");
        }
        if (child == 0) {
            close(sockets[0]);
            const pid_t parent = getppid();
            if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent) {
                _exit(125);
            }
            setpgid(0, 0);
            if (sockets[1] != 3) {
                if (dup2(sockets[1], 3) < 0) {
                    _exit(126);
                }
                close(sockets[1]);
            }
            execl(runner.c_str(), runner.c_str(), "3", config_dir.c_str(), speaker_file.c_str(),
                  talker.c_str(), static_cast<char *>(nullptr));
            _exit(127);
        }

        close(sockets[1]);
        fd_ = sockets[0];
        pid_ = child;
        try {
            setpgid(child, child);
            if (fcntl(fd_, F_SETFD, FD_CLOEXEC) != 0) {
                throw std::runtime_error("failed to protect Qwen3-TTS runner socket");
            }
            set_socket_timeout(fd_);
            qwen3_tts::protocol::message message;
            if (!qwen3_tts::protocol::receive(fd_, message)) {
                throw std::runtime_error("Qwen3-TTS runner exited during startup");
            }
            if (message.type == qwen3_tts::protocol::message_type::error) {
                throw std::runtime_error(std::string(message.payload.begin(), message.payload.end()));
            }
            if (message.type != qwen3_tts::protocol::message_type::ready) {
                throw std::runtime_error("Qwen3-TTS runner did not become ready");
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    runner_process(const runner_process &) = delete;
    runner_process & operator=(const runner_process &) = delete;

    ~runner_process() {
        stop();
    }

    server_speech_result synthesize(const std::string & text) {
        std::lock_guard<std::mutex> lock(mutex_);
        qwen3_tts::protocol::send(fd_, qwen3_tts::protocol::message_type::synthesize, text);
        qwen3_tts::protocol::message message;
        if (!qwen3_tts::protocol::receive(fd_, message)) {
            throw std::runtime_error("Qwen3-TTS runner exited during synthesis");
        }
        if (message.type == qwen3_tts::protocol::message_type::error) {
            throw std::runtime_error(std::string(message.payload.begin(), message.payload.end()));
        }
        if (message.type != qwen3_tts::protocol::message_type::synthesis) {
            throw std::runtime_error("unexpected Qwen3-TTS runner response");
        }

        qwen3_tts::protocol::reader reader(message.payload);
        const uint32_t segments = reader.u32();
        const uint32_t sample_rate = reader.u32();
        const uint64_t samples = reader.u64();
        const double wall_seconds = reader.f64();
        if (sample_rate != 24000 || samples == 0 ||
            samples > (qwen3_tts::protocol::max_payload_size - 44) / 2 ||
            !std::isfinite(wall_seconds) || wall_seconds < 0.0 || reader.remaining() < 44) {
            throw std::runtime_error("invalid Qwen3-TTS runner result");
        }
        const size_t wav_size = reader.remaining();
        const uint8_t * wav = reader.bytes(wav_size);
        const size_t pcm_size = static_cast<size_t>(samples) * 2;
        if (wav_size != 44 + pcm_size || std::memcmp(wav, "RIFF", 4) != 0 ||
            std::memcmp(wav + 8, "WAVE", 4) != 0 ||
            qwen3_tts::protocol::get_u32(wav + 24) != sample_rate ||
            qwen3_tts::protocol::get_u32(wav + 40) != pcm_size) {
            throw std::runtime_error("Qwen3-TTS runner returned invalid WAV data");
        }

        server_speech_result result;
        result.wav.assign(reinterpret_cast<const char *>(wav), wav_size);
        result.backend = backend_name;
        result.segments = static_cast<int>(segments);
        result.audio_seconds = static_cast<double>(samples) / sample_rate;
        result.wall_seconds = wall_seconds;
        return result;
    }

  private:
    void stop() noexcept {
        if (fd_ >= 0) {
            try {
                qwen3_tts::protocol::send(fd_, qwen3_tts::protocol::message_type::shutdown);
            } catch (...) {
            }
            close(fd_);
            fd_ = -1;
        }
        if (pid_ <= 0) {
            return;
        }
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (waitpid(pid_, nullptr, WNOHANG) == pid_) {
                pid_ = -1;
                return;
            }
            usleep(20000);
        }
        if (kill(-pid_, SIGTERM) != 0) {
            kill(pid_, SIGTERM);
        }
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }

    int fd_ = -1;
    pid_t pid_ = -1;
    std::mutex mutex_;
};

class qwen3_tts_backend final : public server_speech_backend {
  public:
    explicit qwen3_tts_backend(const common_params & params) :
        runner_(params.smt_config_dir, params.vocoder.speaker_file) {}

    const char * name() const override {
        return backend_name;
    }

    server_speech_result synthesize(const json & body) override {
        return runner_.synthesize(request_text(body));
    }

  private:
    runner_process runner_;
};

}  // namespace

bool server_speech_qwen3_tts_config_matches(const common_params & params) {
    if ((params.media_backend != "smt" && params.media_backend != "auto") || params.smt_config_dir.empty()) {
        return false;
    }
    return config_has_qwen3_tts_architecture(std::filesystem::path(params.smt_config_dir) / "config.json");
}

const char * server_speech_qwen3_tts_name() {
    return backend_name;
}

std::unique_ptr<server_speech_backend> server_speech_qwen3_tts_create(const common_params & params) {
    return std::make_unique<qwen3_tts_backend>(params);
}
