#include "qwen3_tts_protocol.h"
#include "qwen3_tts_runtime.h"

#include "llama.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

void log_errors(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

int parse_fd(const char * value) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 1024) {
        throw std::runtime_error("invalid Qwen3-TTS runner protocol fd");
    }
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "internal Qwen3-TTS runner invocation is invalid\n");
        return 2;
    }
    std::signal(SIGPIPE, SIG_IGN);
    int fd = -1;
    bool backend_initialized = false;
    try {
        fd = parse_fd(argv[1]);
        llama_log_set(log_errors, nullptr);
        llama_backend_init();
        backend_initialized = true;
        {
            qwen3_tts::runtime runtime(argv[2], argv[3], argv[4]);
            qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::ready);
            qwen3_tts::protocol::message message;
            while (qwen3_tts::protocol::receive(fd, message)) {
                if (message.type == qwen3_tts::protocol::message_type::shutdown) {
                    break;
                }
                if (message.type != qwen3_tts::protocol::message_type::synthesize) {
                    throw std::runtime_error("unexpected Qwen3-TTS runner message");
                }
                try {
                    const std::string text(message.payload.begin(), message.payload.end());
                    const auto result = runtime.synthesize(text);
                    std::vector<uint8_t> payload;
                    payload.reserve(32 + result.wav.size());
                    qwen3_tts::protocol::append_u32(payload, result.stats.segments);
                    qwen3_tts::protocol::append_u32(payload, result.stats.sample_rate);
                    qwen3_tts::protocol::append_u64(payload, result.stats.samples);
                    qwen3_tts::protocol::append_f64(payload, result.stats.wall_seconds);
                    qwen3_tts::protocol::append_bytes(payload, result.wav.data(), result.wav.size());
                    qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::synthesis, payload);
                } catch (const std::exception & error) {
                    qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::error, error.what());
                }
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3-tts runner: %s\n", error.what());
        if (fd >= 0) {
            try {
                qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::error, error.what());
            } catch (...) {
            }
        }
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
