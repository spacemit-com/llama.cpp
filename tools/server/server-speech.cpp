#include "server-speech.h"

#include "server-speech-backend.h"
#include "server-speech-qwen3-tts.h"

#include <memory>
#include <stdexcept>

namespace {

std::unique_ptr<server_speech_backend> create_backend(const common_params & params) {
    if (server_speech_qwen3_tts_config_matches(params)) {
        return server_speech_qwen3_tts_create(params);
    }
    return nullptr;
}

}  // namespace

bool server_speech_config_matches(const common_params & params) {
    return server_speech_qwen3_tts_config_matches(params);
}

std::string server_speech_backend_name(const common_params & params) {
    return server_speech_qwen3_tts_config_matches(params) ? server_speech_qwen3_tts_name() : "";
}

struct server_speech_service::impl {
    explicit impl(const common_params & params) : backend(create_backend(params)) {
        if (!backend) {
            throw std::runtime_error("speech backend is not enabled");
        }
    }

    std::unique_ptr<server_speech_backend> backend;
};

server_speech_service::server_speech_service(const common_params & params) :
    pimpl_(std::make_unique<impl>(params)) {}

server_speech_service::~server_speech_service() = default;

server_speech_result server_speech_service::synthesize(const nlohmann::ordered_json & body) {
    server_speech_result result = pimpl_->backend->synthesize(body);
    if (result.backend.empty()) {
        result.backend = pimpl_->backend->name();
    }
    return result;
}
