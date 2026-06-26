#include "server-speech.h"

#include "server-speech-backend.h"
#include "server-speech-qwen3-tts.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <stdexcept>

using json = nlohmann::ordered_json;

static std::string server_speech_backend_name_for_config(const common_params & params) {
    if (server_speech_qwen3_tts_config_matches(params)) {
        return server_speech_qwen3_tts_name();
    }
    return {};
}

static std::unique_ptr<server_speech_backend> server_speech_create_backend(const common_params & params) {
    if (server_speech_qwen3_tts_config_matches(params)) {
        return server_speech_qwen3_tts_create(params);
    }
    return nullptr;
}

bool server_speech_config_matches(const common_params & params) {
    return !server_speech_backend_name_for_config(params).empty();
}

std::string server_speech_backend_name(const common_params & params) {
    return server_speech_backend_name_for_config(params);
}

struct server_speech_service::impl {
    explicit impl(const common_params & params) :
        backend(server_speech_create_backend(params)) {
    }

    server_speech_result synthesize(const json & body) {
        if (!backend) {
            throw std::runtime_error("speech backend is not enabled");
        }
        server_speech_result result = backend->synthesize(body);
        if (result.backend.empty()) {
            result.backend = backend->name();
        }
        return result;
    }

    std::unique_ptr<server_speech_backend> backend;
};

server_speech_service::server_speech_service(const common_params & params) :
    pimpl(new impl(params)) {
}

server_speech_service::~server_speech_service() = default;

server_speech_result server_speech_service::synthesize(const json & body) {
    return pimpl->synthesize(body);
}
