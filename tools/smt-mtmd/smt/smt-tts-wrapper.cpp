#include "smt-tts-wrapper.h"

#include "qwen3-tts/qwen3_tts_runtime.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace {

constexpr const char * qwen3_tts_arch = "Qwen3TTSForConditionalGeneration";

bool config_has_architecture(const std::string & config_dir, const char * target) {
    try {
        std::ifstream input(std::filesystem::path(config_dir) / "config.json");
        if (!input) {
            return false;
        }
        const nlohmann::json config = nlohmann::json::parse(input);
        if (!config.contains("architectures")) {
            return false;
        }
        const auto & architectures = config.at("architectures");
        if (architectures.is_string()) {
            return architectures.get<std::string>() == target;
        }
        if (architectures.is_array()) {
            for (const auto & architecture : architectures) {
                if (architecture.is_string() && architecture.get<std::string>() == target) {
                    return true;
                }
            }
        }
    } catch (...) {
    }
    return false;
}

void configure_qwen3_tts_backend() {
    if (setenv("LLAMA_CTX_PAD", "16", 0) != 0) {
        throw std::runtime_error("failed to configure Qwen3-TTS context padding");
    }
}

}  // namespace

struct smt_tts_context::impl {
    std::unique_ptr<qwen3_tts::runtime> runtime;
};

smt_tts_context::smt_tts_context() : pimpl_(new impl()) {}

smt_tts_context::~smt_tts_context() = default;

bool smt_tts_context::matches(const std::string & config_dir) {
    return config_has_architecture(config_dir, qwen3_tts_arch);
}

std::unique_ptr<smt_tts_context> smt_tts_context::create(const std::string & config_dir,
                                                         const std::string & speaker_file) {
    if (!matches(config_dir)) {
        return nullptr;
    }

    configure_qwen3_tts_backend();
    auto context             = std::unique_ptr<smt_tts_context>(new smt_tts_context());
    context->pimpl_->runtime = std::make_unique<qwen3_tts::runtime>(config_dir, speaker_file);
    return context;
}

const char * smt_tts_context::backend_name() const {
    return "qwen3-tts";
}

smt_tts_result smt_tts_context::synthesize(const std::string & text) {
    if (!pimpl_->runtime) {
        throw std::runtime_error("SMT TTS backend is not initialized");
    }
    qwen3_tts::synthesis_result synthesis = pimpl_->runtime->synthesize(text);
    smt_tts_result              result;
    result.wav          = std::move(synthesis.wav);
    result.backend      = backend_name();
    result.segments     = synthesis.stats.segments;
    result.sample_rate  = synthesis.stats.sample_rate;
    result.samples      = synthesis.stats.samples;
    result.wall_seconds = synthesis.stats.wall_seconds;
    return result;
}
