#include "multi-asr-service.h"

#include "common.h"

#include <algorithm>
#include <cstring>

smt_multi_asr_service::~smt_multi_asr_service() {
    if (orchestrator_) {
        orchestrator_->stop();
    }
}

void smt_multi_asr_service::init(llama_model * model, const common_params & server_params) {
    multi_asr_params p;
    p.model_path = server_params.model.path;
    p.smt_config_dir = server_params.smt_config_dir;
    p.warmup = server_params.warmup;
    p.n_parallel = std::max(1, server_params.n_parallel);
    p.queue_max = std::max(p.n_parallel, p.n_parallel * 8);
    p.n_predict = server_params.n_predict > 0 ? server_params.n_predict : 256;
    // The ASR context is sized as a server-style total context. Each slot gets
    // its own sequence; generation is batched across those sequences.
    p.n_ctx = server_params.n_ctx;
    p.n_batch = std::max(1, server_params.n_batch);

    orchestrator_ = std::make_unique<multi_asr_orchestrator>();
    orchestrator_->init_shared(model, p);
    orchestrator_->start();
}

bool smt_multi_asr_service::submit(const std::vector<uint8_t> & audio, const std::string & prompt,
                                   int32_t n_predict, multi_asr_request & result) {
    if (!orchestrator_) {
        result.error = "SMT multi-ASR service is not initialized";
        result.stage = multi_asr_stage::failed;
        return false;
    }

    multi_asr_request req;
    req.audio = audio;
    req.prompt = prompt.empty() ? "language Chinese<asr_text>" : prompt;
    req.n_predict = n_predict > 0 ? n_predict : 256;

    // The OAI parser may provide a rendered ChatML prompt. Only the short
    // Qwen3-ASR instruction belongs after the assistant marker.
    const std::string assistant = "<|im_start|>assistant\n";
    if (const size_t pos = req.prompt.rfind(assistant); pos != std::string::npos) {
        const std::string tail = req.prompt.substr(pos + assistant.size());
        if (!tail.empty()) {
            req.prompt = tail;
        }
    }
    const size_t lang = req.prompt.find("language ");
    const size_t marker = lang == std::string::npos ? std::string::npos : req.prompt.find("<asr_text>", lang);
    if (lang != std::string::npos && marker != std::string::npos) {
        const std::string candidate = req.prompt.substr(lang, marker + std::strlen("<asr_text>") - lang);
        if (candidate.find("None") == std::string::npos && candidate.find("null") == std::string::npos) {
            req.prompt = candidate;
        } else {
            req.prompt = "language Chinese<asr_text>";
        }
    } else {
        req.prompt = "language Chinese<asr_text>";
    }
    return orchestrator_->submit_and_wait(req, result);
}

llama_context * smt_multi_asr_service::context() const {
    return orchestrator_ ? orchestrator_->context() : nullptr;
}
