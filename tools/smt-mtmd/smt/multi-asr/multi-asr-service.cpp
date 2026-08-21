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
    p.queue_max = std::max(1, server_params.n_parallel * 8);
    p.n_predict = server_params.n_predict > 0 ? server_params.n_predict : 256;
    // Keep the legacy decoder's model/context settings independent from the
    // HTTP server.  In particular, inheriting the server's reduced n_ctx or
    // its HTTP thread count changes the standalone ASR graph and can produce
    // invalid output on SpacemiT's backend.  The old multi-ASR binary uses
    // model-default context and backend-default decoder threading.
    p.n_ctx = 0;
    p.n_batch = 2048;
    p.decoder_n_threads = 0;
    // Preserve the old FIFO intake semantics, but serialize each complete
    // encode->decode transaction inside llama-server.  The standalone binary
    // can overlap E/D; llama-server owns a second GGUF context in the same
    // process, where that overlap corrupts backend state.
    p.enable_pipeline = false;

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
    const std::string assistant = "<|im_start|>assistant\n";
    if (const size_t pos = req.prompt.rfind(assistant); pos != std::string::npos) {
        const std::string tail = req.prompt.substr(pos + assistant.size());
        if (!tail.empty()) {
            req.prompt = tail;
        }
    }
    // Chat-template rendering may place the user instruction before the
    // assistant generation marker.  Recover the legacy Qwen3-ASR instruction
    // instead of passing the whole rendered conversation to the decoder.
    // The server's `prompt` field is usually a fully rendered chat template,
    // and some templates stringify a missing language as `language None`.
    // The legacy decoder expects only the short Qwen3-ASR instruction.  Keep a
    // valid explicit language when it is present; otherwise use the proven
    // Chinese default.  Never pass a rendered template or a `None` language
    // through to the decoder.
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
    req.n_predict = n_predict > 0 ? n_predict : 256;
    return orchestrator_->submit_and_wait(req, result);
}
