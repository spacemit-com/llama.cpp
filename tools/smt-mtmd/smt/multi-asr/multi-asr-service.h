#pragma once

#include "multi-asr-common.h"
#include "multi-asr-orchestrator.h"

#include <memory>

// Server-facing facade for the legacy, correctness-proven ASR pipeline.
// The facade deliberately does not expose llama-server slots or media chunks:
// requests are admitted FIFO and decoded by the private legacy context.
class smt_multi_asr_service {
  public:
    smt_multi_asr_service() = default;
    ~smt_multi_asr_service();

    smt_multi_asr_service(const smt_multi_asr_service &) = delete;
    smt_multi_asr_service & operator=(const smt_multi_asr_service &) = delete;

    void init(llama_model * model, const common_params & server_params);
    bool submit(const std::vector<uint8_t> & audio, const std::string & prompt,
                int32_t n_predict, multi_asr_request & result);

  private:
    std::unique_ptr<multi_asr_orchestrator> orchestrator_;
};
