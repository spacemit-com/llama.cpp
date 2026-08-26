#pragma once

#include "multi-asr-common.h"
#include "multi-asr-orchestrator.h"

#include <memory>

// Thin HTTP-facing facade over the ASR-owned encoder/decoder scheduler.
class smt_multi_asr_service {
  public:
    smt_multi_asr_service() = default;
    ~smt_multi_asr_service();

    smt_multi_asr_service(const smt_multi_asr_service &) = delete;
    smt_multi_asr_service & operator=(const smt_multi_asr_service &) = delete;

    void init(llama_model * model, const common_params & server_params);
    bool submit(const std::vector<uint8_t> & audio, const std::string & prompt,
                int32_t n_predict, multi_asr_request & result);
    llama_context * context() const;

  private:
    std::unique_ptr<multi_asr_orchestrator> orchestrator_;
};
