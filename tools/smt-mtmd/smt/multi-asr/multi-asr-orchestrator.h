#pragma once

// FIFO ASR scheduler:
//   HTTP submit -> serial SMT encoder -> decoder ready FIFO -> native
//   continuous-batching generation.  The decoder thread is the sole owner of
//   its llama_context, so KV sequences and samplers cannot race.

#include "multi-asr-common.h"
#include "multi-asr-decoder.h"
#include "multi-asr-encoder.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct multi_asr_job {
    multi_asr_request req;
    std::mutex mu;
    std::condition_variable cv;
    bool finished = false;
    int64_t t_submit_ms = 0;
    int slot_id = -1;
};

using multi_asr_job_ptr = std::shared_ptr<multi_asr_job>;

class multi_asr_orchestrator {
  public:
    multi_asr_orchestrator() = default;
    ~multi_asr_orchestrator();

    void init_shared(llama_model * model, const multi_asr_params & params);
    void start();
    void stop();

    bool submit_and_wait(const multi_asr_request & req_in, multi_asr_request & out);
    int64_t hidden_size() const { return hidden_size_; }
    llama_context * context() const { return decoder_.context(); }

  private:
    multi_asr_params params_;
    multi_asr_encoder encoder_;
    multi_asr_decoder decoder_;
    int64_t hidden_size_ = 0;

    std::deque<multi_asr_job_ptr> intake_;
    std::mutex intake_mu_;
    std::condition_variable intake_cv_;

    std::deque<multi_asr_job_ptr> ready_;
    std::mutex ready_mu_;
    std::condition_variable ready_cv_;

    std::vector<multi_asr_job_ptr> active_;
    std::vector<bool> slot_busy_;
    std::atomic<int32_t> in_flight_{ 0 };
    std::atomic<int32_t> pending_encode_{ 0 };
    std::atomic<bool> running_{ false };
    std::thread encode_thread_;
    std::thread decode_thread_;

    void encode_loop();
    void decode_loop();
    void finish_job(const multi_asr_job_ptr & job);
};
