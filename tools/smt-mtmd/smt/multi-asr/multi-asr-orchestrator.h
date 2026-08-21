#pragma once

// Orchestrator: implements scheme B1 "错峰流水" (staggered pipeline).
//
//   intake queue ──► [encode thread](encoder cores) ──► ready queue ──► [decode thread](decoder cores)
//
// Two dedicated worker threads run on disjoint core sets, so E(N+1) overlaps
// D(N). Each stage is serial (one at a time), matching the B1 semantics: at any
// instant at most one encode + one decode in flight.
//
// Concurrency control:
//   - submit() blocks-or-rejects when total in-flight+waiting >= queue_max.
//   - `n_parallel` bounds how many requests may be "in the pipeline" before new
//     submissions wait (the FIFO ">N waits" requirement).
//   - enable_pipeline=false => strict serial baseline (decode finishes before
//     next encode starts), for benchmarking variable ②.

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

// A submitted job: shared between the HTTP handler (waiter) and the pipeline.
struct multi_asr_job {
    multi_asr_request       req;
    std::mutex              mu;
    std::condition_variable cv;
    bool                    finished    = false;
    int64_t                 t_submit_ms = 0;  // ggml_time_ms at submit
};

using multi_asr_job_ptr = std::shared_ptr<multi_asr_job>;

class multi_asr_orchestrator {
  public:
    multi_asr_orchestrator();
    ~multi_asr_orchestrator();

    // Initialize encoder + decoder (loads models, pins cores). Throws on failure.
    void init(const multi_asr_params & params);
    void init_shared(llama_model * model, const multi_asr_params & params);

    // Start the encode/decode worker threads.
    void start();

    // Stop workers and drain. Safe to call once.
    void stop();

    // Submit a request and BLOCK until it finishes (encode+decode) or fails.
    // Returns false if rejected due to a full queue (sets job->req.error).
    bool submit_and_wait(const multi_asr_request & req_in, multi_asr_request & out);

    int64_t hidden_size() const { return hidden_size_; }

  private:
    multi_asr_params  params_;
    multi_asr_encoder encoder_;
    multi_asr_decoder decoder_;
    int64_t           hidden_size_ = 0;

    // intake -> encode
    std::deque<multi_asr_job_ptr> intake_;
    std::mutex                    intake_mu_;
    std::condition_variable       intake_cv_;

    // encode -> decode (bounded to 1 for B1: only one encoded result waiting)
    std::deque<multi_asr_job_ptr> ready_;
    std::mutex                    ready_mu_;
    std::condition_variable       ready_cv_;

    std::atomic<int32_t> in_flight_{ 0 };  // total accepted, not yet finished
    std::atomic<bool>    running_{ false };

    std::thread encode_thread_;
    std::thread decode_thread_;

    void encode_loop();
    void decode_loop();
};
