#pragma once

// Encoder stage: wraps a single SMT audio ONNX context (frontend+backend).
// Runs on its own worker thread, pinned to the encoder core set.
// Encodes audio serially (B1: one encode at a time) and produces the audio
// embedding into the request's `embd` field.

#include "multi-asr-common.h"

#include <memory>
#include <string>

struct smt_audio_context;  // fwd (tools/mtmd/smt-audio-wrapper.h)

class multi_asr_encoder {
  public:
    multi_asr_encoder();
    ~multi_asr_encoder();

    multi_asr_encoder(const multi_asr_encoder &)             = delete;
    multi_asr_encoder & operator=(const multi_asr_encoder &) = delete;

    // Load the ONNX encoder from smt_config_dir. Core binding (affinity/threads)
    // for the ONNX EP comes from config.json. Throws std::runtime_error on failure.
    void init(const std::string & smt_config_dir, bool warmup);

    // Encode one request's audio -> fills req.embd, req.n_audio_tokens, timings.
    // Returns true on success; on failure sets req.error and returns false.
    bool encode(multi_asr_request & req);

    int64_t hidden_size() const { return hidden_size_; }

  private:
    std::unique_ptr<smt_audio_context> audio_;
    int64_t                            hidden_size_ = 0;
};
