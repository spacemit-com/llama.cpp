#pragma once

// Decoder stage: owns the gguf text model + a single llama_context (one slot).
// Runs on its own worker thread, pinned to the decoder core set.
// Given a request whose `embd` was filled by the encoder stage, it:
//   1. decodes the Qwen3-ASR prompt prefix text tokens,
//   2. decodes <|audio_start|> + audio embedding + <|audio_end|>,
//   3. decodes the assistant prefix text tokens,
//   4. generates output tokens serially until EOG / n_predict.
//
// This mirrors the faithful path in tools/mtmd/mtmd-cli-smt.cpp (eval_message_smt
// + generate_response) but split so encode already happened on the encoder core set.

#include "common.h"
#include "llama.h"
#include "multi-asr-common.h"

#include <memory>
#include <string>
#include <vector>

class multi_asr_decoder {
  public:
    multi_asr_decoder() = default;
    ~multi_asr_decoder();

    multi_asr_decoder(const multi_asr_decoder &)             = delete;
    multi_asr_decoder & operator=(const multi_asr_decoder &) = delete;

    // Load gguf model + create llama_context. `cpu_range` pins the ggml compute
    // threads to the decoder core set. `hidden_size` must match the encoder's
    // audio embedding dimension (validated against model n_embd).
    // Throws std::runtime_error on failure.
    void init(const multi_asr_params & params, int64_t hidden_size);

    // Server integration variant: reuse the already loaded text model while
    // creating a private llama_context for the legacy ASR decoder.  The model
    // is owned by llama-server; this object only owns the context.
    void init_shared(llama_model * model, const multi_asr_params & params, int64_t hidden_size);

    // Decode one request (prefill audio embd + generate). Fills req.text and
    // timings.prefill_ms / decode_ms / n_out_tokens. Resets KV per request.
    // Returns true on success; on failure sets req.error and returns false.
    bool decode(multi_asr_request & req);

    int64_t hidden_size() const { return hidden_size_; }

  private:
    common_init_result_ptr init_;  // owns model + context lifetime
    llama_model *          model_ = nullptr;
    llama_context *        lctx_  = nullptr;
    bool                   owns_context_ = false;
    const llama_vocab *    vocab_ = nullptr;

    int64_t hidden_size_ = 0;
    int32_t n_batch_     = 2048;

    // Qwen3-ASR audio boundary tokens (<|audio_start|>, <|audio_end|>).
    std::vector<llama_token> tok_audio_beg_;
    std::vector<llama_token> tok_audio_end_;

    common_params params_;  // holds sampling config for per-request samplers

    // helpers (mirrors mtmd-cli-smt.cpp)
    int decode_tokens(const std::vector<llama_token> & tokens, llama_pos & n_past, bool logits_last);
    int decode_embd(const float * embd, int n_tokens, llama_pos & n_past, bool logits_last);
};
