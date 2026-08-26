#pragma once

#include "common.h"
#include "llama.h"
#include "multi-asr-common.h"
#include "sampling.h"

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

// ASR-specific decoder scheduler. The encoder remains serial, while prompt
// embedding prefill and active token generation share one continuous batch.
class multi_asr_decoder {
  public:
    multi_asr_decoder() = default;
    ~multi_asr_decoder();

    multi_asr_decoder(const multi_asr_decoder &) = delete;
    multi_asr_decoder & operator=(const multi_asr_decoder &) = delete;

    // Borrow the server's model and create one private context with one KV
    // sequence per ASR slot.
    void init_shared(llama_model * model, const multi_asr_params & params, int64_t hidden_size);

    // Prepare one encoded request for admission into a free sequence.
    bool start(multi_asr_request & req, int slot_id);

    // Run one dynamic prefill/decode scheduler iteration.
    void step();

    bool has_active() const;
    int  active_count() const;
    int  capacity() const { return n_parallel_; }
    llama_context * context() const { return lctx_; }

  private:
    struct slot {
        struct prefill_row {
            const float * data = nullptr;
        };

        multi_asr_request * req = nullptr;
        std::unique_ptr<common_sampler, common_sampler_deleter> sampler;
        std::vector<llama_token> generated;
        std::vector<prefill_row> prefill;
        size_t prefill_offset = 0;
        llama_pos n_past = 0;
        int i_batch = 0;
        int n_predict = 0;
        int64_t t_decode_start = 0;
        int64_t t_prefill_start = 0;
        uint64_t admission_order = 0;
        bool active = false;
        bool occupied = false;
    };

    llama_model * model_ = nullptr;
    llama_context * lctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    common_params params_;

    int64_t hidden_size_ = 0;
    int32_t n_batch_ = 2048;
    int32_t n_parallel_ = 1;
    std::vector<slot> slots_;
    std::vector<llama_token> tok_audio_beg_;
    std::vector<llama_token> tok_audio_end_;
    std::vector<float> token_embd_;
    uint64_t next_admission_order_ = 0;

    void load_token_embeddings();
    const float * token_embedding(llama_token token) const;
    void finish_slot(slot & state, bool ok, const std::string & error = {});
    int occupied_count() const;
};
