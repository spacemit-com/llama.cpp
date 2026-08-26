#include "multi-asr-decoder.h"

#include "ggml.h"
#include "llama-model.h"
#include "log.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace {
struct qwen3asr_prompt_split {
    std::string before;
    std::string after;
};

qwen3asr_prompt_split build_qwen3asr_prompt(const std::string & instruction) {
    return {
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n" + instruction,
    };
}
}

multi_asr_decoder::~multi_asr_decoder() {
    for (auto & state : slots_) {
        state.sampler.reset();
    }
    if (lctx_ != nullptr) {
        llama_free(lctx_);
    }
    lctx_ = nullptr;
    model_ = nullptr;
}

void multi_asr_decoder::init_shared(llama_model * model, const multi_asr_params & params, int64_t hidden_size) {
    if (model == nullptr) {
        throw std::runtime_error("multi_asr_decoder: null model");
    }
    model_ = model;
    hidden_size_ = hidden_size;
    n_batch_ = std::max(1, params.n_batch);
    n_parallel_ = std::max(1, params.n_parallel);

    params_ = {};
    params_.model.path = params.model_path;
    params_.n_ctx = params.n_ctx;
    params_.n_batch = n_batch_;
    params_.n_ubatch = std::min(n_batch_, 512);
    params_.warmup = false;
    params_.n_parallel = n_parallel_;
    params_.n_outputs_max = n_parallel_;
    params_.sampling.temp = 0.0f;
    if (params.decoder_n_threads > 0) {
        params_.cpuparams.n_threads = params.decoder_n_threads;
        params_.cpuparams_batch.n_threads = params.decoder_n_threads;
    }

    auto cparams = common_context_params_to_llama(params_);
    cparams.n_seq_max = n_parallel_;
    cparams.n_outputs_max = n_parallel_;
    cparams.n_batch = n_batch_;
    cparams.n_ubatch = std::min(n_batch_, 512);
    lctx_ = llama_init_from_model(model_, cparams);
    if (lctx_ == nullptr) {
        throw std::runtime_error("multi_asr_decoder: failed to create context");
    }

    vocab_ = llama_model_get_vocab(model_);
    if (llama_model_n_embd(model_) != hidden_size_) {
        throw std::runtime_error("multi_asr_decoder: model/encoder embedding size mismatch");
    }

    tok_audio_beg_ = common_tokenize(vocab_, "<|audio_start|>", false, true);
    tok_audio_end_ = common_tokenize(vocab_, "<|audio_end|>", false, true);
    load_token_embeddings();
    slots_.resize(n_parallel_);
    llama_memory_clear(llama_get_memory(lctx_), true);
}

void multi_asr_decoder::load_token_embeddings() {
    const ggml_tensor * tok_embd = model_->tok_embd;
    if (tok_embd == nullptr || tok_embd->ne[0] != hidden_size_ || tok_embd->ne[1] != llama_vocab_n_tokens(vocab_)) {
        throw std::runtime_error("multi_asr_decoder: invalid token embedding tensor");
    }

    const size_t nbytes = ggml_nbytes(tok_embd);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(tok_embd, raw.data(), 0, nbytes);

    token_embd_.resize((size_t) tok_embd->ne[0] * (size_t) tok_embd->ne[1]);
    const ggml_type_traits * traits = ggml_get_type_traits(tok_embd->type);
    if (tok_embd->type == GGML_TYPE_F32) {
        std::memcpy(token_embd_.data(), raw.data(), token_embd_.size() * sizeof(float));
    } else if (traits->to_float != nullptr) {
        for (int64_t i = 0; i < tok_embd->ne[1]; ++i) {
            traits->to_float(raw.data() + (size_t) i * tok_embd->nb[1],
                             token_embd_.data() + (size_t) i * (size_t) hidden_size_, hidden_size_);
        }
    } else {
        throw std::runtime_error(std::string("multi_asr_decoder: cannot dequantize token embedding type ") +
                                 ggml_type_name(tok_embd->type));
    }
}

const float * multi_asr_decoder::token_embedding(llama_token token) const {
    if (token < 0 || token >= llama_vocab_n_tokens(vocab_)) {
        throw std::runtime_error("multi_asr_decoder: invalid token embedding id");
    }
    return token_embd_.data() + (size_t) token * (size_t) hidden_size_;
}

bool multi_asr_decoder::start(multi_asr_request & req, int slot_id) {
    if (lctx_ == nullptr || slot_id < 0 || slot_id >= (int) slots_.size() || slots_[slot_id].active) {
        req.error = "decoder slot unavailable";
        req.stage = multi_asr_stage::failed;
        return false;
    }
    if (req.embd.empty() || req.n_audio_tokens <= 0) {
        req.error = "decoder: empty audio embedding";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    auto & state = slots_[slot_id];
    llama_memory_seq_rm(llama_get_memory(lctx_), slot_id, -1, -1);
    state = {};
    state.req = &req;
    state.n_predict = req.n_predict > 0 ? req.n_predict : 256;
    state.occupied = true;
    state.t_prefill_start = ggml_time_ms();
    state.admission_order = next_admission_order_++;

    const auto split = build_qwen3asr_prompt(req.prompt);
    auto append_tokens = [&](const std::vector<llama_token> & tokens) {
        for (const llama_token token : tokens) {
            state.prefill.push_back({ token_embedding(token) });
        }
    };
    append_tokens(common_tokenize(vocab_, split.before, true, true));
    append_tokens(tok_audio_beg_);
    for (int32_t i = 0; i < req.n_audio_tokens; ++i) {
        state.prefill.push_back({ req.embd.data() + (size_t) i * (size_t) hidden_size_ });
    }
    append_tokens(tok_audio_end_);
    append_tokens(common_tokenize(vocab_, split.after, false, true));
    auto sampling = params_.sampling;
    sampling.temp = req.temperature;
    state.sampler.reset(common_sampler_init(model_, sampling));
    if (!state.sampler) {
        req.error = "decoder: sampler initialization failed";
        req.stage = multi_asr_stage::failed;
        llama_memory_seq_rm(llama_get_memory(lctx_), slot_id, -1, -1);
        state = {};
        return false;
    }
    req.stage = multi_asr_stage::decoding;
    return true;
}

void multi_asr_decoder::finish_slot(slot & state, bool ok, const std::string & error) {
    if (state.req == nullptr) {
        return;
    }
    auto & req = *state.req;
    if (state.active) {
        req.timings.decode_ms = (double) (ggml_time_ms() - state.t_decode_start);
    }
    req.timings.n_out_tokens = (int32_t) state.generated.size();
    if (ok) {
        req.text = common_detokenize(lctx_, state.generated, false);
        req.stage = multi_asr_stage::done;
    } else {
        req.error = error.empty() ? "decoder: generation failed" : error;
        req.stage = multi_asr_stage::failed;
    }
    const int slot_id = (int) (&state - slots_.data());
    llama_memory_seq_rm(llama_get_memory(lctx_), slot_id, -1, -1);
    state = {};
}

void multi_asr_decoder::step() {
    if (!has_active()) {
        return;
    }

    // Sample one token for every sequence that already has a decoder state.
    // New requests remain in prefill until their final prompt row produces the
    // first logits; they join generation on the following scheduler tick.
    for (auto & state : slots_) {
        if (!state.active || state.req == nullptr) {
            continue;
        }
        const llama_token tok = common_sampler_sample(state.sampler.get(), lctx_, state.i_batch);
        common_sampler_accept(state.sampler.get(), tok, true);
        if (llama_vocab_is_eog(vocab_, tok)) {
            finish_slot(state, true);
            continue;
        }
        state.generated.push_back(tok);
        if ((int) state.generated.size() >= state.n_predict) {
            finish_slot(state, true);
        }
    }

    const int n_active = active_count();
    int n_pending = 0;
    for (const auto & state : slots_) {
        if (state.occupied && !state.active && state.prefill_offset < state.prefill.size()) {
            ++n_pending;
        }
    }
    if (n_active == 0 && n_pending == 0) {
        return;
    }

    // All rows are F32 embeddings. Generated token ids are looked up in the
    // model's token embedding table before admission, so the transformer sees
    // one homogeneous input representation even when prefill and decode rows
    // coexist in this batch.
    llama_batch batch = llama_batch_init(n_batch_, (int32_t) hidden_size_, n_parallel_);

    int n_batch = 0;
    std::vector<int> prefill_added(slots_.size(), 0);
    std::vector<bool> decode_added(slots_.size(), false);

    auto append_active = [&](slot & state) {
        const int i = n_batch++;
        const int slot_id = (int) (&state - slots_.data());
        std::memcpy(batch.embd + (size_t) i * (size_t) hidden_size_,
                    token_embedding(state.generated.back()), sizeof(float) * (size_t) hidden_size_);
        batch.pos[i] = state.n_past;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = slot_id;
        batch.logits[i] = true;
        state.i_batch = i;
        decode_added[(size_t) slot_id] = true;
    };

    // Existing decode work gets first claim on the batch. New prefills use the
    // remaining budget in FIFO slot order, so a stream of arrivals cannot
    // starve older generation requests.
    for (auto & state : slots_) {
        if (!state.active || state.req == nullptr) {
            continue;
        }
        append_active(state);
    }

    int prefill_budget = n_batch_ - n_batch;
    // Admit one pending prefill request per scheduler tick. Existing decode
    // rows have already claimed their budget; the oldest prompt can use all
    // remaining rows and is split across ticks only when it exceeds n_batch.
    slot * pending = nullptr;
    for (auto & state : slots_) {
        if (state.occupied && !state.active && state.req != nullptr &&
            state.prefill_offset < state.prefill.size() &&
            (pending == nullptr || state.admission_order < pending->admission_order)) {
            pending = &state;
        }
    }
    if (pending != nullptr && prefill_budget > 0) {
        auto & state = *pending;
        const size_t remaining = state.prefill.size() - state.prefill_offset;
        const int count = std::min<int>(prefill_budget, (int) remaining);
        const int slot_id = (int) (&state - slots_.data());
        for (int j = 0; j < count; ++j) {
            const size_t row = state.prefill_offset + (size_t) j;
            const auto & item = state.prefill[row];
            const int i = n_batch++;
            std::memcpy(batch.embd + (size_t) i * (size_t) hidden_size_, item.data,
                        sizeof(float) * (size_t) hidden_size_);
            batch.pos[i] = state.n_past + j;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = slot_id;
            const bool final_row = row + 1 == state.prefill.size();
            batch.logits[i] = final_row;
            if (final_row) {
                state.i_batch = i;
            }
        }
        prefill_added[(size_t) slot_id] = count;
    }
    batch.n_tokens = n_batch;
    if (n_batch == 0) {
        llama_batch_free(batch);
        return;
    }

    if (active_count() > 0 && n_pending > 0) {
        LOG_DBG("[multi-asr] mixed batch: decode_rows=%d prefill_rows=%d total_rows=%d\n",
                (int) std::count(decode_added.begin(), decode_added.end(), true),
                std::accumulate(prefill_added.begin(), prefill_added.end(), 0), n_batch);
    }

    if (llama_decode(lctx_, batch) != 0) {
        for (auto & state : slots_) {
            if (state.occupied) {
                finish_slot(state, false, "decoder: llama_decode failed during continuous batching");
            }
        }
    } else {
        for (auto & state : slots_) {
            if (!state.occupied) {
                continue;
            }
            const int slot_id = (int) (&state - slots_.data());
            if (decode_added[(size_t) slot_id]) {
                state.n_past += 1;
            }
            state.n_past += prefill_added[(size_t) slot_id];
            if (prefill_added[(size_t) slot_id] > 0) {
                state.prefill_offset += prefill_added[(size_t) slot_id];
                if (state.prefill_offset == state.prefill.size()) {
                    state.active = true;
                    state.t_decode_start = ggml_time_ms();
                    state.req->timings.prefill_ms = (double) (state.t_decode_start - state.t_prefill_start);
                }
            }
        }

    }
    llama_batch_free(batch);
}

bool multi_asr_decoder::has_active() const {
    return occupied_count() > 0;
}

int multi_asr_decoder::active_count() const {
    int count = 0;
    for (const auto & state : slots_) {
        count += state.active ? 1 : 0;
    }
    return count;
}

int multi_asr_decoder::occupied_count() const {
    int count = 0;
    for (const auto & state : slots_) {
        count += state.occupied ? 1 : 0;
    }
    return count;
}
