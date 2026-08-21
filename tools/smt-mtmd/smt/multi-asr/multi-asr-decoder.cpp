#include "multi-asr-decoder.h"

#include "ggml.h"
#include "sampling.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

// Build the Qwen3-ASR prompt around the audio, mirroring
// format_qwen3asr_audio_prompt() in tools/mtmd/mtmd-cli-smt.cpp:
//   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n <AUDIO> <|im_end|>\n<|im_start|>assistant\n<user text>
// The audio embedding (+ <|audio_start|>/<|audio_end|>) is spliced where <AUDIO> is.
// We return the text BEFORE the audio and the text AFTER the audio separately.
struct qwen3asr_prompt_split {
    std::string before;  // up to and including "<|im_start|>user\n"
    std::string after;   // "<|im_end|>\n<|im_start|>assistant\n" + user text
};

static qwen3asr_prompt_split build_qwen3asr_prompt(const std::string & user_text) {
    qwen3asr_prompt_split s;
    s.before = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
    s.after  = "<|im_end|>\n<|im_start|>assistant\n" + user_text;
    return s;
}

multi_asr_decoder::~multi_asr_decoder() {
    if (owns_context_ && lctx_ && !init_) {
        llama_free(lctx_);
    }
    lctx_  = nullptr;
    model_ = nullptr;
}

void multi_asr_decoder::init(const multi_asr_params & params, int64_t hidden_size) {
    hidden_size_ = hidden_size;
    n_batch_     = params.n_batch;

    // Build common_params for model load + threading/affinity + sampling.
    params_.model.path = params.model_path;
    params_.n_ctx      = params.n_ctx;  // 0 -> from model
    params_.n_batch    = params.n_batch;
    params_.warmup     = params.warmup;

    // Decoder thread count. Core binding for the ggml IME backend is set via the
    // SPACEMIT_PERFER_CORE_ID environment variable (see run_server.sh), not here.
    if (params.decoder_n_threads > 0) {
        params_.cpuparams.n_threads = params.decoder_n_threads;
    }

    init_ = common_init_from_params(params_);
    if (!init_) {
        throw std::runtime_error("multi_asr_decoder: common_init_from_params returned null");
    }
    model_ = init_->model();
    lctx_  = init_->context();
    if (!model_ || !lctx_) {
        throw std::runtime_error("multi_asr_decoder: failed to load model / create context");
    }
    vocab_ = llama_model_get_vocab(model_);

    // Validate embedding dim.
    const int model_n_embd = llama_model_n_embd(model_);
    if ((int64_t) model_n_embd != hidden_size_) {
        throw std::runtime_error("multi_asr_decoder: model n_embd (" + std::to_string(model_n_embd) +
                                 ") != encoder hidden_size (" + std::to_string(hidden_size_) + ")");
    }

    // Qwen3-ASR audio boundary special tokens.
    tok_audio_beg_ = common_tokenize(vocab_, "<|audio_start|>", /*add_special*/ false, /*parse_special*/ true);
    tok_audio_end_ = common_tokenize(vocab_, "<|audio_end|>", /*add_special*/ false, /*parse_special*/ true);
    owns_context_ = true;
}

void multi_asr_decoder::init_shared(llama_model * model, const multi_asr_params & params, int64_t hidden_size) {
    if (model == nullptr) {
        throw std::runtime_error("multi_asr_decoder: null shared model");
    }
    hidden_size_ = hidden_size;
    n_batch_ = params.n_batch;
    params_.model.path = params.model_path;
    params_.n_ctx = params.n_ctx;
    params_.n_batch = params.n_batch;
    params_.n_ubatch = std::min(params.n_batch, 512);
    params_.warmup = false;
    if (params.decoder_n_threads > 0) {
        params_.cpuparams.n_threads = params.decoder_n_threads;
        params_.cpuparams_batch.n_threads = params.decoder_n_threads;
    }
    auto cparams = common_context_params_to_llama(params_);
    cparams.n_seq_max = 1;
    cparams.n_outputs_max = 1;
    lctx_ = llama_init_from_model(model, cparams);
    if (!lctx_) {
        throw std::runtime_error("multi_asr_decoder: failed to create private context from shared model");
    }
    model_ = model;
    vocab_ = llama_model_get_vocab(model_);
    if (llama_model_n_embd(model_) != hidden_size_) {
        llama_free(lctx_);
        lctx_ = nullptr;
        throw std::runtime_error("multi_asr_decoder: shared model embedding size mismatch");
    }
    tok_audio_beg_ = common_tokenize(vocab_, "<|audio_start|>", false, true);
    tok_audio_end_ = common_tokenize(vocab_, "<|audio_end|>", false, true);
    owns_context_ = true;
}

int multi_asr_decoder::decode_tokens(const std::vector<llama_token> & tokens, llama_pos & n_past, bool logits_last) {
    if (tokens.empty()) {
        return 0;
    }
    llama_batch batch = llama_batch_init(n_batch_, 0, 1);
    size_t      i     = 0;
    while (i < tokens.size()) {
        batch.n_tokens = 0;
        for (; i < tokens.size() && batch.n_tokens < n_batch_; ++i) {
            const int32_t j    = batch.n_tokens;
            batch.token[j]     = tokens[i];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = false;
            batch.n_tokens++;
        }
        if (logits_last && i == tokens.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }
        if (llama_decode(lctx_, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        n_past += batch.n_tokens;
    }
    llama_batch_free(batch);
    return 0;
}

int multi_asr_decoder::decode_embd(const float * embd, int n_tokens, llama_pos & n_past, bool logits_last) {
    const int                   n_embd = (int) hidden_size_;
    std::vector<llama_pos>      pos((size_t) n_tokens);
    std::vector<int32_t>        n_seq_id((size_t) n_tokens);
    std::vector<llama_seq_id>   seq_id_store((size_t) n_tokens);
    std::vector<llama_seq_id *> seq_ids((size_t) n_tokens + 1);
    std::vector<int8_t>         logits((size_t) n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        seq_id_store[i] = 0;
        seq_ids[i]      = &seq_id_store[i];
    }
    seq_ids[n_tokens] = nullptr;

    int processed = 0;
    while (processed < n_tokens) {
        const int  batch_size    = std::min(n_batch_, n_tokens - processed);
        const bool is_last_batch = (processed + batch_size >= n_tokens);
        for (int i = 0; i < batch_size; ++i) {
            pos[processed + i]      = n_past + processed + i;
            n_seq_id[processed + i] = 1;
            logits[processed + i]   = (logits_last && is_last_batch && i == batch_size - 1) ? 1 : 0;
        }
        llama_batch batch = {
            /*n_tokens =*/batch_size,
            /*token    =*/nullptr,
            /*embd     =*/const_cast<float *>(embd) + (size_t) processed * n_embd,
            /*pos      =*/pos.data() + processed,
            /*n_seq_id =*/n_seq_id.data() + processed,
            /*seq_id   =*/seq_ids.data() + processed,
            /*logits   =*/logits.data() + processed,
        };
        if (llama_decode(lctx_, batch) != 0) {
            return 1;
        }
        processed += batch_size;
    }
    n_past += n_tokens;
    return 0;
}

bool multi_asr_decoder::decode(multi_asr_request & req) {
    if (!lctx_) {
        req.error = "decoder not initialized";
        req.stage = multi_asr_stage::failed;
        return false;
    }
    if (req.embd.empty() || req.n_audio_tokens <= 0) {
        req.error = "decoder: empty audio embedding";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    // Fresh KV for this request (single slot, one request at a time).
    llama_memory_clear(llama_get_memory(lctx_), true);

    llama_pos                   n_past = 0;
    const qwen3asr_prompt_split split  = build_qwen3asr_prompt(req.prompt);

    const int64_t t_prefill0 = ggml_time_ms();

    // 1) text before audio (with BOS/special parsing)
    const std::vector<llama_token> toks_before =
        common_tokenize(vocab_, split.before, /*add_special*/ true, /*parse_special*/ true);
    if (decode_tokens(toks_before, n_past, /*logits_last*/ false) != 0) {
        req.error = "decoder: failed to decode prompt prefix";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    // 2) <|audio_start|>
    if (decode_tokens(tok_audio_beg_, n_past, /*logits_last*/ false) != 0) {
        req.error = "decoder: failed to decode audio-begin";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    // 3) audio embedding
    if (decode_embd(req.embd.data(), req.n_audio_tokens, n_past, /*logits_last*/ false) != 0) {
        req.error = "decoder: failed to decode audio embedding";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    // 4) <|audio_end|>
    if (decode_tokens(tok_audio_end_, n_past, /*logits_last*/ false) != 0) {
        req.error = "decoder: failed to decode audio-end";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    // 5) assistant prefix + user text (logits on last -> ready to sample)
    const std::vector<llama_token> toks_after =
        common_tokenize(vocab_, split.after, /*add_special*/ false, /*parse_special*/ true);
    if (decode_tokens(toks_after, n_past, /*logits_last*/ true) != 0) {
        req.error = "decoder: failed to decode assistant prefix";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    req.timings.prefill_ms = (double) (ggml_time_ms() - t_prefill0);

    // 6) generation loop
    const int64_t    t_decode0 = ggml_time_ms();
    common_sampler * smpl      = common_sampler_init(model_, params_.sampling);
    if (!smpl) {
        req.error = "decoder: failed to init sampler";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    std::vector<llama_token> generated;
    const int                n_predict = req.n_predict > 0 ? req.n_predict : 256;
    llama_batch              gen_batch = llama_batch_init(1, 0, 1);
    bool                     ok        = true;
    for (int i = 0; i < n_predict; ++i) {
        const llama_token tok = common_sampler_sample(smpl, lctx_, -1);
        common_sampler_accept(smpl, tok, true);
        if (llama_vocab_is_eog(vocab_, tok)) {
            break;
        }
        generated.push_back(tok);

        gen_batch.n_tokens     = 1;
        gen_batch.token[0]     = tok;
        gen_batch.pos[0]       = n_past++;
        gen_batch.n_seq_id[0]  = 1;
        gen_batch.seq_id[0][0] = 0;
        gen_batch.logits[0]    = true;
        if (llama_decode(lctx_, gen_batch) != 0) {
            req.error = "decoder: llama_decode failed during generation";
            ok        = false;
            break;
        }
    }
    llama_batch_free(gen_batch);

    req.timings.decode_ms    = (double) (ggml_time_ms() - t_decode0);
    req.timings.n_out_tokens = (int32_t) generated.size();

    if (ok) {
        req.text  = common_detokenize(lctx_, generated, /*special*/ false);
        req.stage = multi_asr_stage::done;
    } else {
        req.stage = multi_asr_stage::failed;
    }

    common_sampler_free(smpl);
    return ok;
}
