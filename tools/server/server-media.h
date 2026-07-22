#pragma once

#include "server-common.h"

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <memory>
#include <string>
#include <vector>

class media_worker;

#if defined(LLAMA_SERVER_SMT_MTMD)
struct server_media_tts_result {
    std::vector<uint8_t> wav;
    std::string backend;
    uint32_t segments = 0;
    uint32_t sample_rate = 0;
    uint64_t samples = 0;
    double wall_seconds = 0.0;
};
#endif

enum class server_media_backend {
    none,
    mtmd,
    smt,
};

struct server_media_context {
    server_media_context(const server_media_context &) = delete;
    server_media_context & operator=(const server_media_context &) = delete;
    ~server_media_context();
    struct impl;

    static std::unique_ptr<server_media_context> init(
            llama_context * ctx_llama,
            llama_model * model,
            const llama_vocab * vocab,
            const common_params & params,
            const mtmd_context_params & mtmd_params,
            std::unique_ptr<media_worker> worker = nullptr);

#if defined(LLAMA_SERVER_SMT_MTMD)
    static bool tts_config_matches(const common_params & params);
    static std::unique_ptr<server_media_context> init_tts(const common_params & params);
#endif

    server_media_backend backend() const;
    const char * backend_name() const;

    mtmd_context * mtmd() const;

    bool supports_prompt_embeddings() const;
    bool supports_vision() const;
    bool supports_audio() const;
    bool supports_video() const;

#if defined(LLAMA_SERVER_SMT_MTMD)
    const char * tts_backend_name() const;

    server_media_tts_result synthesize(const std::string & text) const;
#endif

    server_tokens process_prompt(
            const std::string & prompt,
            const std::vector<raw_buffer> & files,
            bool is_placeholder = false);

    int32_t encode_mtmd_batch(mtmd_batch * batch) const;

    int32_t decode_embd_chunk(
            llama_context * lctx,
            const server_media_embd_chunk & chunk,
            llama_pos & n_past,
            int32_t seq_id,
            int32_t n_batch,
            bool logits_last) const;

private:
    server_media_context();

    std::unique_ptr<impl> pimpl;
};
