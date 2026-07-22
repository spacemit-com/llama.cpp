#include "server-media.h"

#include "log.h"
#include "media-worker.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#if defined(LLAMA_SERVER_SMT_MTMD)
#    include "smt-audio-wrapper.h"
#    include "smt-tts-wrapper.h"
#    include "smt-vision-preprocess.h"
#    include "smt-vision-wrapper.h"

#    include <nlohmann/json.hpp>

#    include <cctype>
#    include <cstdio>
#    include <cstdlib>

#    if defined(_WIN32)
#        include <windows.h>
#    else
#        include <unistd.h>
#    endif
#endif

static std::string server_media_fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static void server_media_replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::vector<std::string> server_media_split_keep_marker(const std::string & input, const std::string & marker) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (true) {
        const size_t mpos = input.find(marker, pos);
        if (mpos == std::string::npos) {
            out.emplace_back(input.substr(pos));
            break;
        }
        if (mpos > pos) {
            out.emplace_back(input.substr(pos, mpos - pos));
        }
        out.emplace_back(marker);
        pos = mpos + marker.size();
    }
    if (out.empty()) {
        out.emplace_back("");
    }
    return out;
}

struct server_media_context::impl {
    server_media_backend mode = server_media_backend::none;
    std::unique_ptr<media_worker> worker;
    mtmd::context_ptr mtmd_ctx;
    const llama_vocab * vocab = nullptr;

#if defined(LLAMA_SERVER_SMT_MTMD)
    std::unique_ptr<smt_vision_context> smt_vision;
    std::unique_ptr<smt_audio_context> smt_audio;
    std::unique_ptr<smt_tts_context> smt_tts;

    int32_t hidden_size = 0;
    bool use_mrope_pos = false;
    std::string architecture;
    std::vector<llama_token> tok_img_beg;
    std::vector<llama_token> tok_img_end;
    std::vector<llama_token> tok_audio_beg;
    std::vector<llama_token> tok_audio_end;
#endif
};

server_media_context::server_media_context() : pimpl(new impl()) {
}

server_media_context::~server_media_context() = default;

server_media_backend server_media_context::backend() const {
    return pimpl->mode;
}

const char * server_media_context::backend_name() const {
    switch (pimpl->mode) {
        case server_media_backend::mtmd: return "mtmd";
        case server_media_backend::smt:  return "smt";
        case server_media_backend::none: return "none";
    }
    return "none";
}

mtmd_context * server_media_context::mtmd() const {
    return pimpl->mtmd_ctx.get();
}

bool server_media_context::supports_prompt_embeddings() const {
    if (pimpl->mode == server_media_backend::mtmd) {
        return true;
    }
#if defined(LLAMA_SERVER_SMT_MTMD)
    if (pimpl->mode == server_media_backend::smt) {
        return pimpl->smt_vision != nullptr || pimpl->smt_audio != nullptr;
    }
#endif
    return false;
}

bool server_media_context::supports_vision() const {
    if (pimpl->mode == server_media_backend::mtmd) {
        return mtmd_support_vision(pimpl->mtmd_ctx.get());
    }
#if defined(LLAMA_SERVER_SMT_MTMD)
    if (pimpl->mode == server_media_backend::smt) {
        return pimpl->smt_vision != nullptr;
    }
#endif
    return false;
}

bool server_media_context::supports_audio() const {
    if (pimpl->mode == server_media_backend::mtmd) {
        return mtmd_support_audio(pimpl->mtmd_ctx.get());
    }
#if defined(LLAMA_SERVER_SMT_MTMD)
    if (pimpl->mode == server_media_backend::smt) {
        return pimpl->smt_audio != nullptr;
    }
#endif
    return false;
}

bool server_media_context::supports_video() const {
    return pimpl->mode == server_media_backend::mtmd && mtmd_helper_support_video(pimpl->mtmd_ctx.get());
}

#if defined(LLAMA_SERVER_SMT_MTMD)
const char * server_media_context::tts_backend_name() const {
    if (pimpl->smt_tts) {
        return pimpl->smt_tts->backend_name();
    }
    return "none";
}

server_media_tts_result server_media_context::synthesize(const std::string & text) const {
    if (!pimpl->smt_tts || !pimpl->worker) {
        throw std::runtime_error("media backend does not support TTS");
    }
    smt_tts_result synthesis = pimpl->worker->invoke("smt_tts_synthesize", [&]() {
        return pimpl->smt_tts->synthesize(text);
    });
    server_media_tts_result result;
    result.wav = std::move(synthesis.wav);
    result.backend = std::move(synthesis.backend);
    result.segments = synthesis.segments;
    result.sample_rate = synthesis.sample_rate;
    result.samples = synthesis.samples;
    result.wall_seconds = synthesis.wall_seconds;
    return result;
}

static std::string server_media_to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static bool server_media_contains_icase(const std::string & text, const std::string & pattern) {
    return server_media_to_lower_ascii(text).find(server_media_to_lower_ascii(pattern)) != std::string::npos;
}

static bool server_media_arch_requires_mrope(const std::string & arch_name) {
    return server_media_contains_icase(arch_name, "qwen2vl") ||
           server_media_contains_icase(arch_name, "qwen2_5_vl") ||
           server_media_contains_icase(arch_name, "qwen3vl") ||
           server_media_contains_icase(arch_name, "glm4v") ||
           server_media_contains_icase(arch_name, "paddleocr");
}

static bool server_media_arch_is_qwen3asr(const std::string & arch_name) {
    return server_media_contains_icase(arch_name, "qwen3asr");
}

static bool server_media_arch_is_gemma4_audio(const std::string & arch_name) {
    return arch_name == "Gemma4Audio";
}

struct server_media_config_modalities {
    bool has_explicit_modality = false;
    bool wants_vision = true;
    bool wants_audio = false;
};

static bool server_media_json_has_architecture(const nlohmann::json & config, const std::string & target) {
    if (!config.contains("architectures")) {
        return false;
    }

    const auto & arch = config.at("architectures");
    if (arch.is_array()) {
        for (const auto & value : arch) {
            if (value.is_string() && value.get<std::string>() == target) {
                return true;
            }
        }
    } else if (arch.is_string()) {
        return arch.get<std::string>() == target;
    }
    return false;
}

static server_media_config_modalities server_media_resolve_config_modalities(const std::string & config_dir) {
    server_media_config_modalities result;
    if (config_dir.empty()) {
        return result;
    }

    const std::string config_path = config_dir + "/config.json";
    std::ifstream file(config_path);
    if (!file.is_open()) {
        return result;
    }

    try {
        const nlohmann::json config = nlohmann::json::parse(file);
        const bool has_vision_model = config.contains("vision_model");
        const bool has_audio_model = config.contains("audio_model");
        const bool audio_only_arch =
                server_media_json_has_architecture(config, "Gemma4Audio") ||
                server_media_json_has_architecture(config, "Qwen3ASRForConditionalGeneration");

        result.has_explicit_modality = has_vision_model || has_audio_model;
        if (result.has_explicit_modality) {
            result.wants_vision = has_vision_model;
            result.wants_audio = has_audio_model;
        } else if (audio_only_arch) {
            result.wants_vision = false;
            result.wants_audio = true;
        }
    } catch (...) {
        return result;
    }

    return result;
}

static std::pair<int32_t, int32_t> server_media_infer_image_grid_xy(int32_t n_tokens) {
    if (n_tokens <= 0) {
        return { 0, 0 };
    }

    int32_t best_y = 1;
    int32_t best_x = n_tokens;
    const int32_t root = (int32_t) std::sqrt((double) n_tokens);
    for (int32_t y = root; y >= 1; --y) {
        if (n_tokens % y == 0) {
            best_y = y;
            best_x = n_tokens / y;
            break;
        }
    }
    return { best_x, best_y };
}

static std::vector<llama_token> server_media_tokenize_exact_special(
        const llama_vocab * vocab,
        const std::string & token_text) {
    auto toks = common_tokenize(vocab, token_text, /* add_special */ false, /* parse_special */ true);
    if (toks.size() != 1) {
        return {};
    }
    if (common_token_to_piece(vocab, toks[0]) != token_text) {
        return {};
    }
    return toks;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> server_media_detect_image_boundary_tokens_native(
        const llama_vocab * vocab,
        const std::string & arch_name) {
    if (server_media_contains_icase(arch_name, "qwen2vl") ||
        server_media_contains_icase(arch_name, "qwen2_5_vl") ||
        server_media_contains_icase(arch_name, "qwen3vl") ||
        server_media_contains_icase(arch_name, "youtuvl")) {
        return { server_media_tokenize_exact_special(vocab, "<|vision_start|>"),
                 server_media_tokenize_exact_special(vocab, "<|vision_end|>") };
    }
    if (server_media_contains_icase(arch_name, "llama4") || server_media_contains_icase(arch_name, "lfm2")) {
        return { server_media_tokenize_exact_special(vocab, "<|image_start|>"),
                 server_media_tokenize_exact_special(vocab, "<|image_end|>") };
    }
    if (server_media_contains_icase(arch_name, "gemma3")) {
        return { server_media_tokenize_exact_special(vocab, "<start_of_image>"),
                 server_media_tokenize_exact_special(vocab, "<end_of_image>") };
    }
    if (server_media_contains_icase(arch_name, "internvl")) {
        return { server_media_tokenize_exact_special(vocab, "<img>"),
                 server_media_tokenize_exact_special(vocab, "</img>") };
    }
    if (server_media_contains_icase(arch_name, "glm4v")) {
        return { server_media_tokenize_exact_special(vocab, "<|begin_of_image|>"),
                 server_media_tokenize_exact_special(vocab, "<|end_of_image|>") };
    }
    if (server_media_contains_icase(arch_name, "paddleocr")) {
        return { server_media_tokenize_exact_special(vocab, "<|IMAGE_START|>"),
                 server_media_tokenize_exact_special(vocab, "<|IMAGE_END|>") };
    }
    if (server_media_contains_icase(arch_name, "lightonocr")) {
        return { server_media_tokenize_exact_special(vocab, "<|im_start|>"),
                 server_media_tokenize_exact_special(vocab, "<|im_end|>") };
    }
    return {};
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> server_media_resolve_image_boundary_tokens(
        const llama_vocab * vocab,
        const std::string & arch_name) {
    const char * env = std::getenv("MTMD_SMT_IMAGE_BOUNDARY");
    if (env != nullptr && (std::strcmp(env, "none") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0)) {
        return {};
    }
    return server_media_detect_image_boundary_tokens_native(vocab, arch_name);
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> server_media_resolve_audio_boundary_tokens(
        const llama_vocab * vocab,
        const std::string & arch_name) {
    if (server_media_arch_is_qwen3asr(arch_name)) {
        return { server_media_tokenize_exact_special(vocab, "<|audio_start|>"),
                 server_media_tokenize_exact_special(vocab, "<|audio_end|>") };
    }
    if (server_media_arch_is_gemma4_audio(arch_name)) {
        return { server_media_tokenize_exact_special(vocab, "<|audio>"),
                 server_media_tokenize_exact_special(vocab, "<audio|>") };
    }
    return {};
}

static bool server_media_looks_like_audio_file(const std::vector<uint8_t> & data) {
    if (data.size() < 12) {
        return false;
    }

    const char * buf = reinterpret_cast<const char *>(data.data());
    const bool is_wav = std::memcmp(buf, "RIFF", 4) == 0 && std::memcmp(buf + 8, "WAVE", 4) == 0;
    const bool is_mp3 =
        data.size() >= 3 &&
        (std::memcmp(buf, "ID3", 3) == 0 ||
         (static_cast<unsigned char>(buf[0]) == 0xFF && (static_cast<unsigned char>(buf[1]) & 0xE0) == 0xE0));
    const bool is_flac = std::memcmp(buf, "fLaC", 4) == 0;
    return is_wav || is_mp3 || is_flac;
}

static std::string server_media_write_temp_bin_file(const std::vector<uint8_t> & data) {
#if defined(_WIN32)
    char temp_path[MAX_PATH] = { 0 };
    char temp_file[MAX_PATH] = { 0 };

    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("failed to get temp path");
    }
    if (GetTempFileNameA(temp_path, "lse", 0, temp_file) == 0) {
        throw std::runtime_error("failed to create temp file");
    }

    FILE * fp = std::fopen(temp_file, "wb");
    if (!fp) {
        throw std::runtime_error("failed to open temp file");
    }
    const size_t n = std::fwrite(data.data(), 1, data.size(), fp);
    std::fclose(fp);
    if (n != data.size()) {
        std::remove(temp_file);
        throw std::runtime_error("failed to write temp file");
    }
    return std::string(temp_file);
#else
    char tmpl[] = "/tmp/llama-server-smt-XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) {
        throw std::runtime_error("failed to create temp file");
    }

    size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = write(fd, data.data() + written, data.size() - written);
        if (n <= 0) {
            close(fd);
            std::remove(tmpl);
            throw std::runtime_error("failed to write temp file");
        }
        written += (size_t) n;
    }
    close(fd);
    return std::string(tmpl);
#endif
}

static server_media_embd_chunk server_media_encode_smt_image(
        server_media_context::impl & impl,
        const std::vector<uint8_t> & data) {
    if (impl.smt_vision == nullptr) {
        throw std::runtime_error("SMT vision backend is not initialized");
    }

    std::vector<uint8_t> smt_input = data;
    auto preproc = smt_vision_preprocess_if_image(
            data,
            impl.architecture,
            impl.smt_vision->input_width(),
            impl.smt_vision->input_height(),
            &impl.smt_vision->preprocess_config());
    if (preproc.was_image) {
        smt_input = std::move(preproc.tensor_bytes);
    }

    server_media_embd_chunk out;
    out.type = server_media_chunk_type::image;
    out.hidden_size = impl.hidden_size;
    out.use_mrope_pos = impl.use_mrope_pos;
    out.tokens_begin = impl.tok_img_beg;
    out.tokens_end = impl.tok_img_end;

    const int64_t t0 = ggml_time_us();
    out.embd = impl.smt_vision->encode_image_mem(smt_input.data(), smt_input.size());
    out.t_encode_ms = (ggml_time_us() - t0) / 1e3;

    if (impl.hidden_size <= 0 || out.embd.empty() || out.embd.size() % (size_t) impl.hidden_size != 0) {
        throw std::runtime_error("Invalid SMT image embedding shape");
    }

    const int32_t n_image_tokens = (int32_t) (out.embd.size() / (size_t) impl.hidden_size);
    auto grid = server_media_infer_image_grid_xy(n_image_tokens);

    const int32_t n_pos_img = impl.use_mrope_pos ? std::max(grid.first, grid.second) : n_image_tokens;
    out.n_tokens = (int32_t) out.tokens_begin.size() + n_image_tokens + (int32_t) out.tokens_end.size();
    out.n_pos = (int32_t) out.tokens_begin.size() + n_pos_img + (int32_t) out.tokens_end.size();
    out.grid_nx = grid.first;
    out.grid_ny = grid.second;
    out.id = server_media_fnv_hash(data.data(), data.size());

    return out;
}

static server_media_embd_chunk server_media_encode_smt_audio(
        server_media_context::impl & impl,
        const std::vector<uint8_t> & data) {
    if (impl.smt_audio == nullptr) {
        throw std::runtime_error("SMT audio backend is not initialized");
    }

    const std::string tmp_file = server_media_write_temp_bin_file(data);

    server_media_embd_chunk out;
    out.type = server_media_chunk_type::audio;
    out.hidden_size = impl.hidden_size;
    out.tokens_begin = impl.tok_audio_beg;
    out.tokens_end = impl.tok_audio_end;

    try {
        const int64_t t0 = ggml_time_us();
        out.embd = impl.smt_audio->encode_audio(tmp_file);
        out.t_encode_ms = (ggml_time_us() - t0) / 1e3;
        std::remove(tmp_file.c_str());
    } catch (...) {
        std::remove(tmp_file.c_str());
        throw;
    }

    if (impl.hidden_size <= 0 || out.embd.empty() || out.embd.size() % (size_t) impl.hidden_size != 0) {
        throw std::runtime_error("Invalid SMT audio embedding shape");
    }

    const int32_t n_audio_tokens = (int32_t) (out.embd.size() / (size_t) impl.hidden_size);
    out.n_tokens = (int32_t) out.tokens_begin.size() + n_audio_tokens + (int32_t) out.tokens_end.size();
    out.n_pos = out.n_tokens;
    out.grid_nx = n_audio_tokens;
    out.grid_ny = 1;
    out.id = std::string("audio:") + server_media_fnv_hash(data.data(), data.size());

    return out;
}

static server_media_embd_chunk server_media_encode_smt_media(
        server_media_context::impl & impl,
        const std::vector<uint8_t> & data) {
    if (server_media_looks_like_audio_file(data)) {
        return server_media_encode_smt_audio(impl, data);
    }
    return server_media_encode_smt_image(impl, data);
}

static int server_media_decode_tokens(
        llama_context * lctx,
        const std::vector<llama_token> & tokens,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last) {
    if (tokens.empty()) {
        return 0;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    size_t i = 0;
    while (i < tokens.size()) {
        batch.n_tokens = 0;
        for (; i < tokens.size() && batch.n_tokens < n_batch; ++i) {
            const int32_t j = batch.n_tokens;
            batch.token[j] = tokens[i];
            batch.pos[j] = n_past + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = false;
            batch.n_tokens++;
        }

        if (logits_last && i == tokens.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        n_past += batch.n_tokens;
    }

    llama_batch_free(batch);
    return 0;
}

static int server_media_decode_embd(
        llama_context * lctx,
        const float * embd,
        int32_t n_tokens,
        int32_t n_embd,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last,
        bool use_mrope_pos,
        int32_t nx,
        int32_t ny) {
    const int n_pos_per_embd = use_mrope_pos ? 4 : 1;

    std::vector<llama_pos> pos((size_t) n_tokens * n_pos_per_embd);
    std::vector<int32_t> n_seq_id(n_tokens);
    std::vector<llama_seq_id> seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t> logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = seq_id;
        seq_ids[i] = &seq_id_0[i];
    }

    if (use_mrope_pos) {
        if (nx > 0 && ny > 0 && nx * ny == n_tokens) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int i = y * nx + x;
                    pos[(size_t) i] = n_past;
                    pos[(size_t) i + n_tokens] = n_past + y;
                    pos[(size_t) i + 2 * n_tokens] = n_past + x;
                    pos[(size_t) i + 3 * n_tokens] = 0;
                }
            }
        } else {
            for (int i = 0; i < n_tokens; ++i) {
                pos[(size_t) i] = n_past + i;
                pos[(size_t) i + n_tokens] = n_past + i;
                pos[(size_t) i + 2 * n_tokens] = n_past + i;
                pos[(size_t) i + 3 * n_tokens] = 0;
            }
        }
    } else {
        for (int i = 0; i < n_tokens; ++i) {
            pos[(size_t) i] = n_past + i;
        }
    }

    int processed = 0;
    while (processed < n_tokens) {
        const int batch_size = std::min(n_batch, n_tokens - processed);
        const bool is_last_batch = processed + batch_size >= n_tokens;

        for (int i = 0; i < batch_size; ++i) {
            if (!use_mrope_pos) {
                pos[processed + i] = n_past + processed + i;
            }
            n_seq_id[processed + i] = 1;
            logits[processed + i] = (logits_last && is_last_batch && i == batch_size - 1);
        }

        llama_pos * pos_ptr = nullptr;
        std::vector<llama_pos> pos_view;
        if (use_mrope_pos) {
            pos_view.reserve((size_t) batch_size * n_pos_per_embd);
            for (int d = 0; d < n_pos_per_embd; ++d) {
                const size_t src_idx = (size_t) d * n_tokens + processed;
                pos_view.insert(pos_view.end(), pos.data() + src_idx, pos.data() + src_idx + batch_size);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + processed;
        }

        llama_batch batch = {
            /* n_tokens  */ batch_size,
            /* token     */ nullptr,
            /* embd      */ const_cast<float *>(embd + (size_t) processed * n_embd),
            /* pos       */ pos_ptr,
            /* n_seq_id  */ n_seq_id.data() + processed,
            /* seq_id    */ seq_ids.data() + processed,
            /* logits    */ logits.data() + processed,
        };

        if (llama_decode(lctx, batch) != 0) {
            return 1;
        }
        processed += batch_size;
    }

    if (use_mrope_pos) {
        n_past += std::max(nx, ny);
    } else {
        n_past += n_tokens;
    }

    return 0;
}
#endif

#if defined(LLAMA_SERVER_SMT_MTMD)
bool server_media_context::tts_config_matches(const common_params & params) {
    return !params.smt_config_dir.empty() &&
           (params.media_backend == "auto" || params.media_backend == "smt") &&
           smt_tts_context::matches(params.smt_config_dir);
}

std::unique_ptr<server_media_context> server_media_context::init_tts(const common_params & params) {
    if (!tts_config_matches(params)) {
        throw std::runtime_error("SMT config is not a supported TTS model bundle");
    }

    auto ctx = std::unique_ptr<server_media_context>(new server_media_context());
    ctx->pimpl->mode = server_media_backend::smt;
    ctx->pimpl->worker = std::make_unique<media_worker>("smt-tts");
    ctx->pimpl->smt_tts = ctx->pimpl->worker->invoke("smt_tts_init", [&]() {
        return smt_tts_context::create(params.smt_config_dir, params.vocoder.speaker_file);
    });
    if (!ctx->pimpl->smt_tts) {
        throw std::runtime_error("failed to initialize SMT TTS backend");
    }
    SRV_INF("media backend '%s' worker thread id = %zu\n", ctx->pimpl->smt_tts->backend_name(),
            std::hash<std::thread::id>{}(ctx->pimpl->worker->thread_id()));
    return ctx;
}
#endif

std::unique_ptr<server_media_context> server_media_context::init(
        llama_context * ctx_llama,
        llama_model * model,
        const llama_vocab * vocab,
        const common_params & params,
        const mtmd_context_params & mtmd_params,
        std::unique_ptr<media_worker> worker) {
    const bool has_mmproj = !params.mmproj.path.empty();

    std::string backend_pref = "auto";
#if defined(LLAMA_SERVER_SMT_MTMD)
    backend_pref = params.media_backend;
    const bool has_smt_config = !params.smt_config_dir.empty();
#else
    const bool has_smt_config = false;
#endif

    if (backend_pref != "auto" && backend_pref != "mtmd" && backend_pref != "smt") {
        throw std::runtime_error("media backend must be one of: auto, mtmd, smt");
    }

    server_media_backend selected = server_media_backend::none;
    if (backend_pref == "mtmd") {
        selected = server_media_backend::mtmd;
    } else if (backend_pref == "smt") {
        selected = server_media_backend::smt;
    } else if (has_mmproj) {
        selected = server_media_backend::mtmd;
    } else if (has_smt_config) {
        selected = server_media_backend::smt;
    }

    if (selected == server_media_backend::none) {
        return nullptr;
    }

    auto ctx = std::unique_ptr<server_media_context>(new server_media_context());
    ctx->pimpl->mode = selected;
    ctx->pimpl->vocab = vocab;
    if (!worker) {
        worker.reset(new media_worker(backend_pref == "auto" ? (selected == server_media_backend::mtmd ? "mtmd" : "smt") : backend_pref));
    }
    ctx->pimpl->worker = std::move(worker);

    if (selected == server_media_backend::mtmd) {
        if (!has_mmproj) {
            throw std::runtime_error("media backend 'mtmd' selected but --mmproj is not set");
        }
        ctx->pimpl->mtmd_ctx = ctx->pimpl->worker->invoke("mtmd_init_from_file", [&]() {
            return mtmd::context_ptr(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
        });
        if (ctx->pimpl->mtmd_ctx == nullptr) {
            throw std::runtime_error("failed to load multimodal model: " + params.mmproj.path);
        }
        SRV_INF("media backend '%s' worker thread id = %zu\n", ctx->backend_name(),
                std::hash<std::thread::id>{}(ctx->pimpl->worker->thread_id()));
        return ctx;
    }

#if defined(LLAMA_SERVER_SMT_MTMD)
    if (selected == server_media_backend::smt) {
        if (!has_smt_config) {
            throw std::runtime_error("media backend 'smt' selected but --smt-config-dir is not set");
        }

        const std::string & config_dir = params.smt_config_dir;
        std::string primary_architecture;
        const server_media_config_modalities modalities = server_media_resolve_config_modalities(config_dir);

        if (modalities.wants_vision) {
            try {
                ctx->pimpl->smt_vision = ctx->pimpl->worker->invoke("smt_vision_init", [&]() {
                    return smt_vision_context::create(config_dir, params.warmup);
                });
                ctx->pimpl->hidden_size = (int32_t) ctx->pimpl->smt_vision->hidden_size();
                primary_architecture = ctx->pimpl->smt_vision->architecture();
                auto boundaries = server_media_resolve_image_boundary_tokens(vocab, primary_architecture);
                ctx->pimpl->tok_img_beg = std::move(boundaries.first);
                ctx->pimpl->tok_img_end = std::move(boundaries.second);
            } catch (const std::exception & e) {
                SRV_WRN("[server-media] failed to initialize SMT vision backend from '%s': %s\n", config_dir.c_str(), e.what());
            }
        }

        if (modalities.wants_audio) {
            try {
                ctx->pimpl->smt_audio = ctx->pimpl->worker->invoke("smt_audio_init", [&]() {
                    return smt_audio_context::create(config_dir, params.warmup);
                });
                if (ctx->pimpl->hidden_size == 0) {
                    ctx->pimpl->hidden_size = (int32_t) ctx->pimpl->smt_audio->hidden_size();
                } else if (ctx->pimpl->hidden_size != ctx->pimpl->smt_audio->hidden_size()) {
                    throw std::runtime_error("SMT image/audio hidden size mismatch");
                }
                if (primary_architecture.empty()) {
                    primary_architecture = ctx->pimpl->smt_audio->architecture();
                }
                auto audio_boundaries = server_media_resolve_audio_boundary_tokens(vocab, ctx->pimpl->smt_audio->architecture());
                ctx->pimpl->tok_audio_beg = std::move(audio_boundaries.first);
                ctx->pimpl->tok_audio_end = std::move(audio_boundaries.second);
            } catch (const std::exception & e) {
                SRV_WRN("[server-media] failed to initialize SMT audio backend from '%s': %s\n", config_dir.c_str(), e.what());
            }
        }

        if (!ctx->pimpl->smt_vision && !ctx->pimpl->smt_audio) {
            throw std::runtime_error("Neither SMT vision nor SMT audio backend is available");
        }

        ctx->pimpl->architecture = primary_architecture;
        ctx->pimpl->use_mrope_pos = server_media_arch_requires_mrope(primary_architecture);

        const int32_t model_n_embd = llama_model_n_embd_inp(model);
        if (ctx->pimpl->hidden_size > 0 && model_n_embd > 0 && ctx->pimpl->hidden_size != model_n_embd) {
            throw std::runtime_error(
                    "SMT hidden size (" + std::to_string(ctx->pimpl->hidden_size) +
                    ") does not match model embedding size (" + std::to_string(model_n_embd) + ")");
        }

        GGML_UNUSED(ctx_llama);
        SRV_INF("media backend '%s' worker thread id = %zu\n", ctx->backend_name(),
                std::hash<std::thread::id>{}(ctx->pimpl->worker->thread_id()));
        return ctx;
    }
#else
    if (selected == server_media_backend::smt) {
        GGML_UNUSED(ctx_llama);
        throw std::runtime_error("SMT media backend is not compiled. Rebuild with LLAMA_SERVER_SMT_MTMD=ON.");
    }
#endif

    return nullptr;
}

server_tokens server_media_context::process_prompt(
        const std::string & prompt_in,
        const std::vector<raw_buffer> & files,
        bool is_placeholder) {
    if (pimpl->mode == server_media_backend::mtmd) {
        return pimpl->worker->invoke("mtmd_process_prompt", [&]() {
            mtmd::bitmaps bitmaps;
            std::vector<mtmd_helper::video_ptr> videos;
            for (auto & file : files) {
                auto out = mtmd_helper_bitmap_init_from_buf(pimpl->mtmd_ctx.get(), file.data(), file.size(), is_placeholder);
                if (!out.bitmap) {
                    throw std::runtime_error("Failed to load image or audio file");
                }
                if (!is_placeholder) {
                    const std::string hash = server_media_fnv_hash(mtmd_bitmap_get_data(out.bitmap), mtmd_bitmap_get_n_bytes(out.bitmap));
                    mtmd_bitmap_set_id(out.bitmap, hash.c_str());
                }
                bitmaps.entries.emplace_back(out.bitmap);
                if (out.video_ctx) {
                    videos.emplace_back(out.video_ctx);
                }
            }

            mtmd_input_text inp_txt = {
                prompt_in.data(),
                prompt_in.size(),
                /* add_special */   true,
                /* parse_special */ true,
            };
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t tokenized = mtmd_tokenize(
                    pimpl->mtmd_ctx.get(),
                    chunks.ptr.get(),
                    &inp_txt,
                    bitmaps_c_ptr.data(),
                    bitmaps_c_ptr.size());
            if (tokenized != 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }
            return server_tokens(chunks, true);
        });
    }

#if defined(LLAMA_SERVER_SMT_MTMD)
    if (pimpl->mode == server_media_backend::smt) {
        if (is_placeholder) {
            throw std::runtime_error("SMT media placeholder token counting is not implemented");
        }

        static const std::string marker = "<__media__>";
        static const std::string legacy_marker = "<__image__>";
        std::string prompt = prompt_in;
        server_media_replace_all(prompt, get_media_marker(), marker);
        server_media_replace_all(prompt, legacy_marker, marker);
        if (!files.empty() && prompt.find(marker) == std::string::npos) {
            for (size_t i = 0; i < files.size(); ++i) {
                prompt = marker + prompt;
            }
        }

        auto parts = server_media_split_keep_marker(prompt, marker);
        size_t marker_count = 0;
        for (const auto & part : parts) {
            if (part == marker) {
                marker_count++;
            }
        }
        if (marker_count != files.size()) {
            throw std::runtime_error("Number of media inputs does not match number of media markers");
        }

        server_tokens out(llama_tokens{}, true);

        size_t media_idx = 0;
        bool add_special_once = true;
        for (const auto & part : parts) {
            if (part == marker) {
                const std::vector<uint8_t> & media_data = files[media_idx++];
                auto chunk = pimpl->worker->invoke("smt_encode_media", [&]() {
                    return server_media_encode_smt_media(*pimpl, media_data);
                });
                out.push_back(chunk);
                add_special_once = false;
                continue;
            }

            auto toks = common_tokenize(pimpl->vocab, part, add_special_once, /* parse_special */ true);
            for (auto tok : toks) {
                out.push_back(tok);
            }
            add_special_once = false;
        }

        return out;
    }
#endif

    throw std::runtime_error("media backend is not initialized");
}

int32_t server_media_context::encode_mtmd_batch(mtmd_batch * batch) const {
    if (pimpl->mode != server_media_backend::mtmd || pimpl->worker == nullptr) {
        return -1;
    }
    return pimpl->worker->invoke("mtmd_batch_encode", [&]() {
        return mtmd_batch_encode(batch);
    });
}

int32_t server_media_context::decode_embd_chunk(
        llama_context * lctx,
        const server_media_embd_chunk & chunk,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last) const {
#if defined(LLAMA_SERVER_SMT_MTMD)
    if (chunk.hidden_size <= 0 || chunk.embd.empty() || chunk.embd.size() % (size_t) chunk.hidden_size != 0) {
        return -1;
    }
    const int32_t n_embd_tokens = (int32_t) (chunk.embd.size() / (size_t) chunk.hidden_size);

    if (!chunk.tokens_begin.empty()) {
        if (server_media_decode_tokens(lctx, chunk.tokens_begin, n_past, seq_id, n_batch, false) != 0) {
            return -1;
        }
    }

    const bool logits_on_embd = logits_last && chunk.tokens_end.empty();
    if (server_media_decode_embd(
                lctx,
                chunk.embd.data(),
                n_embd_tokens,
                chunk.hidden_size,
                n_past,
                seq_id,
                n_batch,
                logits_on_embd,
                chunk.use_mrope_pos,
                chunk.grid_nx,
                chunk.grid_ny) != 0) {
        return -1;
    }

    if (!chunk.tokens_end.empty()) {
        if (server_media_decode_tokens(lctx, chunk.tokens_end, n_past, seq_id, n_batch, logits_last) != 0) {
            return -1;
        }
    }

    return 0;
#else
    GGML_UNUSED(lctx);
    GGML_UNUSED(chunk);
    GGML_UNUSED(n_past);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(n_batch);
    GGML_UNUSED(logits_last);
    return -1;
#endif
}
