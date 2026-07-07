// Multimodal CLI SMT backend using the SpacemiT SMT ONNX vision engine.
// Integrated into llama-mtmd-cli and selected via --vision-backend smt
// LLM inference logic (loading, sampling, generation) is reused from the original mtmd-cli

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "smt-audio-wrapper.h"
#include "smt-vision-wrapper.h"
#include "smt-vision-preprocess.h"

#include <vector>
#include <string>
#include <algorithm>
#include <limits.h>
#include <cinttypes>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Usage: %s [options] -m <model> --media-backend smt --smt-config-dir <dir> [--image <image.jpg|image.bin> | --audio <audio.wav>] -p <prompt>\n\n"
        "  -m and --smt-config-dir are required\n"
        "  --image/--audio and -p are optional, if NOT provided, the CLI will run in chat mode\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// ============================================================
// Prompt chunking utilities
// text/image are evaluated in order, aligned with mtmd chunk strategy
// ============================================================

static const std::string k_media_marker = "<__media__>";
static const std::string k_legacy_image_marker = "<__image__>";

static void replace_all(std::string & s, const std::string & from, const std::string & to);
static std::vector<std::string> split_keep_marker(const std::string & input, const std::string & marker);

enum class smt_chunk_type {
    text,
    image,
    audio,
};

enum class smt_init_mode {
    auto_probe,
    image_only,
    audio_only,
    mixed,
};

struct smt_prompt_chunk {
    smt_chunk_type type = smt_chunk_type::text;
    std::vector<llama_token> tokens_text;
    std::string media_path;
};

struct smt_pending_media {
    smt_chunk_type type = smt_chunk_type::image;
    std::string path;
};

enum class smt_image_boundary_mode {
    native,      // follow native mtmd model-family rules
    auto_detect, // probe vocab for known token pairs
    none,        // do not inject image boundary tokens
};

enum class smt_media_anchor_mode {
    auto_mode, // apply on known architectures for multi-image prompts
    on,        // always apply for multi-image prompts
    off,       // disable prompt canonicalization
};

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static smt_image_boundary_mode smt_image_boundary_mode_from_env() {
    const char * env = std::getenv("MTMD_SMT_IMAGE_BOUNDARY");
    if (env == nullptr || env[0] == '\0') {
        return smt_image_boundary_mode::native;
    }

    const std::string value = to_lower_ascii(env);
    if (value == "native") {
        return smt_image_boundary_mode::native;
    }
    if (value == "auto" || value == "detect") {
        return smt_image_boundary_mode::auto_detect;
    }
    if (value == "none" || value == "off" || value == "0") {
        return smt_image_boundary_mode::none;
    }

    LOG_WRN("[SMT] unknown MTMD_SMT_IMAGE_BOUNDARY='%s', fallback to 'native'\n", env);
    return smt_image_boundary_mode::native;
}

static const char * smt_image_boundary_mode_name(smt_image_boundary_mode mode) {
    switch (mode) {
        case smt_image_boundary_mode::native:      return "native";
        case smt_image_boundary_mode::auto_detect: return "auto";
        case smt_image_boundary_mode::none:        return "none";
    }
    return "native";
}

static bool contains_icase(const std::string & text, const std::string & pattern) {
    return to_lower_ascii(text).find(to_lower_ascii(pattern)) != std::string::npos;
}

static bool arch_requires_mrope(const std::string & arch_name) {
    return contains_icase(arch_name, "qwen2vl") ||
           contains_icase(arch_name, "qwen2_5_vl") ||
           contains_icase(arch_name, "qwen3vl") ||
           contains_icase(arch_name, "glm4v") ||
           contains_icase(arch_name, "paddleocr");
}

static bool arch_is_qwen3asr(const std::string & arch_name) {
    return contains_icase(arch_name, "qwen3asr");
}

static bool arch_is_gemma4_audio(const std::string & arch_name) {
    return arch_name == "Gemma4Audio";
}

static std::pair<int, int> infer_image_grid_xy(int n_tokens) {
    if (n_tokens <= 0) {
        return {0, 0};
    }

    // Choose the factor pair closest to square to approximate (nx, ny).
    int best_y = 1;
    int best_x = n_tokens;
    int root = (int) std::sqrt((double) n_tokens);
    for (int y = root; y >= 1; --y) {
        if (n_tokens % y == 0) {
            best_y = y;
            best_x = n_tokens / y;
            break;
        }
    }
    return {best_x, best_y};
}

static smt_media_anchor_mode smt_media_anchor_mode_from_env() {
    const char * env = std::getenv("MTMD_SMT_MEDIA_ANCHOR");
    if (env == nullptr || env[0] == '\0') {
        return smt_media_anchor_mode::off;
    }
    const std::string value = to_lower_ascii(env);
    if (value == "auto") {
        return smt_media_anchor_mode::auto_mode;
    }
    if (value == "on" || value == "1" || value == "true") {
        return smt_media_anchor_mode::on;
    }
    if (value == "off" || value == "0" || value == "false") {
        return smt_media_anchor_mode::off;
    }
    LOG_WRN("[SMT] unknown MTMD_SMT_MEDIA_ANCHOR='%s', fallback to 'off'\n", env);
    return smt_media_anchor_mode::off;
}

static const char * smt_media_anchor_mode_name(smt_media_anchor_mode mode) {
    switch (mode) {
        case smt_media_anchor_mode::auto_mode: return "auto";
        case smt_media_anchor_mode::on:        return "on";
        case smt_media_anchor_mode::off:       return "off";
    }
    return "auto";
}

static std::vector<llama_token> tokenize_exact_special(llama_context * lctx, const std::string & token_text) {
    auto toks = common_tokenize(lctx, token_text, /*add_special*/ false, /*parse_special*/ true);
    if (toks.size() != 1) {
        return {};
    }
    // Ensure this is a true special-token hit, not byte-level fallback.
    if (common_token_to_piece(lctx, toks[0]) != token_text) {
        return {};
    }
    return toks;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> detect_image_boundary_tokens(llama_context * lctx) {
    // Keep order aligned with mtmd.cpp init_vision() common projector families.
    static const std::vector<std::pair<std::string, std::string>> candidates = {
        {"<|vision_start|>", "<|vision_end|>"},
        {"<|image_start|>",  "<|image_end|>"},
        {"<start_of_image>", "<end_of_image>"},
        {"<img>",            "</img>"},
        {"<|begin_of_image|>", "<|end_of_image|>"},
        {"<|IMAGE_START|>",  "<|IMAGE_END|>"},
        {"<|im_start|>",     "<|im_end|>"},
        {"<image>",          "</image>"},
    };

    for (const auto & [beg_s, end_s] : candidates) {
        auto beg = tokenize_exact_special(lctx, beg_s);
        auto end = tokenize_exact_special(lctx, end_s);
        if (!beg.empty() && !end.empty()) {
            return {std::move(beg), std::move(end)};
        }
    }
    return {};
}

static bool should_apply_media_anchor(const std::string & arch_name, size_t n_images, smt_media_anchor_mode mode) {
    if (n_images < 2) {
        return false;
    }
    switch (mode) {
        case smt_media_anchor_mode::off:
            return false;
        case smt_media_anchor_mode::on:
            return true;
        case smt_media_anchor_mode::auto_mode:
            return contains_icase(arch_name, "llavaqwen2forcausallm") || contains_icase(arch_name, "llavaqwen2");
    }
    return false;
}

static std::string canonicalize_multimage_prompt(
        const std::string & prompt,
        size_t n_images,
        const std::string & arch_name,
        smt_media_anchor_mode mode,
        bool & changed) {
    changed = false;
    if (!should_apply_media_anchor(arch_name, n_images, mode)) {
        return prompt;
    }

    std::string normalized = prompt;
    replace_all(normalized, k_legacy_image_marker, k_media_marker);

    const auto parts = split_keep_marker(normalized, k_media_marker);
    size_t marker_count = 0;
    for (const auto & p : parts) {
        if (p == k_media_marker) {
            marker_count++;
        }
    }

    if (marker_count != n_images || marker_count < 2) {
        return normalized;
    }

    std::string output;
    output.reserve(normalized.size() + marker_count * 48 + 64);
    output += "Please align images by order index. ";
    output += "Image #1 maps to the first media slot, Image #2 to the second, and so on.\n";

    size_t image_idx = 0;
    for (const auto & part : parts) {
        if (part == k_media_marker) {
            image_idx++;
            output += "[Image ";
            output += std::to_string(image_idx);
            output += " Begin]\n";
            output += k_media_marker;
            output += "\n[Image ";
            output += std::to_string(image_idx);
            output += " End]\n";
        } else {
            output += part;
        }
    }

    changed = (output != normalized);
    return output;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
detect_image_boundary_tokens_native(llama_context * lctx, const std::string & arch_name) {
    // Match native mtmd behavior in mtmd.cpp::init_vision().
    if (contains_icase(arch_name, "qwen2vl") || contains_icase(arch_name, "qwen2_5_vl") ||
        contains_icase(arch_name, "qwen3vl") || contains_icase(arch_name, "youtuvl")) {
        return {
            tokenize_exact_special(lctx, "<|vision_start|>"),
            tokenize_exact_special(lctx, "<|vision_end|>")
        };
    }
    if (contains_icase(arch_name, "llama4")) {
        return {
            tokenize_exact_special(lctx, "<|image_start|>"),
            tokenize_exact_special(lctx, "<|image_end|>")
        };
    }
    if (contains_icase(arch_name, "gemma3")) {
        return {
            tokenize_exact_special(lctx, "<start_of_image>"),
            tokenize_exact_special(lctx, "<end_of_image>")
        };
    }
    if (contains_icase(arch_name, "internvl")) {
        return {
            tokenize_exact_special(lctx, "<img>"),
            tokenize_exact_special(lctx, "</img>")
        };
    }
    if (contains_icase(arch_name, "glm4v")) {
        return {
            tokenize_exact_special(lctx, "<|begin_of_image|>"),
            tokenize_exact_special(lctx, "<|end_of_image|>")
        };
    }
    if (contains_icase(arch_name, "paddleocr")) {
        return {
            tokenize_exact_special(lctx, "<|IMAGE_START|>"),
            tokenize_exact_special(lctx, "<|IMAGE_END|>")
        };
    }
    if (contains_icase(arch_name, "lightonocr")) {
        return {
            tokenize_exact_special(lctx, "<|im_start|>"),
            tokenize_exact_special(lctx, "<|im_end|>")
        };
    }
    return {};
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
resolve_image_boundary_tokens(llama_context * lctx,
                              const std::string & arch_name,
                              smt_image_boundary_mode mode) {
    if (mode == smt_image_boundary_mode::none) {
        return {};
    }
    if (mode == smt_image_boundary_mode::auto_detect) {
        return detect_image_boundary_tokens(lctx);
    }
    return detect_image_boundary_tokens_native(lctx, arch_name);
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
resolve_audio_boundary_tokens(llama_context * lctx, const std::string & arch_name) {
    if (arch_is_qwen3asr(arch_name)) {
        return {
            tokenize_exact_special(lctx, "<|audio_start|>"),
            tokenize_exact_special(lctx, "<|audio_end|>")
        };
    }
    if (arch_is_gemma4_audio(arch_name)) {
        return {
            tokenize_exact_special(lctx, "<|audio>"),
            tokenize_exact_special(lctx, "<audio|>")
        };
    }
    return {};
}

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::vector<std::string> split_keep_marker(const std::string & input, const std::string & marker) {
    std::vector<std::string> out;
    if (input.empty()) {
        return out;
    }

    size_t start = 0;
    while (true) {
        size_t pos = input.find(marker, start);
        if (pos == std::string::npos) {
            out.push_back(input.substr(start));
            break;
        }
        if (pos > start) {
            out.push_back(input.substr(start, pos - start));
        }
        out.push_back(marker);
        start = pos + marker.size();
    }
    return out;
}

static std::vector<uint8_t> read_binary_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open image file: " + path);
    }

    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static bool looks_like_audio_file(const std::vector<uint8_t> & bytes) {
    if (bytes.size() < 12) {
        return false;
    }

    const char * buf = reinterpret_cast<const char *>(bytes.data());
    const bool is_wav = std::memcmp(buf, "RIFF", 4) == 0 && std::memcmp(buf + 8, "WAVE", 4) == 0;
    const bool is_mp3 = bytes.size() >= 3 && (
        std::memcmp(buf, "ID3", 3) == 0 ||
        (static_cast<unsigned char>(buf[0]) == 0xFF && (static_cast<unsigned char>(buf[1]) & 0xE0) == 0xE0)
    );
    const bool is_flac = std::memcmp(buf, "fLaC", 4) == 0;
    return is_wav || is_mp3 || is_flac;
}

static smt_chunk_type detect_media_type_from_file(const std::string & path) {
    const auto bytes = read_binary_file(path);
    return looks_like_audio_file(bytes) ? smt_chunk_type::audio : smt_chunk_type::image;
}

static smt_init_mode infer_init_mode_from_params(const common_params & params) {
    bool need_image = false;
    bool need_audio = false;

    for (const auto & media : params.image) {
        const auto type = detect_media_type_from_file(media);
        need_image = need_image || type == smt_chunk_type::image;
        need_audio = need_audio || type == smt_chunk_type::audio;
    }

    if (need_image && need_audio) {
        return smt_init_mode::mixed;
    }
    if (need_image) {
        return smt_init_mode::image_only;
    }
    if (need_audio) {
        return smt_init_mode::audio_only;
    }

    return smt_init_mode::auto_probe;
}

static std::string write_temp_bin_file(const std::vector<uint8_t> & data) {
#if defined(_WIN32)
    char temp_path[MAX_PATH] = {0};
    char temp_file[MAX_PATH] = {0};

    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("failed to get temp path");
    }
    if (GetTempFileNameA(temp_path, "mep", 0, temp_file) == 0) {
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
    char tmpl[] = "/tmp/llama-mtmd-smt-XXXXXX";
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

static void append_text_chunk(std::vector<smt_prompt_chunk> & chunks, std::vector<llama_token> && tokens) {
    if (tokens.empty()) {
        return;
    }
    if (!chunks.empty() && chunks.back().type == smt_chunk_type::text) {
        auto & dst = chunks.back().tokens_text;
        dst.insert(dst.end(), tokens.begin(), tokens.end());
        return;
    }

    smt_prompt_chunk chunk;
    chunk.type = smt_chunk_type::text;
    chunk.tokens_text = std::move(tokens);
    chunks.emplace_back(std::move(chunk));
}

static bool tokenize_to_smt_chunks(
        const llama_vocab * vocab,
        const std::string & formatted_chat,
        bool add_special,
        bool parse_special,
        const std::vector<smt_pending_media> & media_inputs,
        std::vector<smt_prompt_chunk> & out_chunks) {
    out_chunks.clear();

    std::string input = formatted_chat;
    replace_all(input, k_legacy_image_marker, k_media_marker);

    size_t n_images_used = 0;
    const auto parts = split_keep_marker(input, k_media_marker);
    for (const auto & part : parts) {
        if (part == k_media_marker) {
            if (n_images_used >= media_inputs.size()) {
                LOG_ERR("Number of media inputs (%zu) does not match number of media markers in prompt\n", media_inputs.size());
                return false;
            }
            smt_prompt_chunk chunk;
            chunk.type = media_inputs[n_images_used].type;
            chunk.media_path = media_inputs[n_images_used].path;
            n_images_used++;
            out_chunks.emplace_back(std::move(chunk));
        } else {
            append_text_chunk(out_chunks, common_tokenize(vocab, part, /*add_special*/ false, parse_special));
        }
    }

    if (n_images_used != media_inputs.size()) {
        LOG_ERR("Number of media inputs (%zu) does not match number of media markers in prompt (%zu)\n",
                media_inputs.size(), n_images_used);
        return false;
    }

    if (add_special && llama_vocab_get_add_bos(vocab)) {
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            if (!out_chunks.empty() && out_chunks.front().type == smt_chunk_type::text) {
                out_chunks.front().tokens_text.insert(out_chunks.front().tokens_text.begin(), bos);
            } else {
                smt_prompt_chunk bos_chunk;
                bos_chunk.type = smt_chunk_type::text;
                bos_chunk.tokens_text = { bos };
                out_chunks.insert(out_chunks.begin(), std::move(bos_chunk));
            }
        }
    }

    if (add_special && llama_vocab_get_add_eos(vocab)) {
        const llama_token eos = llama_vocab_eos(vocab);
        if (eos != LLAMA_TOKEN_NULL) {
            if (!out_chunks.empty() && out_chunks.back().type == smt_chunk_type::text) {
                out_chunks.back().tokens_text.push_back(eos);
            } else {
                smt_prompt_chunk eos_chunk;
                eos_chunk.type = smt_chunk_type::text;
                eos_chunk.tokens_text = { eos };
                out_chunks.emplace_back(std::move(eos_chunk));
            }
        }
    }

    return true;
}

// ============================================================
// Decode embedding batch into LLM
// (Adapted from mtmd-helper.cpp decode_embd_batch + mtmd_helper_decode_image_chunk)
// Supports both normal positional encoding and M-RoPE positional layout.
// ============================================================

static int decode_embd(llama_context * lctx,
                       float * embd,
                       int n_tokens,
                       int n_embd,
                       llama_pos & n_past,
                       int n_batch,
                       bool logits_last,
                       bool use_mrope_pos,
                       int nx,
                       int ny) {
    const int n_pos_per_embd = use_mrope_pos ? 4 : 1;

    // Allocate auxiliary arrays (managed manually, not via llama_batch_init).
    std::vector<llama_pos>      pos((size_t) n_tokens * n_pos_per_embd);
    std::vector<int32_t>        n_seq_id(n_tokens);
    std::vector<llama_seq_id>   seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = 0;
        seq_ids[i]  = &seq_id_0[i];
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
            // Fallback to 1D M-RoPE-style positions if grid inference is inconsistent.
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
        int batch_size = std::min(n_batch, n_tokens - processed);
        bool is_last_batch = (processed + batch_size >= n_tokens);

        for (int i = 0; i < batch_size; ++i) {
            if (!use_mrope_pos) {
                pos[processed + i] = n_past + processed + i;
            }
            n_seq_id[processed + i] = 1;
            logits[processed + i]   = (logits_last && is_last_batch && i == batch_size - 1);
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
            /*n_tokens  =*/ batch_size,
            /*token     =*/ nullptr,
            /*embd      =*/ embd + (size_t)processed * n_embd,
            /*pos       =*/ pos_ptr,
            /*n_seq_id  =*/ n_seq_id.data() + processed,
            /*seq_id    =*/ seq_ids.data()  + processed,
            /*logits    =*/ logits.data()   + processed,
        };

        int ret = llama_decode(lctx, batch);
        if (ret != 0) {
            LOG_ERR("Failed to decode embedding batch at offset %d\n", processed);
            return ret;
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

static int decode_tokens(llama_context * lctx,
                         const std::vector<llama_token> & tokens,
                         llama_pos & n_past,
                         int n_batch,
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

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        n_past += batch.n_tokens;
    }

    llama_batch_free(batch);
    return 0;
}

// ============================================================
// Context structure
// ============================================================

struct mtmd_cli_smt_context {
    std::unique_ptr<smt_vision_context> smt_vision_ctx;
    std::unique_ptr<smt_audio_context>  smt_audio_ctx;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    int64_t hidden_size = 0;
    bool use_mrope_pos = false;
    std::vector<llama_token> tok_img_beg;
    std::vector<llama_token> tok_img_end;
    std::vector<llama_token> tok_audio_beg;
    std::vector<llama_token> tok_audio_end;
    smt_image_boundary_mode img_boundary_mode = smt_image_boundary_mode::native;
    smt_media_anchor_mode media_anchor_mode = smt_media_anchor_mode::off;

    // Pending media paths
    std::vector<smt_pending_media> pending_media;

    // Chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;

    // Legacy template antiprompt
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    mtmd_cli_smt_context(common_params & params, const std::string & smt_config_dir)
        : llama_init(common_init_from_params(params))
    {
        model = llama_init->model();
        lctx  = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl  = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (!model || !lctx) {
            LOG_ERR("Failed to initialize LLM model\n");
            exit(1);
        }

        // Chat template
        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();

        std::string primary_architecture;
        const smt_init_mode init_mode = infer_init_mode_from_params(params);
        std::string vision_error;
        std::string audio_error;

        const bool try_vision = init_mode == smt_init_mode::auto_probe ||
                                init_mode == smt_init_mode::image_only ||
                                init_mode == smt_init_mode::mixed;
        const bool try_audio = init_mode == smt_init_mode::auto_probe ||
                               init_mode == smt_init_mode::audio_only ||
                               init_mode == smt_init_mode::mixed;

        if (try_vision) {
            try {
                smt_vision_ctx = smt_vision_context::create(smt_config_dir);
                hidden_size = smt_vision_ctx->hidden_size();
                primary_architecture = smt_vision_ctx->architecture();
            } catch (const std::exception & e) {
                vision_error = e.what();
            }
        }

        if (try_audio) {
            try {
                smt_audio_ctx = smt_audio_context::create(smt_config_dir, params.warmup);
                if (hidden_size == 0) {
                    hidden_size = smt_audio_ctx->hidden_size();
                } else if (hidden_size != smt_audio_ctx->hidden_size()) {
                    LOG_ERR("FATAL: SMT audio hidden_size (%" PRId64 ") != current SMT hidden_size (%" PRId64 ")\n",
                            smt_audio_ctx->hidden_size(), hidden_size);
                    exit(1);
                }
                if (primary_architecture.empty()) {
                    primary_architecture = smt_audio_ctx->architecture();
                }
            } catch (const std::exception & e) {
                audio_error = e.what();
            }
        }

        if (init_mode == smt_init_mode::image_only && !smt_vision_ctx) {
            LOG_ERR("FATAL: failed to initialize SMT vision backend from %s: %s\n",
                    smt_config_dir.c_str(), vision_error.empty() ? "unknown error" : vision_error.c_str());
            exit(1);
        }

        if (init_mode == smt_init_mode::audio_only && !smt_audio_ctx) {
            LOG_ERR("FATAL: failed to initialize SMT audio backend from %s: %s\n",
                    smt_config_dir.c_str(), audio_error.empty() ? "unknown error" : audio_error.c_str());
            exit(1);
        }

        if (init_mode == smt_init_mode::mixed && (!smt_vision_ctx || !smt_audio_ctx)) {
            LOG_ERR("FATAL: mixed SMT request requires both vision and audio backends\n");
            if (!smt_vision_ctx) {
                LOG_ERR("FATAL: vision backend init failed: %s\n",
                        vision_error.empty() ? "unknown error" : vision_error.c_str());
            }
            if (!smt_audio_ctx) {
                LOG_ERR("FATAL: audio backend init failed: %s\n",
                        audio_error.empty() ? "unknown error" : audio_error.c_str());
            }
            exit(1);
        }

        if (!smt_vision_ctx && !smt_audio_ctx) {
            LOG_ERR("FATAL: neither SMT vision nor SMT audio backend is available in %s\n", smt_config_dir.c_str());
            exit(1);
        }

        use_mrope_pos = arch_requires_mrope(primary_architecture);
        img_boundary_mode = smt_image_boundary_mode_from_env();
        media_anchor_mode = smt_media_anchor_mode_from_env();
        if (hidden_size <= 0 || hidden_size > INT_MAX) {
            LOG_ERR("FATAL: invalid SMT hidden_size (%" PRId64 ")\n", hidden_size);
            exit(1);
        }

        // Validate n_embd matches
        int model_n_embd = llama_model_n_embd(model);
        if (model_n_embd != hidden_size) {
            LOG_ERR("FATAL: model n_embd (%d) != SMT hidden_size (%" PRId64 ")\n", model_n_embd, hidden_size);
            exit(1);
        }

        // Align image boundary tokens with native mtmd behavior by default.
        if (smt_vision_ctx) {
            auto boundaries = resolve_image_boundary_tokens(lctx, smt_vision_ctx->architecture(), img_boundary_mode);
            tok_img_beg = std::move(boundaries.first);
            tok_img_end = std::move(boundaries.second);
            if (tok_img_beg.empty() || tok_img_end.empty() ||
                tok_img_beg.front() == LLAMA_TOKEN_NULL || tok_img_end.front() == LLAMA_TOKEN_NULL) {
                tok_img_beg.clear();
                tok_img_end.clear();
            }
        }

        if (smt_audio_ctx) {
            auto audio_boundaries = resolve_audio_boundary_tokens(lctx, smt_audio_ctx->architecture());
            tok_audio_beg = std::move(audio_boundaries.first);
            tok_audio_end = std::move(audio_boundaries.second);
            if (tok_audio_beg.empty() || tok_audio_end.empty() ||
                tok_audio_beg.front() == LLAMA_TOKEN_NULL || tok_audio_end.front() == LLAMA_TOKEN_NULL) {
                tok_audio_beg.clear();
                tok_audio_end.clear();
            }
        }

        // Antiprompt for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_smt_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    void add_image(const std::string & binary_path) {
        pending_media.push_back({ smt_chunk_type::image, binary_path });
    }

    void add_audio(const std::string & audio_path) {
        pending_media.push_back({ smt_chunk_type::audio, audio_path });
    }

    void add_media_auto(const std::string & media_path) {
        pending_media.push_back({ detect_media_type_from_file(media_path), media_path });
    }

    size_t pending_image_count() const {
        size_t count = 0;
        for (const auto & media : pending_media) {
            if (media.type == smt_chunk_type::image) {
                count++;
            }
        }
        return count;
    }

    bool has_pending_audio_only() const {
        if (pending_media.empty()) {
            return false;
        }
        for (const auto & media : pending_media) {
            if (media.type != smt_chunk_type::audio) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================
// Generate response (reused from mtmd-cli.cpp)
// ============================================================

static int generate_response(mtmd_cli_smt_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break;
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // Eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

// ============================================================
// Chat formatting (reused from mtmd-cli.cpp)
// ============================================================

static std::string chat_add_and_format(mtmd_cli_smt_context & ctx, common_chat_msg & new_msg) {
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static std::string collect_system_text(const mtmd_cli_smt_context & ctx) {
    std::string system_text;
    for (const auto & msg : ctx.chat_history) {
        if (msg.role == "system") {
            system_text += msg.content;
        }
    }
    return system_text;
}

static std::string strip_media_markers_from_prompt(const std::string & prompt) {
    std::string text = prompt;
    replace_all(text, k_legacy_image_marker, k_media_marker);
    replace_all(text, k_media_marker, "");
    return string_strip(text);
}

static std::string format_qwen3asr_audio_prompt(const mtmd_cli_smt_context & ctx, const common_chat_msg & msg) {
    std::string prompt;
    prompt.reserve(msg.content.size() + ctx.pending_media.size() * 16 + 128);
    prompt += "<|im_start|>system\n";
    prompt += collect_system_text(ctx);
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>user\n";
    for (const auto & media : ctx.pending_media) {
        GGML_ASSERT(media.type == smt_chunk_type::audio);
        prompt += k_media_marker;
    }
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    prompt += strip_media_markers_from_prompt(msg.content);
    return prompt;
}

static std::string format_gemma4_audio_prompt(const mtmd_cli_smt_context & ctx, const common_chat_msg & msg) {
    std::string prompt;
    prompt.reserve(msg.content.size() + ctx.pending_media.size() * 16 + 128);
    prompt += "<|turn>user\n";
    for (const auto & media : ctx.pending_media) {
        GGML_ASSERT(media.type == smt_chunk_type::audio);
        prompt += k_media_marker;
    }
    const std::string user_text = strip_media_markers_from_prompt(msg.content);
    if (!user_text.empty()) {
        prompt += user_text;
    }
    prompt += "<turn|>\n";
    prompt += "<|turn>model\n";
    return prompt;
}

// ============================================================
// Eval message - core multimodal processing
// ============================================================

static int eval_message_smt(mtmd_cli_smt_context & ctx, common_chat_msg & msg) {
    struct pending_media_guard {
        mtmd_cli_smt_context & ctx;

        ~pending_media_guard() {
            ctx.pending_media.clear();
        }
    } guard{ctx};

    bool add_bos = ctx.chat_history.empty();

    if (msg.role == "user" && ctx.pending_image_count() > 0) {
        bool changed = false;
        msg.content = canonicalize_multimage_prompt(
            msg.content,
            ctx.pending_image_count(),
            ctx.smt_vision_ctx ? ctx.smt_vision_ctx->architecture() : std::string(),
            ctx.media_anchor_mode,
            changed);
        GGML_UNUSED(changed);
    }

    std::string formatted_chat;
    const bool use_qwen3asr_prompt =
        msg.role == "user" &&
        ctx.smt_audio_ctx &&
        arch_is_qwen3asr(ctx.smt_audio_ctx->architecture()) &&
        ctx.has_pending_audio_only();
    const bool use_gemma4_audio_prompt =
        msg.role == "user" &&
        ctx.smt_audio_ctx &&
        arch_is_gemma4_audio(ctx.smt_audio_ctx->architecture()) &&
        ctx.has_pending_audio_only();
    if (use_qwen3asr_prompt) {
        formatted_chat = format_qwen3asr_audio_prompt(ctx, msg);
        ctx.chat_history.push_back(msg);
        add_bos = false;
    } else if (use_gemma4_audio_prompt) {
        formatted_chat = format_gemma4_audio_prompt(ctx, msg);
        ctx.chat_history.push_back(msg);
        add_bos = ctx.chat_history.size() == 1;
    } else {
        formatted_chat = chat_add_and_format(ctx, msg);
    }

    if (g_is_interrupted) return 0;

    std::vector<smt_prompt_chunk> chunks;
    if (!tokenize_to_smt_chunks(ctx.vocab, formatted_chat, add_bos, true, ctx.pending_media, chunks)) {
        return 1;
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
        const bool logits_last = (i == chunks.size() - 1);
        const auto & chunk = chunks[i];
        if (chunk.type == smt_chunk_type::text) {
            if (decode_tokens(ctx.lctx, chunk.tokens_text, ctx.n_past, ctx.n_batch, logits_last) != 0) {
                LOG_ERR("Failed to decode text chunk %zu\n", i);
                return 1;
            }
            continue;
        }

        if (chunk.type == smt_chunk_type::audio) {
            if (!ctx.smt_audio_ctx) {
                LOG_ERR("No SMT audio backend is available for '%s'\n", chunk.media_path.c_str());
                return 1;
            }

            std::vector<float> audio_embd;
            try {
                audio_embd = ctx.smt_audio_ctx->encode_audio(chunk.media_path);
            } catch (const std::exception & e) {
                LOG_ERR("Failed to encode audio chunk %zu (%s): %s\n", i, chunk.media_path.c_str(), e.what());
                return 1;
            }

            if (audio_embd.empty() || audio_embd.size() % (size_t) ctx.hidden_size != 0) {
                LOG_ERR("Invalid audio embedding shape from SMT (size=%zu, hidden_size=%" PRId64 ")\n",
                        audio_embd.size(), ctx.hidden_size);
                return 1;
            }

            const int n_audio_tokens = (int) (audio_embd.size() / (size_t) ctx.hidden_size);
            if (!ctx.tok_audio_beg.empty()) {
                if (decode_tokens(ctx.lctx, ctx.tok_audio_beg, ctx.n_past, ctx.n_batch, /*logits_last*/ false) != 0) {
                    LOG_ERR("Failed to decode audio-begin token chunk %zu\n", i);
                    return 1;
                }
            }

            const bool logits_on_embd = logits_last && ctx.tok_audio_end.empty();
            if (decode_embd(ctx.lctx, audio_embd.data(), n_audio_tokens,
                            (int) ctx.hidden_size, ctx.n_past, ctx.n_batch, logits_on_embd,
                            /*use_mrope_pos*/ false, n_audio_tokens, 1) != 0) {
                LOG_ERR("Failed to decode audio chunk %zu\n", i);
                return 1;
            }

            if (!ctx.tok_audio_end.empty()) {
                if (decode_tokens(ctx.lctx, ctx.tok_audio_end, ctx.n_past, ctx.n_batch, logits_last) != 0) {
                    LOG_ERR("Failed to decode audio-end token chunk %zu\n", i);
                    return 1;
                }
            }
            continue;
        }

        // image chunk
        if (!ctx.smt_vision_ctx) {
            LOG_ERR("No SMT vision backend is available for '%s'\n", chunk.media_path.c_str());
            return 1;
        }

        std::string smt_input_path = chunk.media_path;
        std::string temp_input_path;
        try {
            const auto image_bytes = read_binary_file(chunk.media_path);
            auto preproc = smt_vision_preprocess_if_image(
                image_bytes,
                ctx.smt_vision_ctx->architecture(),
                ctx.smt_vision_ctx->input_width(),
                ctx.smt_vision_ctx->input_height());
            if (preproc.was_image) {
                temp_input_path = write_temp_bin_file(preproc.tensor_bytes);
                smt_input_path = temp_input_path;
            }
        } catch (const std::exception & e) {
            LOG_ERR("Failed to prepare image '%s': %s\n", chunk.media_path.c_str(), e.what());
            return 1;
        }

        std::vector<float> image_embd;
        try {
            image_embd = ctx.smt_vision_ctx->encode_image(smt_input_path);
            if (!temp_input_path.empty()) {
                std::remove(temp_input_path.c_str());
            }
        } catch (const std::exception & e) {
            if (!temp_input_path.empty()) {
                std::remove(temp_input_path.c_str());
            }
            LOG_ERR("Failed to encode image chunk %zu (%s): %s\n", i, chunk.media_path.c_str(), e.what());
            return 1;
        }

        if (image_embd.empty() || image_embd.size() % (size_t)ctx.hidden_size != 0) {
            LOG_ERR("Invalid image embedding shape from SMT (size=%zu, hidden_size=%" PRId64 ")\n",
                    image_embd.size(), ctx.hidden_size);
            return 1;
        }

        const int n_image_tokens = (int)(image_embd.size() / (size_t)ctx.hidden_size);
        int grid_nx = n_image_tokens;
        int grid_ny = 1;
        if (ctx.use_mrope_pos) {
            auto grid_xy = infer_image_grid_xy(n_image_tokens);
            grid_nx = grid_xy.first;
            grid_ny = grid_xy.second;
        }

        if (!ctx.tok_img_beg.empty()) {
            if (decode_tokens(ctx.lctx, ctx.tok_img_beg, ctx.n_past, ctx.n_batch, /*logits_last*/ false) != 0) {
                LOG_ERR("Failed to decode image-begin token chunk %zu\n", i);
                return 1;
            }
        }

        const bool logits_on_embd = logits_last && ctx.tok_img_end.empty();
        if (decode_embd(ctx.lctx, image_embd.data(), n_image_tokens,
                        (int)ctx.hidden_size, ctx.n_past, ctx.n_batch, logits_on_embd,
                        ctx.use_mrope_pos, grid_nx, grid_ny) != 0) {
            LOG_ERR("Failed to decode image chunk %zu\n", i);
            return 1;
        }

        if (!ctx.tok_img_end.empty()) {
            if (decode_tokens(ctx.lctx, ctx.tok_img_end, ctx.n_past, ctx.n_batch, logits_last) != 0) {
                LOG_ERR("Failed to decode image-end token chunk %zu\n", i);
                return 1;
            }
        }
    }

    LOG("\n");
    return 0;
}

// ============================================================
// Main function
// ============================================================

int mtmd_cli_smt_run(int argc, char ** argv, common_params params) {
    ggml_time_init();
    LOG_INF("MTMD_CLI_SMT_BUILD_TAG: spacemit-smt (%s %s)\n", __DATE__, __TIME__);

    common_init();

    if (params.smt_config_dir.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --smt-config-dir argument\n");
        return 1;
    }

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();
    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    mtmd_cli_smt_context ctx(params, params.smt_config_dir);

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    // Evaluate system prompt if present
    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }
        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        if (ctx.smt_audio_ctx && arch_is_qwen3asr(ctx.smt_audio_ctx->architecture())) {
            ctx.chat_history.push_back(msg);
            return 0;
        }
        return eval_message_smt(ctx, msg);
    };

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;

        // Insert <__media__> marker if not present (same logic as mtmd-cli.cpp)
        if (params.prompt.find("<__media__>") == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                params.prompt = std::string("<__media__>") + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;

        for (const auto & media : params.image) {
            ctx.add_media_auto(media);
        }

        if (eval_message_smt(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }
    } else {
        // Chat mode
        LOG("\n Running in chat mode (SMT media), available commands:");
        LOG("\n   /image <path>    load an image (.jpg/.png/...) or preprocessed .bin");
        LOG("\n   /audio <path>    load an audio file (.wav/.mp3/.flac)");
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;

            const bool is_image = line == "/image" || line.find("/image ") == 0;
            const bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            if (is_image || is_audio) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (is_image) {
                    ctx.add_image(media_path);
                    LOG("%s image loaded\n", media_path.c_str());
                } else {
                    ctx.add_audio(media_path);
                    LOG("%s audio loaded\n", media_path.c_str());
                }
                content += "<__media__>";
                continue;
            } else {
                content += line;
            }

            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message_smt(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }

    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
