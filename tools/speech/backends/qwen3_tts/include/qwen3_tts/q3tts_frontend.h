#pragma once

#include "gguf.h"
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace q3tts_frontend {

static constexpr int HID = 1024;
static constexpr int TEXT_IM_START = 151644;
static constexpr int TEXT_IM_END = 151645;
static constexpr int TTS_PAD = 151671;
static constexpr int TTS_BOS = 151672;
static constexpr int TTS_EOS = 151673;
static constexpr int CODEC_PAD = 2148;
static constexpr int CODEC_BOS = 2149;
static constexpr int CODEC_THINK = 2154;
static constexpr int CODEC_NOTHINK = 2155;
static constexpr int CODEC_THINK_BOS = 2156;
static constexpr int CODEC_THINK_EOS = 2157;
static constexpr int CODEC_LANGUAGE_ENGLISH = 2050;
static constexpr int CODEC_LANGUAGE_CHINESE = 2055;
static constexpr int CODE_GROUPS = 16;
static constexpr int CODE_VOCAB = 2048;
static constexpr int REF_CODEC_SAMPLES = 192000;
static constexpr int REF_CODEC_SAMPLES_PER_FRAME = 1920;

struct FrontendInput {
    std::vector<float> prefill;
    std::vector<float> trailing;
    std::vector<float> pad;
    int64_t n_prefill = 0;
    int64_t n_trailing = 0;
};

struct FrontendConfig {
    std::string model_dir = ".";
    std::string text;
    std::string ref_wav;
    std::string ref_bin;
    std::string language = "auto";
    std::string talker_gguf = "qwen3-tts-0.6b-talker-qkv-gateup-q8_0-side.gguf";
    std::string cp_gguf = "qwen3-tts-0.6b-cp-qkv-gateup-rawq4.gguf";
    int frontend_threads = 2;
    bool full_prompt_non_streaming = false;
};

struct ReferencePrompt {
    std::vector<float> speaker;
    std::string ref_text;
    std::vector<std::array<int32_t, CODE_GROUPS>> ref_codes;

    bool full_prompt() const {
        return !ref_text.empty() && !ref_codes.empty();
    }
};

inline bool file_exists(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

inline std::string path_join(const std::string &a, const std::string &b) {
    if (a.empty() || a == ".") {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

inline std::string first_existing(const std::vector<std::string> &paths) {
    for (const auto &p : paths) {
        if (!p.empty() && file_exists(p)) {
            return p;
        }
    }
    throw std::runtime_error("missing model asset: " + (paths.empty() ? std::string("") : paths.front()));
}

inline std::string read_text_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open failed: " + path);
    }
    f.seekg(0, std::ios::end);
    const size_t n = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::string s(n, '\0');
    f.read(&s[0], n);
    return s;
}

inline std::vector<uint8_t> read_bytes_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open failed: " + path);
    }
    f.seekg(0, std::ios::end);
    const size_t n = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(n);
    f.read(reinterpret_cast<char *>(out.data()), n);
    return out;
}

inline void write_bytes_file(const std::string &path, const uint8_t *data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("write failed: " + path);
    }
    f.write(reinterpret_cast<const char *>(data), n);
}

inline void append_utf8(uint32_t cp, std::string &out) {
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

inline uint32_t parse_hex4(const std::string &s, size_t pos) {
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = s[pos + i];
        v <<= 4;
        if (c >= '0' && c <= '9') {
            v |= static_cast<uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            v |= static_cast<uint32_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            v |= static_cast<uint32_t>(c - 'A' + 10);
        } else {
            throw std::runtime_error("bad json unicode escape");
        }
    }
    return v;
}

inline std::string parse_json_string(const std::string &s, size_t &i) {
    if (i >= s.size() || s[i] != '"') {
        throw std::runtime_error("expected json string");
    }
    ++i;
    std::string out;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i++]);
        if (c == '"') {
            return out;
        }
        if (c != '\\') {
            out.push_back(static_cast<char>(c));
            continue;
        }
        if (i >= s.size()) {
            throw std::runtime_error("bad json escape");
        }
        char e = s[i++];
        switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > s.size()) {
                    throw std::runtime_error("short json unicode escape");
                }
                uint32_t cp = parse_hex4(s, i);
                i += 4;
                if (cp >= 0xd800 && cp <= 0xdbff && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                    i += 2;
                    uint32_t lo = parse_hex4(s, i);
                    i += 4;
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                }
                append_utf8(cp, out);
                break;
            }
            default:
                throw std::runtime_error("unsupported json escape");
        }
    }
    throw std::runtime_error("unterminated json string");
}

inline void skip_json_ws(const std::string &s, size_t &i) {
    while (i < s.size()) {
        char c = s[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            ++i;
        } else {
            break;
        }
    }
}

inline std::vector<std::string> make_byte_encoder() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::vector<std::string> enc(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        append_utf8(static_cast<uint32_t>(cs[i]), enc[static_cast<size_t>(bs[i])]);
    }
    return enc;
}

inline uint32_t decode_utf8_one(const std::string &s, size_t &i, std::string *bytes = nullptr) {
    const size_t start = i;
    unsigned char c = static_cast<unsigned char>(s[i++]);
    uint32_t cp = c;
    if (c < 0x80) {
        cp = c;
    } else if ((c >> 5) == 0x6 && i < s.size()) {
        const uint32_t b1 = static_cast<unsigned char>(s[i++]) & 0x3f;
        cp = ((c & 0x1f) << 6) | b1;
    } else if ((c >> 4) == 0xe && i + 1 < s.size()) {
        const uint32_t b1 = static_cast<unsigned char>(s[i++]) & 0x3f;
        const uint32_t b2 = static_cast<unsigned char>(s[i++]) & 0x3f;
        cp = ((c & 0x0f) << 12) | (b1 << 6) | b2;
    } else if ((c >> 3) == 0x1e && i + 2 < s.size()) {
        const uint32_t b1 = static_cast<unsigned char>(s[i++]) & 0x3f;
        const uint32_t b2 = static_cast<unsigned char>(s[i++]) & 0x3f;
        const uint32_t b3 = static_cast<unsigned char>(s[i++]) & 0x3f;
        cp = ((c & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3;
    }
    if (bytes) {
        *bytes = s.substr(start, i - start);
    }
    return cp;
}

inline bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\n' || cp == '\r' || cp == '\t' || cp == '\v' || cp == '\f';
}

inline bool is_crlf_cp(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

inline bool is_digit_cp(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

inline bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
        return true;
    }
    if ((cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
        (cp >= 0x3040 && cp <= 0x30ff) || (cp >= 0xac00 && cp <= 0xd7af)) {
        return true;
    }
    if (cp >= 0x80 && !is_space_cp(cp) &&
        !(cp >= 0x3000 && cp <= 0x303f) &&
        !(cp >= 0xff00 && cp <= 0xff65)) {
        return true;
    }
    return false;
}

enum class CharKind {
    Space,
    Digit,
    Letter,
    Punct,
};

inline CharKind char_kind(uint32_t cp) {
    if (is_space_cp(cp)) return CharKind::Space;
    if (is_digit_cp(cp)) return CharKind::Digit;
    if (is_letter_cp(cp)) return CharKind::Letter;
    return CharKind::Punct;
}

inline char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline std::string lower_ascii(std::string s) {
    for (char &c : s) {
        c = ascii_lower(c);
    }
    return s;
}

inline int codec_language_id_for(const std::string &language) {
    const std::string l = lower_ascii(language);
    if (l.empty() || l == "auto") {
        return 0;
    }
    if (l == "chinese" || l == "zh" || l == "zh-cn") {
        return CODEC_LANGUAGE_CHINESE;
    }
    if (l == "english" || l == "en" || l == "en-us") {
        return CODEC_LANGUAGE_ENGLISH;
    }
    throw std::runtime_error("unsupported --language: " + language);
}

inline std::vector<std::pair<uint32_t, std::string>> utf8_chars(const std::string &s) {
    std::vector<std::pair<uint32_t, std::string>> out;
    size_t i = 0;
    while (i < s.size()) {
        std::string b;
        uint32_t cp = decode_utf8_one(s, i, &b);
        out.emplace_back(cp, std::move(b));
    }
    return out;
}

inline std::vector<std::string> split_utf8_strings(const std::string &s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        std::string b;
        (void)decode_utf8_one(s, i, &b);
        out.push_back(std::move(b));
    }
    return out;
}

class BpeTokenizer {
public:
    BpeTokenizer(const std::string &vocab_path, const std::string &merges_path) {
        byte_encoder_ = make_byte_encoder();
        load_vocab(vocab_path);
        load_merges(merges_path);
        specials_["<|im_start|>"] = TEXT_IM_START;
        specials_["<|im_end|>"] = TEXT_IM_END;
    }

    std::vector<int64_t> encode(const std::string &text) const {
        std::vector<int64_t> out;
        size_t pos = 0;
        while (pos < text.size()) {
            std::string special;
            int special_id = -1;
            for (const auto &kv : specials_) {
                const std::string &tok = kv.first;
                if (text.compare(pos, tok.size(), tok) == 0) {
                    if (tok.size() > special.size()) {
                        special = tok;
                        special_id = kv.second;
                    }
                }
            }
            if (special_id >= 0) {
                out.push_back(special_id);
                pos += special.size();
                continue;
            }

            size_t next = text.size();
            for (const auto &kv : specials_) {
                size_t p = text.find(kv.first, pos);
                if (p != std::string::npos) {
                    next = std::min(next, p);
                }
            }
            encode_plain(text.substr(pos, next - pos), out);
            pos = next;
        }
        return out;
    }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<std::string, int> ranks_;
    std::unordered_map<std::string, int> specials_;
    std::vector<std::string> byte_encoder_;

    static std::string pair_key(const std::string &a, const std::string &b) {
        return a + "\001" + b;
    }

    void load_vocab(const std::string &path) {
        const std::string s = read_text_file(path);
        size_t i = 0;
        skip_json_ws(s, i);
        if (i >= s.size() || s[i] != '{') {
            throw std::runtime_error("bad vocab json");
        }
        ++i;
        while (i < s.size()) {
            skip_json_ws(s, i);
            if (i < s.size() && s[i] == '}') {
                break;
            }
            std::string key = parse_json_string(s, i);
            skip_json_ws(s, i);
            if (i >= s.size() || s[i++] != ':') {
                throw std::runtime_error("bad vocab json colon");
            }
            skip_json_ws(s, i);
            bool neg = false;
            if (s[i] == '-') {
                neg = true;
                ++i;
            }
            int64_t val = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                val = val * 10 + (s[i++] - '0');
            }
            vocab_[key] = neg ? -val : val;
            skip_json_ws(s, i);
            if (i < s.size() && s[i] == ',') {
                ++i;
            }
        }
    }

    void load_merges(const std::string &path) {
        const std::string s = read_text_file(path);
        size_t line_begin = 0;
        int rank = 0;
        while (line_begin < s.size()) {
            size_t line_end = s.find('\n', line_begin);
            if (line_end == std::string::npos) {
                line_end = s.size();
            }
            std::string line = s.substr(line_begin, line_end - line_begin);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty() && line[0] != '#') {
                size_t sp = line.find(' ');
                if (sp != std::string::npos && sp + 1 < line.size()) {
                    ranks_[pair_key(line.substr(0, sp), line.substr(sp + 1))] = rank++;
                }
            }
            line_begin = line_end + 1;
        }
    }

    std::vector<std::string> pre_tokenize(const std::string &text) const {
        auto chars = utf8_chars(text);
        std::vector<std::string> toks;
        size_t i = 0;
        while (i < chars.size()) {
            if (chars[i].first == '\'' && i + 1 < chars.size()) {
                std::string suf;
                size_t j = i + 1;
                while (j < chars.size() && chars[j].first < 128 && is_letter_cp(chars[j].first) &&
                       suf.size() < 2) {
                    suf.push_back(ascii_lower(chars[j].second[0]));
                    ++j;
                }
                const bool one = suf == "s" || suf == "t" || suf == "m" || suf == "d";
                const bool two = suf == "re" || suf == "ve" || suf == "ll";
                if ((one && j == i + 2) || (two && j == i + 3)) {
                    std::string tok;
                    for (size_t k = i; k < j; ++k) {
                        tok += chars[k].second;
                    }
                    toks.push_back(std::move(tok));
                    i = j;
                    continue;
                }
            }

            size_t j = i;
            std::string tok;
            if (!is_crlf_cp(chars[j].first) && !is_letter_cp(chars[j].first) && !is_digit_cp(chars[j].first) &&
                j + 1 < chars.size() && is_letter_cp(chars[j + 1].first)) {
                tok += chars[j].second;
                ++j;
            }
            if (j < chars.size() && is_letter_cp(chars[j].first)) {
                while (j < chars.size() && is_letter_cp(chars[j].first)) {
                    tok += chars[j].second;
                    ++j;
                }
                toks.push_back(std::move(tok));
                i = j;
                continue;
            }

            if (is_digit_cp(chars[i].first)) {
                toks.push_back(chars[i].second);
                ++i;
                continue;
            }

            j = i;
            tok.clear();
            if (chars[j].first == ' ' && j + 1 < chars.size() &&
                !is_space_cp(chars[j + 1].first) && !is_letter_cp(chars[j + 1].first) &&
                !is_digit_cp(chars[j + 1].first)) {
                tok += chars[j].second;
                ++j;
            }
            if (j < chars.size() && !is_space_cp(chars[j].first) &&
                !is_letter_cp(chars[j].first) && !is_digit_cp(chars[j].first)) {
                while (j < chars.size() && !is_space_cp(chars[j].first) &&
                       !is_letter_cp(chars[j].first) && !is_digit_cp(chars[j].first)) {
                    tok += chars[j].second;
                    ++j;
                }
                while (j < chars.size() && is_crlf_cp(chars[j].first)) {
                    tok += chars[j].second;
                    ++j;
                }
                toks.push_back(std::move(tok));
                i = j;
                continue;
            }

            j = i;
            tok.clear();
            while (j < chars.size() && is_space_cp(chars[j].first) && !is_crlf_cp(chars[j].first)) {
                tok += chars[j].second;
                ++j;
            }
            if (j < chars.size() && is_crlf_cp(chars[j].first)) {
                while (j < chars.size() && is_crlf_cp(chars[j].first)) {
                    tok += chars[j].second;
                    ++j;
                }
                toks.push_back(std::move(tok));
                i = j;
                continue;
            }

            tok.clear();
            while (i < chars.size() && is_space_cp(chars[i].first)) {
                tok += chars[i].second;
                ++i;
            }
            if (tok.empty()) {
                tok = chars[i].second;
                ++i;
            }
            toks.push_back(std::move(tok));
        }
        return toks;
    }

    std::vector<std::string> bpe(const std::string &token) const {
        std::string encoded;
        encoded.reserve(token.size() * 2);
        for (unsigned char c : token) {
            encoded += byte_encoder_[c];
        }
        std::vector<std::string> word = split_utf8_strings(encoded);
        if (word.size() <= 1) {
            return word;
        }
        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = ranks_.find(pair_key(word[i], word[i + 1]));
                if (it != ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best = i;
                }
            }
            if (best == std::numeric_limits<size_t>::max()) {
                break;
            }
            std::vector<std::string> merged;
            merged.reserve(word.size() - 1);
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && i == best) {
                    merged.push_back(word[i] + word[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(word[i]);
                    ++i;
                }
            }
            word.swap(merged);
        }
        return word;
    }

    void encode_plain(const std::string &text, std::vector<int64_t> &out) const {
        for (const auto &tok : pre_tokenize(text)) {
            for (const auto &piece : bpe(tok)) {
                auto it = vocab_.find(piece);
                if (it == vocab_.end()) {
                    throw std::runtime_error("token not found in vocab");
                }
                out.push_back(it->second);
            }
        }
    }
};

inline std::vector<float> load_gguf_tensor_f32(const std::string &path, const std::string &name, size_t count) {
    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context *ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("open gguf failed: " + path);
    }
    int64_t tid = gguf_find_tensor(ctx, name.c_str());
    if (tid < 0) {
        gguf_free(ctx);
        throw std::runtime_error("tensor not found in gguf: " + name);
    }
    if (gguf_get_tensor_type(ctx, tid) != GGML_TYPE_F32 ||
        gguf_get_tensor_size(ctx, tid) != count * sizeof(float)) {
        gguf_free(ctx);
        throw std::runtime_error("bad gguf tensor shape/type: " + name);
    }
    const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tid);
    std::vector<float> out(count);
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        gguf_free(ctx);
        throw std::runtime_error("open gguf data failed: " + path);
    }
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0 ||
        std::fread(out.data(), sizeof(float), count, f) != count) {
        std::fclose(f);
        gguf_free(ctx);
        throw std::runtime_error("read gguf tensor failed: " + name);
    }
    std::fclose(f);
    gguf_free(ctx);
    return out;
}

inline std::vector<float> run_text_embed(Ort::Env &env,
                                         const std::string &onnx_path,
                                         const std::vector<int64_t> &ids,
                                         int threads) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::max(1, threads));
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    Ort::Session sess(env, onnx_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;
    auto in = sess.GetInputNameAllocated(0, allocator);
    auto out = sess.GetOutputNameAllocated(0, allocator);
    std::array<int64_t, 2> shape {1, static_cast<int64_t>(ids.size())};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<int64_t>(mem,
                                                   const_cast<int64_t *>(ids.data()),
                                                   ids.size(),
                                                   shape.data(),
                                                   shape.size());
    const char *in_names[] = {in.get()};
    const char *out_names[] = {out.get()};
    auto outputs = sess.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    float *p = outputs[0].GetTensorMutableData<float>();
    const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (n != ids.size() * static_cast<size_t>(HID)) {
        throw std::runtime_error("text_embed_proj output shape mismatch");
    }
    return std::vector<float>(p, p + n);
}

inline std::vector<float> run_text_embed_session(Ort::Session &sess,
                                                 const std::string &input_name,
                                                 const std::string &output_name,
                                                 const std::vector<int64_t> &ids) {
    std::array<int64_t, 2> shape {1, static_cast<int64_t>(ids.size())};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<int64_t>(mem,
                                                   const_cast<int64_t *>(ids.data()),
                                                   ids.size(),
                                                   shape.data(),
                                                   shape.size());
    const char *in_names[] = {input_name.c_str()};
    const char *out_names[] = {output_name.c_str()};
    auto outputs = sess.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    float *p = outputs[0].GetTensorMutableData<float>();
    const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (n != ids.size() * static_cast<size_t>(HID)) {
        throw std::runtime_error("text_embed_proj output shape mismatch");
    }
    return std::vector<float>(p, p + n);
}

inline std::vector<float> read_wav_mono_24k(const std::string &path) {
    const auto d = read_bytes_file(path);
    if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) != 0 || std::memcmp(d.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("expected RIFF/WAVE wav: " + path);
    }
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits = 0;
    size_t data_pos = 0;
    size_t data_bytes = 0;
    size_t p = 12;
    while (p + 8 <= d.size()) {
        const char *id = reinterpret_cast<const char *>(d.data() + p);
        uint32_t sz = static_cast<uint32_t>(d[p + 4]) |
                      (static_cast<uint32_t>(d[p + 5]) << 8) |
                      (static_cast<uint32_t>(d[p + 6]) << 16) |
                      (static_cast<uint32_t>(d[p + 7]) << 24);
        p += 8;
        if (p + sz > d.size()) {
            throw std::runtime_error("bad wav chunk size");
        }
        if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
            audio_format = static_cast<uint16_t>(d[p] | (d[p + 1] << 8));
            channels = static_cast<uint16_t>(d[p + 2] | (d[p + 3] << 8));
            sample_rate = static_cast<uint32_t>(d[p + 4]) |
                          (static_cast<uint32_t>(d[p + 5]) << 8) |
                          (static_cast<uint32_t>(d[p + 6]) << 16) |
                          (static_cast<uint32_t>(d[p + 7]) << 24);
            bits = static_cast<uint16_t>(d[p + 14] | (d[p + 15] << 8));
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_pos = p;
            data_bytes = sz;
        }
        p += sz + (sz & 1u);
    }
    if (audio_format != 1 || channels < 1 || bits != 16 || data_pos == 0) {
        throw std::runtime_error("ref wav must be PCM16");
    }
    const size_t frames = data_bytes / (2 * channels);
    std::vector<float> audio(frames);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const size_t p16 = data_pos + (i * channels + ch) * 2;
            int16_t s = static_cast<int16_t>(d[p16] | (d[p16 + 1] << 8));
            sum += static_cast<float>(s) / 32768.0f;
        }
        audio[i] = sum / channels;
    }
    if (sample_rate == 24000) {
        return audio;
    }
    if (sample_rate == 0) {
        throw std::runtime_error("bad wav sample rate");
    }
    const size_t out_frames = std::max<size_t>(1, static_cast<size_t>(
        std::llround(static_cast<double>(audio.size()) * 24000.0 / sample_rate)));
    std::vector<float> resampled(out_frames);
    for (size_t i = 0; i < out_frames; ++i) {
        const double src = static_cast<double>(i) * sample_rate / 24000.0;
        const size_t j = std::min(static_cast<size_t>(src), audio.size() - 1);
        const size_t j1 = std::min(j + 1, audio.size() - 1);
        const float frac = static_cast<float>(src - j);
        resampled[i] = audio[j] * (1.0f - frac) + audio[j1] * frac;
    }
    return resampled;
}

inline double hz_to_mel(double hz) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (hz < min_log_hz) {
        return hz / f_sp;
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

inline double mel_to_hz(double mel) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (mel < min_log_mel) {
        return mel * f_sp;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

inline std::vector<float> mel_filterbank() {
    constexpr int n_fft = 1024;
    constexpr int n_mels = 128;
    constexpr int n_freq = n_fft / 2 + 1;
    std::vector<double> mel_pts(n_mels + 2);
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(12000.0);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    }
    std::vector<float> fb(n_mels * n_freq, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double left = mel_pts[m];
        const double center = mel_pts[m + 1];
        const double right = mel_pts[m + 2];
        const double enorm = 2.0 / (right - left);
        for (int f = 0; f < n_freq; ++f) {
            const double hz = 12000.0 * f / (n_freq - 1);
            double w = 0.0;
            if (hz >= left && hz <= center) {
                w = (hz - left) / (center - left);
            } else if (hz >= center && hz <= right) {
                w = (right - hz) / (right - center);
            }
            fb[static_cast<size_t>(m * n_freq + f)] = static_cast<float>(std::max(0.0, w) * enorm);
        }
    }
    return fb;
}

inline void fft1024(std::array<std::complex<float>, 1024> &a) {
    constexpr int n = 1024;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = a[i + j];
                std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

inline std::vector<float> wav_to_mel_128(const std::string &path) {
    constexpr int n_fft = 1024;
    constexpr int hop = 256;
    constexpr int pad = 384;
    constexpr int n_mels = 128;
    constexpr int n_freq = 513;
    auto audio = read_wav_mono_24k(path);
    if (audio.size() <= static_cast<size_t>(pad + 1)) {
        throw std::runtime_error("ref wav is too short");
    }
    const int n = static_cast<int>(audio.size());
    std::vector<float> padded(static_cast<size_t>(n + 2 * pad));
    for (int i = 0; i < static_cast<int>(padded.size()); ++i) {
        int j = i - pad;
        while (j < 0 || j >= n) {
            if (j < 0) {
                j = -j;
            } else {
                j = 2 * n - 2 - j;
            }
        }
        padded[static_cast<size_t>(i)] = audio[static_cast<size_t>(j)];
    }
    const int frames = 1 + (static_cast<int>(padded.size()) - n_fft) / hop;
    std::vector<float> mel(static_cast<size_t>(frames * n_mels), 0.0f);
    std::array<float, n_fft> window {};
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / n_fft);
    }
    const auto fb = mel_filterbank();
    std::array<std::complex<float>, n_fft> buf {};
    std::array<float, n_freq> mag {};
    for (int t = 0; t < frames; ++t) {
        const int off = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            buf[i] = std::complex<float>(padded[static_cast<size_t>(off + i)] * window[i], 0.0f);
        }
        fft1024(buf);
        for (int f = 0; f < n_freq; ++f) {
            const float re = buf[f].real();
            const float im = buf[f].imag();
            mag[f] = std::sqrt(re * re + im * im + 1e-9f);
        }
        for (int m = 0; m < n_mels; ++m) {
            double v = 0.0;
            for (int f = 0; f < n_freq; ++f) {
                v += static_cast<double>(fb[static_cast<size_t>(m * n_freq + f)]) * mag[f];
            }
            mel[static_cast<size_t>(t * n_mels + m)] = std::log(static_cast<float>(std::max(v, 1e-5)));
        }
    }
    return mel;
}

inline std::vector<float> run_speaker_encoder(Ort::Env &env,
                                              const std::string &onnx_path,
                                              const std::string &wav_path,
                                              int threads) {
    auto mel = wav_to_mel_128(wav_path);
    const int64_t frames = static_cast<int64_t>(mel.size() / 128);
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::max(1, threads));
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    Ort::Session sess(env, onnx_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;
    auto in = sess.GetInputNameAllocated(0, allocator);
    auto out = sess.GetOutputNameAllocated(0, allocator);
    std::array<int64_t, 3> shape {1, frames, 128};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(), shape.data(), shape.size());
    const char *in_names[] = {in.get()};
    const char *out_names[] = {out.get()};
    auto outputs = sess.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    float *p = outputs[0].GetTensorMutableData<float>();
    const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (n != HID) {
        throw std::runtime_error("speaker encoder output shape mismatch");
    }
    return std::vector<float>(p, p + n);
}

inline std::vector<std::array<int32_t, CODE_GROUPS>> run_codec_encoder(Ort::Env &env,
                                                                        const std::string &onnx_path,
                                                                        const std::string &wav_path,
                                                                        int threads) {
    auto audio = read_wav_mono_24k(wav_path);
    if (audio.empty()) {
        throw std::runtime_error("ref wav is empty");
    }
    const size_t valid_samples = std::min(audio.size(), static_cast<size_t>(REF_CODEC_SAMPLES));
    const int valid_frames = std::max(1, static_cast<int>(
        (valid_samples + REF_CODEC_SAMPLES_PER_FRAME - 1) / REF_CODEC_SAMPLES_PER_FRAME));

    std::vector<float> input(REF_CODEC_SAMPLES, 0.0f);
    std::copy_n(audio.data(), valid_samples, input.data());

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::max(1, threads));
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    Ort::Session sess(env, onnx_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;
    auto in = sess.GetInputNameAllocated(0, allocator);
    auto out = sess.GetOutputNameAllocated(0, allocator);
    std::array<int64_t, 2> shape {1, REF_CODEC_SAMPLES};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());
    const char *in_names[] = {in.get()};
    const char *out_names[] = {out.get()};
    auto outputs = sess.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    int64_t *p = outputs[0].GetTensorMutableData<int64_t>();
    const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (n != 100UL * CODE_GROUPS) {
        throw std::runtime_error("codec encoder output shape mismatch");
    }
    const int frames = std::min(valid_frames, 100);
    std::vector<std::array<int32_t, CODE_GROUPS>> codes(static_cast<size_t>(frames));
    for (int t = 0; t < frames; ++t) {
        for (int g = 0; g < CODE_GROUPS; ++g) {
            const int64_t v = p[static_cast<size_t>(t * CODE_GROUPS + g)];
            if (v < 0 || v >= CODE_VOCAB) {
                throw std::runtime_error("codec encoder produced out-of-range code");
            }
            codes[static_cast<size_t>(t)][g] = static_cast<int32_t>(v);
        }
    }
    return codes;
}

inline uint32_t read_u32_le(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline void append_u32_le(std::vector<uint8_t> &out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

inline void append_raw(std::vector<uint8_t> &out, const void *p, size_t n) {
    const auto *b = reinterpret_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);
}

inline std::vector<float> read_speaker_bin(const std::string &path) {
    auto bytes = read_bytes_file(path);
    if (bytes.size() != static_cast<size_t>(HID) * sizeof(float)) {
        throw std::runtime_error("speaker bin must be raw float32[1024]: " + path);
    }
    std::vector<float> out(HID);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

inline void write_speaker_bin(const std::string &path, const std::vector<float> &spk) {
    if (spk.size() != HID) {
        throw std::runtime_error("speaker bin expects float32[1024]");
    }
    write_bytes_file(path, reinterpret_cast<const uint8_t *>(spk.data()), spk.size() * sizeof(float));
}

inline ReferencePrompt read_reference_prompt_bin(const std::string &path) {
    auto bytes = read_bytes_file(path);
    if (bytes.size() == static_cast<size_t>(HID) * sizeof(float)) {
        ReferencePrompt prompt;
        prompt.speaker.resize(HID);
        std::memcpy(prompt.speaker.data(), bytes.data(), bytes.size());
        return prompt;
    }

    static constexpr char kMagic[8] = {'Q', '3', 'T', 'P', 'R', 'M', 'P', 'T'};
    if (bytes.size() < 40 || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("bad reference prompt bin: " + path);
    }
    size_t p = sizeof(kMagic);
    const uint32_t version = read_u32_le(bytes.data() + p); p += 4;
    const uint32_t hid = read_u32_le(bytes.data() + p); p += 4;
    const uint32_t groups = read_u32_le(bytes.data() + p); p += 4;
    const uint32_t frames = read_u32_le(bytes.data() + p); p += 4;
    const uint32_t text_bytes = read_u32_le(bytes.data() + p); p += 4;
    p += 12;  // reserved
    if (version != 1 || hid != HID || groups != CODE_GROUPS || frames == 0 || text_bytes == 0) {
        throw std::runtime_error("unsupported reference prompt bin: " + path);
    }
    const size_t need = p + static_cast<size_t>(HID) * sizeof(float) +
                        static_cast<size_t>(frames) * CODE_GROUPS * sizeof(int32_t) +
                        static_cast<size_t>(text_bytes);
    if (need != bytes.size()) {
        throw std::runtime_error("reference prompt bin size mismatch: " + path);
    }

    ReferencePrompt prompt;
    prompt.speaker.resize(HID);
    std::memcpy(prompt.speaker.data(), bytes.data() + p, static_cast<size_t>(HID) * sizeof(float));
    p += static_cast<size_t>(HID) * sizeof(float);
    prompt.ref_codes.resize(frames);
    std::memcpy(prompt.ref_codes.data(), bytes.data() + p,
                static_cast<size_t>(frames) * CODE_GROUPS * sizeof(int32_t));
    p += static_cast<size_t>(frames) * CODE_GROUPS * sizeof(int32_t);
    prompt.ref_text.assign(reinterpret_cast<const char *>(bytes.data() + p), text_bytes);
    return prompt;
}

inline void write_reference_prompt_bin(const std::string &path,
                                       const std::vector<float> &spk,
                                       const std::string &ref_text,
                                       const std::vector<std::array<int32_t, CODE_GROUPS>> &ref_codes) {
    if (spk.size() != HID) {
        throw std::runtime_error("reference prompt expects speaker float32[1024]");
    }
    if (ref_text.empty()) {
        throw std::runtime_error("reference prompt expects non-empty ref_text");
    }
    if (ref_codes.empty()) {
        throw std::runtime_error("reference prompt expects non-empty ref_codes");
    }

    std::vector<uint8_t> out;
    static constexpr char kMagic[8] = {'Q', '3', 'T', 'P', 'R', 'M', 'P', 'T'};
    append_raw(out, kMagic, sizeof(kMagic));
    append_u32_le(out, 1);
    append_u32_le(out, HID);
    append_u32_le(out, CODE_GROUPS);
    append_u32_le(out, static_cast<uint32_t>(ref_codes.size()));
    append_u32_le(out, static_cast<uint32_t>(ref_text.size()));
    append_u32_le(out, 0);
    append_u32_le(out, 0);
    append_u32_le(out, 0);
    append_raw(out, spk.data(), spk.size() * sizeof(float));
    append_raw(out, ref_codes.data(), ref_codes.size() * CODE_GROUPS * sizeof(int32_t));
    append_raw(out, ref_text.data(), ref_text.size());
    write_bytes_file(path, out.data(), out.size());
}

inline std::string speaker_encoder_path(const std::string &model_dir) {
    return first_existing({
        path_join(path_join(model_dir, "onnx"), "speaker_encoder.onnx"),
        path_join(model_dir, "speaker_encoder.onnx"),
        "speaker_encoder.onnx",
    });
}

inline std::string codec_encoder_path(const std::string &model_dir) {
    return first_existing({
        path_join(path_join(model_dir, "onnx"), "codec_encoder.onnx"),
        path_join(model_dir, "codec_encoder.onnx"),
        "codec_encoder.onnx",
    });
}

inline void append_vec(std::vector<float> &dst, const float *src) {
    dst.insert(dst.end(), src, src + HID);
}

inline void append_sum(std::vector<float> &dst, const float *a, const float *b) {
    const size_t off = dst.size();
    dst.resize(off + HID);
    for (int i = 0; i < HID; ++i) {
        dst[off + i] = a[i] + b[i];
    }
}

inline std::pair<std::string, std::string> tokenizer_paths(const std::string &model_dir) {
    const std::string tokenizer_dir = path_join(model_dir, "tokenizer");
    const std::string vocab = first_existing({
        path_join(tokenizer_dir, "vocab.json"),
        path_join(model_dir, "vocab.json"),
    });
    const std::string merges = first_existing({
        path_join(tokenizer_dir, "merges.txt"),
        path_join(model_dir, "merges.txt"),
    });
    return {vocab, merges};
}

inline std::string text_embed_path(const std::string &model_dir) {
    std::vector<std::string> candidates = {
        path_join(path_join(model_dir, "onnx"), "text_embed_proj.onnx"),
        "text_embed_proj.onnx",
    };
    if (std::getenv("Q3TTS_ALLOW_DYNQ_TEXT")) {
        candidates.push_back(path_join(path_join(model_dir, "onnx"), "text_embed_proj.dynq.onnx"));
        candidates.push_back("text_embed_proj.dynq.onnx");
    }
    return first_existing(candidates);
}

inline std::string talker_gguf_path(const std::string &model_dir, const std::string &talker_gguf) {
    return first_existing({
        path_join(path_join(model_dir, "gguf"), talker_gguf),
        path_join(model_dir, talker_gguf),
        talker_gguf,
    });
}

inline std::string cp_gguf_path(const std::string &model_dir, const std::string &cp_gguf) {
    return first_existing({
        path_join(path_join(model_dir, "gguf"), cp_gguf),
        path_join(model_dir, cp_gguf),
        cp_gguf,
    });
}

inline std::vector<int64_t> tokenize_prompt(const std::string &model_dir, const std::string &text) {
    const auto paths = tokenizer_paths(model_dir);
    const std::string &vocab = paths.first;
    const std::string &merges = paths.second;
    BpeTokenizer tok(vocab, merges);
    return tok.encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
}

class FrontendRuntime {
public:
    FrontendRuntime(Ort::Env &env, const FrontendConfig &cfg)
        : env_(env),
          model_dir_(cfg.model_dir),
          language_(cfg.language),
          talker_gguf_name_(cfg.talker_gguf),
          cp_gguf_name_(cfg.cp_gguf),
          frontend_threads_(std::max(1, cfg.frontend_threads)) {
        (void)codec_language_id_for(language_);

        const auto tok_paths = tokenizer_paths(model_dir_);
        tokenizer_ = std::make_unique<BpeTokenizer>(tok_paths.first, tok_paths.second);

        text_onnx_ = text_embed_path(model_dir_);
        talker_gguf_ = talker_gguf_path(model_dir_, talker_gguf_name_);
        cp_gguf_ = cp_gguf_path(model_dir_, cp_gguf_name_);

        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(frontend_threads_);
        so.AddConfigEntry("session.intra_op.allow_spinning", "0");
        text_session_ = std::make_unique<Ort::Session>(env_, text_onnx_.c_str(), so);

        Ort::AllocatorWithDefaultOptions allocator;
        auto in = text_session_->GetInputNameAllocated(0, allocator);
        auto out = text_session_->GetOutputNameAllocated(0, allocator);
        text_input_name_ = in.get();
        text_output_name_ = out.get();

        tts_h_ = run_text_embed_session(
            *text_session_, text_input_name_, text_output_name_,
            std::vector<int64_t>{TTS_BOS, TTS_EOS, TTS_PAD});
        if (tts_h_.size() != 3UL * HID) {
            throw std::runtime_error("tts embed output shape mismatch");
        }
        codec_ = load_gguf_tensor_f32(talker_gguf_, "q3tts.codec_embedding.weight", 3072UL * HID);
    }

    FrontendInput build(const FrontendConfig &cfg, std::vector<int64_t> *ids_out = nullptr) {
        if (cfg.model_dir != model_dir_ || cfg.language != language_ ||
            cfg.talker_gguf != talker_gguf_name_ || cfg.cp_gguf != cp_gguf_name_) {
            throw std::runtime_error("FrontendRuntime used with different model config");
        }

        std::vector<int64_t> ids = tokenizer_->encode(
            "<|im_start|>assistant\n" + cfg.text + "<|im_end|>\n<|im_start|>assistant\n");
        if (ids.size() < 9) {
            throw std::runtime_error("tokenized prompt is too short");
        }
        if (ids_out) {
            *ids_out = ids;
        }

        auto text_h = run_text_embed_session(*text_session_, text_input_name_, text_output_name_, ids);
        const float *bos_e = tts_h_.data();
        const float *eos_e = tts_h_.data() + HID;
        const float *pad_e = tts_h_.data() + 2 * HID;

        auto ce = [&](int id) -> const float * {
            if (id < 0 || id >= 3072) {
                throw std::runtime_error("codec id out of range");
            }
            return codec_.data() + static_cast<size_t>(id) * HID;
        };

        std::vector<float> ctrl;
        auto add_ctrl = [&](const float *v) {
            ctrl.insert(ctrl.end(), v, v + HID);
        };
        const int language_id = codec_language_id_for(language_);
        if (language_id == 0) {
            add_ctrl(ce(CODEC_NOTHINK));
            add_ctrl(ce(CODEC_THINK_BOS));
            add_ctrl(ce(CODEC_THINK_EOS));
        } else {
            add_ctrl(ce(CODEC_THINK));
            add_ctrl(ce(CODEC_THINK_BOS));
            add_ctrl(ce(language_id));
            add_ctrl(ce(CODEC_THINK_EOS));
        }
        ReferencePrompt ref_prompt;
        if (!cfg.ref_bin.empty() || !cfg.ref_wav.empty()) {
            std::vector<float> spk;
            if (!cfg.ref_bin.empty()) {
                ref_prompt = read_reference_prompt_bin(cfg.ref_bin);
                spk = ref_prompt.speaker;
            } else {
                spk = run_speaker_encoder(env_, speaker_encoder_path(model_dir_), cfg.ref_wav, frontend_threads_);
            }
            add_ctrl(spk.data());
        }
        add_ctrl(ce(CODEC_PAD));
        add_ctrl(ce(CODEC_BOS));
        const int ctrl_n = static_cast<int>(ctrl.size() / HID);

        FrontendInput out;
        out.pad.assign(pad_e, pad_e + HID);

        for (int i = 0; i < 3; ++i) {
            append_vec(out.prefill, text_h.data() + static_cast<size_t>(i) * HID);
        }

        if (ref_prompt.full_prompt()) {
            for (int i = 0; i < ctrl_n - 1; ++i) {
                const float *base = (i == ctrl_n - 2) ? bos_e : pad_e;
                append_sum(out.prefill, base, ctrl.data() + static_cast<size_t>(i) * HID);
            }

            std::vector<int64_t> ref_ids = tokenizer_->encode(
                "<|im_start|>assistant\n" + ref_prompt.ref_text + "<|im_end|>\n");
            if (ref_ids.size() < 6 || ids.size() < 9) {
                throw std::runtime_error("tokenized full prompt is too short");
            }
            std::vector<int64_t> icl_ids;
            for (size_t i = 3; i < ref_ids.size() - 2; ++i) {
                icl_ids.push_back(ref_ids[i]);
            }
            for (size_t i = 3; i < ids.size() - 5; ++i) {
                icl_ids.push_back(ids[i]);
            }
            if (icl_ids.empty()) {
                throw std::runtime_error("empty ICL text prompt");
            }
            auto icl_text_h = run_text_embed_session(*text_session_, text_input_name_, text_output_name_, icl_ids);
            ensure_cp_embeddings();

            std::vector<float> codec_embed;
            codec_embed.reserve((ref_prompt.ref_codes.size() + 1) * HID);
            append_vec(codec_embed, ce(CODEC_BOS));
            for (const auto &frame : ref_prompt.ref_codes) {
                const size_t off = codec_embed.size();
                codec_embed.resize(off + HID, 0.0f);
                const int c0 = frame[0];
                if (c0 < 0 || c0 >= CODE_VOCAB) {
                    throw std::runtime_error("ref code out of range");
                }
                const float *base = ce(c0);
                for (int h = 0; h < HID; ++h) {
                    codec_embed[off + h] += base[h];
                }
                for (int g = 1; g < CODE_GROUPS; ++g) {
                    const int c = frame[g];
                    if (c < 0 || c >= CODE_VOCAB) {
                        throw std::runtime_error("ref code out of range");
                    }
                    const float *emb = cp_embeddings_[static_cast<size_t>(g - 1)].data() +
                                       static_cast<size_t>(c) * HID;
                    for (int h = 0; h < HID; ++h) {
                        codec_embed[off + h] += emb[h];
                    }
                }
            }

            const size_t text_lens = icl_ids.size() + 1;
            const size_t codec_lens = ref_prompt.ref_codes.size() + 1;
            auto text_ptr = [&](size_t i) -> const float * {
                return (i == icl_ids.size()) ? eos_e : (icl_text_h.data() + i * HID);
            };
            if (cfg.full_prompt_non_streaming) {
                const float *codec_pad = ce(CODEC_PAD);
                for (size_t i = 0; i < text_lens; ++i) {
                    append_sum(out.prefill, text_ptr(i), codec_pad);
                }
                for (size_t i = 0; i < codec_lens; ++i) {
                    append_sum(out.prefill, pad_e, codec_embed.data() + i * HID);
                }
                append_vec(out.trailing, pad_e);
            } else {
                const size_t paired = std::min(text_lens, codec_lens);
                for (size_t i = 0; i < paired; ++i) {
                    append_sum(out.prefill, text_ptr(i), codec_embed.data() + i * HID);
                }
                for (size_t i = paired; i < codec_lens; ++i) {
                    append_sum(out.prefill, pad_e, codec_embed.data() + i * HID);
                }
                if (text_lens > codec_lens) {
                    for (size_t i = codec_lens; i < text_lens; ++i) {
                        append_vec(out.trailing, text_ptr(i));
                    }
                } else {
                    append_vec(out.trailing, pad_e);
                }
            }
        } else {
            for (int i = 0; i < ctrl_n - 1; ++i) {
                const float *base = (i == ctrl_n - 2) ? bos_e : pad_e;
                append_sum(out.prefill, base, ctrl.data() + static_cast<size_t>(i) * HID);
            }
            append_sum(out.prefill, text_h.data() + 3UL * HID, ctrl.data() + static_cast<size_t>(ctrl_n - 1) * HID);

            if (ids.size() > 9) {
                const size_t begin = 4;
                const size_t end = ids.size() - 5;
                for (size_t i = begin; i < end; ++i) {
                    append_vec(out.trailing, text_h.data() + i * HID);
                }
            }
            append_vec(out.trailing, eos_e);
        }
        out.n_prefill = static_cast<int64_t>(out.prefill.size() / HID);
        out.n_trailing = static_cast<int64_t>(out.trailing.size() / HID);
        return out;
    }

private:
    void ensure_cp_embeddings() {
        if (!cp_embeddings_.empty()) {
            return;
        }
        cp_embeddings_.reserve(CODE_GROUPS - 1);
        for (int i = 0; i < CODE_GROUPS - 1; ++i) {
            cp_embeddings_.push_back(load_gguf_tensor_f32(
                cp_gguf_,
                "q3tts.cp_embedding." + std::to_string(i) + ".weight",
                static_cast<size_t>(CODE_VOCAB) * HID));
        }
    }

    Ort::Env &env_;
    std::string model_dir_;
    std::string language_;
    std::string talker_gguf_name_;
    std::string cp_gguf_name_;
    int frontend_threads_ = 1;
    std::unique_ptr<BpeTokenizer> tokenizer_;
    std::string text_onnx_;
    std::string talker_gguf_;
    std::string cp_gguf_;
    std::unique_ptr<Ort::Session> text_session_;
    std::string text_input_name_;
    std::string text_output_name_;
    std::vector<float> tts_h_;
    std::vector<float> codec_;
    std::vector<std::vector<float>> cp_embeddings_;
};

inline FrontendInput build(Ort::Env &env, const FrontendConfig &cfg, std::vector<int64_t> *ids_out = nullptr) {
    FrontendRuntime runtime(env, cfg);
    return runtime.build(cfg, ids_out);
}

}  // namespace q3tts_frontend
