#include <onnxruntime_cxx_api.h>

#include "qwen3_tts_runtime.h"

#include "q3tts_frontend.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <csignal>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#include "q3tts_audio_sdk.h"
#include "q3tts_codec_ort.h"

namespace {

using Clock = std::chrono::steady_clock;

std::string env_str(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

double env_double(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return fallback;
    }
    char *end = nullptr;
    const double parsed = std::strtod(v, &end);
    return end != v && std::isfinite(parsed) ? parsed : fallback;
}

void set_default_env(const char *name, const char *value) {
    if (!std::getenv(name)) {
        setenv(name, value, 0);
    }
}

void set_env_override(const char *name, const std::string &value) {
    setenv(name, value.c_str(), 1);
}

void set_default_runtime_env(const std::string &talker_gguf) {
    (void) talker_gguf;
    if (!std::getenv("Q3TTS_DISABLE_SWIGLU_DOWN_FUSION")) {
        set_default_env("GGML_CPU_FUSE_SWIGLU_DOWN_Q8", "1");
    }
    set_default_env("LLAMA_CTX_PAD", "16");
    set_default_env("SPACEMIT_Q4_HP_M1_N64", "1");
    set_default_env("Q3TTS_CP_CTX", "16");
    set_default_env("Q3TTS_CP_CTX_MIN", "16");
}

void profile_event(bool enabled, Clock::time_point origin, const std::string &msg) {
    if (!enabled) {
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
    const auto old_precision = std::cerr.precision();
    const auto old_flags = std::cerr.flags();
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(2);
    std::cerr << "[q3tts-profile +" << ms << "ms] " << msg << "\n";
    std::cerr.flags(old_flags);
    std::cerr.precision(old_precision);
}

bool exists(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

std::vector<int> parse_int_list(const std::string &raw, const std::vector<int> &fallback) {
    if (raw.empty()) {
        return fallback;
    }
    std::vector<int> out;
    std::string s = raw;
    std::replace(s.begin(), s.end(), ';', ',');
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }
    return out.empty() ? fallback : out;
}

std::vector<uint8_t> read_all(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open failed: " + path);
    }
    f.seekg(0, std::ios::end);
    const auto n = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(n);
    f.read(reinterpret_cast<char *>(data.data()), data.size());
    return data;
}

void write_all(const std::string &path, const uint8_t *data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("write failed: " + path);
    }
    f.write(reinterpret_cast<const char *>(data), n);
}

uint16_t le16(const std::vector<uint8_t> &d, size_t p) {
    return static_cast<uint16_t>(d[p]) | (static_cast<uint16_t>(d[p + 1]) << 8);
}

uint32_t le32(const std::vector<uint8_t> &d, size_t p) {
    return static_cast<uint32_t>(d[p]) |
           (static_cast<uint32_t>(d[p + 1]) << 8) |
           (static_cast<uint32_t>(d[p + 2]) << 16) |
           (static_cast<uint32_t>(d[p + 3]) << 24);
}

uint64_t le64(const std::vector<uint8_t> &d, size_t p) {
    return static_cast<uint64_t>(le32(d, p)) |
           (static_cast<uint64_t>(le32(d, p + 4)) << 32);
}

std::unordered_map<std::string, std::vector<uint8_t>> load_npz_stored(const std::string &path) {
    auto zip = read_all(path);
    std::unordered_map<std::string, std::vector<uint8_t>> out;
    size_t p = 0;
    while (p + 30 <= zip.size()) {
        const uint32_t sig = le32(zip, p);
        if (sig != 0x04034b50u) {
            break;
        }
        const uint16_t flags = le16(zip, p + 6);
        const uint16_t method = le16(zip, p + 8);
        uint64_t comp_size = le32(zip, p + 18);
        uint64_t uncomp_size = le32(zip, p + 22);
        const uint16_t name_len = le16(zip, p + 26);
        const uint16_t extra_len = le16(zip, p + 28);
        const size_t name_pos = p + 30;
        const size_t data_pos = name_pos + name_len + extra_len;
        if ((comp_size == 0xffffffffULL || uncomp_size == 0xffffffffULL) && name_pos + name_len + extra_len <= zip.size()) {
            size_t ep = name_pos + name_len;
            const size_t extra_end = ep + extra_len;
            while (ep + 4 <= extra_end) {
                const uint16_t header_id = le16(zip, ep);
                const uint16_t data_size = le16(zip, ep + 2);
                ep += 4;
                if (ep + data_size > extra_end) {
                    throw std::runtime_error("bad zip64 extra field in " + path);
                }
                if (header_id == 0x0001) {
                    size_t zp = ep;
                    if (uncomp_size == 0xffffffffULL && zp + 8 <= ep + data_size) {
                        uncomp_size = le64(zip, zp);
                        zp += 8;
                    }
                    if (comp_size == 0xffffffffULL && zp + 8 <= ep + data_size) {
                        comp_size = le64(zip, zp);
                    }
                    break;
                }
                ep += data_size;
            }
        }
        if (data_pos + comp_size > zip.size()) {
            throw std::runtime_error("bad zip entry size in " + path);
        }
        std::string name(reinterpret_cast<const char *>(&zip[name_pos]), name_len);
        if (method != 0 || (flags & 0x08u)) {
            throw std::runtime_error("npz entry is compressed or uses data descriptor: " + name);
        }
        out[name] = std::vector<uint8_t>(zip.begin() + data_pos, zip.begin() + data_pos + uncomp_size);
        p = data_pos + comp_size;
    }
    return out;
}

struct NpyArray {
    std::string descr;
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr;
    size_t bytes = 0;
};

std::string trim(std::string s) {
    auto is_ws = [](unsigned char c) { return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](char c) { return !is_ws(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](char c) { return !is_ws(c); }).base(), s.end());
    return s;
}

using Hotwords = std::vector<std::pair<std::string, std::string>>;

void add_hotword_spec(Hotwords &hotwords, const std::string &spec, const char *source) {
    const size_t eq = spec.find('=');
    if (eq == std::string::npos || eq == 0) {
        throw std::runtime_error(std::string("bad ") + source + " hotword, expected FROM=TO");
    }
    hotwords.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
}

void add_hotword_specs(Hotwords &hotwords, const std::string &specs, const char *source) {
    std::stringstream ss(specs);
    std::string item;
    while (std::getline(ss, item, ';')) {
        item = trim(item);
        if (!item.empty()) {
            add_hotword_spec(hotwords, item, source);
        }
    }
}

Hotwords env_hotwords() {
    Hotwords hotwords;
    if (const char *v = std::getenv("Q3TTS_HOTWORDS"); v && *v) {
        add_hotword_specs(hotwords, v, "Q3TTS_HOTWORDS");
    }
    return hotwords;
}

std::string apply_hotwords(std::string text, const Hotwords &hotwords) {
    for (const auto &hw : hotwords) {
        size_t pos = 0;
        while ((pos = text.find(hw.first, pos)) != std::string::npos) {
            text.replace(pos, hw.first.size(), hw.second);
            pos += hw.second.size();
        }
    }
    return text;
}

NpyArray parse_npy(const std::vector<uint8_t> &npy) {
    if (npy.size() < 16 || std::memcmp(npy.data(), "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("bad npy magic");
    }
    const uint8_t major = npy[6];
    size_t header_len = 0;
    size_t header_pos = 0;
    if (major == 1) {
        header_len = le16(npy, 8);
        header_pos = 10;
    } else if (major == 2 || major == 3) {
        header_len = le32(npy, 8);
        header_pos = 12;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (header_pos + header_len > npy.size()) {
        throw std::runtime_error("bad npy header length");
    }
    std::string header(reinterpret_cast<const char *>(&npy[header_pos]), header_len);

    NpyArray arr;
    auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
        descr_pos = header.find("\"descr\"");
    }
    auto q1 = header.find('\'', header.find(':', descr_pos));
    auto q2 = header.find('\'', q1 + 1);
    arr.descr = header.substr(q1 + 1, q2 - q1 - 1);

    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header.find("\"shape\"");
    }
    auto l = header.find('(', shape_pos);
    auto r = header.find(')', l);
    std::stringstream ss(header.substr(l + 1, r - l - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            arr.shape.push_back(std::stoll(item));
        }
    }
    arr.data = npy.data() + header_pos + header_len;
    arr.bytes = npy.size() - header_pos - header_len;
    return arr;
}

int64_t first_dim(const NpyArray &arr, const std::string &name) {
    if (arr.shape.empty()) {
        throw std::runtime_error("missing first dimension for " + name);
    }
    return arr.shape[0];
}

std::string default_model_dir() {
    const char *home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.cache/models/tts/qwen3-tts";
    }
    return "/tmp/.cache/models/tts/qwen3-tts";
}

struct Args {
    std::string npz;
    std::string text;
    std::string ref_wav;
    std::string ref_bin;
    std::string model_dir = env_str("Q3TTS_MODEL_DIR", default_model_dir());
    std::string talker_gguf = env_str("Q3TTS_TALKER_GGUF", "qwen3-tts-0.6b-talker-qkv-gateup-q8_0-side.gguf");
    std::string cp_gguf = env_str("Q3TTS_CP_GGUF", "qwen3-tts-0.6b-cp-qkv-gateup-rawq4.gguf");
    std::string language = env_str("Q3TTS_LANGUAGE", "auto");
    std::string clone_leadin = env_str("Q3TTS_CLONE_LEADIN", "");
    int frames = 60;
    std::string wav = "cpp_driver.wav";
    Hotwords hotwords = env_hotwords();
    int play_rate = env_int("Q3TTS_PLAY_RATE", 24000);
    int play_channels = env_int("Q3TTS_PLAY_CHANNELS", 1);
    int play_device = env_int("Q3TTS_PLAY_DEVICE", -1);
    int play_buffer = env_int("Q3TTS_PLAY_BUFFER", 1024);
    int play_tail_ms = env_int("Q3TTS_PLAY_TAIL_MS", 300);
    int play_drain_ms = env_int("Q3TTS_PLAY_DRAIN_MS", 250);
    int play_segment_pause_ms = env_int("Q3TTS_PLAY_SEGMENT_PAUSE_MS", 120);
    bool dump_ids = false;
    bool dump_segments = false;
    bool frontend_only = false;
    bool no_clone_split = false;
    bool play_segments = false;
    bool stdin_segments = false;
    bool talker_gguf_cli = false;
};

Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--npz") {
            a.npz = need("--npz");
        } else if (k == "--text") {
            a.text = need("--text");
        } else if (k == "--ref-wav") {
            a.ref_wav = need("--ref-wav");
        } else if (k == "--ref-bin") {
            a.ref_bin = need("--ref-bin");
        } else if (k == "--model-dir") {
            a.model_dir = need("--model-dir");
        } else if (k == "--talker-gguf") {
            a.talker_gguf = need("--talker-gguf");
            a.talker_gguf_cli = true;
        } else if (k == "--cp-gguf") {
            a.cp_gguf = need("--cp-gguf");
        } else if (k == "--language") {
            a.language = need("--language");
        } else if (k == "--dump-ids") {
            a.dump_ids = true;
        } else if (k == "--dump-segments") {
            a.dump_segments = true;
        } else if (k == "--frontend-only") {
            a.frontend_only = true;
        } else if (k == "--no-clone-split") {
            a.no_clone_split = true;
        } else if (k == "--play-segments") {
            a.play_segments = true;
        } else if (k == "--stdin-segments") {
            a.stdin_segments = true;
        } else if (k == "--clone-leadin") {
            a.clone_leadin = need("--clone-leadin");
        } else if (k == "--play-rate") {
            a.play_rate = std::stoi(need("--play-rate"));
        } else if (k == "--play-channels") {
            a.play_channels = std::stoi(need("--play-channels"));
        } else if (k == "--play-device") {
            a.play_device = std::stoi(need("--play-device"));
        } else if (k == "--play-buffer") {
            a.play_buffer = std::stoi(need("--play-buffer"));
        } else if (k == "--play-tail-ms") {
            a.play_tail_ms = std::stoi(need("--play-tail-ms"));
        } else if (k == "--play-drain-ms") {
            a.play_drain_ms = std::stoi(need("--play-drain-ms"));
        } else if (k == "--play-segment-pause-ms") {
            a.play_segment_pause_ms = std::stoi(need("--play-segment-pause-ms"));
        } else if (k == "--hotword") {
            add_hotword_spec(a.hotwords, need("--hotword"), "--hotword");
        } else if (k == "--hotwords") {
            add_hotword_specs(a.hotwords, need("--hotwords"), "--hotwords");
        } else if (k == "--frames") {
            a.frames = std::stoi(need("--frames"));
        } else if (k == "--wav") {
            a.wav = need("--wav");
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

bool has_clone_reference(const Args &args) {
    return !args.ref_bin.empty() || !args.ref_wav.empty();
}

bool has_full_reference_prompt(const std::string &ref_bin) {
    if (ref_bin.empty()) {
        return false;
    }
    try {
        return q3tts_frontend::read_reference_prompt_bin(ref_bin).full_prompt();
    } catch (const std::exception &) {
        return false;
    }
}

void maybe_select_full_prompt_talker(Args &args) {
    if (!has_full_reference_prompt(args.ref_bin)) {
        return;
    }
    if (args.talker_gguf_cli || std::getenv("Q3TTS_TALKER_GGUF")) {
        return;
    }
    const std::string full_talker =
        env_str("Q3TTS_FULL_PROMPT_TALKER_GGUF", "qwen3-tts-0.6b-talker-qkv-gateup-q8_0-side.gguf");
    if (full_talker.empty() || full_talker == "0") {
        return;
    }
    try {
        (void)q3tts_frontend::first_existing({
            q3tts_frontend::path_join(q3tts_frontend::path_join(args.model_dir, "gguf"), full_talker),
            q3tts_frontend::path_join(args.model_dir, full_talker),
            full_talker,
        });
        args.talker_gguf = full_talker;
        std::cout << "full_prompt_talker " << full_talker << "\n";
    } catch (const std::exception &) {
        std::cerr << "warning: full prompt talker not found, keeping " << args.talker_gguf << "\n";
    }
}

std::vector<std::array<int32_t, q3tts_frontend::CODE_GROUPS>>
load_ref_decode_prefix(const std::string &ref_bin) {
    if (ref_bin.empty()) {
        return {};
    }
    auto ref_prompt = q3tts_frontend::read_reference_prompt_bin(ref_bin);
    if (!ref_prompt.full_prompt()) {
        return {};
    }
    return ref_prompt.ref_codes;
}

size_t reference_audio_cut_samples(size_t ref_frames, size_t generated_frames, size_t wav_samples) {
    const size_t total_frames = ref_frames + generated_frames;
    if (ref_frames == 0 || total_frames == 0 || wav_samples == 0) {
        return 0;
    }
    return static_cast<size_t>(
        (static_cast<uint64_t>(ref_frames) * static_cast<uint64_t>(wav_samples)) /
        static_cast<uint64_t>(total_frames));
}

std::vector<float> decode_with_reference_prefix(
    q3tts_codec::DecoderPool &codec,
    const std::vector<std::array<int32_t, q3tts_frontend::CODE_GROUPS>> &generated,
    const std::vector<std::array<int32_t, q3tts_frontend::CODE_GROUPS>> &ref_prefix,
    const std::vector<int> &buckets,
    int first_chunk,
    int chunk,
    int ctx_limit) {
    if (ref_prefix.empty()) {
        return codec.decode_chunks(generated, buckets, first_chunk, chunk, ctx_limit);
    }

    std::vector<std::array<int32_t, q3tts_frontend::CODE_GROUPS>> all;
    all.reserve(ref_prefix.size() + generated.size());
    all.insert(all.end(), ref_prefix.begin(), ref_prefix.end());
    all.insert(all.end(), generated.begin(), generated.end());

    auto wav = codec.decode_chunks(all, buckets, first_chunk, chunk, ctx_limit);
    const size_t ref_samples = reference_audio_cut_samples(ref_prefix.size(), generated.size(), wav.size());
    if (wav.size() <= ref_samples) {
        wav.clear();
    } else {
        wav.erase(wav.begin(), wav.begin() + static_cast<ptrdiff_t>(ref_samples));
    }
    return wav;
}

void apply_biquad_lowpass(std::vector<float> &samples, double cutoff_hz) {
    if (samples.size() < 3 || cutoff_hz <= 0.0) {
        return;
    }

    constexpr double sample_rate = 24000.0;
    constexpr double pi = 3.14159265358979323846;
    cutoff_hz = std::max(1000.0, std::min(cutoff_hz, sample_rate * 0.49));

    const double q = 0.7071067811865476;
    const double w0 = 2.0 * pi * cutoff_hz / sample_rate;
    const double c = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha;
    const double b0 = ((1.0 - c) * 0.5) / a0;
    const double b1 = (1.0 - c) / a0;
    const double b2 = ((1.0 - c) * 0.5) / a0;
    const double a1 = (-2.0 * c) / a0;
    const double a2 = (1.0 - alpha) / a0;

    auto pass = [&](std::vector<float> &x) {
        double x1 = x.front();
        double x2 = x.front();
        double y1 = x.front();
        double y2 = x.front();
        for (float &sample : x) {
            const double x0 = sample;
            const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            sample = static_cast<float>(std::max(-1.0, std::min(1.0, y0)));
        }
    };

    pass(samples);
    pass(samples);
}

void apply_peak_normalize(std::vector<float> &samples, double target_db, double max_gain_db) {
    if (samples.empty()) {
        return;
    }
    double peak = 0.0;
    for (float x : samples) {
        peak = std::max(peak, static_cast<double>(std::abs(x)));
    }
    if (peak <= 0.0) {
        return;
    }

    const double target = std::pow(10.0, target_db / 20.0);
    const double max_gain = std::pow(10.0, std::max(0.0, max_gain_db) / 20.0);
    const double gain = std::min(target / peak, max_gain);
    for (float &x : samples) {
        x = static_cast<float>(std::max(-1.0, std::min(1.0, static_cast<double>(x) * gain)));
    }
}

void postprocess_audio_f32(std::vector<float> &samples) {
    if (env_int("Q3TTS_AUDIO_POSTPROCESS", 1) == 0 || samples.empty()) {
        return;
    }
    apply_biquad_lowpass(samples, env_double("Q3TTS_AUDIO_LOWPASS_HZ", 9000.0));
    apply_peak_normalize(
        samples,
        env_double("Q3TTS_AUDIO_PEAK_DB", -3.0),
        env_double("Q3TTS_AUDIO_MAX_GAIN_DB", 9.0));
}

std::vector<int16_t> f32_to_pcm16(std::vector<float> samples, bool postprocess) {
    if (postprocess) {
        postprocess_audio_f32(samples);
    }

    std::vector<int16_t> pcm;
    pcm.reserve(samples.size());
    for (float x : samples) {
        x = std::max(-1.0f, std::min(1.0f, x));
        pcm.push_back(static_cast<int16_t>(x * 32767.0f));
    }
    return pcm;
}

void postprocess_pcm16(std::vector<int16_t> &samples) {
    if (env_int("Q3TTS_AUDIO_POSTPROCESS", 1) == 0 || samples.empty()) {
        return;
    }

    std::vector<float> f32;
    f32.reserve(samples.size());
    for (int16_t sample : samples) {
        f32.push_back(static_cast<float>(sample) / 32768.0f);
    }
    postprocess_audio_f32(f32);
    samples = f32_to_pcm16(std::move(f32), false);
}

void write_wav_i16_samples(const std::string &path, const std::vector<int16_t> &samples);

void write_wav_i16(const std::string &path, const std::vector<float> &samples) {
    write_wav_i16_samples(path, f32_to_pcm16(samples, true));
}

void write_wav_i16_samples(const std::string &path, const std::vector<int16_t> &samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("write wav failed: " + path);
    }
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    auto w16 = [&](uint16_t v) {
        char b[2] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff)};
        f.write(b, 2);
    };
    auto w32 = [&](uint32_t v) {
        char b[4] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff),
                     static_cast<char>((v >> 16) & 0xff), static_cast<char>((v >> 24) & 0xff)};
        f.write(b, 4);
    };
    f.write("RIFF", 4);
    w32(riff_size);
    f.write("WAVEfmt ", 8);
    w32(16);
    w16(1);
    w16(1);
    w32(24000);
    w32(48000);
    w16(2);
    w16(16);
    f.write("data", 4);
    w32(data_bytes);
    for (int16_t s : samples) {
        w16(static_cast<uint16_t>(s));
    }
}

std::vector<int16_t> read_wav_i16_mono_24k(const std::string &path) {
    auto wav = read_all(path);
    if (wav.size() < 44 || std::memcmp(wav.data(), "RIFF", 4) != 0 ||
        std::memcmp(wav.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("bad wav: " + path);
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_pos = 0;
    size_t data_bytes = 0;
    for (size_t p = 12; p + 8 <= wav.size();) {
        const std::string id(reinterpret_cast<const char *>(&wav[p]), 4);
        const uint32_t n = le32(wav, p + 4);
        const size_t body = p + 8;
        if (body + n > wav.size()) {
            throw std::runtime_error("bad wav chunk size: " + path);
        }
        if (id == "fmt " && n >= 16) {
            audio_format = le16(wav, body);
            channels = le16(wav, body + 2);
            sample_rate = le32(wav, body + 4);
            bits_per_sample = le16(wav, body + 14);
        } else if (id == "data") {
            data_pos = body;
            data_bytes = n;
        }
        p = body + n + (n & 1u);
    }
    if (audio_format != 1 || channels != 1 || sample_rate != 24000 || bits_per_sample != 16 ||
        data_pos == 0 || data_bytes == 0) {
        throw std::runtime_error("expected PCM16 mono 24k wav: " + path);
    }
    std::vector<int16_t> samples(data_bytes / sizeof(int16_t));
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(le16(wav, data_pos + i * 2));
    }
    return samples;
}

size_t utf8_char_len(unsigned char c) {
    if ((c & 0x80u) == 0) {
        return 1;
    }
    if ((c & 0xe0u) == 0xc0u) {
        return 2;
    }
    if ((c & 0xf0u) == 0xe0u) {
        return 3;
    }
    if ((c & 0xf8u) == 0xf0u) {
        return 4;
    }
    return 1;
}

size_t utf8_len(const std::string &s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        i += std::min(utf8_char_len(static_cast<unsigned char>(s[i])), s.size() - i);
        ++n;
    }
    return n;
}

uint32_t utf8_codepoint(const std::string &s, size_t i, size_t n) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (n == 1) {
        return b0;
    }
    if (n == 2 && i + 1 < s.size()) {
        return (static_cast<uint32_t>(b0 & 0x1fu) << 6) |
               static_cast<uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3fu);
    }
    if (n == 3 && i + 2 < s.size()) {
        return (static_cast<uint32_t>(b0 & 0x0fu) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3fu) << 6) |
               static_cast<uint32_t>(static_cast<unsigned char>(s[i + 2]) & 0x3fu);
    }
    if (n == 4 && i + 3 < s.size()) {
        return (static_cast<uint32_t>(b0 & 0x07u) << 18) |
               (static_cast<uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3fu) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(s[i + 2]) & 0x3fu) << 6) |
               static_cast<uint32_t>(static_cast<unsigned char>(s[i + 3]) & 0x3fu);
    }
    return b0;
}

std::string lower_ascii_copy(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

bool contains_ascii_letter(const std::string &s) {
    for (unsigned char c : s) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            return true;
        }
    }
    return false;
}

bool contains_cjk(const std::string &s) {
    for (size_t i = 0; i < s.size();) {
        const size_t n = std::min(utf8_char_len(static_cast<unsigned char>(s[i])), s.size() - i);
        const uint32_t cp = utf8_codepoint(s, i, n);
        if ((cp >= 0x3400 && cp <= 0x9fff) ||
            (cp >= 0xf900 && cp <= 0xfaff) ||
            (cp >= 0x20000 && cp <= 0x2ebef)) {
            return true;
        }
        i += n;
    }
    return false;
}

bool in_set(const std::vector<std::string> &items, const std::string &x) {
    return std::find(items.begin(), items.end(), x) != items.end();
}

bool starts_with(const std::string &s, const std::string &prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool prefer_chinese_sentence_mark(const std::string &segment, const std::string &language) {
    const std::string l = lower_ascii_copy(language);
    if (l == "chinese" || l == "zh" || l == "zh-cn") {
        return true;
    }
    if (l == "english" || l == "en" || l == "en-us") {
        return false;
    }
    if (contains_cjk(segment)) {
        return true;
    }
    if (contains_ascii_letter(segment)) {
        return false;
    }
    return true;
}

std::string sentence_mark_for(const std::string &segment, const std::string &language) {
    return prefer_chinese_sentence_mark(segment, language) ? "\xe3\x80\x82" : ".";
}

std::string normalize_segment_end(std::string segment, const std::string &language) {
    const std::string mark = sentence_mark_for(segment, language);
    const std::vector<std::string> weak = {"\xef\xbc\x8c", ",", "\xe3\x80\x81", "\xef\xbc\x9a", ":"};
    for (const auto &punct : weak) {
        if (ends_with(segment, punct)) {
            segment.replace(segment.size() - punct.size(), punct.size(), mark);
            break;
        }
    }
    return segment;
}

std::string ensure_sentence_end(std::string segment, const std::string &language) {
    segment = normalize_segment_end(trim(segment), language);
    if (segment.empty()) {
        return segment;
    }
    const std::vector<std::string> final = {
        "\xe3\x80\x82", ".", "\xef\xbc\x81", "\xef\xbc\x9f", "!", "?", "\xef\xbc\x9b", ";",
        "\xef\xbc\x8c", ",", "\xe3\x80\x81", "\xef\xbc\x9a", ":"};
    for (const auto &punct : final) {
        if (ends_with(segment, punct)) {
            return normalize_segment_end(segment, language);
        }
    }
    segment += sentence_mark_for(segment, language);
    return segment;
}

std::vector<std::string> split_on_punct(const std::string &text,
                                        const std::vector<std::string> &breaks,
                                        size_t min_chars_before_break) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size();) {
        const size_t n = std::min(utf8_char_len(static_cast<unsigned char>(text[i])), text.size() - i);
        const std::string ch = text.substr(i, n);
        cur += ch;
        i += n;
        if (in_set(breaks, ch) && utf8_len(cur) >= min_chars_before_break) {
            std::string item = trim(cur);
            if (!item.empty()) {
                out.push_back(item);
            }
            cur.clear();
        }
    }
    std::string tail = trim(cur);
    if (!tail.empty()) {
        out.push_back(tail);
    }
    return out;
}

std::vector<std::string> split_by_max_chars(const std::string &text, size_t max_chars) {
    std::vector<std::string> out;
    std::string cur;
    size_t cur_chars = 0;
    size_t last_soft_byte = std::string::npos;
    const size_t min_soft_chars = std::min<size_t>(8, max_chars);
    size_t chars_at_last_soft = 0;
    const auto soft = std::vector<std::string>{" ", "\t", "\xef\xbc\x8c", ",", "\xe3\x80\x81", "\xef\xbc\x9a", ":"};

    auto flush = [&]() {
        std::string item = trim(cur);
        if (!item.empty()) {
            out.push_back(item);
        }
        cur.clear();
        cur_chars = 0;
        last_soft_byte = std::string::npos;
        chars_at_last_soft = 0;
    };

    for (size_t i = 0; i < text.size();) {
        const size_t n = std::min(utf8_char_len(static_cast<unsigned char>(text[i])), text.size() - i);
        const std::string ch = text.substr(i, n);
        if (cur_chars >= min_soft_chars && in_set(soft, ch)) {
            last_soft_byte = cur.size() + n;
            chars_at_last_soft = cur_chars + 1;
        }
        cur += ch;
        ++cur_chars;
        i += n;
        if (cur_chars > max_chars) {
            if (last_soft_byte != std::string::npos && chars_at_last_soft >= min_soft_chars) {
                std::string head = trim(cur.substr(0, last_soft_byte));
                std::string tail = trim(cur.substr(last_soft_byte));
                if (!head.empty()) {
                    out.push_back(head);
                }
                cur = tail;
                cur_chars = utf8_len(cur);
            } else {
                flush();
            }
            last_soft_byte = std::string::npos;
            chars_at_last_soft = 0;
        }
    }
    flush();
    return out;
}

std::vector<std::string> split_clone_text(const std::string &text,
                                          bool full_prompt_ref,
                                          const std::string &language) {
    const bool latin_text = !prefer_chinese_sentence_mark(text, language);
    int hard_max = env_int("Q3TTS_CLONE_MAX_CHARS", full_prompt_ref ? (latin_text ? 96 : 16) : 0);
    if (latin_text) {
        hard_max = env_int("Q3TTS_CLONE_MAX_CHARS_LATIN", hard_max);
    }
    const size_t max_chars = hard_max > 0 ? static_cast<size_t>(std::max(8, hard_max)) : 28UL;
    const size_t weak_min_chars =
        static_cast<size_t>(std::max(1, env_int("Q3TTS_FULL_PROMPT_WEAK_MIN_CHARS",
                                               full_prompt_ref ? (latin_text ? 24 : 6) : 12)));
    const auto strong = std::vector<std::string>{
        "\xe3\x80\x82", ".", "\xef\xbc\x81", "\xef\xbc\x9f", "!", "?", "\xef\xbc\x9b", ";"};
    const auto weak = std::vector<std::string>{
        "\xef\xbc\x8c", ",", "\xe3\x80\x81", "\xef\xbc\x9a", ":"};
    std::vector<std::string> first = split_on_punct(text, strong, 1);
    std::vector<std::string> out;
    for (const auto &s : first) {
        if (utf8_len(s) <= max_chars) {
            out.push_back(s);
            continue;
        }
        auto pieces = split_on_punct(s, weak, weak_min_chars);
        if (pieces.size() <= 1) {
            if (hard_max > 0) {
                auto hard = split_by_max_chars(s, max_chars);
                out.insert(out.end(), hard.begin(), hard.end());
            } else {
                out.push_back(s);
            }
        } else {
            for (const auto &piece : pieces) {
                if (hard_max <= 0 || utf8_len(piece) <= max_chars) {
                    out.push_back(piece);
                } else {
                    auto hard = split_by_max_chars(piece, max_chars);
                    out.insert(out.end(), hard.begin(), hard.end());
                }
            }
        }
    }
    if (out.empty()) {
        std::string item = trim(text);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<std::string> split_stdin_text(const std::string &text,
                                          bool full_prompt_ref,
                                          const std::string &language) {
    (void) full_prompt_ref;
    if (env_int("Q3TTS_STDIN_SAFE_SPLIT", 1) == 0) {
        std::string item = ensure_sentence_end(text, language);
        return item.empty() ? std::vector<std::string>{} : std::vector<std::string>{item};
    }
    const bool latin_text = !prefer_chinese_sentence_mark(text, language);
    const int default_max_chars = latin_text ? 96 : 48;
    const size_t max_chars = static_cast<size_t>(std::max(
        8, env_int("Q3TTS_STDIN_MAX_CHARS_PER_SEGMENT", default_max_chars)));
    const size_t weak_min_chars = static_cast<size_t>(
        std::max(1, env_int("Q3TTS_STDIN_WEAK_MIN_CHARS", 24)));
    const auto strong = std::vector<std::string>{
        "\xe3\x80\x82", ".", "\xef\xbc\x81", "\xef\xbc\x9f", "!", "?", "\xef\xbc\x9b", ";"};
    const auto weak = std::vector<std::string>{
        "\xef\xbc\x8c", ",", "\xe3\x80\x81", "\xef\xbc\x9a", ":"};
    std::vector<std::string> out;
    for (const auto &sentence : split_on_punct(text, strong, 1)) {
        std::vector<std::string> pieces;
        if (utf8_len(sentence) <= max_chars) {
            pieces.push_back(sentence);
        } else {
            pieces = split_on_punct(sentence, weak, weak_min_chars);
            if (pieces.size() <= 1) {
                pieces = split_by_max_chars(sentence, max_chars);
            }
        }
        for (const auto &piece : pieces) {
            if (utf8_len(piece) <= max_chars) {
                std::string item = ensure_sentence_end(piece, language);
                if (!item.empty()) {
                    out.push_back(item);
                }
            } else {
                for (const auto &hard : split_by_max_chars(piece, max_chars)) {
                    std::string item = ensure_sentence_end(hard, language);
                    if (!item.empty()) {
                        out.push_back(item);
                    }
                }
            }
        }
    }
    if (out.empty()) {
        std::string item = ensure_sentence_end(text, language);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

bool needs_clone_leadin(const std::string &segment, const std::string &leadin) {
    if (leadin.empty()) {
        return false;
    }
    if (env_int("Q3TTS_CLONE_LEADIN_ALWAYS", 1) != 0) {
        return true;
    }
    const std::vector<std::string> stable_prefixes = {
        "\xe4\xbd\xa0\xe5\xa5\xbd",
        "\xe6\x82\xa8\xe5\xa5\xbd",
        "hello",
        "Hello"};
    for (const auto &prefix : stable_prefixes) {
        if (starts_with(segment, prefix)) {
            return false;
        }
    }
    return true;
}

int clone_leadin_trim_frames(const std::string &leadin) {
    const std::string default_leadin = "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c";
    const int default_frames = (leadin == default_leadin) ? 7 : 0;
    return std::max(0, env_int("Q3TTS_CLONE_LEADIN_TRIM_FRAMES", default_frames));
}

size_t clone_leadin_trim_samples(const std::string &leadin) {
    const int explicit_samples = env_int("Q3TTS_CLONE_LEADIN_TRIM_SAMPLES", -1);
    if (explicit_samples >= 0) {
        return static_cast<size_t>(explicit_samples);
    }
    return static_cast<size_t>(clone_leadin_trim_frames(leadin)) * 1920UL;
}

void trim_leading_samples(std::vector<int16_t> &samples, size_t n) {
    if (n == 0) {
        return;
    }
    if (n >= samples.size()) {
        samples.clear();
        return;
    }
    samples.erase(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(n));
}

std::vector<std::string> prepare_clone_segments(const Args &args) {
    const bool full_prompt_ref = has_full_reference_prompt(args.ref_bin);
    std::vector<std::string> segments = split_clone_text(args.text, full_prompt_ref, args.language);
    for (auto &segment : segments) {
        segment = normalize_segment_end(segment, args.language);
    }
    return segments;
}

std::string shell_quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

double rms_i16(const std::vector<int16_t> &samples, size_t begin, size_t end) {
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) {
        const double x = static_cast<double>(samples[i]) / 32768.0;
        sum += x * x;
    }
    return std::sqrt(sum / static_cast<double>(end - begin));
}

void trim_trailing_silence(std::vector<int16_t> &samples) {
    const size_t block = 2400;
    const size_t keep = 2400;
    const double threshold = 0.003;
    for (size_t end = samples.size(); end > 0;) {
        const size_t begin = end > block ? end - block : 0;
        if (rms_i16(samples, begin, end) > threshold) {
            const size_t trimmed = std::min(samples.size(), end + keep);
            samples.resize(trimmed);
            return;
        }
        end = begin;
    }
}

double seconds_since(Clock::time_point a, Clock::time_point b);
std::string model_file(const std::string &model_dir, const std::string &subdir, const std::string &name);

int run_clone_split(const Args &args, const char *argv0, const std::vector<std::string> &segments) {
    (void)argv0;
    if (segments.empty()) {
        throw std::runtime_error("empty text");
    }

    const auto t0 = Clock::now();
    const bool profile = env_int("Q3TTS_PROFILE", 0) != 0;
    profile_event(profile, t0, "clone_split_start segments=" + std::to_string(segments.size()));
    const std::string pid = std::to_string(static_cast<long long>(getpid()));
    std::vector<std::string> temps;
    auto cleanup = [&]() {
        for (const auto &p : temps) {
            std::remove(p.c_str());
        }
    };

    try {
        if (!std::getenv("Q3TTS_SKIP_TCM_CLEAR")) {
            const int clear_rc = std::system("spacemit-tcm-smi -c >/dev/null 2>&1");
            (void)clear_rc;
        }
        profile_event(profile, t0, "tcm_clear_done");

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "q3tts_cpp_driver");
        std::string ref_bin = args.ref_bin;
        if (ref_bin.empty() && !args.ref_wav.empty()) {
            ref_bin = "/tmp/q3tts_split_" + pid + "_ref.spk.bin";
            temps.push_back(ref_bin);
            auto spk = q3tts_frontend::run_speaker_encoder(
                env, q3tts_frontend::speaker_encoder_path(args.model_dir), args.ref_wav,
                env_int("Q3TTS_FRONTEND_THREADS", 2));
            q3tts_frontend::write_speaker_bin(ref_bin, spk);
        }
        profile_event(profile, t0, "ref_ready");

        q3tts_frontend::FrontendConfig base_fc;
        base_fc.model_dir = args.model_dir;
        base_fc.ref_bin = ref_bin;
        base_fc.language = args.language;
        base_fc.talker_gguf = args.talker_gguf;
        base_fc.cp_gguf = args.cp_gguf;
        base_fc.frontend_threads = env_int("Q3TTS_FRONTEND_THREADS", 2);
        base_fc.full_prompt_non_streaming = env_int("Q3TTS_FULL_PROMPT_NON_STREAMING", 0) != 0;
        q3tts_frontend::FrontendRuntime frontend(env, base_fc);
        profile_event(profile, t0, "frontend_runtime_ready");

        struct SegmentJob {
            std::string prefill;
            std::string trailing;
            std::string pad;
            std::string codes;
            std::string done;
            std::string wav;
            std::string text;
            std::string synth_text;
            size_t leadin_trim_samples = 0;
            int64_t np = 0;
            int64_t nt = 0;
        };
        std::vector<SegmentJob> jobs;
        jobs.reserve(segments.size());
        for (size_t i = 0; i < segments.size(); ++i) {
            SegmentJob job;
            const std::string base = "/tmp/q3tts_split_" + pid + "_" + std::to_string(i);
            job.prefill = base + "_prefill.bin";
            job.trailing = base + "_trailing.bin";
            job.pad = base + "_pad.bin";
            job.codes = base + "_codes.bin";
            job.done = base + "_done";
            job.wav = base + ".wav";
            job.text = segments[i];
            if (needs_clone_leadin(job.text, args.clone_leadin)) {
                job.synth_text = args.clone_leadin + job.text;
                job.leadin_trim_samples = clone_leadin_trim_samples(args.clone_leadin);
            } else {
                job.synth_text = job.text;
            }
            temps.push_back(job.prefill);
            temps.push_back(job.trailing);
            temps.push_back(job.pad);
            temps.push_back(job.codes);
            temps.push_back(job.done);
            temps.push_back(job.wav);

            q3tts_frontend::FrontendConfig fc = base_fc;
            fc.text = job.synth_text;
            auto input = frontend.build(fc);
            write_all(job.prefill,
                      reinterpret_cast<const uint8_t *>(input.prefill.data()),
                      input.prefill.size() * sizeof(float));
            write_all(job.trailing,
                      reinterpret_cast<const uint8_t *>(input.trailing.data()),
                      input.trailing.size() * sizeof(float));
            write_all(job.pad,
                      reinterpret_cast<const uint8_t *>(input.pad.data()),
                      input.pad.size() * sizeof(float));
            job.np = input.n_prefill;
            job.nt = input.n_trailing;
            jobs.push_back(std::move(job));
            profile_event(profile, t0, "frontend_segment_done i=" + std::to_string(i));
        }

        const std::string job_list = "/tmp/q3tts_split_" + pid + "_jobs.txt";
        temps.push_back(job_list);
        {
            std::ofstream jf(job_list);
            if (!jf) {
                throw std::runtime_error("write jobs failed: " + job_list);
            }
            for (const auto &job : jobs) {
                jf << job.prefill << " " << job.np << " "
                   << job.trailing << " " << job.nt << " "
                   << job.pad << " " << args.frames << " "
                   << job.codes << " " << job.done << "\n";
            }
        }
        profile_event(profile, t0, "job_list_ready");

        const int codec_threads = env_int("Q3TTS_CODEC_THREADS", 3);
        if (const char *aff = std::getenv("Q3TTS_CODEC_AFFINITY"); aff && *aff) {
            setenv("SPACEMIT_EP_INTRA_THREAD_AFFINITY", aff, 1);
            if (!std::getenv("SPACEMIT_EP_INTRA_THREAD_NUM")) {
                setenv("SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(codec_threads).c_str(), 1);
            }
            if (!std::getenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD")) {
                setenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "0", 1);
            }
        }

        const std::vector<int> buckets =
            parse_int_list(env_str("Q3TTS_CODEC_BUCKETS", ""), std::vector<int>{50});
        const int chunk = env_int("Q3TTS_CODEC_CHUNK", 50);
        const int first_chunk = env_int("Q3TTS_CODEC_FIRST_CHUNK", chunk);
        const int ctx_limit = env_int("Q3TTS_CODEC_CTX", 25);

#ifndef Q3TTS_ENABLE_SDK_AUDIO
        if (args.play_segments) {
            throw std::runtime_error("--play-segments requires Q3TTS_ENABLE_SDK_AUDIO build");
        }
#else
        std::unique_ptr<q3tts_audio::SdkSegmentPlayer> realtime_player;
        if (args.play_segments) {
            realtime_player = std::make_unique<q3tts_audio::SdkSegmentPlayer>(
                args.play_rate, args.play_channels, args.play_device, args.play_buffer,
                args.play_tail_ms, args.play_drain_ms, args.play_segment_pause_ms);
        }
#endif

        const std::string driver = env_str("TALKER_DRIVER", "./talker_driver");
        const std::string talker_cpuset = env_str("TALKER_CPUSET", "4-7");
        const std::string talker_gguf = model_file(args.model_dir, "gguf", args.talker_gguf);
        const std::string cp_gguf = model_file(args.model_dir, "gguf", args.cp_gguf);
        std::stringstream cmd;
        cmd << "taskset -c " << shell_quote(talker_cpuset)
            << " " << shell_quote(driver)
            << " " << shell_quote(talker_gguf)
            << " " << shell_quote(cp_gguf)
            << " --jobs " << shell_quote(job_list);
        std::atomic<int> talker_rc{-1};
        std::thread talker_thread([&]() {
            talker_rc.store(std::system(cmd.str().c_str()));
        });
        profile_event(profile, t0, "talker_launched");

        q3tts_codec::DecoderPoolConfig codec_cfg;
        codec_cfg.model_dir = args.model_dir;
        codec_cfg.buckets = buckets;
        codec_cfg.intra_threads = codec_threads;
        codec_cfg.on_bucket_warm = [&](int b) {
            profile_event(profile, t0, "codec_warm bucket=" + std::to_string(b));
        };
        q3tts_codec::DecoderPool codec(env, codec_cfg);
        profile_event(profile, t0, "codec_ready");

        const auto ref_decode_prefix = load_ref_decode_prefix(ref_bin);

        auto decode_codes = [&](const std::string &path, size_t *frame_count) -> std::vector<float> {
            auto data = read_all(path);
            const size_t frame_bytes = sizeof(int32_t) * 16;
            if (data.size() % frame_bytes != 0) {
                throw std::runtime_error("bad code file size: " + path);
            }
            if (frame_count) {
                *frame_count = data.size() / frame_bytes;
            }
            std::vector<std::array<int32_t, 16>> frames(data.size() / frame_bytes);
            if (!frames.empty()) {
                std::memcpy(frames.data(), data.data(), data.size());
            }
            return decode_with_reference_prefix(codec, frames, ref_decode_prefix, buckets, first_chunk, chunk, ctx_limit);
        };

        auto decode_job_samples = [&](const SegmentJob &job) -> std::vector<int16_t> {
            size_t code_frames = 0;
            auto wav_f32 = decode_codes(job.codes, &code_frames);
            if (args.frames > 0 && code_frames >= static_cast<size_t>(args.frames)) {
                std::cout << "clone_segment_truncated"
                          << " frames " << code_frames
                          << " max " << args.frames
                          << " text " << job.text << std::endl;
                if (env_int("Q3TTS_ALLOW_TRUNCATED", 0) == 0) {
                    throw std::runtime_error(
                        "clone split segment reached max frames without EOS: " + job.text);
                }
            }
            std::vector<int16_t> samples = f32_to_pcm16(std::move(wav_f32), false);
            trim_leading_samples(samples, job.leadin_trim_samples);
            trim_trailing_silence(samples);
            if (samples.empty() || rms_i16(samples, 0, samples.size()) <= 0.003) {
                throw std::runtime_error("persistent clone split segment produced silent audio");
            }
            postprocess_pcm16(samples);
            return samples;
        };

        if (args.play_segments) {
#ifndef Q3TTS_ENABLE_SDK_AUDIO
            throw std::runtime_error("--play-segments requires Q3TTS_ENABLE_SDK_AUDIO build");
#else
            try {
                std::vector<int16_t> merged;
                double total_gap = 0.0;
                double max_gap = 0.0;
                bool got_first_queue = false;
                Clock::time_point t_first_queue = t0;
                Clock::time_point t_generation_done = t0;

                for (size_t i = 0; i < jobs.size(); ++i) {
                    const auto wait0 = Clock::now();
                    while (!exists(jobs[i].done)) {
                        if (talker_rc.load() != -1) {
                            throw std::runtime_error("talker stopped before segment marker");
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    const double wait_s = seconds_since(wait0, Clock::now());

                    const auto decode0 = Clock::now();
                    auto samples = decode_job_samples(jobs[i]);
                    write_wav_i16_samples(jobs[i].wav, samples);
                    const double decode_s = seconds_since(decode0, Clock::now());
                    const double audio_s = static_cast<double>(samples.size()) / 24000.0;

                    const double gap_s = realtime_player->enqueue_mono24k(samples);
                    if (!got_first_queue) {
                        got_first_queue = true;
                        t_first_queue = Clock::now();
                    }
                    total_gap += gap_s;
                    max_gap = std::max(max_gap, gap_s);

                    if (!merged.empty()) {
                        merged.insert(merged.end(), 4800, 0);
                    }
                    merged.insert(merged.end(), samples.begin(), samples.end());

                    std::cout.setf(std::ios::fixed);
                    std::cout.precision(3);
                    std::cout << "segment " << (i + 1)
                              << " wait " << wait_s << "s"
                              << " decode " << decode_s << "s"
                              << " audio " << audio_s << "s"
                              << " gap " << gap_s << "s"
                              << " wav " << jobs[i].wav << std::endl;
                }
                t_generation_done = Clock::now();

                if (talker_thread.joinable()) {
                    talker_thread.join();
                }
                profile_event(profile, t0, "talker_done");
                if (talker_rc.load() != 0) {
                    throw std::runtime_error("persistent clone split talker failed");
                }
                realtime_player->finish();

                write_wav_i16_samples(args.wav, merged);
                profile_event(profile, t0, "wav_written");
                cleanup();

                const double wall = seconds_since(t0, Clock::now());
                const double audio = static_cast<double>(merged.size()) / 24000.0;
                const double gen_wall = seconds_since(t0, t_generation_done);
                const double warm = got_first_queue ? seconds_since(t_first_queue, t_generation_done) : gen_wall;
                std::cout.setf(std::ios::fixed);
                std::cout.precision(2);
                std::cout << "clone_split_realtime segments " << segments.size()
                          << "  gen_wall " << gen_wall << "s"
                          << "  wall " << wall << "s"
                          << "  audio " << audio << "s"
                          << "  genRTF " << (gen_wall / audio)
                          << "  RTF " << (wall / audio)
                          << "  warmRTF " << (warm / audio)
                          << "  total_gap " << total_gap << "s"
                          << "  max_gap " << max_gap << "s\n";
                std::cout << "wav " << args.wav << "\n";
                return 0;
            } catch (...) {
                if (talker_thread.joinable()) {
                    talker_thread.join();
                }
                if (realtime_player) {
                    try {
                        realtime_player->finish();
                    } catch (...) {
                    }
                }
                throw;
            }
#endif
        }

        talker_thread.join();
        profile_event(profile, t0, "talker_done");
        if (talker_rc.load() != 0) {
            throw std::runtime_error("persistent clone split talker failed");
        }

        std::vector<int16_t> merged;
        for (const auto &job : jobs) {
            auto samples = decode_job_samples(job);
            if (!merged.empty()) {
                merged.insert(merged.end(), 4800, 0);
            }
            merged.insert(merged.end(), samples.begin(), samples.end());
            profile_event(profile, t0, "decode_segment_done codes=" + job.codes);
        }

        write_wav_i16_samples(args.wav, merged);
        profile_event(profile, t0, "wav_written");
        cleanup();

        const double wall = seconds_since(t0, Clock::now());
        const double audio = static_cast<double>(merged.size()) / 24000.0;
        std::cout.setf(std::ios::fixed);
        std::cout.precision(2);
        std::cout << "clone_split_persistent segments " << segments.size()
                  << "  cold_wall " << wall << "s"
                  << "  audio " << audio << "s"
                  << "  coldRTF " << (wall / audio) << "\n";
        std::cout << "wav " << args.wav << "\n";
        return 0;
    } catch (...) {
        cleanup();
        throw;
    }
}

double seconds_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::string model_file(const std::string &model_dir, const std::string &subdir, const std::string &name) {
    return q3tts_frontend::first_existing({
        q3tts_frontend::path_join(q3tts_frontend::path_join(model_dir, subdir), name),
        q3tts_frontend::path_join(model_dir, name),
        name,
    });
}

int run_stdin_segments(Args args, const std::vector<std::string> &input_lines = {}) {
    std::signal(SIGPIPE, SIG_IGN);

    const auto t0 = Clock::now();
    const bool profile = env_int("Q3TTS_PROFILE", 0) != 0;
    profile_event(profile, t0, "stdin_segments_start");
    const bool has_ref = !args.ref_bin.empty() || !args.ref_wav.empty();
    const std::string pid = std::to_string(static_cast<long long>(getpid()));
    std::vector<std::string> temps;
    auto cleanup = [&]() {
        for (const auto &p : temps) {
            std::remove(p.c_str());
        }
    };

    struct StreamJob {
        int index = 0;
        std::string text;
        std::string synth_text;
        std::string prefill;
        std::string trailing;
        std::string pad;
        std::string codes;
        std::string done;
        std::string wav;
        int64_t np = 0;
        int64_t nt = 0;
        int max_frames = 0;
        size_t leadin_trim_samples = 0;
    };

    try {
        if (!std::getenv("Q3TTS_SKIP_TCM_CLEAR")) {
            const int clear_rc = std::system("spacemit-tcm-smi -c >/dev/null 2>&1");
            (void)clear_rc;
        }
        profile_event(profile, t0, "tcm_clear_done");

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "q3tts_cpp_driver");
        std::string ref_bin = args.ref_bin;
        if (ref_bin.empty() && !args.ref_wav.empty()) {
            ref_bin = "/tmp/q3tts_stdin_" + pid + "_ref.spk.bin";
            temps.push_back(ref_bin);
            auto spk = q3tts_frontend::run_speaker_encoder(
                env, q3tts_frontend::speaker_encoder_path(args.model_dir), args.ref_wav,
                env_int("Q3TTS_FRONTEND_THREADS", 2));
            q3tts_frontend::write_speaker_bin(ref_bin, spk);
        }
        profile_event(profile, t0, "ref_ready");

        q3tts_frontend::FrontendConfig base_fc;
        base_fc.model_dir = args.model_dir;
        base_fc.ref_bin = ref_bin;
        base_fc.language = args.language;
        base_fc.talker_gguf = args.talker_gguf;
        base_fc.cp_gguf = args.cp_gguf;
        base_fc.frontend_threads = env_int("Q3TTS_FRONTEND_THREADS", 2);
        base_fc.full_prompt_non_streaming = env_int("Q3TTS_FULL_PROMPT_NON_STREAMING", 0) != 0;
        q3tts_frontend::FrontendRuntime frontend(env, base_fc);
        profile_event(profile, t0, "frontend_runtime_ready");

        const int codec_threads = env_int("Q3TTS_CODEC_THREADS", 3);
        if (const char *aff = std::getenv("Q3TTS_CODEC_AFFINITY"); aff && *aff) {
            setenv("SPACEMIT_EP_INTRA_THREAD_AFFINITY", aff, 1);
            if (!std::getenv("SPACEMIT_EP_INTRA_THREAD_NUM")) {
                setenv("SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(codec_threads).c_str(), 1);
            }
            if (!std::getenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD")) {
                setenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "0", 1);
            }
        }

        const bool no_ref_text = !has_ref;
        const std::vector<int> buckets = no_ref_text
            ? parse_int_list(env_str("Q3TTS_NOREF_CODEC_BUCKETS", ""), std::vector<int>{25})
            : parse_int_list(env_str("Q3TTS_CODEC_BUCKETS", ""), std::vector<int>{50});
        const int chunk = no_ref_text ? env_int("Q3TTS_NOREF_CODEC_CHUNK", 25)
                                      : env_int("Q3TTS_CODEC_CHUNK", 50);
        const int first_chunk = no_ref_text ? env_int("Q3TTS_NOREF_CODEC_FIRST_CHUNK", chunk)
                                            : env_int("Q3TTS_CODEC_FIRST_CHUNK", chunk);
        const int ctx_limit = env_int("Q3TTS_CODEC_CTX", 25);

#ifndef Q3TTS_ENABLE_SDK_AUDIO
        if (args.play_segments) {
            throw std::runtime_error("--play-segments requires Q3TTS_ENABLE_SDK_AUDIO build");
        }
#else
        std::unique_ptr<q3tts_audio::SdkSegmentPlayer> realtime_player;
        if (args.play_segments) {
            realtime_player = std::make_unique<q3tts_audio::SdkSegmentPlayer>(
                args.play_rate, args.play_channels, args.play_device, args.play_buffer,
                args.play_tail_ms, args.play_drain_ms, args.play_segment_pause_ms);
        }
#endif

        q3tts_codec::DecoderPoolConfig codec_cfg;
        codec_cfg.model_dir = args.model_dir;
        codec_cfg.buckets = buckets;
        codec_cfg.intra_threads = codec_threads;
        codec_cfg.on_bucket_warm = [&](int b) {
            profile_event(profile, t0, "codec_warm bucket=" + std::to_string(b));
        };
        q3tts_codec::DecoderPool codec(env, codec_cfg);
        profile_event(profile, t0, "codec_ready");

        const std::string driver = env_str("TALKER_DRIVER", "./talker_driver");
        const std::string talker_cpuset = env_str("TALKER_CPUSET", "4-7");
        const std::string talker_gguf = model_file(args.model_dir, "gguf", args.talker_gguf);
        const std::string cp_gguf = model_file(args.model_dir, "gguf", args.cp_gguf);
        const int stdin_max_prefill = env_int("Q3TTS_STDIN_MAX_PREFILL", 128);
        const int stdin_max_frames = env_int("Q3TTS_STDIN_MAX_FRAMES", args.frames);
        std::stringstream cmd;
        cmd << "taskset -c " << shell_quote(talker_cpuset)
            << " " << shell_quote(driver)
            << " " << shell_quote(talker_gguf)
            << " " << shell_quote(cp_gguf)
            << " --jobs-stdin " << stdin_max_prefill << " " << stdin_max_frames;
        profile_event(profile, t0, "popen_talker_stdin cmd=" + cmd.str());
        FILE *talker = popen(cmd.str().c_str(), "w");
        if (!talker) {
            throw std::runtime_error("popen talker stdin failed");
        }

        std::queue<StreamJob> decode_queue;
        std::mutex mu;
        std::condition_variable cv;
        bool producer_done = false;
        std::exception_ptr decoder_error = nullptr;
        std::atomic<bool> talker_closed{false};
        std::atomic<int> talker_rc{-1};
        std::vector<int16_t> merged;
        double total_gap = 0.0;
        double max_gap = 0.0;
        bool got_first_queue = false;
        bool got_first_input = false;
        Clock::time_point t_first_queue = t0;
        Clock::time_point t_first_input = t0;
        Clock::time_point t_generation_done = t0;
        int decoded_segments = 0;
        int skipped_segments = 0;
        int truncated_segments = 0;
        int written_segments = 0;
        const auto ref_decode_prefix = load_ref_decode_prefix(ref_bin);

        auto decode_codes = [&](const std::string &path, size_t *frame_count) -> std::vector<float> {
            auto data = read_all(path);
            const size_t frame_bytes = sizeof(int32_t) * 16;
            if (data.size() % frame_bytes != 0) {
                throw std::runtime_error("bad code file size: " + path);
            }
            if (frame_count) {
                *frame_count = data.size() / frame_bytes;
            }
            std::vector<std::array<int32_t, 16>> frames(data.size() / frame_bytes);
            if (!frames.empty()) {
                std::memcpy(frames.data(), data.data(), data.size());
            }
            return decode_with_reference_prefix(codec, frames, ref_decode_prefix, buckets, first_chunk, chunk, ctx_limit);
        };

        auto decode_codes_streaming = [&](const StreamJob &job, size_t *frame_count) -> std::vector<float> {
            const size_t frame_bytes = sizeof(int32_t) * 16;
            std::vector<std::array<int32_t, 16>> frames;
            frames.insert(frames.end(), ref_decode_prefix.begin(), ref_decode_prefix.end());
            std::vector<std::pair<int, std::vector<float>>> chunks;
            size_t generated = 0;
            int done = 0;

            auto append_available_frames = [&]() {
                if (!exists(job.codes)) {
                    return;
                }
                auto data = read_all(job.codes);
                const size_t available = data.size() / frame_bytes;
                if (available <= generated) {
                    return;
                }
                const size_t old = generated;
                generated = available;
                frames.resize(ref_decode_prefix.size() + generated);
                std::memcpy(frames.data() + ref_decode_prefix.size() + old,
                            data.data() + old * frame_bytes,
                            (generated - old) * frame_bytes);
            };

            auto submit = [&](int start, int n) -> int {
                const int new_count = n - start;
                auto it = std::find_if(buckets.begin(), buckets.end(), [&](int b) { return b >= new_count; });
                if (it == buckets.end()) {
                    throw std::runtime_error("no codec bucket for stdin chunk");
                }
                const int b = *it;
                const int ctx = std::min({start, ctx_limit, b - new_count});
                std::vector<std::array<int32_t, 16>> codes(frames.begin() + (start - ctx), frames.begin() + n);
                chunks.emplace_back(start, codec.decode(b, codes, ctx));
                return n;
            };

            while (true) {
                append_available_frames();
                while (static_cast<int>(frames.size()) - done >= (done == 0 ? first_chunk : chunk)) {
                    const int next = done == 0 ? first_chunk : chunk;
                    done = submit(done, done + next);
                }
                if (exists(job.done)) {
                    append_available_frames();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (static_cast<int>(frames.size()) > done) {
                submit(done, static_cast<int>(frames.size()));
            }
            if (frame_count) {
                *frame_count = generated;
            }

            std::sort(chunks.begin(), chunks.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
            });
            size_t total_samples = 0;
            for (const auto &chunk_wav : chunks) {
                total_samples += chunk_wav.second.size();
            }
            std::vector<float> wav;
            wav.reserve(total_samples);
            for (auto &chunk_wav : chunks) {
                wav.insert(wav.end(), chunk_wav.second.begin(), chunk_wav.second.end());
            }

            const size_t ref_decode_samples = reference_audio_cut_samples(
                ref_decode_prefix.size(), generated, wav.size());
            if (ref_decode_samples > 0) {
                if (wav.size() <= ref_decode_samples) {
                    wav.clear();
                } else {
                    wav.erase(wav.begin(), wav.begin() + static_cast<ptrdiff_t>(ref_decode_samples));
                }
            }
            return wav;
        };

        auto decode_job_samples = [&](const StreamJob &job) -> std::vector<int16_t> {
            size_t code_frames = 0;
            auto wav_f32 = env_int("Q3TTS_STDIN_STREAM_DECODE", 1) != 0
                ? decode_codes_streaming(job, &code_frames)
                : decode_codes(job.codes, &code_frames);
            if (job.max_frames > 0 && code_frames >= static_cast<size_t>(job.max_frames)) {
                ++truncated_segments;
                std::cout << "stream_segment_truncated " << job.index
                          << " frames " << code_frames
                          << " max " << job.max_frames
                          << " text " << job.text << std::endl;
            }
            std::vector<int16_t> samples = f32_to_pcm16(std::move(wav_f32), false);
            trim_leading_samples(samples, job.leadin_trim_samples);
            trim_trailing_silence(samples);
            if (samples.empty() || rms_i16(samples, 0, samples.size()) <= 0.003) {
                std::cout << "stream_segment_skip " << job.index
                          << " reason silent_audio text " << job.text << std::endl;
                return {};
            }
            postprocess_pcm16(samples);
            return samples;
        };

        auto cleanup_stream_job_inputs = [](const StreamJob &job) {
            std::remove(job.prefill.c_str());
            std::remove(job.trailing.c_str());
            std::remove(job.pad.c_str());
            std::remove(job.codes.c_str());
            std::remove(job.done.c_str());
        };

        std::thread decoder([&]() {
            try {
                while (true) {
                    StreamJob job;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&]() { return producer_done || !decode_queue.empty(); });
                        if (decode_queue.empty()) {
                            return;
                        }
                        job = std::move(decode_queue.front());
                        decode_queue.pop();
                    }

                    const auto decode0 = Clock::now();
                    auto samples = decode_job_samples(job);
                    cleanup_stream_job_inputs(job);
                    if (samples.empty()) {
                        ++skipped_segments;
                        t_generation_done = Clock::now();
                        continue;
                    }
                    write_wav_i16_samples(job.wav, samples);
                    const double decode_s = seconds_since(decode0, Clock::now());
                    const double wait_s = 0.0;
                    const double audio_s = static_cast<double>(samples.size()) / 24000.0;

                    double gap_s = 0.0;
#ifdef Q3TTS_ENABLE_SDK_AUDIO
                    if (realtime_player) {
                        gap_s = realtime_player->enqueue_mono24k(samples);
                        if (!got_first_queue) {
                            got_first_queue = true;
                            t_first_queue = Clock::now();
                        }
                    }
#endif
                    total_gap += gap_s;
                    max_gap = std::max(max_gap, gap_s);

                    if (!merged.empty()) {
                        merged.insert(merged.end(), 4800, 0);
                    }
                    merged.insert(merged.end(), samples.begin(), samples.end());
                    ++decoded_segments;
                    t_generation_done = Clock::now();

                    std::cout.setf(std::ios::fixed);
                    std::cout.precision(3);
                    std::cout << "stream_segment " << job.index
                              << " wait " << wait_s << "s"
                              << " decode " << decode_s << "s"
                              << " audio " << audio_s << "s"
                              << " gap " << gap_s << "s"
                              << " wav " << job.wav << std::endl;
                }
            } catch (...) {
                decoder_error = std::current_exception();
            }
        });

        auto submit_line = [&](std::string line) -> std::pair<int, int> {
            line = trim(line);
            if (line.empty()) {
                return {0, 0};
            }
            const int first_index = written_segments + 1;
            if (!args.hotwords.empty()) {
                line = apply_hotwords(line, args.hotwords);
            }
            Args line_args = args;
            line_args.text = line;
            std::vector<std::string> segments;
            if (args.no_clone_split) {
                segments = split_stdin_text(line, has_full_reference_prompt(ref_bin), line_args.language);
            } else {
                segments = prepare_clone_segments(line_args);
            }
            for (const auto &segment : segments) {
                StreamJob job;
                job.index = ++written_segments;
                job.text = segment;
                if (has_ref && needs_clone_leadin(job.text, args.clone_leadin)) {
                    job.synth_text = args.clone_leadin + job.text;
                    job.leadin_trim_samples = clone_leadin_trim_samples(args.clone_leadin);
                } else {
                    job.synth_text = job.text;
                }
                const std::string base = "/tmp/q3tts_stdin_" + pid + "_" + std::to_string(job.index);
                job.prefill = base + "_prefill.bin";
                job.trailing = base + "_trailing.bin";
                job.pad = base + "_pad.bin";
                job.codes = base + "_codes.bin";
                job.done = base + "_done";
                job.wav = base + ".wav";
                job.max_frames = stdin_max_frames;
                temps.push_back(job.prefill);
                temps.push_back(job.trailing);
                temps.push_back(job.pad);
                temps.push_back(job.codes);
                temps.push_back(job.done);
                temps.push_back(job.wav);

                q3tts_frontend::FrontendConfig fc = base_fc;
                fc.text = job.synth_text;
                auto input = frontend.build(fc);
                write_all(job.prefill,
                          reinterpret_cast<const uint8_t *>(input.prefill.data()),
                          input.prefill.size() * sizeof(float));
                write_all(job.trailing,
                          reinterpret_cast<const uint8_t *>(input.trailing.data()),
                          input.trailing.size() * sizeof(float));
                write_all(job.pad,
                          reinterpret_cast<const uint8_t *>(input.pad.data()),
                          input.pad.size() * sizeof(float));
                job.np = input.n_prefill;
                job.nt = input.n_trailing;
                if (job.np > stdin_max_prefill) {
                    throw std::runtime_error("stdin segment prefill exceeds Q3TTS_STDIN_MAX_PREFILL");
                }

                if (std::fprintf(talker, "%s %lld %s %lld %s %d %s %s\n",
                                 job.prefill.c_str(), static_cast<long long>(job.np),
                                 job.trailing.c_str(), static_cast<long long>(job.nt),
                                 job.pad.c_str(), job.max_frames,
                                 job.codes.c_str(), job.done.c_str()) < 0 ||
                    std::fflush(talker) != 0) {
                    throw std::runtime_error("write talker stdin job failed");
                }
                if (!got_first_input) {
                    got_first_input = true;
                    t_first_input = Clock::now();
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    decode_queue.push(job);
                }
                cv.notify_one();
                std::cout << "stream_text " << job.index << " " << segment << std::endl;
            }
            return {first_index, written_segments};
        };

        if (!input_lines.empty()) {
            for (const auto &line : input_lines) {
                const auto range = submit_line(line);
                std::cout << "stream_request " << range.first << " " << range.second << std::endl;
            }
        } else {
            std::string line;
            while (std::getline(std::cin, line)) {
                const auto range = submit_line(line);
                std::cout << "stream_request " << range.first << " " << range.second << std::endl;
            }
        }

        {
            std::lock_guard<std::mutex> lk(mu);
            producer_done = true;
        }
        cv.notify_all();

        const int rc = pclose(talker);
        talker_rc.store(rc);
        talker_closed.store(true);
        cv.notify_all();

        if (decoder.joinable()) {
            decoder.join();
        }
        if (decoder_error) {
            std::rethrow_exception(decoder_error);
        }
        if (rc != 0) {
            throw std::runtime_error("stdin talker exited with status " + std::to_string(rc));
        }
        if (truncated_segments > 0 && env_int("Q3TTS_ALLOW_TRUNCATED", 0) == 0) {
            throw std::runtime_error(
                "stdin segment reached max frames without EOS; split text or increase --frames "
                "(set Q3TTS_ALLOW_TRUNCATED=1 to keep partial audio)");
        }

#ifdef Q3TTS_ENABLE_SDK_AUDIO
        if (realtime_player) {
            realtime_player->finish();
        }
#endif
        write_wav_i16_samples(args.wav, merged);
        cleanup();

        const double wall = seconds_since(t0, Clock::now());
        const double input_wall = got_first_input ? seconds_since(t_first_input, t_generation_done) : wall;
        const double audio = static_cast<double>(merged.size()) / 24000.0;
        const double warm = got_first_queue ? seconds_since(t_first_queue, t_generation_done) : input_wall;
        std::cout.setf(std::ios::fixed);
        std::cout.precision(2);
        std::cout << "stdin_realtime segments " << decoded_segments
                  << "  skipped " << skipped_segments
                  << "  gen_wall " << input_wall << "s"
                  << "  wall " << wall << "s"
                  << "  audio " << audio << "s"
                  << "  genRTF " << (audio > 0 ? input_wall / audio : 0.0)
                  << "  RTF " << (audio > 0 ? wall / audio : 0.0)
                  << "  warmRTF " << (audio > 0 ? warm / audio : 0.0)
                  << "  total_gap " << total_gap << "s"
                  << "  max_gap " << max_gap << "s\n";
        std::cout << "wav " << args.wav << "\n";
        return 0;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace

namespace qwen3_tts {
int run_cli(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!args.stdin_segments && !args.text.empty() && !args.hotwords.empty()) {
            args.text = apply_hotwords(args.text, args.hotwords);
        }
        if (args.dump_segments) {
            std::vector<std::string> segments = prepare_clone_segments(args);
            for (size_t i = 0; i < segments.size(); ++i) {
                std::cout << (i + 1) << "\t" << utf8_len(segments[i]) << "\t" << segments[i] << "\n";
            }
            return 0;
        }
        maybe_select_full_prompt_talker(args);
        set_default_runtime_env(args.talker_gguf);
        const bool has_ref = has_clone_reference(args);
        if (has_ref) {
            set_env_override(
                "Q3TTS_TALKER_REPETITION_PENALTY",
                env_str("Q3TTS_CLONE_TALKER_REPETITION_PENALTY", "1.15"));
        }
        if (args.stdin_segments) {
            return run_stdin_segments(args);
        }
        if (has_ref && !args.text.empty() && !args.dump_ids && !args.frontend_only) {
            std::vector<std::string> segments = prepare_clone_segments(args);
            if (segments.size() == 1) {
                args.text = segments[0];
                args.no_clone_split = true;
            } else if (!args.no_clone_split || env_int("Q3TTS_CLONE_UNSAFE_NOSPLIT", 0) == 0) {
                if (args.no_clone_split) {
                    std::cerr << "clone_force_split segments " << segments.size()
                              << " text_chars " << utf8_len(args.text) << "\n";
                }
                if (has_full_reference_prompt(args.ref_bin)) {
                    Args stream_args = args;
                    stream_args.no_clone_split = true;
                    return run_stdin_segments(stream_args, segments);
                }
                return run_clone_split(args, argv[0], segments);
            }
        }
        const bool profile = env_int("Q3TTS_PROFILE", 0) != 0;
        const auto t_profile0 = Clock::now();
        profile_event(profile, t_profile0, "driver_start");

        if (!std::getenv("Q3TTS_SKIP_TCM_CLEAR")) {
            const int clear_rc = std::system("spacemit-tcm-smi -c >/dev/null 2>&1");
            (void)clear_rc;
        }
        profile_event(profile, t_profile0, "tcm_clear_done");

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "q3tts_cpp_driver");

        const std::string tmp_tag = std::to_string(static_cast<long long>(getpid()));
        const std::string prefill_bin = "/tmp/q3tts_cpp_" + tmp_tag + "_prefill.bin";
        const std::string trailing_bin = "/tmp/q3tts_cpp_" + tmp_tag + "_trailing.bin";
        const std::string pad_bin = "/tmp/q3tts_cpp_" + tmp_tag + "_pad.bin";
        int64_t np = 0;
        int64_t nt = 0;
        if (!args.text.empty()) {
            if (args.dump_ids) {
                auto ids = q3tts_frontend::tokenize_prompt(args.model_dir, args.text);
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (i != 0) {
                        std::cout << " ";
                    }
                    std::cout << ids[i];
                }
                std::cout << "\n";
                return 0;
            }
            q3tts_frontend::FrontendConfig fc;
            fc.model_dir = args.model_dir;
            fc.text = args.text;
            fc.ref_wav = args.ref_wav;
            fc.ref_bin = args.ref_bin;
            fc.language = args.language;
            fc.talker_gguf = args.talker_gguf;
            fc.cp_gguf = args.cp_gguf;
            fc.frontend_threads = env_int("Q3TTS_FRONTEND_THREADS", 2);
            fc.full_prompt_non_streaming = env_int("Q3TTS_FULL_PROMPT_NON_STREAMING", 0) != 0;
            auto input = q3tts_frontend::build(env, fc);
            write_all(prefill_bin,
                      reinterpret_cast<const uint8_t *>(input.prefill.data()),
                      input.prefill.size() * sizeof(float));
            write_all(trailing_bin,
                      reinterpret_cast<const uint8_t *>(input.trailing.data()),
                      input.trailing.size() * sizeof(float));
            write_all(pad_bin,
                      reinterpret_cast<const uint8_t *>(input.pad.data()),
                      input.pad.size() * sizeof(float));
            np = input.n_prefill;
            nt = input.n_trailing;
            profile_event(profile, t_profile0,
                          "frontend_materialized np=" + std::to_string(np) + " nt=" + std::to_string(nt));
        } else {
            if (args.npz.empty()) {
                args.npz = "e2e_spk.npz";
            }
            auto npz = load_npz_stored(args.npz);
            auto prefill = parse_npy(npz.at("prefill.npy"));
            auto trailing = parse_npy(npz.at("trailing.npy"));
            auto pad = parse_npy(npz.at("pad.npy"));
            if (prefill.descr != "<f4" || trailing.descr != "<f4" || pad.descr != "<f4") {
                throw std::runtime_error("expected float32 prefill/trailing/pad");
            }
            write_all(prefill_bin, prefill.data, prefill.bytes);
            write_all(trailing_bin, trailing.data, trailing.bytes);
            write_all(pad_bin, pad.data, pad.bytes);
            np = first_dim(prefill, "prefill");
            nt = first_dim(trailing, "trailing");
            profile_event(profile, t_profile0, "npz_materialized");
        }
        if (args.frontend_only) {
            std::cout << "frontend np " << np << " nt " << nt << "\n";
            return 0;
        }

        const int codec_threads = env_int("Q3TTS_CODEC_THREADS", 3);
        if (const char *aff = std::getenv("Q3TTS_CODEC_AFFINITY"); aff && *aff) {
            setenv("SPACEMIT_EP_INTRA_THREAD_AFFINITY", aff, 1);
            if (!std::getenv("SPACEMIT_EP_INTRA_THREAD_NUM")) {
                setenv("SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(codec_threads).c_str(), 1);
            }
            if (!std::getenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD")) {
                setenv("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "0", 1);
            }
        }

        const bool no_ref_text = !has_ref && !args.text.empty();
        const std::vector<int> buckets = no_ref_text
            ? parse_int_list(env_str("Q3TTS_NOREF_CODEC_BUCKETS", ""), std::vector<int>{25})
            : parse_int_list(env_str("Q3TTS_CODEC_BUCKETS", ""), std::vector<int>{50});
        const int chunk = no_ref_text ? env_int("Q3TTS_NOREF_CODEC_CHUNK", 25)
                                      : env_int("Q3TTS_CODEC_CHUNK", 50);
        const int first_chunk = no_ref_text ? env_int("Q3TTS_NOREF_CODEC_FIRST_CHUNK", chunk)
                                            : env_int("Q3TTS_CODEC_FIRST_CHUNK", chunk);
        const int ctx_limit = env_int("Q3TTS_CODEC_CTX", 25);
        const int codec_nice = env_int("Q3TTS_CODEC_NICE", 0);
        if (codec_nice > 0) {
            if (setpriority(PRIO_PROCESS, 0, codec_nice) != 0) {
                std::perror("setpriority codec");
            }
        }

        q3tts_codec::DecoderPoolConfig codec_cfg;
        codec_cfg.model_dir = args.model_dir;
        codec_cfg.buckets = buckets;
        codec_cfg.intra_threads = codec_threads;
        codec_cfg.on_bucket_warm = [&](int b) {
            profile_event(profile, t_profile0, "codec_warm bucket=" + std::to_string(b));
        };
        q3tts_codec::DecoderPool codec(env, codec_cfg);
        if (codec_nice > 0 && setpriority(PRIO_PROCESS, 0, 0) != 0) {
            std::perror("setpriority main");
        }

        const auto ref_decode_prefix = load_ref_decode_prefix(args.ref_bin);

        std::vector<std::array<int32_t, 16>> frames;
        frames.insert(frames.end(), ref_decode_prefix.begin(), ref_decode_prefix.end());
        std::vector<std::pair<int, std::vector<float>>> chunks;
        std::mutex mu;
        std::condition_variable cv;
        std::queue<std::tuple<std::vector<std::array<int32_t, 16>>, int, int, int>> jobs;
        bool stop = false;

        std::thread worker([&]() {
            while (true) {
                std::tuple<std::vector<std::array<int32_t, 16>>, int, int, int> job;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return stop || !jobs.empty(); });
                    if (jobs.empty()) {
                        return;
                    }
                    job = std::move(jobs.front());
                    jobs.pop();
                }
                auto &[codes, off, ctx, bucket] = job;
                const auto t_job0 = Clock::now();
                auto wav = codec.decode(bucket, codes, ctx);
                const auto t_job1 = Clock::now();
                if (profile) {
                    std::stringstream msg;
                    msg << "codec_done off=" << off
                        << " input=" << codes.size()
                        << " new=" << (static_cast<int>(codes.size()) - ctx)
                        << " ctx=" << ctx
                        << " bucket=" << bucket
                        << " run_ms=" << (seconds_since(t_job0, t_job1) * 1000.0);
                    profile_event(true, t_profile0, msg.str());
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    chunks.emplace_back(off, std::move(wav));
                }
            }
        });

        auto submit = [&](int done, int n) -> int {
            const int new_count = n - done;
            auto it = std::find_if(buckets.begin(), buckets.end(), [&](int b) { return b >= new_count; });
            if (it == buckets.end()) {
                throw std::runtime_error("no codec bucket for chunk");
            }
            const int b = *it;
            const int ctx = std::min({done, ctx_limit, b - new_count});
            std::vector<std::array<int32_t, 16>> codes(frames.begin() + (done - ctx), frames.begin() + n);
            {
                std::lock_guard<std::mutex> lk(mu);
                jobs.emplace(std::move(codes), done, ctx, b);
            }
            if (profile) {
                std::stringstream msg;
                msg << "codec_submit off=" << done
                    << " input=" << (new_count + ctx)
                    << " new=" << new_count
                    << " ctx=" << ctx
                    << " bucket=" << b;
                profile_event(true, t_profile0, msg.str());
            }
            cv.notify_one();
            return n;
        };

        const std::string driver = env_str("TALKER_DRIVER", "./talker_driver");
        const std::string talker_cpuset = env_str("TALKER_CPUSET", "4-7");
        const std::string talker_gguf = model_file(args.model_dir, "gguf", args.talker_gguf);
        const std::string cp_gguf = model_file(args.model_dir, "gguf", args.cp_gguf);
        std::stringstream cmd;
        cmd << "taskset -c " << talker_cpuset << " " << driver
            << " " << talker_gguf
            << " " << cp_gguf
            << " aux " << prefill_bin << " " << np
            << " " << trailing_bin << " " << nt
            << " " << pad_bin << " " << args.frames;

        profile_event(profile, t_profile0, "popen_talker cmd=" + cmd.str());
        FILE *pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("popen talker failed");
        }
        profile_event(profile, t_profile0, "popen_done");

        auto t0 = Clock::now();
        Clock::time_point t_first;
        bool got_first = false;
        int done = 0;
        int generated_frames = 0;
        while (true) {
            std::array<int32_t, 16> f {};
            const size_t got = fread(f.data(), sizeof(int32_t), 16, pipe);
            if (got < 16) {
                break;
            }
            if (!got_first) {
                t_first = Clock::now();
                got_first = true;
                profile_event(profile, t_profile0, "first_frame");
            }
            frames.push_back(f);
            ++generated_frames;
            const int next_chunk = done == 0 ? first_chunk : chunk;
            if (static_cast<int>(frames.size()) - done >= next_chunk) {
                done = submit(done, done + next_chunk);
            }
        }
        profile_event(profile, t_profile0, "talker_stdout_eof frames=" + std::to_string(generated_frames));
        if (static_cast<int>(frames.size()) > done) {
            submit(done, static_cast<int>(frames.size()));
        }
        if (const char *dump_codes = std::getenv("Q3TTS_DUMP_CODES"); dump_codes && *dump_codes) {
            const auto *dump_begin = frames.data() + static_cast<ptrdiff_t>(ref_decode_prefix.size());
            write_all(dump_codes,
                      reinterpret_cast<const uint8_t *>(dump_begin),
                      static_cast<size_t>(generated_frames) * sizeof(frames[0]));
        }
        const auto t_pclose0 = Clock::now();
        const int rc = pclose(pipe);
        const auto t_pclose1 = Clock::now();
        if (profile) {
            std::stringstream msg;
            msg << "pclose_done rc=" << rc
                << " wait_ms=" << (seconds_since(t_pclose0, t_pclose1) * 1000.0);
            profile_event(true, t_profile0, msg.str());
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_one();
        worker.join();
        profile_event(profile, t_profile0, "codec_worker_join chunks=" + std::to_string(chunks.size()));
        if (rc != 0) {
            std::remove(prefill_bin.c_str());
            std::remove(trailing_bin.c_str());
            std::remove(pad_bin.c_str());
            throw std::runtime_error("talker exited with status " + std::to_string(rc));
        }

        auto t1 = Clock::now();
        std::sort(chunks.begin(), chunks.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });
        std::vector<float> wav;
        size_t total_samples = 0;
        for (const auto &c : chunks) {
            total_samples += c.second.size();
        }
        wav.reserve(total_samples);
        for (auto &c : chunks) {
            wav.insert(wav.end(), c.second.begin(), c.second.end());
        }
        const size_t ref_decode_samples = reference_audio_cut_samples(
            ref_decode_prefix.size(), static_cast<size_t>(generated_frames), wav.size());
        if (ref_decode_samples > 0) {
            if (wav.size() <= ref_decode_samples) {
                wav.clear();
            } else {
                wav.erase(wav.begin(), wav.begin() + static_cast<ptrdiff_t>(ref_decode_samples));
            }
        }
        const double wall = seconds_since(t0, t1);
        const double audio = static_cast<double>(wav.size()) / 24000.0;
        const double warm = got_first ? seconds_since(t_first, t1) : wall;
        if (args.frames > 0 && generated_frames >= args.frames &&
            env_int("Q3TTS_ALLOW_TRUNCATED", 0) == 0) {
            std::remove(prefill_bin.c_str());
            std::remove(trailing_bin.c_str());
            std::remove(pad_bin.c_str());
            throw std::runtime_error(
                "talker reached max frames without EOS; split text or increase --frames "
                "(set Q3TTS_ALLOW_TRUNCATED=1 to keep partial audio)");
        }
        if (profile) {
            std::stringstream msg;
            msg << "wav_assembled samples=" << wav.size()
                << " chunks=" << chunks.size()
                << " wall_ms=" << (wall * 1000.0)
                << " warm_ms=" << (warm * 1000.0);
            profile_event(true, t_profile0, msg.str());
        }
        std::cout.setf(std::ios::fixed);
        std::cout.precision(2);
        std::cout << "frames " << generated_frames
                  << "  E2E wall " << wall << "s"
                  << "  audio " << audio << "s"
                  << "  RTF " << (wall / audio)
                  << "  warmRTF " << (warm / audio) << "\n";
        write_wav_i16(args.wav, wav);
        std::remove(prefill_bin.c_str());
        std::remove(trailing_bin.c_str());
        std::remove(pad_bin.c_str());
        std::cout << "wav " << args.wav << "\n";
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace qwen3_tts
