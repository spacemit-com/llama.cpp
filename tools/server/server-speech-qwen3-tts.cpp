#include "server-speech-qwen3-tts.h"

#include "log.h"
#include "server-speech-backend.h"
#include "server-common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::ordered_json;

static bool speech_path_exists(const std::string & path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::exists(path, ec);
}

static bool qwen3_tts_name_matches(const std::string & value) {
    return value.find("qwen3-tts") != std::string::npos ||
           value.find("Qwen3-TTS") != std::string::npos ||
           value.find("q3tts") != std::string::npos;
}

static bool qwen3_tts_config_matches(const common_params & params) {
    if (params.smt_config_dir.empty()) {
        return false;
    }
    if (params.media_backend != "smt" && params.media_backend != "auto") {
        return false;
    }
    if (qwen3_tts_name_matches(params.model.path) || qwen3_tts_name_matches(params.model.name)) {
        return true;
    }
    for (const auto & alias : params.model_alias) {
        if (qwen3_tts_name_matches(alias)) {
            return true;
        }
    }

    const std::filesystem::path dir(params.smt_config_dir);
    return speech_path_exists((dir / "onnx" / "codec_decoder_t25.q.onnx").string()) ||
           speech_path_exists((dir / "gguf" / "qwen3-tts-0.6b-talker-qkv-gateup-q8_0-side.gguf").string());
}

static std::string speech_current_exe_dir() {
    char path[4096];
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) {
        return {};
    }
    path[n] = '\0';
    return std::filesystem::path(path).parent_path().string();
}

static int speech_env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

static std::string speech_env_str(const char * name, const std::string & fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value;
}

static std::string qwen3_tts_find_runner() {
    if (const char * env = std::getenv("Q3TTS_RUN_BIN"); env != nullptr && env[0] != '\0') {
        return env;
    }
    const std::string exe_dir = speech_current_exe_dir();
    if (!exe_dir.empty()) {
        const std::string colocated = (std::filesystem::path(exe_dir) / "q3tts-run").string();
        if (speech_path_exists(colocated)) {
            return colocated;
        }
    }
    return "q3tts-run";
}

static std::string qwen3_tts_default_ref_bin(const std::string & model_dir) {
    const std::filesystem::path dir(model_dir);
    const std::vector<std::filesystem::path> candidates = {
        dir / "refs" / "default.spk.bin",
        dir / "refs" / "default.prompt.bin",
        dir / "refs" / "warm_female.spk.bin",
        dir / "refs" / "warm_female_full_prompt.spk.bin",
    };
    for (const auto & path : candidates) {
        if (speech_path_exists(path.string())) {
            return path.string();
        }
    }
    return {};
}

static bool speech_write_all_fd(int fd, const std::string & data) {
    const char * ptr = data.data();
    size_t left = data.size();
    while (left > 0) {
        const ssize_t n = write(fd, ptr, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        ptr += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}

static std::vector<uint8_t> speech_read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open wav segment: " + path);
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

static uint16_t speech_u16le(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t speech_u32le(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static void speech_put_u16le(std::string & out, uint16_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
}

static void speech_put_u32le(std::string & out, uint32_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 24) & 0xff));
}

static std::vector<uint8_t> speech_extract_pcm16_mono24k(const std::string & path) {
    const std::vector<uint8_t> wav = speech_read_file(path);
    if (wav.size() < 44 || std::memcmp(wav.data(), "RIFF", 4) != 0 || std::memcmp(wav.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("unsupported wav segment header: " + path);
    }

    bool fmt_ok = false;
    size_t data_offset = 0;
    size_t data_size = 0;
    for (size_t pos = 12; pos + 8 <= wav.size();) {
        const uint8_t * chunk = wav.data() + pos;
        const uint32_t size = speech_u32le(chunk + 4);
        const size_t payload = pos + 8;
        if (payload + size > wav.size()) {
            break;
        }
        if (std::memcmp(chunk, "fmt ", 4) == 0 && size >= 16) {
            const uint16_t format = speech_u16le(wav.data() + payload);
            const uint16_t channels = speech_u16le(wav.data() + payload + 2);
            const uint32_t sample_rate = speech_u32le(wav.data() + payload + 4);
            const uint16_t bits = speech_u16le(wav.data() + payload + 14);
            fmt_ok = format == 1 && channels == 1 && sample_rate == 24000 && bits == 16;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_offset = payload;
            data_size = size;
        }
        pos = payload + size + (size & 1u);
    }
    if (!fmt_ok || data_offset == 0) {
        throw std::runtime_error("unsupported wav segment format: " + path);
    }
    return std::vector<uint8_t>(wav.begin() + static_cast<ptrdiff_t>(data_offset),
                                wav.begin() + static_cast<ptrdiff_t>(data_offset + data_size));
}

static std::string speech_make_wav(const std::vector<uint8_t> & pcm) {
    std::string out;
    out.reserve(44 + pcm.size());
    out.append("RIFF", 4);
    speech_put_u32le(out, static_cast<uint32_t>(36 + pcm.size()));
    out.append("WAVE", 4);
    out.append("fmt ", 4);
    speech_put_u32le(out, 16);
    speech_put_u16le(out, 1);
    speech_put_u16le(out, 1);
    speech_put_u32le(out, 24000);
    speech_put_u32le(out, 24000 * 2);
    speech_put_u16le(out, 2);
    speech_put_u16le(out, 16);
    out.append("data", 4);
    speech_put_u32le(out, static_cast<uint32_t>(pcm.size()));
    out.append(reinterpret_cast<const char *>(pcm.data()), pcm.size());
    return out;
}

static std::string speech_merge_pcm16_mono24k_wavs(const std::vector<std::string> & paths, int pause_ms, double & audio_seconds) {
    std::vector<uint8_t> pcm;
    const size_t pause_bytes = static_cast<size_t>(std::max(0, pause_ms)) * 24 * 2;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (i > 0) {
            pcm.insert(pcm.end(), pause_bytes, 0);
        }
        auto segment = speech_extract_pcm16_mono24k(paths[i]);
        pcm.insert(pcm.end(), segment.begin(), segment.end());
    }
    audio_seconds = static_cast<double>(pcm.size()) / (24000.0 * 2.0);
    return speech_make_wav(pcm);
}

static void speech_replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::vector<std::pair<std::string, std::string>> speech_hotwords_from_json(const json & body) {
    std::vector<std::pair<std::string, std::string>> hotwords;
    const json * raw = nullptr;
    if (body.contains("hotwords")) {
        raw = &body.at("hotwords");
    } else if (body.contains("lexicon")) {
        raw = &body.at("lexicon");
    }
    if (raw == nullptr) {
        return hotwords;
    }
    if (raw->is_object()) {
        for (const auto & item : raw->items()) {
            if (item.value().is_string() && !item.key().empty()) {
                hotwords.emplace_back(item.key(), item.value().get<std::string>());
            }
        }
    } else if (raw->is_array()) {
        for (const auto & item : *raw) {
            if (!item.is_object()) {
                continue;
            }
            std::string from;
            std::string to;
            for (const char * key : {"word", "from", "text"}) {
                if (item.contains(key) && item.at(key).is_string()) {
                    from = item.at(key).get<std::string>();
                    break;
                }
            }
            for (const char * key : {"phoneme", "to", "replacement"}) {
                if (item.contains(key) && item.at(key).is_string()) {
                    to = item.at(key).get<std::string>();
                    break;
                }
            }
            if (!from.empty() && !to.empty()) {
                hotwords.emplace_back(std::move(from), std::move(to));
            }
        }
    }
    std::sort(hotwords.begin(), hotwords.end(), [](const auto & a, const auto & b) {
        return a.first.size() > b.first.size();
    });
    return hotwords;
}

class qwen3_tts_backend : public server_speech_backend {
  public:
    explicit qwen3_tts_backend(const common_params & params);
    ~qwen3_tts_backend() override;

    const char * name() const override {
        return server_speech_qwen3_tts_name();
    }
    server_speech_result synthesize(const json & body) override;

  private:
    struct SegmentResult {
        std::string wav_path;
        std::string skip_reason;
        std::string skip_text;
    };

    bool enabled = false;
    std::string model_dir;
    std::string ref_file;
    int frames;
    int pause_ms;
    int ready_timeout_sec;
    int request_timeout_sec;

    std::mutex start_mutex;
    std::mutex request_mutex;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::deque<std::pair<int, int>> request_ranges;
    std::map<int, SegmentResult> segment_results;
    std::thread reader;
    pid_t child_pid = -1;
    int child_stdin = -1;
    bool ready = false;
    bool child_closed = false;

    server_speech_result synthesize_text(std::string text);
    void ensure_started();
    void start_process();
    void read_loop(int fd);
    void mark_closed();
    void stop();
};

qwen3_tts_backend::qwen3_tts_backend(const common_params & params) :
    enabled(qwen3_tts_config_matches(params)),
    model_dir(params.smt_config_dir),
    ref_file(params.vocoder.speaker_file),
    frames(speech_env_int("Q3TTS_SERVICE_FRAMES", 160)),
    pause_ms(speech_env_int("Q3TTS_SERVICE_PAUSE_MS", 200)),
    ready_timeout_sec(speech_env_int("Q3TTS_SERVICE_READY_TIMEOUT", 60)),
    request_timeout_sec(speech_env_int("Q3TTS_SERVICE_REQUEST_TIMEOUT", 180)) {
    if (ref_file.empty()) {
        ref_file = qwen3_tts_default_ref_bin(model_dir);
    }
    if (enabled && speech_env_int("Q3TTS_SERVICE_PREWARM", 1) != 0) {
        const std::string text = speech_env_str("Q3TTS_SERVICE_PREWARM_TEXT", "你好，这是预热。");
        if (!text.empty()) {
            try {
                (void) synthesize_text(text);
            } catch (const std::exception & e) {
                SRV_WRN("Qwen3-TTS prewarm failed: %s\n", e.what());
            }
        } else {
            try {
                ensure_started();
            } catch (const std::exception & e) {
                SRV_WRN("Qwen3-TTS runner startup failed: %s\n", e.what());
            }
        }
    }
}

qwen3_tts_backend::~qwen3_tts_backend() {
    stop();
}

server_speech_result qwen3_tts_backend::synthesize(const json & body) {
    std::string text;
    if (body.contains("input") && body.at("input").is_string()) {
        text = body.at("input").get<std::string>();
    } else if (body.contains("text") && body.at("text").is_string()) {
        text = body.at("text").get<std::string>();
    }
    if (text.empty()) {
        throw std::invalid_argument("\"input\" must be a non-empty string");
    }

    const std::string response_format = json_value(body, "response_format", std::string("wav"));
    if (!response_format.empty() && response_format != "wav") {
        throw std::invalid_argument("Qwen3-TTS speech currently supports response_format=wav");
    }

    for (const auto & hotword : speech_hotwords_from_json(body)) {
        speech_replace_all(text, hotword.first, hotword.second);
    }
    return synthesize_text(std::move(text));
}

server_speech_result qwen3_tts_backend::synthesize_text(std::string text) {
    if (!enabled) {
        throw std::runtime_error("Qwen3-TTS speech backend is not enabled");
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> req_lock(request_mutex);
    ensure_started();

    if (!speech_write_all_fd(child_stdin, text + "\n")) {
        throw std::runtime_error("failed to write Qwen3-TTS request");
    }

    std::pair<int, int> range;
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(request_timeout_sec);
        state_cv.wait_until(lock, deadline, [&]() {
            return !request_ranges.empty() || child_closed;
        });
        if (request_ranges.empty()) {
            throw std::runtime_error(child_closed ? "Qwen3-TTS runner exited" : "Qwen3-TTS request timeout");
        }
        range = request_ranges.front();
        request_ranges.pop_front();
    }
    if (range.first <= 0 || range.second < range.first) {
        throw std::runtime_error("empty text after Qwen3-TTS segmentation");
    }

    std::vector<std::string> wav_paths;
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(request_timeout_sec);
        state_cv.wait_until(lock, deadline, [&]() {
            if (child_closed) {
                return true;
            }
            for (int i = range.first; i <= range.second; ++i) {
                if (segment_results.find(i) == segment_results.end()) {
                    return false;
                }
            }
            return true;
        });
        for (int i = range.first; i <= range.second; ++i) {
            auto it = segment_results.find(i);
            if (it == segment_results.end()) {
                throw std::runtime_error(child_closed ? "Qwen3-TTS runner exited" : "Qwen3-TTS segment timeout");
            }
            if (!it->second.skip_reason.empty()) {
                throw std::runtime_error("Qwen3-TTS skipped segment: " + it->second.skip_text);
            }
            wav_paths.push_back(it->second.wav_path);
            segment_results.erase(it);
        }
    }

    double audio_seconds = 0.0;
    std::string wav = speech_merge_pcm16_mono24k_wavs(wav_paths, pause_ms, audio_seconds);
    for (const auto & path : wav_paths) {
        std::remove(path.c_str());
    }
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::move(wav),
        name(),
        range.second - range.first + 1,
        audio_seconds,
        std::chrono::duration<double>(t1 - t0).count(),
    };
}

void qwen3_tts_backend::ensure_started() {
    std::lock_guard<std::mutex> lock(start_mutex);
    if (child_pid > 0 && !child_closed) {
        return;
    }
    start_process();
}

void qwen3_tts_backend::start_process() {
    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
        throw std::runtime_error("pipe failed for Qwen3-TTS runner");
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        throw std::runtime_error("fork failed for Qwen3-TTS runner");
    }

    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(out_pipe[1], STDERR_FILENO);
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);

        setenv("Q3TTS_MODEL_DIR", model_dir.c_str(), 1);
        const std::string frames_str = std::to_string(frames);
        setenv("Q3TTS_STDIN_MAX_FRAMES", frames_str.c_str(), 1);

        std::vector<std::string> args = {
            qwen3_tts_find_runner(),
            "--stdin-segments",
            "--no-clone-split",
            "--frames",
            frames_str,
            "--wav",
            "/tmp/qwen3_tts_server_merged.wav",
        };
        if (!model_dir.empty()) {
            args.push_back("--model-dir");
            args.push_back(model_dir);
        }
        if (!ref_file.empty()) {
            const bool is_wav = ref_file.size() >= 4 && ref_file.substr(ref_file.size() - 4) == ".wav";
            args.push_back(is_wav ? "--ref-wav" : "--ref-bin");
            args.push_back(ref_file);
        }

        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (auto & arg : args) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);
        execv(argv[0], argv.data());
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    child_pid = pid;
    child_stdin = in_pipe[1];
    ready = false;
    child_closed = false;
    request_ranges.clear();
    segment_results.clear();
    reader = std::thread([this, fd = out_pipe[0]]() { read_loop(fd); });

    std::unique_lock<std::mutex> state_lock(state_mutex);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(ready_timeout_sec);
    state_cv.wait_until(state_lock, deadline, [&]() {
        return ready || child_closed;
    });
    if (!ready) {
        state_lock.unlock();
        if (child_stdin >= 0) {
            close(child_stdin);
            child_stdin = -1;
        }
        if (child_pid > 0) {
            int status = 0;
            kill(child_pid, SIGTERM);
            waitpid(child_pid, &status, 0);
            child_pid = -1;
        }
        if (reader.joinable()) {
            reader.join();
        }
        mark_closed();
        throw std::runtime_error("Qwen3-TTS runner did not become ready");
    }
}

void qwen3_tts_backend::read_loop(int fd) {
    FILE * stream = fdopen(fd, "r");
    if (stream == nullptr) {
        close(fd);
        mark_closed();
        return;
    }

    char * line = nullptr;
    size_t cap = 0;
    const std::regex segment_re("^stream_segment\\s+([0-9]+)\\b.*\\swav\\s+(\\S+)");
    const std::regex skip_re("^stream_segment_skip\\s+([0-9]+)\\s+reason\\s+(\\S+)\\s+text\\s+(.*)");
    const std::regex truncated_re("^stream_segment_truncated\\s+([0-9]+)\\s+frames\\s+([0-9]+)\\s+max\\s+([0-9]+)\\s+text\\s+(.*)");
    const std::regex request_re("^stream_request\\s+([0-9]+)\\s+([0-9]+)");
    while (getline(&line, &cap, stream) >= 0) {
        std::string msg(line);
        if (!msg.empty() && msg.back() == '\n') {
            msg.pop_back();
        }
        if (msg.find("talker_stdin_ready") != std::string::npos) {
            std::lock_guard<std::mutex> lock(state_mutex);
            ready = true;
            state_cv.notify_all();
            continue;
        }

        std::smatch match;
        if (std::regex_match(msg, match, request_re)) {
            std::lock_guard<std::mutex> lock(state_mutex);
            request_ranges.emplace_back(std::stoi(match[1].str()), std::stoi(match[2].str()));
            state_cv.notify_all();
            continue;
        }
        if (std::regex_match(msg, match, truncated_re)) {
            std::lock_guard<std::mutex> lock(state_mutex);
            auto & result = segment_results[std::stoi(match[1].str())];
            result.skip_reason = "truncated";
            result.skip_text = match[4].str();
            state_cv.notify_all();
            continue;
        }
        if (std::regex_match(msg, match, segment_re)) {
            std::lock_guard<std::mutex> lock(state_mutex);
            segment_results[std::stoi(match[1].str())].wav_path = match[2].str();
            state_cv.notify_all();
            continue;
        }
        if (std::regex_match(msg, match, skip_re)) {
            std::lock_guard<std::mutex> lock(state_mutex);
            auto & result = segment_results[std::stoi(match[1].str())];
            result.skip_reason = match[2].str();
            result.skip_text = match[3].str();
            state_cv.notify_all();
            continue;
        }
    }
    free(line);
    fclose(stream);
    mark_closed();
}

void qwen3_tts_backend::mark_closed() {
    std::lock_guard<std::mutex> lock(state_mutex);
    child_closed = true;
    state_cv.notify_all();
}

void qwen3_tts_backend::stop() {
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        if (child_stdin >= 0) {
            close(child_stdin);
            child_stdin = -1;
        }
        if (child_pid > 0) {
            int status = 0;
            if (waitpid(child_pid, &status, WNOHANG) == 0) {
                kill(child_pid, SIGTERM);
                waitpid(child_pid, &status, 0);
            }
            child_pid = -1;
        }
    }
    if (reader.joinable()) {
        reader.join();
    }
    mark_closed();
}

const char * server_speech_qwen3_tts_name() {
    return "qwen3-tts";
}

bool server_speech_qwen3_tts_config_matches(const common_params & params) {
    return qwen3_tts_config_matches(params);
}

std::unique_ptr<server_speech_backend> server_speech_qwen3_tts_create(const common_params & params) {
    return std::make_unique<qwen3_tts_backend>(params);
}
