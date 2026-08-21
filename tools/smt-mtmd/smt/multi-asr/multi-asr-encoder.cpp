#include "multi-asr-encoder.h"

#include "ggml.h"               // ggml_time_ms
#include "smt-audio-wrapper.h"  // tools/mtmd/smt-audio-wrapper.h

#include <cstdio>
#include <fstream>
#include <stdexcept>

#ifndef _WIN32
#    include <unistd.h>
#endif

multi_asr_encoder::multi_asr_encoder()  = default;
multi_asr_encoder::~multi_asr_encoder() = default;

void multi_asr_encoder::init(const std::string & smt_config_dir, bool warmup) {
    audio_ = smt_audio_context::create(smt_config_dir, warmup);
    if (!audio_) {
        throw std::runtime_error("multi_asr_encoder: failed to create SMT audio context");
    }
    hidden_size_ = audio_->hidden_size();
    if (hidden_size_ <= 0) {
        throw std::runtime_error("multi_asr_encoder: invalid hidden_size");
    }
}

// smt_audio_context::encode_audio() takes a file path, so we spill the raw wav
// bytes to a temp file (same approach as the server path).
static std::string write_temp_wav(const std::vector<uint8_t> & data, std::string & err) {
    char      tmpl[] = "/tmp/multi-asr-XXXXXX";
    const int fd     = mkstemp(tmpl);
    if (fd < 0) {
        err = "mkstemp failed";
        return {};
    }
    const std::string path = tmpl;
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            close(fd);
            std::remove(path.c_str());
            err = "cannot open temp file";
            return {};
        }
        f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    }
    close(fd);
    return path;
}

bool multi_asr_encoder::encode(multi_asr_request & req) {
    if (!audio_) {
        req.error = "encoder not initialized";
        req.stage = multi_asr_stage::failed;
        return false;
    }

    std::string       err;
    const std::string tmp = write_temp_wav(req.audio, err);
    if (tmp.empty()) {
        req.error = "multi_asr_encoder: " + err;
        req.stage = multi_asr_stage::failed;
        return false;
    }

    const int64_t t0 = ggml_time_ms();
    try {
        req.embd = audio_->encode_audio(tmp);
    } catch (const std::exception & e) {
        std::remove(tmp.c_str());
        req.error = std::string("multi_asr_encoder: encode_audio failed: ") + e.what();
        req.stage = multi_asr_stage::failed;
        return false;
    }
    std::remove(tmp.c_str());

    req.timings.encode_ms = (double) (ggml_time_ms() - t0);

    if (req.embd.empty() || (req.embd.size() % (size_t) hidden_size_) != 0) {
        req.error = "multi_asr_encoder: invalid embedding shape";
        req.stage = multi_asr_stage::failed;
        return false;
    }
    req.n_audio_tokens         = (int32_t) (req.embd.size() / (size_t) hidden_size_);
    req.timings.n_audio_tokens = req.n_audio_tokens;
    req.stage                  = multi_asr_stage::decoding;
    return true;
}
