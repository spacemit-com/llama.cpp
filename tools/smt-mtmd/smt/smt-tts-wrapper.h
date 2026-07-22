#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct smt_tts_result {
    std::vector<uint8_t> wav;
    std::string          backend;
    uint32_t             segments     = 0;
    uint32_t             sample_rate  = 0;
    uint64_t             samples      = 0;
    double               wall_seconds = 0.0;
};

struct smt_tts_context {
    smt_tts_context(const smt_tts_context &)             = delete;
    smt_tts_context & operator=(const smt_tts_context &) = delete;
    ~smt_tts_context();

    static bool                             matches(const std::string & config_dir);
    static std::unique_ptr<smt_tts_context> create(const std::string & config_dir, const std::string & speaker_file);

    const char *   backend_name() const;
    smt_tts_result synthesize(const std::string & text);

  private:
    smt_tts_context();
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
