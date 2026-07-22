#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_tts {

struct synthesis_stats {
    uint32_t segments     = 0;
    uint32_t sample_rate  = 24000;
    uint64_t samples      = 0;
    double   wall_seconds = 0.0;
};

struct synthesis_result {
    std::vector<uint8_t> wav;
    synthesis_stats      stats;
};

class runtime {
  public:
    runtime(const std::string & config_dir, std::string speaker_file);
    ~runtime();

    runtime(const runtime &)             = delete;
    runtime & operator=(const runtime &) = delete;

    synthesis_result synthesize(const std::string & text);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

}  // namespace qwen3_tts
