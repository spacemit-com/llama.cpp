#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_tts {

constexpr int hidden_size = 1024;
constexpr int code_groups = 16;

using code_frame     = std::array<int32_t, code_groups>;
using frame_callback = std::function<void(const code_frame &)>;

class talker_engine {
  public:
    talker_engine(const std::string & talker_path,
                  const std::string & code_predictor_path,
                  const std::string & aux_path,
                  int                 max_prefill,
                  int                 max_frames,
                  int                 threads);
    ~talker_engine();

    talker_engine(const talker_engine &)             = delete;
    talker_engine & operator=(const talker_engine &) = delete;

    void generate(const std::vector<float> &             prefill,
                  const std::vector<float> &             trailing,
                  const std::array<float, hidden_size> & pad,
                  uint32_t                               max_frames,
                  const frame_callback &                 on_frame);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

}  // namespace qwen3_tts
