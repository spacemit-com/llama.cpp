#pragma once

#include "common.h"

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <string>

struct server_speech_result {
    std::string wav;
    std::string backend;
    int segments = 0;
    double audio_seconds = 0.0;
    double wall_seconds = 0.0;
};

bool server_speech_config_matches(const common_params & params);
std::string server_speech_backend_name(const common_params & params);

class server_speech_service {
  public:
    explicit server_speech_service(const common_params & params);
    ~server_speech_service();

    server_speech_result synthesize(const nlohmann::ordered_json & body);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
