#pragma once

#include "server-speech.h"

class server_speech_backend {
  public:
    virtual ~server_speech_backend() = default;

    virtual const char * name() const = 0;
    virtual server_speech_result synthesize(const nlohmann::ordered_json & body) = 0;
};
