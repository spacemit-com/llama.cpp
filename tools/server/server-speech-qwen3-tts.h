#pragma once

#include "common.h"

#include <memory>
#include <string>

class server_speech_backend;

const char * server_speech_qwen3_tts_name();
bool server_speech_qwen3_tts_config_matches(const common_params & params);
std::unique_ptr<server_speech_backend> server_speech_qwen3_tts_create(const common_params & params);
