#pragma once

#include "common.h"

#include <memory>

class server_speech_backend;

bool server_speech_qwen3_tts_config_matches(const common_params & params);
const char * server_speech_qwen3_tts_name();
std::unique_ptr<server_speech_backend> server_speech_qwen3_tts_create(const common_params & params);
