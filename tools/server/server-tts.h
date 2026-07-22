#pragma once

#include "server-http.h"
#include "server-media.h"

#include <cstddef>
#include <string>

void                server_tts_validate_body_size(size_t bytes);
std::string         server_tts_request_text(const json & body);
server_http_res_ptr server_tts_make_response(server_media_tts_result result);
