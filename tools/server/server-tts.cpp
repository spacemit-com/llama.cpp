#include "server-tts.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr size_t max_input_bytes   = 16 * 1024;
constexpr size_t max_body_bytes    = 64 * 1024;
constexpr size_t max_hotwords      = 64;
constexpr size_t max_hotword_bytes = 1024;

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        if (to.size() > from.size() && text.size() > max_input_bytes - (to.size() - from.size())) {
            throw std::invalid_argument("input is too large after hotword replacement");
        }
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

std::vector<std::pair<std::string, std::string>> parse_hotwords(const json & body) {
    const json * value = nullptr;
    if (body.contains("hotwords")) {
        value = &body.at("hotwords");
    } else if (body.contains("lexicon")) {
        value = &body.at("lexicon");
    }
    if (value == nullptr) {
        return {};
    }

    std::vector<std::pair<std::string, std::string>> result;
    const auto                                       append = [&result](std::string from, std::string to) {
        if (from.empty() || from.size() > max_hotword_bytes || to.size() > max_hotword_bytes) {
            throw std::invalid_argument("hotword entries have invalid lengths");
        }
        if (result.size() >= max_hotwords) {
            throw std::invalid_argument("too many hotword entries");
        }
        result.emplace_back(std::move(from), std::move(to));
    };
    if (value->is_object()) {
        for (const auto & item : value->items()) {
            if (!item.value().is_string()) {
                throw std::invalid_argument("hotword values must be strings");
            }
            append(item.key(), item.value().get<std::string>());
        }
        return result;
    }
    if (value->is_array()) {
        for (const auto & item : *value) {
            if (!item.is_object()) {
                throw std::invalid_argument("hotwords array entries must be objects");
            }
            const auto from = item.find("from");
            const auto to   = item.find("to");
            if (from == item.end() || to == item.end() || !from->is_string() || !to->is_string()) {
                throw std::invalid_argument("hotwords entries require string from and to fields");
            }
            append(from->get<std::string>(), to->get<std::string>());
        }
        return result;
    }
    throw std::invalid_argument("hotwords must be an object or array");
}

}  // namespace

void server_tts_validate_body_size(size_t bytes) {
    if (bytes > max_body_bytes) {
        throw std::invalid_argument("request body is too large");
    }
}

std::string server_tts_request_text(const json & body) {
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    const auto input = body.find("input");
    if (input == body.end() || !input->is_string() || input->get_ref<const std::string &>().empty()) {
        throw std::invalid_argument("input must be a non-empty string");
    }
    if (input->get_ref<const std::string &>().size() > max_input_bytes) {
        throw std::invalid_argument("input is too large");
    }
    if (const auto format = body.find("response_format");
        format != body.end() && (!format->is_string() || format->get<std::string>() != "wav")) {
        throw std::invalid_argument("response_format must be wav");
    }
    if (const auto speed = body.find("speed");
        speed != body.end() &&
        (!speed->is_number() || !std::isfinite(speed->get<double>()) || std::abs(speed->get<double>() - 1.0) > 1e-9)) {
        throw std::invalid_argument("speed must be 1.0");
    }
    for (const char * field : { "model", "voice" }) {
        if (const auto value = body.find(field); value != body.end() && !value->is_string()) {
            throw std::invalid_argument(std::string(field) + " must be a string");
        }
    }

    std::string text = input->get<std::string>();
    for (const auto & [from, to] : parse_hotwords(body)) {
        replace_all(text, from, to);
    }
    if (text.empty() || text.size() > max_input_bytes) {
        throw std::invalid_argument("input is invalid after hotword replacement");
    }
    return text;
}

server_http_res_ptr server_tts_make_response(server_media_tts_result result) {
    if (result.sample_rate == 0 || result.samples == 0 || result.wav.size() < 44) {
        throw std::runtime_error("TTS backend returned invalid audio");
    }
    const double audio_seconds = static_cast<double>(result.samples) / result.sample_rate;
    auto         response      = std::make_unique<server_http_res>();
    response->content_type     = "audio/wav";
    response->data.assign(reinterpret_cast<const char *>(result.wav.data()), result.wav.size());
    response->headers["X-TTS-Backend"]       = result.backend;
    response->headers["X-TTS-Segments"]      = std::to_string(result.segments);
    response->headers["X-TTS-Audio-Seconds"] = std::to_string(audio_seconds);
    response->headers["X-TTS-Wall-Seconds"]  = std::to_string(result.wall_seconds);
    response->headers["X-TTS-RTF"]           = std::to_string(result.wall_seconds / audio_seconds);
    return response;
}
