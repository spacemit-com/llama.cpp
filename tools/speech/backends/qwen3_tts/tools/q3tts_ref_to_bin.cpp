#include <onnxruntime_cxx_api.h>

#include "q3tts_frontend.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string env_str(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

std::string default_model_dir() {
    const char *home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.cache/models/tts/qwen3-tts";
    }
    return "/tmp/.cache/models/tts/qwen3-tts";
}

struct Args {
    std::string ref_wav;
    std::string ref_text;
    std::string ref_text_file;
    std::string out;
    std::string model_dir = env_str("Q3TTS_MODEL_DIR", default_model_dir());
    int threads = env_int("Q3TTS_FRONTEND_THREADS", 2);
};

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--ref-wav") {
            args.ref_wav = need("--ref-wav");
        } else if (k == "--ref-text") {
            args.ref_text = need("--ref-text");
        } else if (k == "--ref-text-file") {
            args.ref_text_file = need("--ref-text-file");
        } else if (k == "--out") {
            args.out = need("--out");
        } else if (k == "--model-dir") {
            args.model_dir = need("--model-dir");
        } else if (k == "--threads") {
            args.threads = std::stoi(need("--threads"));
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (args.ref_wav.empty()) {
        throw std::runtime_error("missing --ref-wav");
    }
    if (args.out.empty()) {
        throw std::runtime_error("missing --out");
    }
    if (!args.ref_text_file.empty()) {
        args.ref_text = q3tts_frontend::read_text_file(args.ref_text_file);
        while (!args.ref_text.empty() &&
               (args.ref_text.back() == '\n' || args.ref_text.back() == '\r')) {
            args.ref_text.pop_back();
        }
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "q3tts_ref_to_bin");

        const auto ref_audio = q3tts_frontend::read_wav_mono_24k(args.ref_wav);
        const double ref_seconds = static_cast<double>(ref_audio.size()) / 24000.0;
        if (ref_seconds < 4.0) {
            std::cerr << "warning: reference audio is short (" << ref_seconds
                      << "s); use a clean 5-10s clip for stable voice cloning\n";
        }

        const std::string speaker_onnx = q3tts_frontend::speaker_encoder_path(args.model_dir);
        auto spk = q3tts_frontend::run_speaker_encoder(env, speaker_onnx, args.ref_wav, args.threads);
        if (args.ref_text.empty()) {
            q3tts_frontend::write_speaker_bin(args.out, spk);
            std::cout << "ref_bin " << args.out << " floats " << spk.size()
                      << " bytes " << (spk.size() * sizeof(float)) << "\n";
        } else {
            const std::string codec_onnx = q3tts_frontend::codec_encoder_path(args.model_dir);
            auto codes = q3tts_frontend::run_codec_encoder(env, codec_onnx, args.ref_wav, args.threads);
            q3tts_frontend::write_reference_prompt_bin(args.out, spk, args.ref_text, codes);
            std::cout << "ref_prompt_bin " << args.out
                      << " speaker_floats " << spk.size()
                      << " ref_frames " << codes.size()
                      << " ref_text_bytes " << args.ref_text.size()
                      << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
