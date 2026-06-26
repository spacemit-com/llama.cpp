#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace q3tts_codec {

inline bool file_exists(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

inline std::string path_join(const std::string &a, const std::string &b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

inline std::string first_existing(const std::vector<std::string> &paths) {
    for (const auto &p : paths) {
        if (!p.empty() && file_exists(p)) {
            return p;
        }
    }
    throw std::runtime_error("missing file");
}

inline std::string codec_model_file(const std::string &model_dir, int bucket) {
    const std::string name = "codec_decoder_t" + std::to_string(bucket) + ".q.onnx";
    return first_existing({
        path_join(path_join(model_dir, "onnx"), name),
        path_join(model_dir, name),
        name,
    });
}

inline std::string first_existing_ep_lib() {
    const char *env = std::getenv("Q3TTS_SPACEMIT_EP_LIB");
    const std::vector<std::string> paths = {
        env && *env ? std::string(env) : std::string(),
        "/usr/lib/python3.14/dist-packages/spacemit_ort/libspacemit_ep.so.2.0.3",
        "/usr/lib/python3.14/dist-packages/spacemit_ort/libspacemit_ep.so.2",
        "/usr/lib/python3/dist-packages/spacemit_ort/libspacemit_ep.so.2.0.3",
        "/usr/lib/python3/dist-packages/spacemit_ort/libspacemit_ep.so.2",
    };
    return first_existing(paths);
}

inline void init_spacemit_ep(Ort::SessionOptions &so, const std::string &ep_lib) {
    using InitFn = OrtStatus *(ORT_API_CALL *)(OrtSessionOptions *,
                                               const char *const *,
                                               const char *const *,
                                               size_t);
    static void *handle = nullptr;
    static InitFn init_fn = nullptr;
    if (!handle) {
        handle = dlopen(ep_lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            throw std::runtime_error(std::string("dlopen spacemit ep failed: ") + dlerror());
        }
        init_fn = reinterpret_cast<InitFn>(dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
        if (!init_fn) {
            throw std::runtime_error(std::string("dlsym OrtSessionOptionsSpaceMITEnvInit failed: ") + dlerror());
        }
    }
    Ort::ThrowOnError(init_fn(so, nullptr, nullptr, 0));
}

struct Decoder {
    int bucket = 0;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
};

struct DecoderPoolConfig {
    std::string model_dir = ".";
    std::vector<int> buckets;
    int intra_threads = 3;
    std::string ep_lib;
    std::function<void(int)> on_bucket_warm;
};

class DecoderPool {
public:
    DecoderPool(Ort::Env &env, DecoderPoolConfig config) {
        if (config.buckets.empty()) {
            throw std::runtime_error("empty codec bucket list");
        }
        const std::string ep_lib = config.ep_lib.empty() ? first_existing_ep_lib() : config.ep_lib;
        Ort::AllocatorWithDefaultOptions allocator;
        for (int b : config.buckets) {
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(config.intra_threads);
            so.AddConfigEntry("session.intra_op.allow_spinning", "0");
            init_spacemit_ep(so, ep_lib);

            Decoder d;
            d.bucket = b;
            const std::string model = codec_model_file(config.model_dir, b);
            d.session = std::make_unique<Ort::Session>(env, model.c_str(), so);
            auto in = d.session->GetInputNameAllocated(0, allocator);
            auto out = d.session->GetOutputNameAllocated(0, allocator);
            d.input_name = in.get();
            d.output_name = out.get();

            std::vector<std::array<int32_t, 16>> warm(static_cast<size_t>(b));
            for (auto &x : warm) {
                x.fill(0);
            }
            (void)decode_with(d, warm, 0);
            decoders_.emplace(b, std::move(d));
            if (config.on_bucket_warm) {
                config.on_bucket_warm(b);
            }
        }
    }

    std::vector<float> decode(int bucket,
                              const std::vector<std::array<int32_t, 16>> &codes,
                              int ctx) {
        auto it = decoders_.find(bucket);
        if (it == decoders_.end()) {
            throw std::runtime_error("codec bucket not initialized");
        }
        return decode_with(it->second, codes, ctx);
    }

    std::vector<float> decode_chunks(const std::vector<std::array<int32_t, 16>> &frames,
                                     const std::vector<int> &buckets,
                                     int first_chunk,
                                     int chunk,
                                     int ctx_limit) {
        std::vector<float> wav;
        int done = 0;
        while (done < static_cast<int>(frames.size())) {
            const int next_chunk = done == 0 ? first_chunk : chunk;
            const int n = std::min(static_cast<int>(frames.size()), done + next_chunk);
            const int new_count = n - done;
            auto it = std::find_if(buckets.begin(), buckets.end(), [&](int b) { return b >= new_count; });
            if (it == buckets.end()) {
                throw std::runtime_error("no codec bucket for chunk");
            }
            const int b = *it;
            const int ctx = std::min({done, ctx_limit, b - new_count});
            std::vector<std::array<int32_t, 16>> codes(frames.begin() + (done - ctx), frames.begin() + n);
            auto chunk_wav = decode(b, codes, ctx);
            wav.insert(wav.end(), chunk_wav.begin(), chunk_wav.end());
            done = n;
        }
        return wav;
    }

private:
    static std::vector<float> decode_with(Decoder &dec,
                                          const std::vector<std::array<int32_t, 16>> &codes,
                                          int ctx) {
        std::vector<int64_t> input(static_cast<size_t>(16 * dec.bucket), 0);
        for (size_t t = 0; t < codes.size(); ++t) {
            for (int c = 0; c < 16; ++c) {
                input[static_cast<size_t>(c * dec.bucket) + t] = codes[t][c];
            }
        }
        std::array<int64_t, 3> shape {1, 16, dec.bucket};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<int64_t>(mem, input.data(), input.size(), shape.data(), shape.size());
        const char *in_names[] = {dec.input_name.c_str()};
        const char *out_names[] = {dec.output_name.c_str()};
        auto outputs = dec.session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        float *out = outputs[0].GetTensorMutableData<float>();
        const size_t total = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const size_t begin = static_cast<size_t>(ctx) * 1920;
        const size_t end = std::min(total, codes.size() * static_cast<size_t>(1920));
        if (begin > end) {
            throw std::runtime_error("bad codec slice");
        }
        return std::vector<float>(out + begin, out + end);
    }

    std::unordered_map<int, Decoder> decoders_;
};

}  // namespace q3tts_codec
