#include "qwen3_tts_gguf.h"
#include "qwen3_tts_protocol.h"

#include "ggml.h"
#include "llama.h"
#include "vec.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int hidden_size = 1024;
constexpr int code_groups = 16;
constexpr int code_vocab = 2048;
constexpr int talker_eos = 2150;
constexpr float repetition_penalty = 1.15f;

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

void log_errors(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

float dot_f16(const ggml_fp16_t * weight, const ggml_fp16_t * input) {
    float sum = 0.0f;
    ggml_vec_dot_f16(hidden_size, &sum, 0,
                     const_cast<ggml_fp16_t *>(weight), 0,
                     const_cast<ggml_fp16_t *>(input), 0, 1);
    return sum;
}

class f16_gemv {
  public:
    int argmax(const ggml_fp16_t * weight, int rows, const float * input, float * best_value = nullptr) {
        prepare(input);
        int best_index = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int row = 0; row < rows; ++row) {
            const float value = dot_f16(weight + static_cast<size_t>(row) * hidden_size, input_f16_.data());
            if (value > best) {
                best = value;
                best_index = row;
            }
        }
        if (best_value != nullptr) {
            *best_value = best;
        }
        return best_index;
    }

    void multiply(const ggml_fp16_t * weight, int rows, const float * input, float * output) {
        prepare(input);
        for (int row = 0; row < rows; ++row) {
            output[row] = dot_f16(weight + static_cast<size_t>(row) * hidden_size, input_f16_.data());
        }
    }

  private:
    void prepare(const float * input) {
        ggml_fp32_to_fp16_row(input, input_f16_.data(), hidden_size);
    }

    std::array<ggml_fp16_t, hidden_size> input_f16_{};
};

class batch_holder {
  public:
    batch_holder(int capacity, int embeddings) : batch_(llama_batch_init(capacity, embeddings, 1)) {}
    ~batch_holder() { llama_batch_free(batch_); }
    llama_batch & get() { return batch_; }

  private:
    llama_batch batch_{};
};

struct talker_job {
    uint32_t n_prefill = 0;
    uint32_t n_trailing = 0;
    uint32_t max_frames = 0;
    std::vector<float> prefill;
    std::vector<float> trailing;
    std::array<float, hidden_size> pad{};
};

talker_job parse_job(const std::vector<uint8_t> & payload, uint32_t max_prefill, uint32_t max_frames) {
    qwen3_tts::protocol::reader reader(payload);
    talker_job job;
    job.n_prefill = reader.u32();
    job.n_trailing = reader.u32();
    job.max_frames = reader.u32();
    if (job.n_prefill == 0 || job.n_prefill > max_prefill || job.max_frames == 0 || job.max_frames > max_frames) {
        throw std::runtime_error("Qwen3-TTS talker job exceeds configured limits");
    }
    const size_t prefill_count = static_cast<size_t>(job.n_prefill) * hidden_size;
    const size_t trailing_count = static_cast<size_t>(job.n_trailing) * hidden_size;
    const size_t expected = (prefill_count + trailing_count + hidden_size) * sizeof(float);
    if (reader.remaining() != expected) {
        throw std::runtime_error("invalid Qwen3-TTS talker job size");
    }
    job.prefill.resize(prefill_count);
    job.trailing.resize(trailing_count);
    std::memcpy(job.prefill.data(), reader.bytes(prefill_count * sizeof(float)), prefill_count * sizeof(float));
    std::memcpy(job.trailing.data(), reader.bytes(trailing_count * sizeof(float)), trailing_count * sizeof(float));
    std::memcpy(job.pad.data(), reader.bytes(hidden_size * sizeof(float)), hidden_size * sizeof(float));
    return job;
}

class talker_engine {
  public:
    talker_engine(const std::string & talker_path,
                  const std::string & cp_path,
                  const std::string & aux_path,
                  int max_prefill,
                  int max_frames,
                  int threads) :
        aux_(aux_path),
        max_prefill_(max_prefill),
        max_frames_(max_frames),
        talker_batch_(max_prefill, hidden_size),
        cp_batch_(2, hidden_size) {
        codec_embedding_ = static_cast<const float *>(aux_.tensor(
            "q3tts.codec_embedding.weight", GGML_TYPE_F32,
            3072ULL * hidden_size * sizeof(float)));
        talker_head_ = static_cast<const ggml_fp16_t *>(aux_.tensor(
            "q3tts.talker_head_f16.weight", GGML_TYPE_F16,
            3072ULL * hidden_size * sizeof(ggml_fp16_t)));
        for (int i = 0; i < code_groups - 1; ++i) {
            const std::string suffix = std::to_string(i) + ".weight";
            cp_embeddings_[i] = static_cast<const float *>(aux_.tensor(
                "q3tts.cp_embedding." + suffix, GGML_TYPE_F32,
                static_cast<size_t>(code_vocab) * hidden_size * sizeof(float)));
            cp_heads_[i] = static_cast<const ggml_fp16_t *>(aux_.tensor(
                "q3tts.cp_head_f16." + suffix, GGML_TYPE_F16,
                static_cast<size_t>(code_vocab) * hidden_size * sizeof(ggml_fp16_t)));
        }

        llama_model_params model_params = llama_model_default_params();
        talker_model_.reset(llama_model_load_from_file(talker_path.c_str(), model_params));
        if (!talker_model_) {
            throw std::runtime_error("failed to load Qwen3-TTS talker model");
        }
        cp_model_.reset(llama_model_load_from_file(cp_path.c_str(), model_params));
        if (!cp_model_) {
            throw std::runtime_error("failed to load Qwen3-TTS code predictor model");
        }

        llama_context_params talker_params = llama_context_default_params();
        talker_params.n_ctx = static_cast<uint32_t>(max_prefill + max_frames + 1);
        talker_params.n_batch = static_cast<uint32_t>(max_prefill);
        talker_params.n_ubatch = static_cast<uint32_t>(std::min(max_prefill, 16));
        talker_params.n_threads = threads;
        talker_params.n_threads_batch = threads;
        talker_params.embeddings = true;
        talker_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        talker_context_.reset(llama_init_from_model(talker_model_.get(), talker_params));
        if (!talker_context_) {
            throw std::runtime_error("failed to create Qwen3-TTS talker context");
        }

        llama_context_params cp_params = llama_context_default_params();
        cp_params.n_ctx = 64;
        cp_params.n_batch = 16;
        cp_params.n_ubatch = 16;
        cp_params.n_threads = threads;
        cp_params.n_threads_batch = threads;
        cp_params.embeddings = true;
        cp_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp_context_.reset(llama_init_from_model(cp_model_.get(), cp_params));
        if (!cp_context_) {
            throw std::runtime_error("failed to create Qwen3-TTS code predictor context");
        }
    }

    void run(int fd, const talker_job & job) {
        llama_memory_clear(llama_get_memory(talker_context_.get()), true);
        llama_memory_clear(llama_get_memory(cp_context_.get()), true);

        auto & talker_batch = talker_batch_.get();
        std::memcpy(talker_batch.embd, job.prefill.data(), job.prefill.size() * sizeof(float));
        for (uint32_t i = 0; i < job.n_prefill; ++i) {
            talker_batch.pos[i] = static_cast<llama_pos>(i);
            talker_batch.n_seq_id[i] = 1;
            talker_batch.seq_id[i][0] = 0;
            talker_batch.logits[i] = i + 1 == job.n_prefill;
        }
        talker_batch.n_tokens = static_cast<int32_t>(job.n_prefill);
        if (llama_decode(talker_context_.get(), talker_batch) != 0) {
            throw std::runtime_error("Qwen3-TTS talker prefill failed");
        }

        std::array<uint8_t, code_vocab> seen{};
        std::array<int32_t, code_groups> codes{};
        std::array<float, talker_eos + 1> talker_logits{};
        bool ended_by_eos = false;
        uint32_t frame_count = 0;
        const auto start = clock_type::now();

        for (uint32_t frame = 0; frame < job.max_frames; ++frame) {
            const float * hidden = llama_get_embeddings_ith(talker_context_.get(), talker_batch.n_tokens - 1);
            if (hidden == nullptr) {
                throw std::runtime_error("Qwen3-TTS talker returned no embedding");
            }
            gemv_.multiply(talker_head_, talker_eos + 1, hidden, talker_logits.data());
            int first_code = 0;
            float best = -std::numeric_limits<float>::infinity();
            for (int code = 0; code < code_vocab; ++code) {
                float value = talker_logits[code];
                if (seen[code]) {
                    value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
                }
                if (value > best) {
                    best = value;
                    first_code = code;
                }
            }
            if (frame >= 2 && talker_logits[talker_eos] > best) {
                ended_by_eos = true;
                break;
            }
            seen[first_code] = 1;
            codes[0] = first_code;

            llama_memory_clear(llama_get_memory(cp_context_.get()), false);
            auto & cp_batch = cp_batch_.get();
            std::memcpy(cp_batch.embd, hidden, hidden_size * sizeof(float));
            std::memcpy(cp_batch.embd + hidden_size,
                        codec_embedding_ + static_cast<size_t>(first_code) * hidden_size,
                        hidden_size * sizeof(float));
            for (int i = 0; i < 2; ++i) {
                cp_batch.pos[i] = i;
                cp_batch.n_seq_id[i] = 1;
                cp_batch.seq_id[i][0] = 0;
                cp_batch.logits[i] = i == 1;
            }
            cp_batch.n_tokens = 2;
            if (llama_decode(cp_context_.get(), cp_batch) != 0) {
                throw std::runtime_error("Qwen3-TTS code predictor prefill failed");
            }

            for (int group = 0; group < code_groups - 1; ++group) {
                const float * cp_hidden = llama_get_embeddings_ith(cp_context_.get(), cp_batch.n_tokens - 1);
                if (cp_hidden == nullptr) {
                    throw std::runtime_error("Qwen3-TTS code predictor returned no embedding");
                }
                const int code = gemv_.argmax(cp_heads_[group], code_vocab, cp_hidden);
                codes[group + 1] = code;
                if (group + 1 == code_groups - 1) {
                    break;
                }
                std::memcpy(cp_batch.embd,
                            cp_embeddings_[group] + static_cast<size_t>(code) * hidden_size,
                            hidden_size * sizeof(float));
                cp_batch.pos[0] = 2 + group;
                cp_batch.n_seq_id[0] = 1;
                cp_batch.seq_id[0][0] = 0;
                cp_batch.logits[0] = 1;
                cp_batch.n_tokens = 1;
                if (llama_decode(cp_context_.get(), cp_batch) != 0) {
                    throw std::runtime_error("Qwen3-TTS code predictor step failed");
                }
            }

            std::vector<uint8_t> frame_payload;
            frame_payload.reserve(code_groups * sizeof(uint32_t));
            for (int32_t code : codes) {
                qwen3_tts::protocol::append_u32(frame_payload, static_cast<uint32_t>(code));
            }
            qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::talker_frame, frame_payload);
            ++frame_count;

            const float * trailing = frame < job.n_trailing ?
                job.trailing.data() + static_cast<size_t>(frame) * hidden_size : job.pad.data();
            float * embedding = talker_batch.embd;
            for (int dim = 0; dim < hidden_size; ++dim) {
                float value = trailing[dim] +
                    codec_embedding_[static_cast<size_t>(codes[0]) * hidden_size + dim];
                for (int group = 0; group < code_groups - 1; ++group) {
                    value += cp_embeddings_[group][static_cast<size_t>(codes[group + 1]) * hidden_size + dim];
                }
                embedding[dim] = value;
            }
            talker_batch.pos[0] = static_cast<llama_pos>(job.n_prefill + frame);
            talker_batch.n_seq_id[0] = 1;
            talker_batch.seq_id[0][0] = 0;
            talker_batch.logits[0] = 1;
            talker_batch.n_tokens = 1;
            if (llama_decode(talker_context_.get(), talker_batch) != 0) {
                throw std::runtime_error("Qwen3-TTS talker step failed");
            }
        }

        std::vector<uint8_t> done;
        qwen3_tts::protocol::append_u32(done, frame_count);
        qwen3_tts::protocol::append_u32(done, ended_by_eos ? 0U : 1U);
        qwen3_tts::protocol::append_f64(done, seconds_since(start));
        qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::talker_done, done);
        std::fprintf(stderr, "qwen3-tts: generated %u frames in %.3f s%s\n",
                     frame_count, seconds_since(start), ended_by_eos ? "" : " (frame limit)");
    }

  private:
    struct model_deleter { void operator()(llama_model * model) const { llama_model_free(model); } };
    struct context_deleter { void operator()(llama_context * context) const { llama_free(context); } };

    qwen3_tts::mapped_gguf aux_;
    int max_prefill_;
    int max_frames_;
    const float * codec_embedding_ = nullptr;
    const ggml_fp16_t * talker_head_ = nullptr;
    std::array<const float *, code_groups - 1> cp_embeddings_{};
    std::array<const ggml_fp16_t *, code_groups - 1> cp_heads_{};
    std::unique_ptr<llama_model, model_deleter> talker_model_;
    std::unique_ptr<llama_model, model_deleter> cp_model_;
    std::unique_ptr<llama_context, context_deleter> talker_context_;
    std::unique_ptr<llama_context, context_deleter> cp_context_;
    batch_holder talker_batch_;
    batch_holder cp_batch_;
    f16_gemv gemv_;
};

int parse_positive(const char * value, const char * name) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "internal Qwen3-TTS talker invocation is invalid\n");
        return 2;
    }
    std::signal(SIGPIPE, SIG_IGN);
    int fd = -1;
    bool backend_initialized = false;
    try {
        fd = parse_positive(argv[1], "protocol fd");
        const int max_prefill = parse_positive(argv[5], "max prefill");
        const int max_frames = parse_positive(argv[6], "max frames");
        const int threads = parse_positive(argv[7], "thread count");
        llama_log_set(log_errors, nullptr);
        llama_backend_init();
        backend_initialized = true;
        {
            talker_engine engine(argv[2], argv[3], argv[4], max_prefill, max_frames, threads);
            qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::ready);
            qwen3_tts::protocol::message message;
            while (qwen3_tts::protocol::receive(fd, message)) {
                if (message.type == qwen3_tts::protocol::message_type::shutdown) {
                    break;
                }
                if (message.type != qwen3_tts::protocol::message_type::talker_job) {
                    throw std::runtime_error("unexpected Qwen3-TTS talker message");
                }
                engine.run(fd, parse_job(message.payload, max_prefill, max_frames));
            }
        }
        llama_backend_free();
        backend_initialized = false;
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3-tts talker: %s\n", error.what());
        if (fd >= 0) {
            try {
                qwen3_tts::protocol::send(fd, qwen3_tts::protocol::message_type::error, error.what());
            } catch (...) {
            }
        }
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
