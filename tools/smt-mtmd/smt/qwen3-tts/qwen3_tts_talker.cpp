#include "qwen3_tts_talker.h"

#include "ggml.h"
#include "llama.h"
#include "qwen3_tts_gguf.h"

#if defined(__riscv_vector) && defined(__riscv_zvfh)
#    include <riscv_vector.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qwen3_tts {
namespace {

constexpr int   code_vocab         = 2048;
constexpr int   talker_eos         = 2150;
constexpr float repetition_penalty = 1.15f;

float dot_f16(const ggml_fp16_t * weight, const ggml_fp16_t * input) {
#if defined(__riscv_vector) && defined(__riscv_zvfh)
    const size_t vl          = __riscv_vsetvl_e16m4(128);
    vfloat32m8_t accumulator = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    for (int offset = 0; offset < hidden_size; offset += static_cast<int>(vl)) {
        accumulator = __riscv_vfwmacc_vv_f32m8(
            accumulator, __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16 *>(weight + offset), vl),
            __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16 *>(input + offset), vl), vl);
    }
    const size_t       vl32    = __riscv_vsetvl_e32m8(vl);
    const vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m8_f32m1(accumulator, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl32);
    return __riscv_vfmv_f_s_f32m1_f32(reduced);
#else
    float sum = 0.0f;
    for (int index = 0; index < hidden_size; ++index) {
        sum += ggml_fp16_to_fp32(weight[index]) * ggml_fp16_to_fp32(input[index]);
    }
    return sum;
#endif
}

class f16_gemv {
  public:
    f16_gemv() {
        int started = 0;
        try {
            for (; started < worker_count; ++started) {
                workers_[started] = std::thread([this, started] { worker_loop(started); });
            }
        } catch (...) {
            stopping_.store(true, std::memory_order_release);
            start_cv_.notify_all();
            for (int id = 0; id < started; ++id) {
                workers_[id].join();
            }
            throw;
        }
    }

    ~f16_gemv() {
        stopping_.store(true, std::memory_order_release);
        start_cv_.notify_all();
        for (auto & worker : workers_) {
            worker.join();
        }
    }

    f16_gemv(const f16_gemv &)             = delete;
    f16_gemv & operator=(const f16_gemv &) = delete;

    void begin() {
        active_.store(true, std::memory_order_release);
        start_cv_.notify_all();
    }

    void end() { active_.store(false, std::memory_order_release); }

    int argmax(const ggml_fp16_t * weight, int rows, const float * input) {
        prepare(input);
        return execute(weight, rows, nullptr).index;
    }

    void multiply(const ggml_fp16_t * weight, int rows, const float * input, float * output) {
        prepare(input);
        execute(weight, rows, output);
    }

  private:
    static constexpr int worker_count = 4;

    struct job {
        const ggml_fp16_t * weight        = nullptr;
        float *             output        = nullptr;
        int                 rows          = 0;
        int                 rows_per_part = 0;
    };

    struct result {
        int   index = 0;
        float value = -std::numeric_limits<float>::infinity();
    };

    void prepare(const float * input) {
#if defined(__riscv_vector) && defined(__riscv_zvfh)
        for (int offset = 0; offset < hidden_size;) {
            const size_t       vl       = __riscv_vsetvl_e32m4(hidden_size - offset);
            const vfloat16m2_t narrowed = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(input + offset, vl), vl);
            __riscv_vse16_v_f16m2(reinterpret_cast<_Float16 *>(input_f16_.data() + offset), narrowed, vl);
            offset += static_cast<int>(vl);
        }
#else
        ggml_fp32_to_fp16_row(input, input_f16_.data(), hidden_size);
#endif
    }

    result compute_part(const job & current, int part) const {
        const int first = std::min(part * current.rows_per_part, current.rows);
        const int last  = std::min(first + current.rows_per_part, current.rows);
        result    best;
        best.index = first;
        for (int row = first; row < last; ++row) {
            __builtin_prefetch(current.weight + static_cast<size_t>(row + 1) * hidden_size);
            const float value = dot_f16(current.weight + static_cast<size_t>(row) * hidden_size, input_f16_.data());
            if (current.output != nullptr) {
                current.output[row] = value;
            } else if (value > best.value) {
                best.value = value;
                best.index = row;
            }
        }
        return best;
    }

    result execute(const ggml_fp16_t * weight, int rows, float * output) {
        job current;
        current.weight        = weight;
        current.output        = output;
        current.rows          = rows;
        current.rows_per_part = (rows + worker_count - 1) / worker_count;

        job_                      = current;
        const uint64_t generation = generation_.fetch_add(1, std::memory_order_release) + 1;

        for (int id = 0; id < worker_count; ++id) {
            while (completed_[id].load(std::memory_order_acquire) != generation) {
#if defined(__riscv_zihintpause)
                __asm__ volatile("pause");
#else
                std::this_thread::yield();
#endif
            }
        }
        result best;
        if (output == nullptr) {
            for (const result & candidate : results_) {
                if (candidate.value > best.value || (candidate.value == best.value && candidate.index < best.index)) {
                    best = candidate;
                }
            }
        }
        return best;
    }

    void worker_loop(int id) {
        uint64_t observed_generation = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(start_mutex_);
                start_cv_.wait(lock, [this] {
                    return stopping_.load(std::memory_order_acquire) || active_.load(std::memory_order_acquire);
                });
            }
            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }
            while (active_.load(std::memory_order_acquire)) {
                if (stopping_.load(std::memory_order_acquire)) {
                    return;
                }
                const uint64_t generation = generation_.load(std::memory_order_acquire);
                if (generation == observed_generation) {
#if defined(__riscv_zihintpause)
                    __asm__ volatile("pause");
#else
                    std::this_thread::yield();
#endif
                    continue;
                }
                const job current = job_;
                results_[id]      = compute_part(current, id);
                completed_[id].store(generation, std::memory_order_release);
                observed_generation = generation;
            }
        }
    }

    std::array<ggml_fp16_t, hidden_size>            input_f16_{};
    std::array<std::thread, worker_count>           workers_{};
    std::array<result, worker_count>                results_{};
    std::array<std::atomic<uint64_t>, worker_count> completed_{};
    std::mutex                                      start_mutex_;
    std::condition_variable                         start_cv_;
    job                                             job_;
    std::atomic<uint64_t>                           generation_{ 0 };
    std::atomic<bool>                               active_{ false };
    std::atomic<bool>                               stopping_{ false };
};

class batch_holder {
  public:
    batch_holder(int capacity, int embeddings) : batch_(llama_batch_init(capacity, embeddings, 1)) {}

    ~batch_holder() { llama_batch_free(batch_); }

    llama_batch & get() { return batch_; }

  private:
    llama_batch batch_{};
};

}  // namespace

struct talker_engine::impl {
  public:
    impl(const std::string & talker_path,
         const std::string & cp_path,
         const std::string & aux_path,
         int                 max_prefill,
         int                 max_frames,
         int                 threads) :
        aux_(aux_path),
        max_prefill_(max_prefill),
        max_frames_(max_frames),
        talker_batch_(max_prefill, hidden_size),
        cp_batch_(2, hidden_size) {
        codec_embedding_ = static_cast<const float *>(
            aux_.tensor("q3tts.codec_embedding.weight", GGML_TYPE_F32, 3072ULL * hidden_size * sizeof(float)));
        talker_head_ = static_cast<const ggml_fp16_t *>(
            aux_.tensor("q3tts.talker_head_f16.weight", GGML_TYPE_F16, 3072ULL * hidden_size * sizeof(ggml_fp16_t)));
        for (int i = 0; i < code_groups - 1; ++i) {
            const std::string suffix = std::to_string(i) + ".weight";
            cp_embeddings_[i] =
                static_cast<const float *>(aux_.tensor("q3tts.cp_embedding." + suffix, GGML_TYPE_F32,
                                                       static_cast<size_t>(code_vocab) * hidden_size * sizeof(float)));
            cp_heads_[i] = static_cast<const ggml_fp16_t *>(
                aux_.tensor("q3tts.cp_head_f16." + suffix, GGML_TYPE_F16,
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
        talker_params.n_ctx                = static_cast<uint32_t>(max_prefill + max_frames + 1);
        talker_params.n_batch              = static_cast<uint32_t>(max_prefill);
        talker_params.n_ubatch             = static_cast<uint32_t>(std::min(max_prefill, 16));
        talker_params.n_threads            = threads;
        talker_params.n_threads_batch      = threads;
        talker_params.embeddings           = true;
        talker_params.pooling_type         = LLAMA_POOLING_TYPE_NONE;
        talker_context_.reset(llama_init_from_model(talker_model_.get(), talker_params));
        if (!talker_context_) {
            throw std::runtime_error("failed to create Qwen3-TTS talker context");
        }

        llama_context_params cp_params = llama_context_default_params();
        cp_params.n_ctx                = code_groups;
        cp_params.n_batch              = 16;
        cp_params.n_ubatch             = 16;
        cp_params.n_threads            = threads;
        cp_params.n_threads_batch      = threads;
        cp_params.embeddings           = true;
        cp_params.pooling_type         = LLAMA_POOLING_TYPE_NONE;
        cp_context_.reset(llama_init_from_model(cp_model_.get(), cp_params));
        if (!cp_context_) {
            throw std::runtime_error("failed to create Qwen3-TTS code predictor context");
        }
    }

    void generate(const std::vector<float> &             prefill,
                  const std::vector<float> &             trailing,
                  const std::array<float, hidden_size> & pad,
                  uint32_t                               max_frames,
                  const frame_callback &                 on_frame) {
        if (prefill.empty() || prefill.size() % hidden_size != 0 || trailing.size() % hidden_size != 0) {
            throw std::runtime_error("invalid Qwen3-TTS talker input shape");
        }
        const uint32_t n_prefill  = static_cast<uint32_t>(prefill.size() / hidden_size);
        const uint32_t n_trailing = static_cast<uint32_t>(trailing.size() / hidden_size);
        if (n_prefill > static_cast<uint32_t>(max_prefill_) || max_frames == 0 ||
            max_frames > static_cast<uint32_t>(max_frames_)) {
            throw std::runtime_error("Qwen3-TTS talker input exceeds configured limits");
        }

        llama_memory_clear(llama_get_memory(talker_context_.get()), true);
        llama_memory_clear(llama_get_memory(cp_context_.get()), true);

        auto & talker_batch = talker_batch_.get();
        std::memcpy(talker_batch.embd, prefill.data(), prefill.size() * sizeof(float));
        for (uint32_t i = 0; i < n_prefill; ++i) {
            talker_batch.pos[i]       = static_cast<llama_pos>(i);
            talker_batch.n_seq_id[i]  = 1;
            talker_batch.seq_id[i][0] = 0;
            talker_batch.logits[i]    = 1;
        }
        talker_batch.n_tokens = static_cast<int32_t>(n_prefill);
        if (llama_decode(talker_context_.get(), talker_batch) != 0) {
            throw std::runtime_error("Qwen3-TTS talker prefill failed");
        }

        std::array<uint8_t, code_vocab>   seen{};
        std::array<int32_t, code_groups>  codes{};
        std::array<float, talker_eos + 1> talker_logits{};
        bool                              ended_by_eos = false;
        gemv_.begin();

        struct gemv_guard {
            f16_gemv & gemv;

            ~gemv_guard() { gemv.end(); }
        } guard{ gemv_ };

        for (uint32_t frame = 0; frame < max_frames; ++frame) {
            const float * hidden = llama_get_embeddings_ith(talker_context_.get(), talker_batch.n_tokens - 1);
            if (hidden == nullptr) {
                throw std::runtime_error("Qwen3-TTS talker returned no embedding");
            }
            gemv_.multiply(talker_head_, talker_eos + 1, hidden, talker_logits.data());
            int   first_code = 0;
            float best       = -std::numeric_limits<float>::infinity();
            for (int code = 0; code < code_vocab; ++code) {
                float value = talker_logits[code];
                if (seen[code]) {
                    value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
                }
                if (value > best) {
                    best       = value;
                    first_code = code;
                }
            }
            if (frame >= 2 && talker_logits[talker_eos] > best) {
                ended_by_eos = true;
                break;
            }
            seen[first_code] = 1;
            codes[0]         = first_code;

            llama_memory_clear(llama_get_memory(cp_context_.get()), false);
            auto & cp_batch = cp_batch_.get();
            std::memcpy(cp_batch.embd, hidden, hidden_size * sizeof(float));
            std::memcpy(cp_batch.embd + hidden_size, codec_embedding_ + static_cast<size_t>(first_code) * hidden_size,
                        hidden_size * sizeof(float));
            for (int i = 0; i < 2; ++i) {
                cp_batch.pos[i]       = i;
                cp_batch.n_seq_id[i]  = 1;
                cp_batch.seq_id[i][0] = 0;
                cp_batch.logits[i]    = 1;
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
                const int code   = gemv_.argmax(cp_heads_[group], code_vocab, cp_hidden);
                codes[group + 1] = code;
                if (group + 1 == code_groups - 1) {
                    break;
                }
                std::memcpy(cp_batch.embd, cp_embeddings_[group] + static_cast<size_t>(code) * hidden_size,
                            hidden_size * sizeof(float));
                cp_batch.pos[0]       = 2 + group;
                cp_batch.n_seq_id[0]  = 1;
                cp_batch.seq_id[0][0] = 0;
                cp_batch.logits[0]    = 1;
                cp_batch.n_tokens     = 1;
                if (llama_decode(cp_context_.get(), cp_batch) != 0) {
                    throw std::runtime_error("Qwen3-TTS code predictor step failed");
                }
            }

            on_frame(codes);

            const float * trailing_embedding =
                frame < n_trailing ? trailing.data() + static_cast<size_t>(frame) * hidden_size : pad.data();
            float * embedding = talker_batch.embd;
            for (int dim = 0; dim < hidden_size; ++dim) {
                float value =
                    trailing_embedding[dim] + codec_embedding_[static_cast<size_t>(codes[0]) * hidden_size + dim];
                for (int group = 0; group < code_groups - 1; ++group) {
                    value += cp_embeddings_[group][static_cast<size_t>(codes[group + 1]) * hidden_size + dim];
                }
                embedding[dim] = value;
            }
            talker_batch.pos[0]       = static_cast<llama_pos>(n_prefill + frame);
            talker_batch.n_seq_id[0]  = 1;
            talker_batch.seq_id[0][0] = 0;
            talker_batch.logits[0]    = 1;
            talker_batch.n_tokens     = 1;
            if (llama_decode(talker_context_.get(), talker_batch) != 0) {
                throw std::runtime_error("Qwen3-TTS talker step failed");
            }
        }
        if (!ended_by_eos) {
            throw std::runtime_error("Qwen3-TTS generation reached the frame limit without EOS");
        }
    }

  private:
    struct model_deleter {
        void operator()(llama_model * model) const { llama_model_free(model); }
    };

    struct context_deleter {
        void operator()(llama_context * context) const { llama_free(context); }
    };

    mapped_gguf                                      aux_;
    int                                              max_prefill_;
    int                                              max_frames_;
    const float *                                    codec_embedding_ = nullptr;
    const ggml_fp16_t *                              talker_head_     = nullptr;
    std::array<const float *, code_groups - 1>       cp_embeddings_{};
    std::array<const ggml_fp16_t *, code_groups - 1> cp_heads_{};
    std::unique_ptr<llama_model, model_deleter>      talker_model_;
    std::unique_ptr<llama_model, model_deleter>      cp_model_;
    std::unique_ptr<llama_context, context_deleter>  talker_context_;
    std::unique_ptr<llama_context, context_deleter>  cp_context_;
    batch_holder                                     talker_batch_;
    batch_holder                                     cp_batch_;
    f16_gemv                                         gemv_;
};

talker_engine::talker_engine(const std::string & talker_path,
                             const std::string & code_predictor_path,
                             const std::string & aux_path,
                             int                 max_prefill,
                             int                 max_frames,
                             int                 threads) :
    pimpl_(std::make_unique<impl>(talker_path, code_predictor_path, aux_path, max_prefill, max_frames, threads)) {}

talker_engine::~talker_engine() = default;

void talker_engine::generate(const std::vector<float> &             prefill,
                             const std::vector<float> &             trailing,
                             const std::array<float, hidden_size> & pad,
                             uint32_t                               max_frames,
                             const frame_callback &                 on_frame) {
    pimpl_->generate(prefill, trailing, pad, max_frames, on_frame);
}

}  // namespace qwen3_tts
