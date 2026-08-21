#pragma once

// Multi-ASR 4-way concurrent orchestrator — common types & config.
//
// Design (see docs/../multi-asr-design.md, scheme B1 "错峰流水"):
//   - encoder stage : 1 ONNX session (smt_audio_context), pinned to encoder cores,
//                     serial encode -> produces audio embedding chunk.
//   - decoder stage : 1 llama_context slot, pinned to decoder cores, serial decode.
//   - orchestrator  : FIFO queue (>N waits) + encode/decode threads that overlap
//                     E(N+1) with D(N).
// Core split (encoder cores vs decoder cores) is a RUNTIME INPUT, not hardcoded,
// so the optimal ratio can be swept experimentally.

#include "ggml.h"  // GGML_MAX_N_THREADS
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct multi_asr_params {
    // model / backend
    std::string model_path;      // gguf text model (decoder)
    std::string smt_config_dir;  // dir with config.json + encoder ONNX
    bool        warmup = true;

    // HTTP
    std::string host = "0.0.0.0";
    int32_t     port = 8080;

    // concurrency
    int32_t n_parallel = 4;   // max in-flight requests before queueing (>N waits)
    int32_t queue_max  = 64;  // hard cap on total (in-flight + waiting); reject beyond

    // --- core split (the tunable input variable) ---
    // encoder cores: set via the ONNX EP fields in config.json (spacemit_ep_*).
    // decoder cores: applied to the llama_context ggml thread affinity (cpumask).
    // Accepts a CPU range string like "10,11,14,15" or "12-15". Empty = leave backend default.
    std::string decoder_cpu_range;      // e.g. "10,11,14,15"                or "12-15"
    int32_t     decoder_n_threads = 0;  // 0 = derive from decoder_cpu_range size

    // generation
    int32_t n_predict = 256;  // max output tokens per request
    int32_t n_ctx     = 0;    // 0 = from model / config.json context_size
    int32_t n_batch   = 2048;

    // pipeline
    bool enable_pipeline = true;  // false = strict serial baseline (variable ②)
};

// ---------------------------------------------------------------------------
// Per-request lifecycle
// ---------------------------------------------------------------------------

enum class multi_asr_stage : uint8_t {
    queued   = 0,
    encoding = 1,
    decoding = 2,
    done     = 3,
    failed   = 4,
};

// Per-stage timing (ms), filled as the request flows through the pipeline.
struct multi_asr_timings {
    double  queue_ms       = 0.0;  // time spent waiting in FIFO before encode start
    double  encode_ms      = 0.0;  // ONNX encoder wall time
    double  prefill_ms     = 0.0;  // audio embedding prefill into gguf
    double  decode_ms      = 0.0;  // token generation
    double  total_ms       = 0.0;  // end-to-end (accept -> response)
    int32_t n_audio_tokens = 0;
    int32_t n_out_tokens   = 0;
};

// A single ASR request as it moves through encode -> decode.
struct multi_asr_request {
    uint64_t             id = 0;
    std::vector<uint8_t> audio;   // raw wav bytes (decoded input)
    std::string          prompt;  // text prompt, e.g. "language Chinese<asr_text>"
    int32_t              n_predict = 256;

    // filled by encoder stage: audio embedding (n_audio_tokens * hidden_size floats)
    std::vector<float> embd;
    int32_t            n_audio_tokens = 0;

    // result
    std::string       text;
    multi_asr_stage   stage = multi_asr_stage::queued;
    std::string       error;
    multi_asr_timings timings;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Parse a CPU range string ("8,9,12,13" or "8-11") into a boolean affinity mask.
// Returns the number of selected CPUs, or -1 on parse error.
int multi_asr_parse_cpu_range(const std::string & range, bool (&mask)[GGML_MAX_N_THREADS]);

// Convert a boolean affinity mask back to a compact "8,9,12,13" string (for logging).
std::string multi_asr_cpu_mask_to_string(const bool (&mask)[GGML_MAX_N_THREADS]);
