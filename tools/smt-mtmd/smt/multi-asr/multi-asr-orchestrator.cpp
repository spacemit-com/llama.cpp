#include "multi-asr-orchestrator.h"

#include "smt-audio-wrapper.h"  // full type for smt_audio_context (unique_ptr dtor)
#include "ggml.h"
#include "log.h"

#include <stdexcept>
#include <utility>

multi_asr_orchestrator::multi_asr_orchestrator() = default;

multi_asr_orchestrator::~multi_asr_orchestrator() {
    stop();
}

void multi_asr_orchestrator::init(const multi_asr_params & params) {
    params_ = params;

    // Encoder must init first so we know hidden_size, then validate decoder.
    LOG_INF("[multi-asr] initializing encoder (cores from config.json)...\n");
    encoder_.init(params.smt_config_dir, params.warmup);
    hidden_size_ = encoder_.hidden_size();

    LOG_INF("[multi-asr] initializing decoder (cores='%s')...\n", params.decoder_cpu_range.c_str());
    decoder_.init(params, hidden_size_);

    LOG_INF("[multi-asr] ready. hidden_size=%lld n_parallel=%d queue_max=%d pipeline=%s\n", (long long) hidden_size_,
            params.n_parallel, params.queue_max, params.enable_pipeline ? "on" : "off");
}

void multi_asr_orchestrator::init_shared(llama_model * model, const multi_asr_params & params) {
    params_ = params;
    LOG_INF("[multi-asr] initializing fully isolated legacy decoder\n");
    encoder_.init(params.smt_config_dir, params.warmup);
    hidden_size_ = encoder_.hidden_size();
    decoder_.init(params, hidden_size_);
    LOG_INF("[multi-asr] isolated legacy decoder ready. hidden_size=%lld\n", (long long) hidden_size_);
    GGML_UNUSED(model);
}

void multi_asr_orchestrator::start() {
    if (running_.exchange(true)) {
        return;
    }
    encode_thread_ = std::thread(&multi_asr_orchestrator::encode_loop, this);
    decode_thread_ = std::thread(&multi_asr_orchestrator::decode_loop, this);
}

void multi_asr_orchestrator::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    intake_cv_.notify_all();
    ready_cv_.notify_all();
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }
}

bool multi_asr_orchestrator::submit_and_wait(const multi_asr_request & req_in, multi_asr_request & out) {
    // Backpressure: reject if the system is already at its hard cap.
    if (in_flight_.load() >= params_.queue_max) {
        out       = req_in;
        out.error = "server busy: queue full";
        out.stage = multi_asr_stage::failed;
        return false;
    }

    auto job         = std::make_shared<multi_asr_job>();
    job->req         = req_in;
    job->t_submit_ms = ggml_time_ms();

    in_flight_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(intake_mu_);
        intake_.push_back(job);
    }
    intake_cv_.notify_one();

    // Block until the pipeline finishes this job.
    {
        std::unique_lock<std::mutex> lk(job->mu);
        job->cv.wait(lk, [&] { return job->finished; });
    }

    out = std::move(job->req);
    in_flight_.fetch_sub(1);
    return out.stage != multi_asr_stage::failed;
}

// Encode worker: pulls from intake, runs ONNX encode on the encoder cores,
// pushes the encoded job to the ready queue for the decoder.
void multi_asr_orchestrator::encode_loop() {
    while (running_.load()) {
        multi_asr_job_ptr job;
        {
            std::unique_lock<std::mutex> lk(intake_mu_);
            intake_cv_.wait(lk, [&] { return !intake_.empty() || !running_.load(); });
            if (!running_.load() && intake_.empty()) {
                break;
            }
            job = intake_.front();
            intake_.pop_front();
        }

        // In strict-serial (no pipeline) mode, only one job is ever in intake at
        // a time because submit blocks; the encode->decode handoff below still
        // serializes naturally since ready_ holds at most one item.

        const int64_t t0          = ggml_time_ms();
        job->req.timings.queue_ms = (double) (t0 - job->t_submit_ms);
        job->req.stage            = multi_asr_stage::encoding;

        const bool ok = encoder_.encode(job->req);
        if (!ok) {
            // finish immediately with error
            {
                std::lock_guard<std::mutex> lk(job->mu);
                job->req.timings.total_ms = (double) (ggml_time_ms() - job->t_submit_ms);
                job->finished             = true;
            }
            job->cv.notify_one();
            continue;
        }

        // In the server facade, keep the legacy FIFO queue but execute the
        // complete encode->decode transaction on this worker.  The server
        // process owns a second GGUF context; overlapping SMT encoder work
        // with GGUF decode corrupts backend state even though the standalone
        // binary is safe to pipeline.  This path also avoids a cross-worker
        // wait cycle when strict mode is selected.
        if (!params_.enable_pipeline) {
            decoder_.decode(job->req);
            job->req.timings.total_ms = (double) (ggml_time_ms() - job->t_submit_ms);
            {
                std::lock_guard<std::mutex> lk(job->mu);
                job->finished = true;
            }
            job->cv.notify_one();
            continue;
        }

        // Hand off to decoder. Bound the ready queue to depth 1 so the encoder
        // runs at most ONE step ahead of the decoder — the clean B1 staggered
        // overlap E(N+1)‖D(N), not unbounded runahead.
        {
            std::unique_lock<std::mutex> lk(ready_mu_);
            ready_cv_.wait(lk, [&] { return ready_.empty() || !running_.load(); });
            if (!running_.load()) {
                break;
            }
            ready_.push_back(job);
        }
        ready_cv_.notify_one();

    }
}

// Decode worker: pulls encoded jobs, runs gguf decode on the decoder cores.
void multi_asr_orchestrator::decode_loop() {
    while (running_.load()) {
        multi_asr_job_ptr job;
        {
            std::unique_lock<std::mutex> lk(ready_mu_);
            ready_cv_.wait(lk, [&] { return !ready_.empty() || !running_.load(); });
            if (!running_.load() && ready_.empty()) {
                break;
            }
            job = ready_.front();
            ready_.pop_front();
        }
        // Wake the encoder: the ready slot is now free, so E(N+1) can hand off
        // while we decode this job (the staggered overlap).
        ready_cv_.notify_one();

        decoder_.decode(job->req);
        job->req.timings.total_ms = (double) (ggml_time_ms() - job->t_submit_ms);

        {
            std::lock_guard<std::mutex> lk(job->mu);
            job->finished = true;
        }
        job->cv.notify_one();
    }
}
