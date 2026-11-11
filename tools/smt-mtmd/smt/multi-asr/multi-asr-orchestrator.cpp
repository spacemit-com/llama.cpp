#include "multi-asr-orchestrator.h"

#include "ggml.h"
#include "log.h"
#include "smt-audio-wrapper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

// Unsupported or malformed WAVs use the arrival deadline.
// Parse chunk bounds and odd-byte padding without decoding audio.
int64_t wav_duration_ms(const std::vector<uint8_t> & bytes) {
    auto u16 = [&](size_t p) { return uint32_t(bytes[p]) | uint32_t(bytes[p + 1]) << 8; };
    auto u32 = [&](size_t p) { return u16(p) | u16(p + 2) << 16; };
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) ||
        std::memcmp(bytes.data() + 8, "WAVE", 4)) {
        return 0;
    }
    uint32_t rate = 0, data_bytes = 0;
    bool have_data = false;
    for (size_t p = 12; p + 8 <= bytes.size();) {
        const uint32_t length = u32(p + 4);
        const size_t payload = p + 8;
        if (length > bytes.size() - payload) return 0;
        if (!std::memcmp(bytes.data() + p, "fmt ", 4) && length >= 16) {
            const uint32_t format = u16(payload);
            const uint32_t channels = u16(payload + 2);
            const uint32_t samples = u32(payload + 4);
            const uint32_t bits = u16(payload + 14);
            const uint64_t expected = uint64_t(samples) * channels * bits / 8;
            if ((format != 1 && format != 3) || !channels || !samples || !bits || bits % 8 ||
                expected != u32(payload + 8)) return 0;
            rate = u32(payload + 8);
        } else if (!std::memcmp(bytes.data() + p, "data", 4)) {
            data_bytes = length;
            have_data = true;
        }
        p = payload + length;
        if ((length & 1) && p < bytes.size()) ++p;
    }
    return rate && have_data ? (uint64_t(data_bytes) * 1000 + rate - 1) / rate : 0;
}

} // namespace

multi_asr_orchestrator::~multi_asr_orchestrator() {
    stop();
}

void multi_asr_orchestrator::init_shared(llama_model * model, const multi_asr_params & params) {
    if (model == nullptr) {
        throw std::runtime_error("multi_asr_orchestrator: null model");
    }
    params_ = params;
    encoder_.init(params_.smt_config_dir, params_.warmup);
    hidden_size_ = encoder_.hidden_size();
    decoder_.init_shared(model, params_, hidden_size_);
    slot_busy_.assign((size_t) std::max(1, params_.n_parallel), false);
    LOG_INF("[multi-asr] ready: serial encoder, continuous-batch decoder, slots=%d\n",
            std::max(1, params_.n_parallel));
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
    if (!running_.load()) {
        out = req_in;
        out.error = "multi-ASR service is stopped";
        out.stage = multi_asr_stage::failed;
        return false;
    }
    if (in_flight_.load() >= std::max(1, params_.queue_max)) {
        out = req_in;
        out.error = "multi-ASR queue is full";
        out.stage = multi_asr_stage::failed;
        return false;
    }

    auto job = std::make_shared<multi_asr_job>();
    job->req = req_in;
    job->t_submit_ms = ggml_time_ms();
    job->deadline_ms = job->t_submit_ms + wav_duration_ms(req_in.audio);
    in_flight_.fetch_add(1);
    pending_encode_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(intake_mu_);
        intake_.push_back(job);
    }
    intake_cv_.notify_one();

    std::unique_lock<std::mutex> lock(job->mu);
    job->cv.wait(lock, [&] { return job->finished; });
    out = job->req;
    in_flight_.fetch_sub(1);
    return out.stage != multi_asr_stage::failed;
}

void multi_asr_orchestrator::finish_job(const multi_asr_job_ptr & job) {
    job->req.timings.total_ms = (double) (ggml_time_ms() - job->t_submit_ms);
    {
        std::lock_guard<std::mutex> lock(job->mu);
        job->finished = true;
    }
    job->cv.notify_one();
}

void multi_asr_orchestrator::encode_loop() {
    while (running_.load()) {
        multi_asr_job_ptr job;
        {
            std::unique_lock<std::mutex> lock(intake_mu_);
            intake_cv_.wait(lock, [&] { return !intake_.empty() || !running_.load(); });
            if (!running_.load() && intake_.empty()) {
                return;
            }
            // Order queued jobs by arrival + audio duration; do not wait
            // for additional requests.
            auto next = std::min_element(intake_.begin(), intake_.end(), [](const auto & a, const auto & b) {
                return a->deadline_ms < b->deadline_ms;
            });
            job = *next;
            intake_.erase(next);
        }

        job->req.stage = multi_asr_stage::encoding;
        job->req.timings.queue_ms = (double) (ggml_time_ms() - job->t_submit_ms);
        // Allow ORT encoding to overlap decoder execution.
        const bool encode_ok = encoder_.encode(job->req);
        if (!encode_ok) {
            pending_encode_.fetch_sub(1);
            finish_job(job);
            ready_cv_.notify_one();
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(ready_mu_);
            ready_.push_back(job);
        }
        pending_encode_.fetch_sub(1);
        ready_cv_.notify_one();
    }
}

void multi_asr_orchestrator::decode_loop() {
    while (running_.load()) {
        // Fill free decoder slots from the FIFO as soon as encoded requests
        // arrive. `decoder_.start()` only prepares the request's prompt rows;
        // its first embedding prefill is admitted by the next mixed batch.
        while ((int) active_.size() < decoder_.capacity()) {
            multi_asr_job_ptr job;
            {
                std::lock_guard<std::mutex> lock(ready_mu_);
                if (ready_.empty()) {
                    break;
                }
                job = ready_.front();
                ready_.pop_front();
            }

            int slot_id = -1;
            for (size_t i = 0; i < slot_busy_.size(); ++i) {
                if (!slot_busy_[i]) {
                    slot_id = (int) i;
                    break;
                }
            }
            if (slot_id < 0) {
                std::lock_guard<std::mutex> lock(ready_mu_);
                ready_.push_front(job);
                break;
            }

            slot_busy_[(size_t) slot_id] = true;
            job->slot_id = slot_id;
            if (!decoder_.start(job->req, slot_id)) {
                slot_busy_[(size_t) slot_id] = false;
                finish_job(job);
            } else {
                active_.push_back(job);
            }
        }

        // Every tick admits existing decode rows and the earliest pending audio
        // prefill rows into one mixed llama_batch. This removes the old wave
        // barrier while preserving FIFO admission and decode priority.
        if (decoder_.has_active()) {
            decoder_.step();
            // Keep the completion fence for this experimental run, but do not
            // serialize it with ORT encoder execution.
            llama_synchronize(decoder_.context());
            for (auto it = active_.begin(); it != active_.end();) {
                const auto & job = *it;
                if (job->req.stage == multi_asr_stage::done || job->req.stage == multi_asr_stage::failed) {
                    slot_busy_[(size_t) job->slot_id] = false;
                    finish_job(job);
                    it = active_.erase(it);
                } else {
                    ++it;
                }
            }
            continue;
        }

        if (active_.empty()) {
            std::unique_lock<std::mutex> lock(ready_mu_);
            ready_cv_.wait(lock, [&] { return !running_.load() || !ready_.empty(); });
            if (!running_.load()) {
                return;
            }
        }
    }
}
